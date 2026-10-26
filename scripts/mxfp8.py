"""GEMM MXFP8 (FP8 e4m3, mỗi khối 32 phần tử một scale e8m0) cho trọng số ĐÓNG BĂNG của các lớp LoRA.

RTX 5060 Laptop (sm120): bf16 cộng dồn fp32 chạy ~37 TFLOPS (trần), MXFP8 qua torch._scaled_mm ~115-123 TFLOPS
(đo 2026-10-20). FP8 theo hàng (rowwise) không chạy trên sm120, theo tensor chỉ ~71 TFLOPS và sai số kém hơn khi có
kênh ngoại lai. Chỉ đổi GEMM gốc x @ W^T (forward, tính lại khi checkpoint) và dy @ W (lan ngược ra x); nhánh LoRA,
attention, DeltaNet, CE vẫn bf16. Sai số mỗi GEMM ~3.7% tương đối (bf16 0.17%) -> KHÔNG còn chính xác như bf16:
chỉ dùng sau khi A/B cho thấy chất lượng không giảm.

Lượng tử hoá đầu vào bằng kernel Triton tự viết (mx_quant): đọc bf16 một lần, ghi fp8 + scale đã xếp theo layout
khối 128x4 của cuBLAS trong cùng kernel (bản torchao eager + compile trước đó chạy ~80 GB/s, chiếm ~1.2 s mỗi 8 hội
thoại). Phép tính giống hệt torchao `_to_mx_rceil` (chia 448, cvt.rp.satfinite.ue8m0, nhân nghịch đảo, kẹp ±448).
Đầu vào chung của qkv/z, q/k/v, gate/up chỉ lượng tử một lần (_quant_act nhớ tensor vừa lượng tử).

Trọng số bf16 được chuyển sang RAM trong lúc train (bản MX hai chiều chiếm ~bằng chỗ bf16 trên VRAM);
disable_mxfp8() đưa về lại trước khi merge.

Lan ngược của nhóm qkv|z, q|k|v (2026-10-23): Wᵀ lưu một bản ghép theo nhóm (_Cat), dy của các nhánh lượng tử thẳng
vào một bộ đệm ghép (lora_bwd, kèm d(xa)/dB của LoRA trong cùng lượt đọc) -> một GEMM dx. Xem fused_linear.

gate_proj + up_proj của cùng một MLP lượng tử CHUNG: W_gu = [W_gate; W_up] [2N, K] và W_guᵀ [K, 2N] (fused_mlp.py chạy
một GEMM forward và một GEMM dx K = 2N). Khối 32 không vắt qua ranh giới gate/up nên fp8 + scale giống từng byte bản
riêng; _mx_w/_mx_ws của gate/up là view vào bản chung, bản chuyển vị riêng (đường cũ) cắt ra khi cần (mx_wt) -> VRAM
không tăng. MXFP8_FUSE_GU=0: lượng tử riêng từng lớp như trước.
"""
import os
import types
import weakref

import torch
import triton
import triton.language as tl
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import to_blocked

F8, E8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
STATS = {"quant": 0, "reuse": 0}  # số lần lượng tử hoá đầu vào / dùng lại (để kiểm tra)


def quant_ref(x2d):
    """Bản tham chiếu (torchao eager, RCEIL): dùng cho trọng số (một lần mỗi lượt train) và để kiểm tra kernel."""
    s, q = to_mx(x2d.contiguous(), F8, 32, scaling_mode=ScaleCalculationMode.RCEIL)
    return q, to_blocked(s.view(x2d.shape[0], x2d.shape[1] // 32)).view(E8)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, n_rows, n_cols, n_col_tiles, ROWS: tl.constexpr):
    # Một chương trình = một ô 128 hàng x 128 cột = đúng một khối scale 128x4 của layout cuBLAS
    # (https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROWS + tl.arange(0, ROWS)[:, None]
    cols = pid_c * 128 + tl.arange(0, 128)[None, :]
    offs = rows.to(tl.int64) * n_cols + cols
    mask = (rows < n_rows) & (cols < n_cols)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (ROWS * 4, 32))
    amax = tl.max(tl.abs(xb), axis=1)
    nan_row = tl.max((xb != xb).to(tl.int32), axis=1)
    amax = tl.where(nan_row > 0, float("nan"), amax)
    descale = amax / 448.0
    e = tl.inline_asm_elementwise(asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;", constraints="=h,r",
                                  args=[descale], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)
    rcp = tl.where(e == 255, descale, tl.where(e == 254, 2.0 ** -127,
                   tl.where(e == 0, 1.0, tl.exp2(127.0 - e.to(tl.float32)))))
    q = tl.clamp(xb * rcp[:, None], -448.0, 448.0)
    q = tl.reshape(q, (ROWS, 128)).to(tl.float8e4nv)
    tl.store(q_ptr + offs, q, mask=mask)
    # scale: vị trí trong khối 128x4 là (r % 32) * 16 + (r // 32) * 4 + c; hàng đệm ghi 0 như to_blocked
    e2 = tl.reshape(e, (ROWS, 4))
    r = tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, 4)[None, :]
    e2 = tl.where(pid_r * ROWS + r < n_rows, e2, 0)
    dest = (pid_r * n_col_tiles + pid_c) * 512 + (r % 32) * 16 + (r // 32) * 4 + c
    tl.store(s_ptr + dest, e2)


def mx_quant(x2d):
    """[R, K] bf16 liền khối (K chia hết 128) -> (fp8 [R, K], scale e8m0 phẳng theo layout khối của cuBLAS)."""
    R, K = x2d.shape
    assert x2d.dtype == torch.bfloat16 and x2d.is_contiguous() and K % 128 == 0, (x2d.dtype, x2d.shape)
    q = torch.empty((R, K), dtype=F8, device=x2d.device)
    nr, nc = triton.cdiv(R, 128), K // 128
    s = torch.empty(nr * nc * 512, dtype=torch.uint8, device=x2d.device)
    _quant_kernel[(nr, nc)](x2d, q, s, R, K, nc, ROWS=128, num_warps=4)
    return q, s.view(E8)


# ---------------------------------------------------------------- lan ngược LoRA gộp (lora-bwd-fused + mx-dx-kconcat)
# Trước đây mỗi nhánh LoRA có trọng số gốc MX đọc dy [T, N] ba lần (lượng tử MX, dB = dyᵀ·xa, d(xa) = dy·B), cả ba
# đã ở trần băng thông (~330 GB/s) nên chỉ bớt được bằng cách bớt lượt đọc. Kernel dưới đọc mỗi ô dy MỘT lần rồi làm
# cả ba việc; fp8 + scale ghi thẳng vào bộ đệm ghép theo K của cả nhóm (qkv|z, q|k|v), để dx của nhóm là MỘT
# _scaled_mm với Wᵀ ghép (cộng fp32 qua các nhánh, bỏ dx.add_ bf16 [T, K]).
# Phần lượng tử chép đúng thân _quant_kernel -> fp8 + scale giống từng byte mx_quant(dy). dB / d(xa) cộng dồn fp32
# (như cuBLAS, chỉ khác thứ tự cộng); phần cộng giữa các chương trình ghi bộ đệm phần rồi cộng theo thứ tự cố định
# (_sum_parts) -> tất định, không dùng atomic.

@triton.jit
def _quant_tile(dy_ptr, dy_stride, q_ptr, q_stride, s_ptr, rows, live, c0, q_c0, tile, n_col_tiles,
                BM: tl.constexpr):
    # = thân _quant_kernel cho ô [BM hàng, 128 cột] bắt đầu ở cột c0 của dy, ghi vào cột q_c0 của q và ô cột `tile`
    # của layout scale khối 128x4; hàng đệm (>= n_rows) ghi scale 0 như mx_quant
    cols = tl.arange(0, 128)
    x = tl.load(dy_ptr + rows[:, None] * dy_stride + (c0 + cols)[None, :], mask=live[:, None], other=0.0)
    x = x.to(tl.float32)
    xb = tl.reshape(x, (BM * 4, 32))
    amax = tl.max(tl.abs(xb), axis=1)
    nan_row = tl.max((xb != xb).to(tl.int32), axis=1)
    amax = tl.where(nan_row > 0, float("nan"), amax)
    descale = amax / 448.0
    e = tl.inline_asm_elementwise(asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;", constraints="=h,r",
                                  args=[descale], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)
    rcp = tl.where(e == 255, descale, tl.where(e == 254, 2.0 ** -127,
                   tl.where(e == 0, 1.0, tl.exp2(127.0 - e.to(tl.float32)))))
    q = tl.clamp(xb * rcp[:, None], -448.0, 448.0)
    q = tl.reshape(q, (BM, 128)).to(tl.float8e4nv)
    tl.store(q_ptr + rows[:, None] * q_stride + (q_c0 + cols)[None, :], q, mask=live[:, None])
    e2 = tl.reshape(e, (BM, 4))
    e2 = tl.where(live[:, None], e2, 0)
    c = tl.arange(0, 4)[None, :]
    r = rows[:, None]
    dest = ((r // 128) * n_col_tiles + tile) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c
    tl.store(s_ptr + dest, e2)


@triton.jit
def _lora_bwd_kernel(dy_ptr, dy_stride, xa_ptr, xa_stride, b_ptr, q_ptr, q_stride, s_ptr, dxa_ptr, dxa_stride,
                     db_ptr, n_rows, N, rch, col_tile_off, n_col_tiles, scale,
                     BM: tl.constexpr, WN: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                     DO_QUANT: tl.constexpr, HAS_LORA: tl.constexpr, DXA_DIRECT: tl.constexpr):
    # Chương trình (pid_r, pid_c): khúc hàng [pid_r*rch, +rch) (bội 128; khúc cuối chạy tới hết hàng đệm của khối
    # 128 để ghi scale 0) x WN cột của nhánh. Mỗi vòng BM hàng: lượng tử từng ô 128 cột (nạp riêng), rồi nạp lại cả
    # ô [BM, WN] (vừa đọc nên trúng cache) cho hai tích: dB [WN, r] cộng dồn trong thanh ghi qua cả khúc (một bộ
    # đệm phần mỗi khúc hàng), d(xa) [BM, r] xong ngay (một bộ đệm phần mỗi khúc cột). Tách hai lần nạp nhanh hơn
    # hẳn dùng chung một ô (đo 2026-10-23, T=6400 N=6144: ~396 µs so với 500-900; riêng lượng tử 360): ô dùng chung
    # buộc giữ cả bản fp32 lẫn toán hạng dot trong thanh ghi -> tràn thanh ghi, ít chương trình mỗi SM.
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r0 = pid_r * rch
    r1 = tl.minimum(r0 + rch, tl.cdiv(n_rows, 128) * 128)
    c0 = pid_c * WN
    cols = c0 + tl.arange(0, WN)
    rr = tl.arange(0, RP)
    if HAS_LORA:
        if R == RP:
            b = tl.load(b_ptr + cols[:, None] * R + rr[None, :])  # B [WN, r]
        else:
            b = tl.load(b_ptr + cols[:, None] * R + rr[None, :], mask=rr[None, :] < R, other=0.0)
        acc = tl.zeros((WN, RP), dtype=tl.float32)
    for m0 in range(r0, r1, BM):
        rows = m0 + tl.arange(0, BM)
        live = rows < n_rows
        if DO_QUANT:
            for k in tl.static_range(WN // 128):
                _quant_tile(dy_ptr, dy_stride, q_ptr, q_stride, s_ptr, rows, live, c0 + k * 128,
                            col_tile_off * 128 + c0 + k * 128, col_tile_off + c0 // 128 + k, n_col_tiles, BM)
        if HAS_LORA:
            xb = tl.load(dy_ptr + rows[:, None] * dy_stride + cols[None, :], mask=live[:, None], other=0.0)
            if R == RP:
                xa = tl.load(xa_ptr + rows[:, None] * xa_stride + rr[None, :], mask=live[:, None], other=0.0)
            else:
                xa = tl.load(xa_ptr + rows[:, None] * xa_stride + rr[None, :],
                             mask=live[:, None] & (rr[None, :] < R), other=0.0)
            acc += tl.dot(tl.trans(xb), xa)             # dB [WN, RP]
            d = tl.dot(xb, b)                           # d(xa) [BM, RP], fp32
            if DXA_DIRECT:  # một khúc cột: khỏi bộ đệm phần, ghi thẳng bf16 vào cột của dS
                if R == RP:
                    dm = live[:, None]
                else:
                    dm = live[:, None] & (rr[None, :] < R)
                tl.store(dxa_ptr + rows[:, None] * dxa_stride + rr[None, :], (d * scale).to(dxa_ptr.dtype.element_ty),
                         mask=dm)
            else:
                tl.store(dxa_ptr + (pid_c * n_rows + rows[:, None]) * RP + rr[None, :], d, mask=live[:, None])
    if HAS_LORA:
        tl.store(db_ptr + (pid_r * N + cols[:, None]) * RP + rr[None, :], acc)


@triton.jit
def _sum_rows(src_ptr, n_parts, n_rows, out_ptr, out_stride, scale, pid, R: tl.constexpr, RP: tl.constexpr,
              BM: tl.constexpr):
    # out[m, :R] = scale * Σ_p src[p, m, :R], cộng theo p tăng dần (tất định), ép về dtype của out
    rows = pid * BM + tl.arange(0, BM)
    rr = tl.arange(0, RP)
    mask = (rows[:, None] < n_rows) & (rr[None, :] < R)
    acc = tl.zeros((BM, RP), dtype=tl.float32)
    for p in range(0, n_parts):
        acc += tl.load(src_ptr + (p * n_rows + rows[:, None]) * RP + rr[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + rows[:, None] * out_stride + rr[None, :], (acc * scale).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sum_parts_kernel(src1, n1, m1, out1, ld1, src2, n2, m2, out2, ld2, scale, g1,
                      R: tl.constexpr, RP: tl.constexpr, BM: tl.constexpr):
    # hai phép cộng phần (d(xa) rồi dB) trong MỘT lần phóng: chương trình < g1 làm việc 1, còn lại việc 2
    pid = tl.program_id(0)
    if pid < g1:
        _sum_rows(src1, n1, m1, out1, ld1, scale, pid, R, RP, BM)
    else:
        _sum_rows(src2, n2, m2, out2, ld2, scale, pid - g1, R, RP, BM)


# Cấu hình (quét 2026-10-23, sweep_lorabwd.py): BM hàng mỗi vòng, WN cột mỗi chương trình (N <= 512: cả N, khỏi bộ
# đệm phần d(xa)), số chương trình nhắm ~_PROGS (kernel chạm trần băng thông; nhiều hơn chỉ thêm bộ đệm phần dB)
_LB = dict(BM=64, WN=256, num_warps=4)
_PROGS = (192, 96)  # (dy nhỏ: nhiều chương trình hơn để giấu độ trễ, dy >= 24M phần tử)


def _sum_parts(jobs, scale, R):
    """jobs: 1-2 cặp (bộ đệm phần [P, M, RP] fp32, đầu ra [M, R] cho phép stride hàng)."""
    BM = 32
    (s1, o1), (s2, o2) = jobs if len(jobs) == 2 else ((jobs[0][0][:0], jobs[0][1]), jobs[0])
    g1 = triton.cdiv(s1.shape[1], BM) if s1.numel() else 0
    _sum_parts_kernel[(g1 + triton.cdiv(s2.shape[1], BM),)](
        s1, s1.shape[0], s1.shape[1], o1, o1.stride(0), s2, s2.shape[0], s2.shape[1], o2, o2.stride(0), scale, g1,
        R=R, RP=s2.shape[2], BM=BM, num_warps=4)


def lora_bwd(dy, xa=None, B=None, scale=1.0, q=None, s=None, col_tile_off=0, dxa_out=None):
    """Một lượt đọc dy [T, N] bf16 (hàng liền khối): nếu q/s khác None thì lượng tử MX vào cột
    col_tile_off*128.. của q [T, ΣN] (scale vào layout khối của q, giống từng byte mx_quant); nếu có LoRA (xa [T, r]
    - cột của xS, B [N, r] bf16 liền khối) thì ghi scale * dy @ B (bf16) vào dxa_out [T, r] (cho phép stride hàng) và
    trả dB = scale * dyᵀ @ xa (fp32: đúng dtype tham số LoRA, autograd khỏi ép; chính xác hơn bản bf16 cũ)."""
    T, N = dy.shape
    assert dy.dtype == torch.bfloat16 and dy.stride(1) == 1 and N % 128 == 0, (dy.dtype, dy.shape, dy.stride())
    has_lora, do_q = xa is not None, q is not None
    if not (has_lora or do_q):
        return None
    assert T * max(dy.stride(0), q.shape[1] if do_q else 0) < 2**31  # chỉ số int32 trong kernel
    BM, WN, nw = _LB["BM"], _LB["WN"], _LB["num_warps"]
    if N <= 512:
        WN = N
    while N % WN:
        WN //= 2
    R = xa.shape[1] if has_lora else 16
    RP = max(16, triton.next_power_of_2(R))
    cs = N // WN
    nb = triton.cdiv(T, 128)
    P = _PROGS if isinstance(_PROGS, int) else _PROGS[T * N >= 24 * 2**20]
    blk = triton.cdiv(nb, max(1, min(nb, round(P / cs))))  # số khối 128 hàng mỗi chương trình
    rs = triton.cdiv(nb, blk)
    dev = dy.device
    db_part = dxa_part = None
    direct = False
    if has_lora:
        assert xa.stride(1) == 1 and B.is_contiguous() and B.shape == (N, R) and B.dtype == torch.bfloat16
        assert dxa_out.shape == (T, R) and dxa_out.stride(1) == 1
        db_part = torch.empty((rs, N, RP), dtype=torch.float32, device=dev)
        direct = cs == 1
        dxa_part = dxa_out if direct else torch.empty((cs, T, RP), dtype=torch.float32, device=dev)
    dummy = dy  # con trỏ không dùng tới (nhánh constexpr tắt)
    _lora_bwd_kernel[(rs, cs)](
        dy, dy.stride(0), xa if has_lora else dummy, xa.stride(0) if has_lora else 0, B if has_lora else dummy,
        q if do_q else dummy, q.shape[1] if do_q else 0, s if do_q else dummy,
        dxa_part if has_lora else dummy, dxa_out.stride(0) if direct else 0,
        db_part if has_lora else dummy, T, N, blk * 128, col_tile_off, q.shape[1] // 128 if do_q else 0,
        float(scale), BM=BM, WN=WN, R=R, RP=RP, DO_QUANT=do_q, HAS_LORA=has_lora, DXA_DIRECT=direct, num_warps=nw)
    if not has_lora:
        return None
    dB = torch.empty((N, R), dtype=torch.float32, device=dev)
    _sum_parts(((db_part, dB),) if direct else ((dxa_part, dxa_out), (db_part, dB)), float(scale), R)
    return dB


# ---------------------------------------------------------------- residual-lora
@triton.jit
def _res_lora_kernel(x_ptr, y_ptr, xa_ptr, xa_stride, b_ptr, out_ptr, n_rows, N, scale,
                     BM: tl.constexpr, BN: tl.constexpr, R: tl.constexpr, RP: tl.constexpr):
    # out = bf16(x + bf16(y + scale * xa @ Bᵀ)): đúng hai lần làm tròn của đường cũ - addmm_ (cộng fp32 rồi làm
    # tròn một lần) rồi phép cộng residual bf16; chỉ khác thứ tự cộng r tích (đo: trùng từng bit 100%)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rr = tl.arange(0, RP)
    live = rows < n_rows
    xa = tl.load(xa_ptr + rows[:, None] * xa_stride + rr[None, :], mask=live[:, None] & (rr[None, :] < R), other=0.0)
    b = tl.load(b_ptr + cols[:, None] * R + rr[None, :], mask=rr[None, :] < R, other=0.0)
    acc = tl.dot(xa, tl.trans(b))
    offs = rows[:, None] * N + cols[None, :]
    y = tl.load(y_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
    t = (y + scale * acc).to(tl.bfloat16).to(tl.float32)
    xv = tl.load(x_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (xv + t).to(tl.bfloat16), mask=live[:, None])


def res_lora(x, y, xa, B, scale):
    """x + (y + scale * xa @ Bᵀ) cho x, y [T, N] bf16 liền khối, xa [T, r] (stride hàng tuỳ ý), B [N, r] liền khối."""
    T, N = x.shape
    assert T * N < 2**31 and N % 128 == 0
    out = torch.empty_like(x)
    R = xa.shape[1]
    _res_lora_kernel[(triton.cdiv(T, 64), N // 128)](x, y, xa, xa.stride(0), B, out, T, N, float(scale), BM=64,
                                                    BN=128, R=R, RP=max(16, triton.next_power_of_2(R)), num_warps=4)
    return out


_last = [None, 0, None, None]  # (weakref tensor, _version, q, s) của lần lượng tử gần nhất


def _quant_act(x):
    """Lượng tử hoá đầu vào một lớp tuyến tính; nếu vẫn là chính tensor vừa lượng tử (qkv/z, q/k/v, gate/up nhận
    cùng một đầu vào) thì dùng lại. weakref: không giữ x sống thêm (VRAM sát trần - từng OOM ở đỉnh 6.2 GiB);
    x đã bị giải phóng thì ref() trả None nên không thể nhầm với tensor mới cấp lại cùng địa chỉ."""
    if _last[0] is not None and _last[0]() is x and _last[1] == x._version:
        STATS["reuse"] += 1
        return _last[2], _last[3]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.bfloat16 or not x2.is_contiguous():
        x2 = x2.to(torch.bfloat16).contiguous()
    q, s = mx_quant(x2)
    STATS["quant"] += 1
    _last[:] = [weakref.ref(x), x._version, q, s]
    return q, s


class _MXLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, s, lin):
        out = torch._scaled_mm(q, lin._mx_w.t(), s, lin._mx_ws, out_dtype=torch.bfloat16)
        ctx.lin = lin
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        lin = ctx.lin
        n = dy.shape[-1]
        dy2 = dy.reshape(-1, n)
        if dy2.dtype != torch.bfloat16 or not dy2.is_contiguous():
            dy2 = dy2.to(torch.bfloat16).contiguous()
        q, s = mx_quant(dy2)
        STATS["quant"] += 1
        wt, wts = wt_of(lin)
        dx = torch._scaled_mm(q, wt, s, wts, out_dtype=torch.bfloat16)
        return dx.view(*dy.shape[:-1], dx.shape[-1]), None, None, None


@torch.compiler.disable
def _forward(self, x):
    # Ngoài mọi vùng torch.compile: khi MLP đã compile của Unsloth trace xuyên qua hàm này, lần chạy đầu tiên của
    # graph đó cho kết quả khác (loss 0.614 thay vì 0.636 ở mọi bước sau và ở bản không compile) - không rõ vì sao
    q, s = _quant_act(x)
    return _MXLinear.apply(x, q, s, self)


_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"}


def _base_linears(model):
    """Các lớp tuyến tính gốc đóng băng của 24 khối: base_layer của lớp đã bọc LoRA và cả lớp không có LoRA
    (vd. --no-gdn). Bỏ in_proj_b/a (nhỏ), lm_head."""
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or m.weight.requires_grad or ".layers." not in name:
            continue
        parts = name.split(".")
        leaf = parts[-2] if parts[-1] == "base_layer" else parts[-1]
        if leaf in _NAMES:
            yield name, m


# Nhóm Linear chung đầu vào mà _Group gọi cùng lúc (đúng thứ tự group_linear trong shared_prefix): Wᵀ của nhóm lưu
# MỘT bản ghép theo N (khối 32 không vắt qua ranh giới nhánh vì mọi N chia hết 128 -> fp8 + scale của từng nhánh
# y hệt bản riêng) để dx của cả nhóm là một _scaled_mm. Bản Wᵀ riêng của từng lớp là view cột của bản ghép (cuBLAS
# nhận ld > số hàng), scale riêng tạo khi cần lần đầu: VRAM không tăng. MLP (gate/up/down) để mlp-fused lo, chưa ghép.
_CATS = (("in_proj_qkv", "in_proj_z"), ("q_proj", "k_proj", "v_proj"))
# Lớp đi đường lan ngược gộp (lora_bwd) trong fused_linear._Group; thêm "gate_proj", "up_proj", "down_proj" (và
# ("gate_proj", "up_proj") vào _CATS) để dùng cho cả MLP nếu không làm mlp-fused.
FUSED_BWD = {"in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}


class _Cat:
    """Wᵀ ghép [K, ΣN] fp8 + scale (layout khối 128x4 của cuBLAS) của một nhóm lớp."""

    def __init__(self, lins):
        w = torch.cat([lin.weight.data for lin in lins], 0)
        self.qt, self.st = quant_ref(w.t().contiguous())
        del w
        self.offs = [0]
        for lin in lins:
            self.offs.append(self.offs[-1] + lin.weight.shape[0])
        self.n = len(lins)
        self._sub = {}

    def range(self, a, b):
        """(mat2 [N_a+..+N_{b-1}, K] cột-chính, scale) cho các lớp a..b-1 liên tiếp. Scale của một dải cột là các
        ô 128x4 tương ứng của mỗi khối 128 hàng (ô xếp theo khối hàng rồi khối cột) -> chép lát, giống từng byte
        quant_ref của riêng dải đó."""
        c0, c1 = self.offs[a], self.offs[b]
        mat2 = self.qt[:, c0:c1].t()
        if (a, b) == (0, self.n):
            return mat2, self.st
        sc = self._sub.get((a, b))
        if sc is None:
            K, ntot = self.qt.shape
            blk = self.st.view(torch.uint8).view(triton.cdiv(K, 128), ntot // 128, 512)
            sc = self._sub[(a, b)] = blk[:, c0 // 128:c1 // 128].contiguous().view(-1).view(E8)
        return mat2, sc


def is_mx(lin):
    return getattr(lin, "_mx_w", None) is not None


def mx_wt(lin):
    """(fp8 Wᵀ [K, N], scale) cho dx = dy @ W. gate/up ghép: view cột của W_guᵀ (cuBLAS nhận ld = 2N) + scale cắt
    ra liền khối (layout khối 128x4 xếp theo hàng-khối nên phần của một nửa không liền), tạo lần đầu cần rồi giữ."""
    gu = lin.__dict__.get("_mx_gu")
    if gu is None:
        m, s = wt_of(lin)
        return m.t(), s
    hold, i = gu
    if hold.half_t[i] is None:
        n, k = hold.n, hold.wt.shape[0]
        s3 = hold.wts.view(torch.uint8).view(k // 128, 2 * n // 128, 512)
        hold.half_t[i] = (hold.wt[:, i * n:(i + 1) * n],
                          s3[:, i * (n // 128):(i + 1) * (n // 128)].reshape(-1).view(E8))
    return hold.half_t[i]


def wt_of(lin):
    """(mat2, scale) của dy @ W cho một lớp (mat2 = Wᵀ fp8 dạng [N, K] cột-chính)."""
    c = lin.__dict__.get("_mx_cat")
    if c is not None:
        return c[0].range(c[1], c[1] + 1)
    if "_mx_gu" in lin.__dict__:
        wt, s = mx_wt(lin)
        return wt.t(), s
    return lin._mx_wt.t(), lin._mx_wts


def group_wt(lins):
    """(mat2 [ΣN, K], scale) cho dx = [dy_1|dy_2|..] @ [W_1; W_2; ..] của các lớp liên tiếp trong một bản ghép (hoặc
    một lớp lẻ); None nếu không có bản ghép phù hợp."""
    c = lins[0].__dict__.get("_mx_cat")
    if c is None:
        return wt_of(lins[0]) if len(lins) == 1 else None
    cat, k = c
    for d, lin in enumerate(lins):
        if lin.__dict__.get("_mx_cat") != (cat, k + d):
            return None
    return cat.range(k, k + len(lins))


def _leaf_parent(name):
    parts = name.split(".")
    if parts[-1] == "base_layer":
        return parts[-2], ".".join(parts[:-2])
    return parts[-1], ".".join(parts[:-1])


FUSE_GU = os.environ.get("MXFP8_FUSE_GU", "1") != "0"


def enable_mxfp8(model):
    lins = list(_base_linears(model))
    fam = {}  # module cha -> {tên lá: Linear}
    for name, lin in lins:
        assert lin.bias is None
        w = lin.weight.data
        assert w.is_cuda and w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0, w.shape
        leaf, parent = _leaf_parent(name)
        fam.setdefault(parent, {})[leaf] = lin
    # Trước vòng dưới (cần trọng số bf16 của cả nhóm còn trên GPU): Wᵀ ghép của qkv|z, q|k|v (lan ngược gộp) và
    # W_gu, W_guᵀ chung của gate|up (fused_mlp)
    in_cat, paired = set(), set()
    for d in fam.values():
        for names in _CATS:
            if all(k in d for k in names):
                grp = [d[k] for k in names]
                cat = _Cat(grp)
                for k, lin in enumerate(grp):
                    lin._mx_cat = (cat, k)
                    in_cat.add(id(lin))
        gb, ub = d.get("gate_proj"), d.get("up_proj")
        if FUSE_GU and gb is not None and ub is not None and gb.weight.shape == ub.weight.shape:
            w = torch.cat([gb.weight.data, ub.weight.data])           # [2N, K]
            n = gb.weight.shape[0]
            hold = types.SimpleNamespace(n=n, half_t=[None, None])
            hold.w, hold.ws = quant_ref(w)
            hold.wt, hold.wts = quant_ref(w.t().contiguous())
            del w
            ns = hold.ws.numel() // 2                                 # scale forward xếp theo hàng-khối: nửa đầu = gate
            for i, lin in enumerate((gb, ub)):
                lin._mx_w, lin._mx_ws = hold.w[i * n:(i + 1) * n], hold.ws[i * ns:(i + 1) * ns]
                lin._mx_gu = (hold, i)
                paired.add(id(lin))
    for name, lin in lins:
        w = lin.weight.data
        if id(lin) not in paired:
            lin._mx_w, lin._mx_ws = quant_ref(w.contiguous())            # forward:  x @ W^T, khối theo K
            if id(lin) not in in_cat:
                lin._mx_wt, lin._mx_wts = quant_ref(w.t().contiguous())  # backward: dy @ W, khối theo N
        lin._mx_fbwd = _leaf_parent(name)[0] in FUSED_BWD
        lin._bf16_cpu = w.to("cpu", copy=True)
        lin.weight.data = torch.empty((0,), dtype=w.dtype, device=w.device)  # giữ dtype/device cho code đọc nó
        lin.forward = types.MethodType(_forward, lin)
    torch.cuda.empty_cache()
    print(f"MXFP8: {len(lins)} lớp tuyến tính đóng băng chạy GEMM FP8 (trọng số bf16 gửi sang RAM;"
          f" {len(in_cat)} lớp có Wᵀ ghép theo nhóm, {len(paired) // 2} cặp gate/up ghép)", flush=True)
    return len(lins)


def disable_mxfp8(model):
    for _, lin in _base_linears(model):
        if hasattr(lin, "_bf16_cpu"):
            lin.weight.data = lin._bf16_cpu.to(lin.weight.device)
            for a in ("_mx_w", "_mx_ws", "_mx_wt", "_mx_wts", "_mx_cat", "_mx_gu", "_mx_fbwd", "_bf16_cpu", "forward"):
                if a in lin.__dict__:
                    delattr(lin, a)
    _last[:] = [None, 0, None, None]
    torch.cuda.empty_cache()
