"""MLP gộp trọn (gate/up + SwiGLU + LoRA + lượng tử MX + dB/dxa/dA) trong MỘT autograd.Function, 2 kernel Triton.

Đường cũ (shared_prefix._mlp: _Group gate/up -> _SWIGLU compile -> _Group down) đọc/ghi [T, 6144] nhiều lượt:
forward addmm_ LoRA-B của gate và up (mỗi cái đọc + ghi [T, 6144]), SwiGLU (đọc g, u, ghi h), lượng tử h, xS = h @ A_downᵀ
(đọc lại h); backward dx.addmm_ của down (RMW [T, 6144]), SwiGLU backward (đọc g, u, dh, ghi dg, du), lượng tử dg, du,
dB/dxa của gate/up (đọc lại dg, du), dA của down (đọc lại h), dx.add_ của gate+up. Ở đây:
  forward : một GEMM MX gate|up (W_gu [12288, 2048], mxfp8.enable_mxfp8 ghép sẵn) -> kernel _fwd đọc gu một lần: cộng
            nhánh LoRA-B (tl.dot hạng r), g/u làm tròn bf16 như epilogue addmm_ cũ, h = silu(g)*u như kernel inductor,
            ghi thẳng fp8 + scale MX của h (đúng thuật toán mxfp8._quant_kernel) và xS_down = h @ A_downᵀ.
            KHÔNG lưu h (backward tính lại từ gu) -> lớp giữ (không checkpoint) bớt 12 KB/token.
  backward: kernel _bwd đọc gu + dh_mx một lần: tính lại g, u, h; dh = dh_mx + dxa_down @ A_down (thay dx.addmm_);
            dg/du đúng biểu thức kernel inductor; lượng tử dg|du vào MỘT bộ đệm fp8 [T, 12288] cho một GEMM dx
            (K = 12288, bỏ dx.add_); cùng lúc dxa của gate/up, dB của gate/up, dA của down và xS_down cho dB_down
            (cộng dồn fp32 theo khối, cộng các phần bằng .sum(0) -> tất định, không atomic).
need_out=False (tính lại checkpoint / bước C): chỉ x @ A_gu, lượng tử x, GEMM gate|up (gu là thứ backward cần) - bỏ
cả kernel fwd lẫn GEMM down, vì xS_down cũng được kernel bwd tính lại (chỉ khác thứ tự cộng fp32 so với bản forward).
Toán học y hệt đường cũ, chỉ khác thứ tự cộng fp32 (tích hạng r, các khối), dB/dA giữ fp32 thay vì làm tròn bf16,
và dx cộng fp32 qua cả gate lẫn up rồi làm tròn một lần (chính xác hơn). Chỉ dùng khi gate/up/down đều là LoRA + MX và
tính bf16; ngoài ra mlp() trả None để shared_prefix chạy đường cũ.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # cùng exp với kernel inductor của _SWIGLU cũ

import mxfp8
from fused_linear import STATS, _cast, _part

_F8, _E8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
ENABLED = [True]  # tắt (False) để so với đường cũ trong cùng tiến trình
ACTIVE = [False]  # đã chạy đường gộp ít nhất một lần (shared_prefix._auto_keep dùng để bớt byte activation mỗi lớp)
SAVED_LESS = 12288  # byte/token mỗi lớp giữ không còn lưu so với đường cũ (h [T, 6144] bf16)


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _mx_store(x, rows, col0, q_ptr, q_stride, s_ptr, n_col_tiles, T, ROWS: tl.constexpr, BN: tl.constexpr):
    """Thân mxfp8._quant_kernel cho một ô [ROWS, BN] fp32 (giá trị bf16) bắt đầu ở cột col0 của bộ đệm đích: fp8 vào
    q[rows, col0:col0+BN] và scale e8m0 vào khối 128x4 (rows // 128, col0 // 128) của layout cuBLAS; hàng >= T ghi
    scale 0 như to_blocked. amax lan truyền NaN trong MỘT phép rút gọn (= amax rồi gán NaN nếu khối có NaN như
    _quant_kernel)."""
    xb = tl.reshape(x, (ROWS * (BN // 32), 32))
    amax = tl.reduce(tl.abs(xb), 1, _max_nan)
    descale = amax / 448.0
    e = tl.inline_asm_elementwise(asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;", constraints="=h,r",
                                  args=[descale], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)
    rcp = tl.where(e == 255, descale, tl.where(e == 254, 2.0 ** -127,
                   tl.where(e == 0, 1.0, tl.exp2(127.0 - e.to(tl.float32)))))
    q = tl.clamp(xb * rcp[:, None], -448.0, 448.0)
    q = tl.reshape(q, (ROWS, BN)).to(tl.float8e4nv)
    rmask = rows < T
    tl.store(q_ptr + rows[:, None] * q_stride + (col0 + tl.arange(0, BN))[None, :], q, mask=rmask[:, None])
    e2 = tl.reshape(e, (ROWS, BN // 32))
    e2 = tl.where(rmask[:, None], e2, 0)
    rr = (rows % 128)[:, None]
    c = ((col0 % 128) // 32 + tl.arange(0, BN // 32))[None, :]
    dest = ((rows // 128)[:, None] * n_col_tiles + col0 // 128) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + c
    tl.store(s_ptr + dest, e2)


@triton.jit
def _gu_tile(gb, ub, xa_g, xa_u, bgT, buT, s_g, s_u):
    """g, u của một ô: đầu ra GEMM MX (gb, ub bf16) + s * xa @ Bᵀ cộng fp32 rồi làm tròn bf16 một lần (= addmm_ beta=1
    của đường cũ; kiểm: fp8 của h trùng từng byte)."""
    g = (gb.to(tl.float32) + s_g * tl.dot(xa_g, bgT)).to(tl.bfloat16).to(tl.float32)
    u = (ub.to(tl.float32) + s_u * tl.dot(xa_u, buT)).to(tl.bfloat16).to(tl.float32)
    return g, u


# Bố cục: Triton thấy các tl.dot nối nhau qua luồng dữ liệu (cùng vùng) thì dùng CHUNG một bố cục mma, với mọi warp xếp
# dọc chiều dài hơn của dot được xử lý trước ([nw, 1] nếu chiều 0 >= chiều 1, không thì [1, nw]). Một dot "lệch" chiều
# (vd. kết quả [ROWS, R] trong khi ô chính [ROWS, 128]) làm ô chính bị nhân bản nhiều lần trong thanh ghi (đo: 255 thanh
# ghi, tràn, chậm 2-4 lần). Nên mỗi kernel viết MỌI dot cùng một hướng:
#   _fwd_kernel: chiều 0 < chiều 1 (tích hạng r ở dạng chuyển vị [R, ROWS]) -> warp dọc cột, ROWS 32;
#   _bwd_kernel: chiều 0 >= chiều 1 -> warp dọc hàng, ROWS = 16 * num_warps (mỗi warp đủ 16 hàng x BN cột: rút gọn
#                amax của khối 32 nằm gọn trong warp, không qua shared memory + bar.sync).
@triton.jit
def _fwd_kernel(gu_ptr, xs_ptr, bg_ptr, bu_ptr, ad_ptr, q_ptr, s_ptr, xsd_ptr, T, s_g, s_u, TPP,
                N: tl.constexpr, R: tl.constexpr, ROWS: tl.constexpr, BN: tl.constexpr, WRITE_Q: tl.constexpr):
    # Chương trình (khối ROWS hàng, nhóm TPP ô cột BN): xS_down cộng dồn trong thanh ghi qua các ô của nhóm
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROWS + tl.arange(0, ROWS)
    rmask = rows < T
    rk = tl.arange(0, R)
    xoff = rows[:, None] * (2 * R) + rk[None, :]
    xa_g = tl.load(xs_ptr + xoff, mask=rmask[:, None], other=0.0)
    xa_u = tl.load(xs_ptr + xoff + R, mask=rmask[:, None], other=0.0)
    accT = tl.zeros((R, ROWS), dtype=tl.float32)
    for t in range(TPP):
        j = pid_c * TPP + t
        cols = j * BN + tl.arange(0, BN)
        bgT = tl.load(bg_ptr + cols[None, :] * R + rk[:, None])   # [R, BN]
        buT = tl.load(bu_ptr + cols[None, :] * R + rk[:, None])
        off = rows[:, None] * (2 * N) + cols[None, :]
        gb = tl.load(gu_ptr + off, mask=rmask[:, None], other=0.0)
        ub = tl.load(gu_ptr + off + N, mask=rmask[:, None], other=0.0)
        g, u = _gu_tile(gb, ub, xa_g, xa_u, bgT, buT, s_g, s_u)
        # đúng biểu thức triton_poi_fused_mul_silu_0: silu = g / (exp(-g) + 1), h = silu * u, lưu bf16
        h = ((g / (libdevice.exp(-g) + 1.0)) * u).to(tl.bfloat16)
        if WRITE_Q:
            _mx_store(h.to(tl.float32), rows, j * BN, q_ptr, N, s_ptr, N // 128, T, ROWS, BN)
        ad_t = tl.load(ad_ptr + rk[:, None] * N + cols[None, :])  # [R, BN]
        accT = tl.dot(ad_t, tl.trans(h), accT)                   # xS_downᵀ [R, ROWS]
    tl.store(xsd_ptr + (pid_c * T + rows)[None, :] * R + rk[:, None], accT, mask=rmask[None, :])


@triton.jit
def _bwd_kernel(gu_ptr, xs_ptr, dhm_ptr, dxd_ptr, bg_ptr, bu_ptr, ad_ptr, q_ptr, s_ptr,
                dxa_ptr, dbg_ptr, dbu_ptr, dad_ptr, T, Tpad, s_g, s_u, SB, TPG,
                N: tl.constexpr, R: tl.constexpr, ROWS: tl.constexpr, BN: tl.constexpr, DO_QUANT: tl.constexpr):
    # Chương trình (siêu khối SB hàng, nhóm TPG ô cột BN). Vòng ngoài theo ô cột: dB/dA của ô cộng dồn trong thanh ghi
    # qua mọi hàng của siêu khối rồi ghi một phần [số siêu khối, ...]; vòng trong theo khối ROWS hàng: dxa_gate,
    # dxa_up và xS_down (= h @ A_downᵀ, tính lại ở đây nên forward không phải lưu nó) cộng qua các ô của nhóm bằng
    # đọc-cộng-ghi vào phần riêng [T, 3R] của chương trình (nằm trong L2). Không atomic -> tất định.
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    r_begin = pid_s * SB
    r_end = tl.minimum(r_begin + SB, Tpad)  # tới hết khối 128 cuối: scale hàng đệm phải ghi 0
    rk = tl.arange(0, R)
    for t in range(TPG):
        j = pid_c * TPG + t
        cols = j * BN + tl.arange(0, BN)
        bgT = tl.load(bg_ptr + cols[None, :] * R + rk[:, None])    # [R, BN]
        buT = tl.load(bu_ptr + cols[None, :] * R + rk[:, None])
        ad_t = tl.load(ad_ptr + rk[:, None] * N + cols[None, :])   # [R, BN]
        acc_bg = tl.zeros((BN, R), dtype=tl.float32)                # dB_gate của ô
        acc_bu = tl.zeros((BN, R), dtype=tl.float32)
        acc_ad = tl.zeros((BN, R), dtype=tl.float32)                # dA_downᵀ của ô
        for r0 in range(r_begin, r_end, ROWS):
            rows = r0 + tl.arange(0, ROWS)
            rmask = rows < T
            # mọi lần nạp global ở đầu vòng (độc lập nhau) để độ trễ chồng lên nhau
            xoff = rows[:, None] * (2 * R) + rk[None, :]
            xa_g = tl.load(xs_ptr + xoff, mask=rmask[:, None], other=0.0)
            xa_u = tl.load(xs_ptr + xoff + R, mask=rmask[:, None], other=0.0)
            dxd = tl.load(dxd_ptr + rows[:, None] * R + rk[None, :], mask=rmask[:, None], other=0.0)
            off = rows[:, None] * (2 * N) + cols[None, :]
            gb = tl.load(gu_ptr + off, mask=rmask[:, None], other=0.0)
            ub = tl.load(gu_ptr + off + N, mask=rmask[:, None], other=0.0)
            dhm = tl.load(dhm_ptr + rows[:, None] * N + cols[None, :], mask=rmask[:, None], other=0.0)
            poff = (pid_c * T + rows)[:, None] * (3 * R) + rk[None, :]
            pmask = rmask[:, None] & (t > 0)
            pg = tl.load(dxa_ptr + poff, mask=pmask, other=0.0)        # phần dxa / xS_down của các ô trước trong nhóm
            pu = tl.load(dxa_ptr + poff + R, mask=pmask, other=0.0)
            px = tl.load(dxa_ptr + poff + 2 * R, mask=pmask, other=0.0)
            g, u = _gu_tile(gb, ub, xa_g, xa_u, bgT, buT, s_g, s_u)
            silu = g / (libdevice.exp(-g) + 1.0)
            h = (silu * u).to(tl.bfloat16)
            # dh = bf16(dh_mx + dxa_down @ A_down): như dx.addmm_ của down ở đường cũ
            dh = (dhm.to(tl.float32) + tl.dot(dxd, ad_t)).to(tl.bfloat16).to(tl.float32)
            # đúng biểu thức triton_poi_fused_mul_silu_silu_backward_0 (du = dh*silu; dg = dh*u*sig*(g*(1-sig)+1))
            du = (dh * silu).to(tl.bfloat16)
            sig = tl.sigmoid(g)
            dg = (((dh * u) * sig) * (g * (1.0 - sig) + 1.0)).to(tl.bfloat16)
            if DO_QUANT:  # dg vào cột j*BN, du vào cột N + j*BN của bộ đệm [T, 2N] (một GEMM dx K = 2N)
                _mx_store(dg.to(tl.float32), rows, j * BN, q_ptr, 2 * N, s_ptr, 2 * N // 128, T, ROWS, BN)
                _mx_store(du.to(tl.float32), rows, N + j * BN, q_ptr, 2 * N, s_ptr, 2 * N // 128, T, ROWS, BN)
            acc_bg = tl.dot(tl.trans(dg), xa_g, acc_bg)
            acc_bu = tl.dot(tl.trans(du), xa_u, acc_bu)
            acc_ad = tl.dot(tl.trans(h), dxd, acc_ad)
            pg = tl.dot(dg, tl.trans(bgT), pg)                      # dxa_gate [ROWS, R] cộng dồn qua các ô
            pu = tl.dot(du, tl.trans(buT), pu)
            px = tl.dot(h, tl.trans(ad_t), px)                      # xS_down = h @ A_downᵀ (cho dB_down)
            tl.store(dxa_ptr + poff, pg, mask=rmask[:, None])
            tl.store(dxa_ptr + poff + R, pu, mask=rmask[:, None])
            tl.store(dxa_ptr + poff + 2 * R, px, mask=rmask[:, None])
        boff = (pid_s * N + cols)[:, None] * R + rk[None, :]
        tl.store(dbg_ptr + boff, acc_bg)
        tl.store(dbu_ptr + boff, acc_bu)
        tl.store(dad_ptr + (pid_s * R + rk)[None, :] * N + cols[:, None], acc_ad)


# Cấu hình kernel (quét 2026-10-23 trên RTX 5060 Laptop, GPU dùng chung nên lấy min nhiều vòng xen kẽ):
#   fwd ROWS 32, BN 128, 4 warp: T 2400 0.24 ms (~310 GB/s, gần trần); 8 warp / BN 64 / ROWS 64 chậm hơn 10-50%.
#   bwd BN 64, 4 warp (ROWS 64): T 2400 0.57 ms, T 6400 1.32 ms (~230 GB/s; bị giới hạn bởi số lệnh + bar.sync của các
#   lần đổi bố cục, không phải băng thông). BN 128 tràn thanh ghi; 8 warp chậm hơn ~10%; maxnreg 128 / num_stages 2 tràn.
FWD_ROWS, FWD_BN, FWD_WARPS, FWD_STAGES, FWD_MAXNREG = 32, 128, 4, 1, None
BWD_BN, BWD_WARPS, BWD_STAGES, BWD_MAXNREG = 64, 4, 1, None  # ROWS = 16 * BWD_WARPS
_SMS = [None]
_LAST = {}  # kernel vừa chạy (xem n_regs / n_spills khi chỉnh cấu hình)


def _n_sm():
    if _SMS[0] is None:
        _SMS[0] = torch.cuda.get_device_properties(0).multi_processor_count
    return _SMS[0]


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def mlp_fwd(gu, xs, bg, bu, ad, s_g, s_u, write_q):
    """gu [T, 2N] bf16, xs [T, 2R] bf16 (xa_gate | xa_up), bg/bu [N, R], ad [R, N] bf16 -> (q_h, s_h, xS_down bf16)."""
    T, N2 = gu.shape
    N, R = N2 // 2, ad.shape[0]
    nt = N // FWD_BN
    nrb = triton.cdiv(triton.cdiv(T, 128) * 128, FWD_ROWS)  # phủ tới hết khối 128 cuối (scale hàng đệm = 0)
    cs = next((d for d in _divisors(nt) if nrb * d >= 4 * _n_sm()), nt)
    if write_q:
        q = torch.empty((T, N), dtype=_F8, device=gu.device)
        s = torch.empty(triton.cdiv(T, 128) * (N // 128) * 512, dtype=torch.uint8, device=gu.device)
    else:
        q = s = gu  # không ghi (WRITE_Q=False)
    part = torch.empty((cs, T, R), dtype=torch.float32, device=gu.device)
    _LAST["fwd"] = _fwd_kernel[(nrb, cs)](gu, xs, bg, bu, ad, q, s, part, T, float(s_g), float(s_u), nt // cs,
                           N=N, R=R, ROWS=FWD_ROWS, BN=FWD_BN, WRITE_Q=write_q, num_warps=FWD_WARPS, num_stages=FWD_STAGES,
                                         **({"maxnreg": FWD_MAXNREG} if FWD_MAXNREG else {}))
    xsd = (part[0] if cs == 1 else part.sum(0)).to(torch.bfloat16)
    return (q, s.view(_E8), xsd) if write_q else (None, None, xsd)


def _bwd_grid(T, nt, R, N, step=128, min_progs=8):
    """(SB, số nhóm cột) ít byte phần cộng nhất mà vẫn có >= min_progs chương trình mỗi SM (đo: ~8/SM nhanh hơn
    ~4/SM khoảng 10% nhờ chia đều tải): phần dB/dA tỉ lệ số siêu khối, phần dxa tỉ lệ số nhóm cột x T. SB là bội của
    step (= ROWS của kernel, >= 128)."""
    tpad = triton.cdiv(T, 128) * 128
    best = None
    for ncg in _divisors(nt):
        for sb in range(step, triton.cdiv(tpad, step) * step + step, step):
            nsb = triton.cdiv(tpad, sb)
            if nsb * ncg < min_progs * _n_sm() and not (sb == step and ncg == nt):
                continue
            cost = nsb * N * R * 3 * 4 + ncg * T * 3 * R * 4
            if best is None or cost < best[0]:
                best = (cost, sb, ncg)
    return best[1], best[2]


def mlp_bwd(gu, xs, dhm, dxd, bg, bu, ad, s_g, s_u, need_q):
    """-> (q_dgu [T, 2N] fp8, s_dgu, [dxa_gate | dxa_up | xS_down] [T, 3R] fp32 (dxa chưa nhân s), dBg, dBu [N, R] fp32
    (chưa nhân s), dAd [R, N])."""
    T, N2 = gu.shape
    N, R = N2 // 2, ad.shape[0]
    bn = BWD_BN if R <= 16 else 32  # r = 32: ô 64 cột cần 102 KB shared memory (> 101 KB của sm120)
    nt = N // bn
    sb, ncg = _bwd_grid(T, nt, R, N, max(128, 16 * BWD_WARPS))
    nsb = triton.cdiv(triton.cdiv(T, 128) * 128, sb)
    dev = gu.device
    if need_q:
        q = torch.empty((T, N2), dtype=_F8, device=dev)
        s = torch.empty(triton.cdiv(T, 128) * (N2 // 128) * 512, dtype=torch.uint8, device=dev)
    else:
        q = s = gu
    dxa = torch.empty((ncg, T, 3 * R), dtype=torch.float32, device=dev)
    dbg = torch.empty((nsb, N, R), dtype=torch.float32, device=dev)
    dbu = torch.empty((nsb, N, R), dtype=torch.float32, device=dev)
    dad = torch.empty((nsb, R, N), dtype=torch.float32, device=dev)
    _LAST["bwd"] = _bwd_kernel[(nsb, ncg)](gu, xs, dhm, dxd, bg, bu, ad, q, s, dxa, dbg, dbu, dad, T,
                            triton.cdiv(T, 128) * 128, float(s_g), float(s_u), sb, nt // ncg,
                            N=N, R=R, ROWS=16 * BWD_WARPS, BN=bn, DO_QUANT=need_q, num_warps=BWD_WARPS, num_stages=BWD_STAGES,
                                         **({"maxnreg": BWD_MAXNREG} if BWD_MAXNREG else {}))
    red = lambda p: p[0] if p.shape[0] == 1 else p.sum(0)  # noqa: E731
    return (q if need_q else None, s.view(_E8) if need_q else None, red(dxa), red(dbg), red(dbu), red(dad))


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, meta, need_out, Ag, Bg, Au, Bu, Ad, Bd):
        ctx.set_materialize_grads(False)
        hold, down, (s_g, s_u, s_d), dt = meta
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        if x2.dtype != dt or not x2.is_contiguous():
            x2 = x2.to(dt).contiguous()
        # cùng khoá cache với _Group (fused_linear) để hai đường dùng chung bản bf16 trong một bước
        S = _cast(((id(Ag), id(Au)), dt), lambda: torch.cat([Ag.detach().to(dt), Au.detach().to(dt)]))
        bg = _cast((id(Bg), dt), lambda: Bg.detach().to(dt))
        bu = _cast((id(Bu), dt), lambda: Bu.detach().to(dt))
        ad = _cast(((id(Ad),), dt), lambda: torch.cat([Ad.detach().to(dt)]))
        bd = _cast((id(Bd), dt), lambda: Bd.detach().to(dt))
        xs = x2 @ S.t()                                           # [T, 2R] = xa_gate | xa_up
        q, sc = mxfp8.mx_quant(x2)
        gu = torch._scaled_mm(q, hold.w.t(), sc, hold.ws, out_dtype=torch.bfloat16)   # [T, 2N] gate | up
        del q, sc
        if need_out:
            qh, sh, xsd = mlp_fwd(gu, xs, bg, bu, ad, s_g, s_u, True)
            y = torch._scaled_mm(qh, down._mx_w.t(), sh, down._mx_ws, out_dtype=torch.bfloat16)
            y.addmm_(xsd, bd.t(), alpha=s_d)
            del qh, sh, xsd
        else:  # chỉ cần đồ thị (tính lại checkpoint / bước C): giá trị đầu ra không ai đọc -> khỏi chạy kernel fwd
            STATS["bỏ đầu ra"] = STATS.get("bỏ đầu ra", 0) + 1
            y = x2.new_empty(x2.shape[0], down.out_features)
        # cùng tập tensor lưu ở cả hai chế độ (checkpoint non-reentrant so metadata khi tính lại); xS_down không lưu,
        # kernel bwd tính lại từ h
        ctx.save_for_backward(x2, xs, gu, S, bg, bu, ad, bd)
        ctx.meta = (hold, down, (s_g, s_u, s_d), dt, lead, K, x.dtype)
        return y.view(*lead, y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return (None,) * 9
        x2, xs, gu, S, bg, bu, ad, bd = ctx.saved_tensors
        hold, down, (s_g, s_u, s_d), dt, lead, K, xdt = ctx.meta
        need_x = ctx.needs_input_grad[0]
        R = ad.shape[0]
        dy2 = dy.reshape(-1, dy.shape[-1]).to(dt)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()
        # down: như _Group.backward cũ (dh_mx = MX(dy) @ W_down, dxa_down; dB_down sau kernel vì cần xS_down)
        qd, sd = mxfp8.mx_quant(dy2)
        wt, wts = mxfp8.mx_wt(down)
        dhm = torch._scaled_mm(qd, wt.t(), sd, wts, out_dtype=torch.bfloat16)
        del qd, sd
        dxd = dy2 @ bd
        if s_d != 1:
            dxd.mul_(s_d)
        qg, sg, dxa, dB_g, dB_u, dA_d = mlp_bwd(gu, xs, dhm, dxd, bg, bu, ad, s_g, s_u, need_x)
        del dhm
        dB_d = dy2.t() @ dxa[:, 2 * R:].to(dt)                   # xS_down bf16 như bản đường cũ lưu
        if s_d != 1:
            dB_d.mul_(s_d)
        dxa = dxa[:, :2 * R]
        if s_g != 1:
            dxa[:, :R].mul_(s_g)
            dB_g.mul_(s_g)
        if s_u != 1:
            dxa[:, R:].mul_(s_u)
            dB_u.mul_(s_u)
        dxa = dxa.to(dt)                                          # [T, 2R] như dS bf16 của _Group cũ (liền khối)
        dx = None
        if need_x:
            dx = torch._scaled_mm(qg, hold.wt.t(), sg, hold.wts, out_dtype=torch.bfloat16)  # K = 2N: gate + up
            del qg, sg
            dx.addmm_(dxa, S)
            dx = dx.view(*lead, K).to(xdt)
        dA = dxa.t() @ x2
        grads = [dA[:R], dB_g, dA[R:], dB_u, dA_d, dB_d]
        ws_in = ctx.needs_input_grad[3:]
        return (dx, None, None, *[g if ws_in[k] else None for k, g in enumerate(grads)])


def _r_ok(r):
    return r in (16, 32)  # tl.dot cần K >= 16, tl.arange cần lũy thừa 2; r = 64 chưa thử (shared memory)


def mlp(mlp_mod, x, need_out=True):
    """down(silu(gate(x)) * up(x)) qua _MLP nếu gate/up/down đều là LoRA (PEFT) trên trọng số MX và gate/up đã ghép
    (mxfp8.enable_mxfp8), tính bf16. Không áp dụng được -> None (người gọi chạy đường cũ)."""
    if not ENABLED[0]:
        return None
    parts = [_part(m) for m in (mlp_mod.gate_proj, mlp_mod.up_proj, mlp_mod.down_proj)]
    if any(p is None or p[1] is None for p in parts):
        return None
    (gb, Ag, Bg, s_g), (ub, Au, Bu, s_u), (db, Ad, Bd, s_d) = parts
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if dt != torch.bfloat16 or not x.is_cuda:
        return None
    pg, pu = getattr(gb, "_mx_gu", None), getattr(ub, "_mx_gu", None)
    if pg is None or pu is None or pg[0] is not pu[0] or (pg[1], pu[1]) != (0, 1) or not mxfp8.is_mx(db):
        return None
    hold = pg[0]
    r = Ag.shape[0]
    T = x.numel() // x.shape[-1]
    if not (_r_ok(r) and Au.shape[0] == r and Ad.shape[0] == r and x.shape[-1] == hold.w.shape[1]
            and hold.n % 128 == 0 and Ad.shape[1] == hold.n and T > 0 and T * 2 * hold.n < 2 ** 31):
        return None  # (kernel dùng chỉ số int32)
    STATS["mlp gộp"] = STATS.get("mlp gộp", 0) + 1
    ACTIVE[0] = True
    return _MLP.apply(x, (hold, db, (s_g, s_u, s_d), dt), need_out, Ag, Bg, Au, Bu, Ad, Bd)
