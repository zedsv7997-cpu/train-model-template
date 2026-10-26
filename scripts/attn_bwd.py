"""Backward tự viết (Triton) cho lõi attention của shared_prefix: 5 matmul / cặp (query, kv) thay vì 7 của FlexAttention.

Forward vẫn là FlexAttention (lấy thêm lse qua return_lse). Backward của Flex có hai nhánh: nhánh dK/dV (4 matmul:
qkᵀ, dV, dPᵀ, dK) và nhánh dQ tính LẠI qkᵀ và dP (3 matmul: qk, dp, dq). Ở đây:
  K1 = nhánh dK/dV y như Flex, thêm ghi dS (bf16, 2 byte/cặp-head) ra bộ đệm theo khúc kv;
  K2 = dQ = dS·K, một matmul kiểu GEMM đọc lại dS (không cần atomic, tất định).
Không làm kiểu FA2 cộng dồn dQ bằng atomic: head_dim 256 -> lưu lượng atomic 256/BLOCK_N số fp32 mỗi cặp-head, gấp 8-16
lần 4 byte ghi + đọc dS.

Mask: query thứ i thấy kv <= qpos[i] (qpos tăng dần, int32 trên GPU): arange(s) + p cho lớp thường / prefix,
rows + p cho lớp trên cùng chỉ tính ở vị trí có nhãn. GQA: HQ / HKV head q dùng chung một head kv.
Số học giống template Flex (qkᵀ*scale rồi *RCP_LN2, p = exp2(s - lse2), dS làm tròn về dtype trước dot, dK*scale,
dQ*scale); chỉ khác thứ tự cộng fp32 của dQ (theo khúc kv và tile của K2), và lse2 = lse*RCP_LN2 có thể lệch 1 ulp
so với lse cơ số 2 nội bộ của Flex.
"""
import os
import warnings

import torch
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.44269504)  # như template Flex

# Bộ đệm dS cho một khúc kv (MB): khúc dài thì ít lần chạy K1/K2 hơn (ít đuôi sóng trên 26 SM, K2 ít lần đọc/ghi lại
# bộ cộng dồn dQ fp32): phần riêng 2600 nhìn 9000 bwd 14.97 ms ở 128 MB, 14.03 ở 512; prefix 6400 bước C 17.15 ->
# 14.27 (~0.07 s/bước tổng cộng). Đỉnh VRAM của bước nằm ở backward phần riêng, nên mặc định 128 MB (+21 MB bộ
# cộng dồn dQ fp32 khi > 1 khúc): ở lớp trên cùng nhỏ hơn khúc CE 127 MB của forward, ở các lớp checkpoint khác nhỏ
# hơn phần tính lại MLP ngay trước nó. Đặt SHARED_PREFIX_ATTN_DS_MAX_MB > 128 để nới tới mức đó khi còn trống dưới
# PEAK_BYTES (shared_prefix đặt = đỉnh nhắm tới) quá _RESERVE; mặc định tắt vì cỡ bộ đệm đổi theo VRAM trống làm
# bộ cấp phát (Windows, không expandable_segments) dễ phân mảnh, mà vượt trần là tràn RAM chậm 5-7 lần.
_DS_MB = float(os.environ.get("SHARED_PREFIX_ATTN_DS_MB", "128"))
_DS_MAX_MB = float(os.environ.get("SHARED_PREFIX_ATTN_DS_MAX_MB", str(_DS_MB)))
_RESERVE = 1.0 * 2**30
PEAK_BYTES = None
_ALIGN = 128  # khúc kv và số cột bộ đệm dS là bội 128 -> mọi tile của K1/K2 nằm gọn trong khúc
# Cấu hình khối (quét 2026-10-23 trên RTX 5060 sm120, shared tối đa 101,376 B/block, bench_attn_bwd.py k1|k2):
# K1 32x32, 4 warps, 3 stages 15.70 ms cả backward (8 warps 19.5; BN 64 hoặc BM 64 vượt shared hoặc chậm gấp đôi);
# K2 64x32, 4 warps, 2 stages (các cấu hình hợp lệ khác chỉ chậm hơn 0-5%).
K1_CFG = dict(BM=32, BN=32, num_warps=4, num_stages=3)
K2_CFG = dict(BM=64, BN=32, num_warps=4, num_stages=2)


def _ds_budget():
    b = _DS_MB * 2**20
    if PEAK_BYTES is not None:
        b = max(b, min(_DS_MAX_MB * 2**20, PEAK_BYTES - torch.cuda.memory_allocated() - _RESERVE))
    return b


@triton.jit
def _delta_kernel(O, DO, DELTA, so_h, so_m, sdo_h, sdo_m, S_LEN, D: tl.constexpr, BM: tl.constexpr):
    """Δ = rowsum(O·dO) fp32 [HQ, S]."""
    h = tl.program_id(1).to(tl.int64)
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mm = offs_m < S_LEN
    o = tl.load(O + h * so_h + offs_m[:, None] * so_m + offs_d[None, :], mask=mm[:, None], other=0.).to(tl.float32)
    do = tl.load(DO + h * sdo_h + offs_m[:, None] * sdo_m + offs_d[None, :], mask=mm[:, None], other=0.).to(tl.float32)
    tl.store(DELTA + h * S_LEN + offs_m, tl.sum(o * do, 1), mask=mm)


@triton.jit
def _dkdv_step(dk, dv, k, v, Qh, DOh, LSEh, DELh, DSh, QPOS, m0, offs_n, n_loc, offs_d, sq_m, sdo_m, sds_n, S_LEN,
               SCALE: tl.constexpr, BM: tl.constexpr, MASKED: tl.constexpr, PREC: tl.constexpr):
    offs_m = m0 + tl.arange(0, BM)
    mm = offs_m < S_LEN
    # hàng đệm (m >= S) nạp 0 cả q, dO, lse, Δ -> pT = 1 nhưng dO = 0 và dsT = 0: đóng góp đúng bằng 0
    q = tl.load(Qh + offs_m[:, None] * sq_m + offs_d[None, :], mask=mm[:, None], other=0.)
    lse = tl.load(LSEh + offs_m, mask=mm, other=0.) * RCP_LN2
    lse = tl.where(lse == float("-inf"), 0.0, lse)
    qkT = tl.dot(k, tl.trans(q), input_precision=PREC)
    qkT *= SCALE
    if MASKED:
        qp = tl.load(QPOS + offs_m, mask=mm, other=-1)
        vis = offs_n[:, None] <= qp[None, :]
        qkT = tl.where(vis, qkT, float("-inf"))
    qkT *= RCP_LN2
    pT = tl.math.exp2(qkT - lse[None, :])
    do = tl.load(DOh + offs_m[:, None] * sdo_m + offs_d[None, :], mask=mm[:, None], other=0.)
    dv += tl.dot(pT.to(do.dtype), do, input_precision=PREC)
    di = tl.load(DELh + offs_m, mask=mm, other=0.)
    dpT = tl.dot(v, tl.trans(do), input_precision=PREC)
    dsT = pT * (dpT - di[None, :])
    if MASKED:
        dsT = tl.where(vis, dsT, 0.0)
    dsT = dsT.to(q.dtype)
    dk += tl.dot(dsT, q, input_precision=PREC)
    # bố cục dS [HQ, khúc kv, S_pad]: hàng n liền theo m -> K2 đọc tile [BN, BM2] liền khối
    tl.store(DSh + n_loc[:, None] * sds_n + offs_m[None, :], dsT)
    return dk, dv


@triton.jit(do_not_specialize=["S_LEN", "KV_LEN", "J0"])  # độ dài đổi mỗi hội thoại: khỏi biên dịch nhiều bản
def _bwd_dkdv_ds(Q, K, V, DO, LSE, DELTA, QPOS, PLAN, DK, DV, DS,
                 sq_h, sq_m, sk_h, sk_n, sv_h, sv_n, sdo_h, sdo_m, sdk_h, sdk_n, sdv_h, sdv_n, sds_h, sds_n,
                 S_LEN, KV_LEN, J0,
                 SCALE: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 PREC: tl.constexpr):
    """K1: một chương trình = một khối kv BN hàng của một head kv; lặp GQA head q x các khối q nhìn thấy nó."""
    g = tl.program_id(1).to(tl.int64)
    nb = J0 // BN + tl.program_id(0)
    offs_n = nb * BN + tl.arange(0, BN)
    n_loc = offs_n - J0
    offs_d = tl.arange(0, D)
    nm = offs_n[:, None] < KV_LEN
    k = tl.load(K + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=nm, other=0.)
    v = tl.load(V + g * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=nm, other=0.)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    # khối q đầu tiên thấy (một phần) khối kv này / khối q đầu tiên thấy trọn khối kv (bội BM, tính sẵn trên GPU)
    m_start = tl.load(PLAN + 2 * nb)
    m_full = tl.load(PLAN + 2 * nb + 1)
    for hq in range(GQA):
        h = g * GQA + hq
        Qh = Q + h * sq_h
        DOh = DO + h * sdo_h
        LSEh = LSE + h * S_LEN
        DELh = DELTA + h * S_LEN
        DSh = DS + h * sds_h
        for m0 in range(m_start, m_full, BM):
            dk, dv = _dkdv_step(dk, dv, k, v, Qh, DOh, LSEh, DELh, DSh, QPOS, m0, offs_n, n_loc, offs_d,
                                sq_m, sdo_m, sds_n, S_LEN, SCALE, BM, True, PREC)
        for m0 in range(m_full, S_LEN, BM):
            dk, dv = _dkdv_step(dk, dv, k, v, Qh, DOh, LSEh, DELh, DSh, QPOS, m0, offs_n, n_loc, offs_d,
                                sq_m, sdo_m, sds_n, S_LEN, SCALE, BM, False, PREC)
    tl.store(DV + g * sdv_h + offs_n[:, None] * sdv_n + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=nm)
    dk *= SCALE
    tl.store(DK + g * sdk_h + offs_n[:, None] * sdk_n + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=nm)


@triton.jit(do_not_specialize=["S_LEN", "KV_LEN", "J0", "J1"])
def _bwd_dq(DS, K, QPOS, DQACC, DQ, sk_h, sk_n, sds_h, sds_n, sdq_h, sdq_m, S_LEN, KV_LEN, J0, J1,
            SCALE: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            FIRST: tl.constexpr, LAST: tl.constexpr, PREC: tl.constexpr):
    """K2: dQ[khối BM hàng q, head h] += Σ_{kv trong khúc [J0, J1)} dS·K; khúc cuối nhân scale rồi ghi dtype gốc."""
    h = tl.program_id(1).to(tl.int64)
    g = h // GQA
    m0 = tl.program_id(0) * BM
    offs_m = m0 + tl.arange(0, BM)
    mm = offs_m < S_LEN
    offs_d = tl.arange(0, D)
    acc_ptr = DQACC + h * S_LEN * D + offs_m[:, None] * D + offs_d[None, :]
    if FIRST:
        acc = tl.zeros([BM, D], dtype=tl.float32)
    else:
        acc = tl.load(acc_ptr, mask=mm[:, None], other=0.)
    qp = tl.load(QPOS + offs_m, mask=mm, other=-1)
    lo = tl.load(QPOS + m0)                                  # qpos nhỏ nhất của khối
    hi = tl.load(QPOS + tl.minimum(m0 + BM, S_LEN) - 1)      # qpos lớn nhất của khối
    # tile kv trọn vẹn (mọi hàng q của khối thấy hết): n0 + BN - 1 <= lo, không cần mask
    n_full = J0 + tl.maximum((tl.minimum(lo + 1, J1) - J0) // BN, 0) * BN
    n_end = tl.minimum(J1, hi + 1)
    DSh = DS + h * sds_h
    Kg = K + g * sk_h
    for n0 in range(J0, n_full, BN):
        offs_n = n0 + tl.arange(0, BN)
        ds = tl.load(DSh + (offs_n - J0)[:, None] * sds_n + offs_m[None, :])
        kt = tl.load(Kg + offs_n[:, None] * sk_n + offs_d[None, :])
        acc += tl.dot(tl.trans(ds), kt, input_precision=PREC)
    for n0 in range(n_full, n_end, BN):
        offs_n = n0 + tl.arange(0, BN)
        # chỉ đọc phần tử K1 đã ghi (cùng điều kiện nhìn thấy); phần còn lại coi là 0
        vis = offs_n[:, None] <= qp[None, :]
        ds = tl.load(DSh + (offs_n - J0)[:, None] * sds_n + offs_m[None, :], mask=vis, other=0.)
        kt = tl.load(Kg + offs_n[:, None] * sk_n + offs_d[None, :], mask=offs_n[:, None] < KV_LEN, other=0.)
        acc += tl.dot(tl.trans(ds), kt, input_precision=PREC)
    if LAST:
        acc *= SCALE
        tl.store(DQ + h * sdq_h + offs_m[:, None] * sdq_m + offs_d[None, :], acc.to(DQ.dtype.element_ty),
                 mask=mm[:, None])
    else:
        tl.store(acc_ptr, acc, mask=mm[:, None])


def supported(q, k, v):
    """Đường Triton áp dụng được không (còn lại rơi về backward của Flex)."""
    if os.environ.get("SHARED_PREFIX_ATTN_BWD", "1") == "0" or not q.is_cuda or _BROKEN:
        return False
    if q.dim() != 4 or q.shape[0] != 1 or k.shape[0] != 1 or v.shape != k.shape:
        return False
    d = q.shape[-1]
    if d != k.shape[-1] or d < 16 or d > 256 or d & (d - 1):
        return False
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    return q.shape[1] % k.shape[1] == 0


def _rowmajor(t):
    return t if t.stride(-1) == 1 else t.contiguous()


def _plan(qpos, kv_len, bm, bn):
    """[số khối kv, 2] int32: (khối q đầu tiên thấy một phần, khối q đầu tiên thấy trọn) của từng khối kv BN hàng,
    làm tròn về bội BM. Nhớ theo tensor qpos (6 lớp attention của một hội thoại dùng chung)."""
    cache = qpos.__dict__.setdefault("_attn_plan", {})
    key = (kv_len, bm, bn)
    if key not in cache:
        n0 = torch.arange(0, kv_len, bn, device=qpos.device, dtype=torch.int32)
        first = torch.searchsorted(qpos, n0)               # hàng q đầu tiên có qpos >= n0
        full = torch.searchsorted(qpos, n0 + (bn - 1))     # hàng q đầu tiên có qpos >= n0 + BN - 1
        cache[key] = torch.stack([first // bm * bm, (full + bm - 1) // bm * bm], 1).to(torch.int32).contiguous()
    return cache[key]


def attn_bwd(q, k, v, out, lse, qpos, do, scale, k1=None, k2=None):
    """dq, dk, dv của o = softmax(q kᵀ·scale + mask) v với mask kv <= qpos[i]. q [1, HQ, S, D], k/v [1, HKV, KV, D],
    out/do [1, HQ, S, D], lse [1, HQ, S] fp32 (cơ số e, từ Flex), qpos [S] int32 tăng dần."""
    k1, k2 = k1 or K1_CFG, k2 or K2_CFG
    q, k, v, out, do = (_rowmajor(t) for t in (q, k, v, out, do))
    _, hq, s, d = q.shape
    hkv, kv_len = k.shape[1], k.shape[2]
    gqa = hq // hkv
    dev = q.device
    prec = "ieee" if q.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32 else "tf32"
    lse = lse.reshape(hq, s).float().contiguous()
    qpos = qpos.to(torch.int32).contiguous()

    delta = torch.empty(hq, s, device=dev, dtype=torch.float32)
    _delta_kernel[(triton.cdiv(s, 64), hq)](out, do, delta, out.stride(1), out.stride(2), do.stride(1), do.stride(2),
                                            s, D=d, BM=64, num_warps=4)
    plan = _plan(qpos, kv_len, k1["BM"], k1["BN"])

    s_pad = triton.cdiv(s, _ALIGN) * _ALIGN
    row = hq * s_pad * q.element_size()
    kv_pad = triton.cdiv(kv_len, _ALIGN) * _ALIGN
    c = max(_ALIGN, int(_ds_budget() // row) // _ALIGN * _ALIGN)
    n_chunk = triton.cdiv(kv_pad, c)
    c = triton.cdiv(triton.cdiv(kv_pad, n_chunk), _ALIGN) * _ALIGN  # chia đều các khúc
    ds = torch.empty(hq, c, s_pad, device=dev, dtype=q.dtype)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dqacc = torch.empty(hq, s, d, device=dev, dtype=torch.float32) if n_chunk > 1 else dq  # dq: chỉ để có con trỏ
    scale = float(scale)
    for i in range(n_chunk):
        j0, j1 = i * c, min(kv_len, (i + 1) * c)
        KERNELS["k1"] = _bwd_dkdv_ds[(triton.cdiv(j1 - j0, k1["BN"]), hkv)](
            q, k, v, do, lse, delta, qpos, plan, dk, dv, ds,
            q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2), do.stride(1), do.stride(2),
            dk.stride(1), dk.stride(2), dv.stride(1), dv.stride(2), ds.stride(0), ds.stride(1),
            s, kv_len, j0, SCALE=scale, GQA=gqa, D=d, BM=k1["BM"], BN=k1["BN"], PREC=prec,
            num_warps=k1["num_warps"], num_stages=k1["num_stages"])
        KERNELS["k2"] = _bwd_dq[(triton.cdiv(s, k2["BM"]), hq)](
            ds, k, qpos, dqacc, dq, k.stride(1), k.stride(2), ds.stride(0), ds.stride(1), dq.stride(1), dq.stride(2),
            s, kv_len, j0, j1, SCALE=scale, GQA=gqa, D=d, BM=k2["BM"], BN=k2["BN"],
            FIRST=i == 0, LAST=i == n_chunk - 1, PREC=prec, num_warps=k2["num_warps"], num_stages=k2["num_stages"])
    KERNELS["n_chunk"] = n_chunk
    return dq, dk, dv


KERNELS = {}  # kernel đã biên dịch của lần gọi gần nhất (xem shared / thanh ghi / spill khi quét cấu hình)


def _mute_lse_warning():
    """flex_attention cảnh báo return_lse deprecated một lần qua tập _WARNINGS_SHOWN, và dynamo guard luôn trên tập đó:
    lần đầu cảnh báo xong là guard hỏng -> biên dịch lại thêm một bản (mỗi bản tính vào recompile_limit 8 của
    flex_attention; vượt là dynamo thôi compile hàm này). Đánh dấu "đã cảnh báo" trước lần gọi đầu tiên."""
    try:
        from torch.nn.attention import flex_attention as fa
        fa._WARNINGS_SHOWN.add("deprecated_return_lse")
    except (ImportError, AttributeError):
        pass


class AttnCore(torch.autograd.Function):
    """o = flex_attention(q, k, v, block_mask) (forward của Flex, lấy thêm lse); backward = attn_bwd."""

    @staticmethod
    def forward(ctx, q, k, v, block_mask, qpos, scale, flex, opts):
        # return_lse (đã deprecated) chứ không return_aux=AuxRequest(lse=True): bản sau làm torch 2.14 + dynamo
        # đệ quy vô hạn khi dựng lại namedtuple AuxOutput (tràn stack 0xC00000FD, đo 2026-10-23)
        _mute_lse_warning()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out, lse = flex(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True, kernel_options=opts,
                            return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse, qpos)
        ctx.scale, ctx.block_mask, ctx.flex, ctx.opts = scale, block_mask, flex, opts
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse, qpos = ctx.saved_tensors
        if not _BROKEN:
            try:
                dq, dk, dv = attn_bwd(q, k, v, out, lse, qpos, do, ctx.scale)
                return dq, dk, dv, None, None, None, None, None
            except triton.runtime.errors.OutOfResources as ex:
                # K1 dùng 101,120 / 101,376 B shared: Triton / driver khác có thể vượt -> tắt hẳn đường này
                _BROKEN.append(ex)
                warnings.warn(f"attn_bwd: {ex} -> từ giờ dùng backward của FlexAttention")
        # Dự phòng: chạy lại forward Flex có đồ thị rồi lấy gradient (đắt hơn một lần forward, gradient như đường cũ)
        with torch.enable_grad():
            qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
            o = ctx.flex(qq, kk, vv, block_mask=ctx.block_mask, scale=ctx.scale, enable_gqa=True, kernel_options=ctx.opts)
            dq, dk, dv = torch.autograd.grad(o, (qq, kk, vv), do)
        return dq, dk, dv, None, None, None, None, None


_BROKEN = []  # lỗi tài nguyên lần đầu (nếu có): khỏi biên dịch lại mỗi lần
