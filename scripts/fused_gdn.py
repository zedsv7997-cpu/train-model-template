"""Lõi Gated DeltaNet (conv nhân quả + SiLU + tách q/k/v + L2norm + chunk gated delta rule) trong MỘT autograd.Function.

Đường cũ (shared_prefix._gdn): causal_conv1d của fla ghi mixed [T, 6144], torch.split ra 3 view stride 6144, rồi
chunk_gated_delta_rule: input_guard chép q/k/v cho liền khối (3 bản chép), l2norm_fwd riêng cho q và k. Lan ngược:
l2norm_bwd x2, dk.add_(dk2), cat của SplitBackward, causal_conv1d_bwd chạy lại conv forward (lấy tiền kích hoạt) rồi
kernel bwd (BD=32, kèm dW thừa vì conv1d.weight đóng băng), thêm compute_dh0. Profile 2026-10-23: ~0.42 s/bước bwd,
~0.21 s/bước chép + l2norm ở forward, ~0.1 s/bước fla tự tính lại w/u/h trong bwd.

Ở đây:
  - forward: một kernel conv + SiLU + L2norm đọc qkv một lượt, ghi thẳng q̂, k̂, v liền khối + rstd (mỗi chương trình
    = 64 hàng x 1 head 128 kênh, đủ cả hàng cho L2norm); rồi gọi đúng chuỗi hàm nội bộ của fla như
    ChunkGatedDeltaRuleFunction (cumsum, intra, fwd_h, fwd_o);
  - backward: chuỗi bwd của fla như chunk_gated_delta_rule_bwd, rồi MỘT kernel gộp cộng dk2 + L2norm-bwd + SiLU'
    (tính lại tiền kích hoạt trong thanh ghi) + conv chuyển vị + gradient 3 token đuôi conv vào/ra; bỏ dW (đóng băng);
  - save_inner: lưu luôn w, h, v_new từ forward (chỉ dùng cho lớp checkpoint / bước C, nơi forward tính lại chạy ngay
    trước backward), khỏi chạy lại recompute_w_u + fwd_h trong bwd;
  - lora (tùy chọn, GDN_LORA_FOLD=1): qkv vào là phần gốc chưa cộng LoRA-B (group_linear(..., defer=(0,))), kernel
    forward tự cộng s * xA @ Bᵀ (tl.dot K=16) và, khi cần backward, ghi luôn qkv đầy đủ (cho bwd) -> bỏ addmm_ tại chỗ
    (đọc + ghi [T, 6144]) đổi lấy một lượt ghi; bước A (không grad) bỏ hẳn. Không trùng từng bit: cuBLAS addmm_ đôi khi
    làm tròn khác tl.dot (vài phần tử / 15M lệch 1 ulp bf16) -> gradient lệch ~1e-3, nhỏ hơn nhiễu nền bf16 ~7 lần.
    Bản bwd tự cộng LoRA (16 tl.dot / chương trình + đổi layout) chậm gấp ~3.5 lần nên không dùng.
Cùng điểm làm tròn với đường cũ (tiền kích hoạt bf16 như fla lưu y, dk+dk2 làm tròn bf16 như add_, đầu ra L2norm-bwd
bf16 như cat, bố cục ô [hàng, 128 kênh] như l2norm của fla nên rút gọn cùng thứ tự): fused_gdn_test.py và
fused_gdn_layer_test.py cho kết quả TRÙNG TỪNG BIT với đường cũ (đầu ra, đuôi conv, trạng thái, mọi gradient).

Chỉ dùng khi: conv kernel 4, không bias, weight đóng băng, head q/k/v cùng cỡ (lũy thừa 2), num_v_heads = num_k_heads,
batch 1, cùng dtype; còn lại shared_prefix._gdn rơi về đường cũ. FUSED_GDN=0 tắt hẳn.
Dựa vào hàm nội bộ của fla 0.5.2 (ghim phiên bản).

Đo 2026-10-23 (RTX 5060 Laptop, xen kẽ cũ / mới, GPU rảnh), một lớp DeltaNet cỡ 2B + LoRA r16 + MXFP8, fwd+bwd:
T=2400: keep 17.30 -> 16.66 ms, checkpoint 22.90 -> 21.47, bước C 17.25 -> 15.92; T=6400: keep 52.42 -> 49.77,
checkpoint 69.54 -> 65.03, bước C 50.50 -> 46.84 (riêng lõi: -26% lớp keep, -35% lớp checkpoint / bước C).
GDN_LORA_FOLD=1 chỉ thêm 0.1-0.4 ms/lớp. VRAM: lớp checkpoint / bước C +16 KB/token lúc backward (save_inner,
đã cộng vào margin của shared_prefix._auto_keep).
"""
import os

import fla
import torch
import triton
import triton.language as tl
from fla.ops.gated_delta_rule import chunk as _fc
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as FLA_CGDR

# gọi thẳng hàm nội bộ của fla theo đúng trình tự của bản 0.5.2: bản khác thì chạy đường cũ (FUSED_GDN=1 vẫn tắt)
ENABLED = os.environ.get("FUSED_GDN", "1") != "0" and fla.__version__ == "0.5.2"
LORA_FOLD = False  # GDN_LORA_FOLD (nhánh LoRA-B của qkv trong kernel conv) chưa gộp với cơ chế defer của fused_linear
STATS = {"gộp": 0, "gộp + LoRA": 0, "đường cũ": 0}
_EPS = 1e-6  # như l2norm_fwd mặc định trong ChunkGatedDeltaRuleFunction
_BT = 64     # chunk của fla


# ------------------------------------------------------------------ đọc đầu vào conv

@triton.jit
def _ld_in(x, tail, xs, bt, s, rr, cols, o_r, T, stride_xt, stride_xs,
           HAS_TAIL: tl.constexpr, HAS_LORA: tl.constexpr):
    """Đầu vào conv (fp32) ở các hàng rr: qkv (+ s * xA @ Bᵀ làm tròn như addmm_) cho rr >= 0, 3 token đuôi của đoạn
    trước (tail [3, C], cùng bố cục với x) cho -3 <= rr < 0, 0 ở ngoài."""
    m = (rr >= 0) & (rr < T)
    xv = tl.load(x + rr[:, None] * stride_xt + cols[None, :], mask=m[:, None], other=0.0)
    if HAS_LORA:
        a = tl.load(xs + rr[:, None] * stride_xs + o_r[None, :], mask=m[:, None], other=0.0)
        xv = (xv.to(tl.float32) + s * tl.dot(a, bt)).to(x.dtype.element_ty)
    xv = xv.to(tl.float32)
    if HAS_TAIL:
        xv += tl.load(tail + (rr + 3)[:, None] * stride_xt + cols[None, :], mask=((rr < 0) & (rr >= -3))[:, None],
                      other=0.0).to(tl.float32)
    return xv


@triton.jit
def _w_col(j: tl.constexpr, w0, w1, w2, w3):
    if j == 0:
        r = w0
    elif j == 1:
        r = w1
    elif j == 2:
        r = w2
    else:
        r = w3
    return r


# ------------------------------------------------------------------ kernel forward

@triton.jit
def _conv_l2_fwd_kernel(x, w, tail, xs, bq, s, qn, kn, v, rq, rk, tail_out, x_out, T, stride_xt, stride_xs, IT0,
                        HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BT: tl.constexpr,
                        HAS_TAIL: tl.constexpr, HAS_LORA: tl.constexpr, WANT_TAIL: tl.constexpr,
                        WRITE_FULL: tl.constexpr, EPS: tl.constexpr):
    # Một chương trình = BT hàng x một head D kênh (hb 0..HK-1: q, HK..2HK-1: k, còn lại v)
    it = tl.program_id(0) + IT0
    hb = tl.program_id(1)
    o_t = it * BT + tl.arange(0, BT)
    o_d = tl.arange(0, D)
    o_r = tl.arange(0, R)
    cols = hb * D + o_d
    bt = 0
    if HAS_LORA:
        bt = tl.load(bq + cols[None, :] * R + o_r[:, None])  # Bᵀ [R, D]
    acc = tl.zeros((BT, D), dtype=tl.float32)
    # cộng 4 tap theo thứ tự cũ -> mới như causal_conv1d_fwd_kernel (cùng phép fma, cùng làm tròn)
    for j in tl.static_range(4):
        xv = _ld_in(x, tail, xs, bt, s, o_t - 3 + j, cols, o_r, T, stride_xt, stride_xs, HAS_TAIL, HAS_LORA)
        if j == 3:  # tap j = 3 là chính các hàng của khối (đã cộng LoRA nếu có)
            if WANT_TAIL:  # 3 token cuối = đuôi conv cho đoạn sau
                k = o_t - (T - 3)
                tl.store(tail_out + k[:, None] * stride_xt + cols[None, :], xv.to(x.dtype.element_ty),
                         mask=((k >= 0) & (o_t < T))[:, None])
            if WRITE_FULL:  # qkv đầy đủ cho backward
                tl.store(x_out + o_t[:, None] * stride_xt + cols[None, :], xv.to(x.dtype.element_ty),
                         mask=(o_t < T)[:, None])
        wj = tl.load(w + cols * 4 + j).to(tl.float32)
        xv *= wj[None, :]
        acc += xv
    acc = acc * tl.sigmoid(acc)
    m_t = o_t < T
    if hb < 2 * HK:
        # L2norm trên giá trị đã làm tròn (fla lưu mixed rồi mới l2norm_fwd), rstd = 1/sqrt(sum(y²)+eps)
        y = acc.to(qn.dtype.element_ty).to(tl.float32)
        rs = 1 / tl.sqrt(tl.sum(y * y, 1) + EPS)
        yn = (y * rs[:, None]).to(qn.dtype.element_ty)
        if hb < HK:
            tl.store(qn + o_t[:, None] * (HK * D) + hb * D + o_d[None, :], yn, mask=m_t[:, None])
            tl.store(rq + o_t * HK + hb, rs, mask=m_t)
        else:
            tl.store(kn + o_t[:, None] * (HK * D) + (hb - HK) * D + o_d[None, :], yn, mask=m_t[:, None])
            tl.store(rk + o_t * HK + (hb - HK), rs, mask=m_t)
    else:
        tl.store(v + o_t[:, None] * (HV * D) + (hb - 2 * HK) * D + o_d[None, :], acc.to(v.dtype.element_ty),
                 mask=m_t[:, None])


def _row_blocks(T, BT, has_tail):
    """(khối hàng đầu, số khối, có đọc đuôi) cho mỗi lần phóng: chỉ khối hàng đầu chạm 3 token đuôi conv nên chỉ nó
    dùng bản kernel có đuôi (bản đó thêm 4-16 lần load có mask, tốn thanh ghi; phóng chung cả lưới thì chậm hẳn)."""
    nt = triton.cdiv(T, BT)
    if not has_tail:
        return [(0, nt, False)]
    return [(0, 1, True)] + ([(1, nt - 1, False)] if nt > 1 else [])


def _lora_args(x, lora):
    if lora is None:
        return x, 0, x, 1.0, 16
    xs, bq, s = lora
    return xs, xs.stride(0), bq, float(s), bq.shape[1]


def conv_l2_fwd(x, w, tail_r, hk, hv, d, lora=None, want_tail=False, write_full=False):
    """x [1, T, C] liền khối, w [C, 4] liền khối, tail_r [3, C] liền khối hoặc None, lora (xA [T, r], B [C, r], s)
    hoặc None -> q̂, k̂ [1, T, hk, d], v [1, T, hv, d], rstd q/k [1, T, hk] fp32, đuôi ra [3, C] | None,
    qkv đầy đủ [1, T, C] (write_full, chỉ khi có lora) | None."""
    T, C = x.shape[1], x.shape[2]
    qn = x.new_empty(1, T, hk, d)
    kn = x.new_empty(1, T, hk, d)
    v = x.new_empty(1, T, hv, d)
    rq = x.new_empty(1, T, hk, dtype=torch.float32)
    rk = x.new_empty(1, T, hk, dtype=torch.float32)
    tail_out = x.new_empty(3, C) if want_tail else None
    write_full = write_full and lora is not None
    x_out = torch.empty_like(x) if write_full else None
    xs, sxs, bq, s, R = _lora_args(x, lora)
    for it0, n, has_tail in _row_blocks(T, _FWD_BT, tail_r is not None):
        _conv_l2_fwd_kernel[(n, 2 * hk + hv)](
            x, w, tail_r if has_tail else x, xs, bq, s, qn, kn, v, rq, rk,
            tail_out if want_tail else x, x_out if write_full else x, T, x.stride(1), sxs, it0,
            HK=hk, HV=hv, D=d, R=R, BT=_FWD_BT, HAS_TAIL=has_tail, HAS_LORA=lora is not None, WANT_TAIL=want_tail,
            WRITE_FULL=write_full, EPS=_EPS, num_warps=_FWD_WARPS)
    return qn, kn, v, rq, rk, tail_out, x_out


# ------------------------------------------------------------------ kernel backward

@triton.jit
def _conv_l2_bwd_kernel(x, w, tail, qn, kn, rq, rk, dq, dk, dk2, dv, dtail, dx, dtail0, T, stride_xt, IT0,
                        HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
                        HAS_TAIL: tl.constexpr, HAS_DTAIL: tl.constexpr, NEED_DTAIL0: tl.constexpr):
    """dx[t] = sum_i dpre[t+i] * w[3-i] (+ gradient của 3 token đuôi ra), dpre = d_post * SiLU'(pre);
    d_post: q -> L2norm-bwd(dq), k -> L2norm-bwd(bf16(dk + dk2)), v -> dv. Mỗi lần dịch i tính lại pre và d_post ở
    hàng t+i (đọc lại từ L1/L2) thay vì ghi d_post / pre ra bộ nhớ như đường cũ (Triton không dịch hàng trong thanh ghi
    được; bản đọc 7 ô x một lần chỉ nhanh hơn ~4%, bản vòng lặp động chậm hơn)."""
    it = tl.program_id(0) + IT0
    hb = tl.program_id(1)
    o_t = it * BT + tl.arange(0, BT)
    o_d = tl.arange(0, D)
    cols = hb * D + o_d
    w0 = tl.load(w + cols * 4 + 0).to(tl.float32)
    w1 = tl.load(w + cols * 4 + 1).to(tl.float32)
    w2 = tl.load(w + cols * 4 + 2).to(tl.float32)
    w3 = tl.load(w + cols * 4 + 3).to(tl.float32)
    dt = dx.dtype.element_ty
    b_dx = tl.zeros((BT, D), dtype=tl.float32)
    for i in tl.static_range(4):
        r = o_t + i
        m_r = r < T
        # tiền kích hoạt ở hàng r, cùng phép cộng như causal_conv1d_fwd_kernel (activation=None), làm tròn như y bf16
        # mà causal_conv1d_bwd của fla tính lại
        pre = tl.zeros((BT, D), dtype=tl.float32)
        for j in tl.static_range(4):
            xv = _ld_in(x, tail, x, 0, 1.0, r - 3 + j, cols, o_d, T, stride_xt, 0, HAS_TAIL, False)
            xv *= _w_col(j, w0, w1, w2, w3)[None, :]
            pre += xv
        pre = pre.to(dt).to(tl.float32)
        if hb < 2 * HK:
            if hb < HK:
                off = r[:, None] * (HK * D) + hb * D + o_d[None, :]
                b_dy = tl.load(dq + off, mask=m_r[:, None], other=0.0).to(tl.float32)
                b_y = tl.load(qn + off, mask=m_r[:, None], other=0.0).to(tl.float32)
                b_rs = tl.load(rq + r * HK + hb, mask=m_r, other=0.0).to(tl.float32)
            else:
                off = r[:, None] * (HK * D) + (hb - HK) * D + o_d[None, :]
                # dk.add_(dk2) của fla: cộng fp32 rồi làm tròn về dtype của dk
                b_dy = (tl.load(dk + off, mask=m_r[:, None], other=0.0).to(tl.float32)
                        + tl.load(dk2 + off, mask=m_r[:, None], other=0.0).to(tl.float32)).to(dt).to(tl.float32)
                b_y = tl.load(kn + off, mask=m_r[:, None], other=0.0).to(tl.float32)
                b_rs = tl.load(rk + r * HK + (hb - HK), mask=m_r, other=0.0).to(tl.float32)
            # l2norm_bwd_kernel, đầu ra làm tròn như tensor bf16 nó ghi
            b_dp = b_dy * b_rs[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rs[:, None]
            b_dp = b_dp.to(dt).to(tl.float32)
        else:
            b_dp = tl.load(dv + r[:, None] * (HV * D) + (hb - 2 * HK) * D + o_d[None, :], mask=m_r[:, None],
                           other=0.0).to(tl.float32)
        b_ys = tl.sigmoid(pre)
        b_dp = b_dp * b_ys * (1 + pre * (1 - b_ys))
        if NEED_DTAIL0:
            if i == 0:
                if it == 0:
                    # gradient 3 token đuôi vào: dtail0[k] = sum_{t<=k} dpre[t] * w[k-t] (như compute_dh0_kernel)
                    d0 = tl.sum(tl.where((o_t == 0)[:, None], b_dp, 0.0), 0)
                    d1 = tl.sum(tl.where((o_t == 1)[:, None], b_dp, 0.0), 0)
                    d2 = tl.sum(tl.where((o_t == 2)[:, None], b_dp, 0.0), 0)
                    z = tl.zeros((D,), dtype=tl.float32)
                    g0 = z + tl.where(0 < T, d0 * w0, 0.0)
                    g1 = z + tl.where(0 < T, d0 * w1, 0.0)
                    g1 += tl.where(1 < T, d1 * w0, 0.0)
                    g2 = z + tl.where(0 < T, d0 * w2, 0.0)
                    g2 += tl.where(1 < T, d1 * w1, 0.0)
                    g2 += tl.where(2 < T, d2 * w0, 0.0)
                    ot = dtail0.dtype.element_ty
                    tl.store(dtail0 + cols, g0.to(ot))
                    tl.store(dtail0 + stride_xt + cols, g1.to(ot))
                    tl.store(dtail0 + 2 * stride_xt + cols, g2.to(ot))
        b_dp *= _w_col(3 - i, w0, w1, w2, w3)[None, :]
        b_dx += b_dp
    if HAS_DTAIL:
        # gradient của 3 token đuôi ra (tail = x[T-3:]) cộng vào 3 hàng cuối
        if it * BT + BT >= T - 3:
            k = o_t - (T - 3)
            m_k = (k >= 0) & (o_t < T)
            b_dx += tl.load(dtail + k[:, None] * stride_xt + cols[None, :], mask=m_k[:, None],
                            other=0.0).to(tl.float32)
    tl.store(dx + o_t[:, None] * stride_xt + cols[None, :], b_dx.to(dt), mask=(o_t < T)[:, None])


def conv_l2_bwd(x, w, tail_r, qn, kn, rq, rk, dq, dk, dk2, dv, dtail_r, need_tail0):
    """x: qkv đầy đủ -> (dx [1, T, C], dtail0 [3, C] | None). tail_r, dtail_r: [3, C] liền khối hoặc None."""
    T, C = x.shape[1], x.shape[2]
    hk, d, hv = qn.shape[2], qn.shape[3], dv.shape[2]
    dx = torch.empty((1, T, C), dtype=x.dtype, device=x.device)
    dtail0 = torch.empty((3, C), dtype=tail_r.dtype, device=x.device) if need_tail0 else None
    for it0, n, has_tail in _row_blocks(T, _BWD_BT, tail_r is not None):
        _conv_l2_bwd_kernel[(n, 2 * hk + hv)](
            x, w, tail_r if has_tail else x, qn, kn, rq, rk, dq, dk, dk2, dv,
            dtail_r if dtail_r is not None else x, dx, dtail0 if dtail0 is not None else x,
            T, x.stride(1), it0, HK=hk, HV=hv, D=d, BT=_BWD_BT, HAS_TAIL=has_tail,
            HAS_DTAIL=dtail_r is not None, NEED_DTAIL0=need_tail0 and has_tail, num_warps=_BWD_WARPS)
    return dx, dtail0


# cấu hình kernel (quét trên RTX 5060 Laptop: fwd gần như không đổi theo BT; bwd BT=16/4 warp nhanh nhất, 0.46 so với
# 0.65 ms của 32/4 ở T=2400; maxnreg làm tràn thanh ghi, chậm hơn)
_FWD_BT, _FWD_WARPS = 64, 4
_BWD_BT, _BWD_WARPS = 16, 4


# ------------------------------------------------------------------ autograd

class _GDNCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, conv_w, tail0, g, beta, h0, want_state, save_inner, hk, hv, d, xs, bq, s, need_bwd):
        ctx.set_materialize_grads(False)
        x = qkv.contiguous()  # đầu ra group_linear vốn liền khối (không chép); dx ghi cùng stride
        lora = (xs, bq, s) if xs is not None else None
        # đuôi conv vào dạng hàng [3, C] (cùng bố cục với x, để L2norm rút gọn đúng thứ tự như l2norm_fwd của fla);
        # đuôi do chính hàm này trả ra (xem dưới) vốn đã là view của [3, C] liền khối nên không chép
        tail_r = tail0[0].t().contiguous() if tail0 is not None else None
        qn, kn, v, rq, rk, tail_out, x_full = conv_l2_fwd(x, conv_w, tail_r, hk, hv, d, lora, want_state, need_bwd)
        if x_full is not None:
            x = x_full  # backward đọc qkv đầy đủ; phần gốc không giữ (như addmm_ tại chỗ của đường cũ)
        g, beta = g.contiguous(), beta.contiguous()
        h0 = h0.contiguous() if h0 is not None else None
        # đúng trình tự chunk_gated_delta_rule_fwd của fla (chunk 64, không cu_seqlens)
        gc = _fc.chunk_local_cumsum(g, chunk_size=_BT, scale=_fc.RCP_LN2)
        w, u, A = _fc.chunk_gated_delta_rule_fwd_intra(k=kn, v=v, g=gc, beta=beta, chunk_size=_BT)
        h, v_new, rec = _fc.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=gc, initial_state=h0,
                                                         output_final_state=want_state, chunk_size=_BT)
        del u
        o = _fc.chunk_fwd_o(q=qn, k=kn, v=v_new, h=h, g=gc, scale=d ** -0.5, chunk_size=_BT)
        # 3 token đuôi conv cho đoạn sau (= final_state[..., 1:] của causal_conv1d; T >= 3 do usable() bảo đảm):
        # [1, C, 3], view của [3, C] liền khối do kernel ghi
        tail = tail_out.t()[None] if want_state else None
        inner = (w, h, v_new) if save_inner else (None, None, None)
        ctx.save_for_backward(x, conv_w, tail_r, qn, kn, v, rq, rk, gc, beta, A, h0, *inner)
        ctx.meta = (d, save_inner, g.dtype)
        return o, tail, rec

    @staticmethod
    def backward(ctx, do, dtail, drec):
        x, conv_w, tail_r, qn, kn, v, rq, rk, gc, beta, A, h0, w, h, v_new = ctx.saved_tensors
        d, inner, g_dt = ctx.meta
        scale = d ** -0.5
        need = ctx.needs_input_grad
        do = torch.zeros_like(v) if do is None else do.contiguous()
        dht = drec.contiguous() if drec is not None else None
        if not inner:  # như chunk_gated_delta_rule_bwd: tính lại w, u, h
            w, u = _fc.recompute_w_u_fwd(k=kn, v=v, beta=beta, A=A, g=gc)
            h, v_new, _ = _fc.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=gc, initial_state=h0,
                                                           output_final_state=False, chunk_size=_BT)
            del u
        dv = _fc.chunk_bwd_dv_local(q=qn, k=kn, g=gc, do=do, scale=scale, chunk_size=_BT)
        dh, dh0, dv = _fc.chunk_gated_delta_rule_bwd_dhu(q=qn, k=kn, w=w, g=gc, h0=h0, dht=dht, do=do, dv=dv,
                                                          scale=scale, chunk_size=_BT)
        dq, dk, dw, dg = _fc.chunk_bwd_dqkwg(q=qn, k=kn, v=v_new, w=w, g=gc, h=h, dv=dv, do=do, dh=dh,
                                             scale=scale, chunk_size=_BT)
        del w, h, v_new, dh, do
        dk2, dv, db, dg2 = _fc.prepare_wy_repr_bwd(k=kn, v=v, beta=beta, g=gc, A=A, dw=dw, du=dv)
        del dw
        dg.add_(dg2)
        dg = _fc.chunk_local_cumsum(dg, chunk_size=_BT, reverse=True)
        dx = dtail0 = None
        need_tail0 = tail_r is not None and need[2]
        if need[0] or need_tail0:
            dtail_r = dtail[0].t().contiguous() if dtail is not None else None
            dx, dtail0 = conv_l2_bwd(x, conv_w, tail_r, qn, kn, rq, rk, dq, dk, dk2, dv, dtail_r, need_tail0)
            if not need[0]:
                dx = None
            if dtail0 is not None:
                dtail0 = dtail0.t()[None]  # [1, C, 3] như tail0
        return (dx, None, dtail0, dg.to(g_dt) if need[3] else None, db.to(beta.dtype) if need[4] else None,
                dh0 if need[5] else None, None, None, None, None, None, None, None, None, None)


def usable(mod, h, state, want_state):
    """Đường gộp áp dụng được cho lớp DeltaNet này không (không thì shared_prefix._gdn chạy đường cũ). Xét TRƯỚC khi
    chiếu qkv (để biết có hoãn LoRA-B của in_proj_qkv được không): dtype của qkv là dtype tính của group_linear."""
    conv = mod.conv1d
    w = conv.weight
    d = mod.head_k_dim
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else h.dtype
    ok = (ENABLED and h.is_cuda and h.dim() == 3 and h.shape[0] == 1 and conv.bias is None
          and not w.requires_grad and w.shape[1:] == (1, 4) and dt == w.dtype
          and mod.head_v_dim == d and d in (16, 32, 64, 128) and mod.num_v_heads == mod.num_k_heads
          and w.shape[0] == 2 * mod.key_dim + mod.value_dim
          and not (want_state and h.shape[1] < 3)
          and (state is None or state[0].shape == (1, w.shape[0], 3)))
    if not ok:
        STATS["đường cũ"] += 1
    return ok


def gdn_core(mod, qkv, g, beta, state, want_state, save_inner, lora=None):
    """-> (o [1, T, H, V], đuôi conv [1, C, 3] | None, trạng thái hồi quy [1, H, K, V] fp32 | None), như chuỗi
    causal_conv1d + split + chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True) của shared_prefix._gdn.
    lora: (xA [T, r], B [C, r] bf16, s) khi qkv là phần gốc chưa cộng LoRA-B (group_linear(..., defer=(0,)))."""
    w = mod.conv1d.weight
    if lora is not None and (lora[1].shape[1] not in (16, 32, 64) or lora[1].dtype != qkv.dtype):
        xs, bq, s = lora  # hạng LoRA không hợp tl.dot: cộng bằng torch như addmm_ cũ (gradient vẫn về qkv gốc)
        qkv, lora = qkv[0].addmm(xs, bq.t(), alpha=s)[None], None
    assert qkv.shape[2] == w.shape[0] and qkv.dtype == w.dtype, (qkv.shape, qkv.dtype, w.dtype)
    STATS["gộp + LoRA" if lora is not None else "gộp"] += 1
    tail0 = None if state is None else state[0].to(w.dtype)
    h0 = None if state is None else state[1]
    xs, bq, s = lora if lora is not None else (None, None, 1.0)
    # có backward không (bước A chạy no_grad): chỉ khi đó kernel mới phải ghi qkv đầy đủ
    need_bwd = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (qkv, tail0, g, beta, h0))
    return _GDNCore.apply(qkv, w.detach().squeeze(1).contiguous(), tail0, g, beta, h0, want_state, save_inner,
                          mod.num_k_heads, mod.num_v_heads, mod.head_k_dim,
                          None if xs is None else xs.detach(), None if bq is None else bq.detach(), s, need_bwd)
