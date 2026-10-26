"""Train một nhóm hội thoại có chung phần đầu (system prompt + tool, vd ~6.4K token) mà chỉ tính phần đó MỘT lần/bước.

Mỗi bước gồm một nhóm hội thoại (mặc định 16) ~9.2K token, trong đó ~6.4K token đầu giống hệt nhau -> ~70% công tính
bị lặp lại ở từng hội thoại. Ở đây mỗi bước (một nhóm hội thoại cùng system prompt + tool):
  A. chạy prefix một lần, không lưu đồ thị; giữ đầu vào từng lớp và "trạng thái" mà phần sau đọc từ prefix:
     K/V của 6 lớp attention; trạng thái hồi quy + 3 cột conv cuối của 18 lớp Gated DeltaNet;
  B. từng hội thoại: chạy phần riêng (~3K token) nối tiếp trạng thái đó, backward -> gradient LoRA, đồng thời
     gradient của trạng thái prefix được cộng dồn qua các hội thoại;
  C. lan ngược qua prefix một lần, từ lớp trên xuống (tính lại từng lớp có gradient như gradient checkpointing).
Gradient bằng đúng cách train từng hội thoại riêng (chỉ khác sai số làm tròn) - không bỏ gradient qua prefix
(bản bỏ bước C, --prefix-no-grad, đã thử: nhanh hơn ~20% nhưng model kém rõ -> đã xoá khỏi code).

Forward từng lớp viết lại theo modeling_qwen3_5.py (transformers 5.17) nhưng gọi đúng các module của model
(LoRA, norm, MLP), nên trọng số/adapter vẫn là của model đang train.
"""
import concurrent.futures
import gc
import os
import random
import sys
import time
import types
from collections import defaultdict

import torch
import torch.nn.functional as F
from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import flex_attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

import attn_bwd
import fused_gdn
from fused_ce import ce_rows_
from fused_linear import STATS, group_linear, new_step, residual_add

_DEBUG = os.environ.get("SHARED_PREFIX_DEBUG") == "1"
_TIME = os.environ.get("SHARED_PREFIX_TIME") == "1"  # in thời gian mỗi bước, không chèn đồng bộ GPU như _DEBUG
# Loss trung bình mỗi token lớn hơn ngưỡng này là vô lý (đoán đều 248K token chỉ ~12.4) -> bỏ bước đó
_LOSS_SKIP = float(os.environ.get("TRAIN_LOSS_SKIP", "20"))
_T = [0.0]

# Attention head_dim 256 trên RTX 5060 (sm120): Windows không có FlashAttention, cuDNN không hỗ trợ, kernel
# mem-efficient (cutlass sm80, nhánh head dim lớn) chậm. FlexAttention (Triton) với block 32 vừa 101 KB shared memory,
# nhanh ~1.5 lần (đo: 62.6 vs 94 ms fwd+bwd, phần riêng 4K token nhìn 10.4K token), có GQA sẵn nên khỏi chép K/V.
# dynamic=True: compile một lần (~6 s) rồi chạy mọi độ dài chuỗi không compile lại.
# Backward: num_stages 3 nhanh hơn 1 khoảng 9% (quét 2026-10-21, phần riêng 3K token nhìn 9.4K: 32.9 -> 29.9 ms; block
# lớn hơn 32 thì hết shared memory với head_dim 256), gradient y hệt. Quét lại 2026-10-23 (torch 2.14, q 2.6K nhìn 9K):
# backward vẫn tốt nhất ở cấu hình này; forward khối 64x32 + num_stages 2 nhanh hơn 11% (7.04 -> 6.28 ms), đầu ra y hệt.
_FLEX = None
_FLEX_OPTS = dict(BLOCK_M=32, BLOCK_N=32, BLOCK_M1=32, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=32, num_stages=1, num_warps=4,
                  bwd_num_stages=3, fwd_BLOCK_M=64, fwd_BLOCK_N=32, fwd_num_stages=2, fwd_num_warps=4)


def checkpoint(fn, *args, use_reentrant=False):
    """torch.utils.checkpoint non-reentrant gốc của torch. use_gradient_checkpointing="unsloth" thay
    torch.utils.checkpoint.checkpoint bằng bản reentrant (UnslothCheckpointFunction), và train.py import module này
    sau khi Unsloth vá nên trước 2026-10-23 các lớp checkpoint chạy bản đó: lúc backward tính lại TOÀN BỘ đoạn (cả
    phép cộng residual cuối lớp, cat K/V prefix), còn bản non-reentrant dừng ngay khi tensor lưu cuối cùng đã có
    (early stop). Gradient như nhau; bench và train giờ cùng một đường. Model không có dropout nên khỏi giữ RNG."""
    ck = getattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None) or torch.utils.checkpoint.checkpoint
    if ck.__name__.startswith("unsloth"):
        raise RuntimeError(f"không tìm thấy torch.utils.checkpoint gốc (đang là {ck.__name__})")
    return ck(fn, *args, use_reentrant=False, preserve_rng_state=False)


def _flex(*args, **kw):
    global _FLEX
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=True)
    return _FLEX(*args, **kw)


def _mem(tag):
    if _DEBUG:
        torch.cuda.synchronize()
        now = time.time()
        print(f"    [mem] {tag}: {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
              f" (đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f}) | {now - _T[0]:.2f}s", flush=True)
        _T[0] = now


def model_parts(model):
    text = next(m for m in model.modules() if type(m).__name__ == "Qwen3_5TextModel")
    head = next(m for n, m in model.named_modules() if n.endswith("lm_head"))
    return text, head


_BLK = 128  # khối thưa mặc định của create_block_mask


def block_mask(lim, q_len, kv_len, mask_mod):
    """BlockMask cho mask dạng "query i thấy kv <= lim[i]" (lim không giảm), dựng thẳng ở mức khối 128x128 thay vì
    create_block_mask tính mask dày đặc [q_len, kv_len] (~7 ms mỗi phần riêng 2.6K x 9K). Phân loại khối như
    create_block_mask (khối chạm phần đệm ngoài q_len / kv_len không bao giờ "full"), rồi cùng _dense_to_ordered +
    from_kv_blocks -> BlockMask giống từng phần tử, kernel chạy y hệt. lim: tensor [q_len] trên GPU."""
    from torch.nn.attention.flex_attention import BlockMask, _dense_to_ordered
    dev = lim.device
    qb = torch.arange(0, q_len, _BLK, device=dev)
    kb = torch.arange(0, kv_len, _BLK, device=dev)[None]
    lo = lim[qb][:, None]                                       # lim nhỏ nhất của khối q
    hi = lim[(qb + _BLK - 1).clamp(max=q_len - 1)][:, None]     # lim lớn nhất của khối q
    full = (kb + _BLK - 1 <= lo) & (kb + _BLK <= kv_len) & (qb + _BLK <= q_len)[:, None]
    part = (kb <= hi) & ~full
    return BlockMask.from_kv_blocks(*_dense_to_ordered(part[None, None]), *_dense_to_ordered(full[None, None]),
                                    BLOCK_SIZE=(_BLK, _BLK), mask_mod=mask_mod, seq_lengths=(q_len, kv_len))


def _ctx(text, x, start, kv_len):
    """RoPE cho vị trí start.. + block mask nhân quả: token thứ i của đoạn thấy mọi token kv có chỉ số <= i + start
    (prefix có start=0 -> nhân quả thường; phần riêng thấy cả prefix). Tính một lần, dùng cho cả 6 lớp attention."""
    seq = x.shape[1]
    pos = torch.arange(start, start + seq, device=x.device).view(1, 1, -1).expand(3, x.shape[0], -1)
    cos, sin = text.rotary_emb(x, pos)
    off = kv_len - seq
    qpos = torch.arange(off, kv_len, device=x.device, dtype=torch.int32)  # query i thấy kv <= qpos[i]

    def causal(b, h, qi, ki):
        return ki <= qi + off
    return cos, sin, block_mask(qpos, seq, kv_len, causal), qpos


def _gdn(mod, h, state, want_state, defer=False, save_inner=False):
    """Qwen3_5GatedDeltaNet.forward, thêm trạng thái vào (state) / ra (want_state). state = (3 token cuối của đầu vào
    conv [B, C, 3], trạng thái hồi quy [B, H, K, V]). defer: đầu ra out_proj có thể là Deferred (cho residual_add).
    save_inner: backward chạy ngay sau lần forward này (lớp checkpoint đang tính lại, bước C) -> lõi gộp giữ luôn
    w, h, v_new của fla (+16 KB/token tới lúc backward lớp đó) thay vì tính lại; lớp keep để False."""
    bsz, seq, _ = h.shape
    # Lõi gộp (fused_gdn.py): conv + SiLU + L2norm một kernel ghi thẳng q̂/k̂/v liền khối, bwd một kernel; kết quả
    # trùng từng bit với đường dưới. check_shared_prefix_fp32.py thay chunk_gated_delta_rule bằng bản PyTorch chính xác
    # -> khi đó (và khi lớp không hợp: conv weight cần grad, GVA...) chạy đường cũ.
    fused = chunk_gated_delta_rule is fused_gdn.FLA_CGDR and fused_gdn.usable(mod, h, state, want_state)
    qkv, z, b, a = group_linear(h, (mod.in_proj_qkv, mod.in_proj_z, mod.in_proj_b, mod.in_proj_a))
    z = z.reshape(bsz, seq, -1, mod.head_v_dim)
    beta = b.sigmoid()
    g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
    if fused:
        o, tail, rec = fused_gdn.gdn_core(mod, qkv, g, beta, state, want_state, save_inner)
        o = mod.norm(o.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(bsz, seq, -1)
        return group_linear(o, (mod.out_proj,), defer=defer)[0], ((tail, rec) if want_state else None)
    # Conv nhân quả theo kênh + SiLU bằng kernel Triton của fla (layout [B, T, C], nhanh ~4 lần F.conv1d depthwise).
    # 3 token đuôi của đoạn trước vào qua initial_state [B, C, W] (cột đầu đệm 0, chỉ W-1 cột cuối được đọc) thay vì
    # nối vào đầu chuỗi: kết quả y hệt từng bit, khỏi chép cả [T, 6144] khi nối và khi cắt gradient (-1 ms/lần, T=3K);
    # final_state = W token cuối, 3 cột cuối là đuôi cho đoạn sau.
    w = mod.conv1d.weight
    init = None if state is None else F.pad(state[0].to(w.dtype), (1, 0))
    mixed, fin = causal_conv1d(qkv.to(w.dtype), w.squeeze(1), mod.conv1d.bias, initial_state=init,
                               output_final_state=want_state, activation="silu")
    tail = fin[..., 1:] if want_state else None
    q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    q = q.reshape(bsz, seq, -1, mod.head_k_dim)
    k = k.reshape(bsz, seq, -1, mod.head_k_dim)
    v = v.reshape(bsz, seq, -1, mod.head_v_dim)
    rep = mod.num_v_heads // mod.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    o, rec = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None if state is None else state[1],
        output_final_state=want_state, use_qk_l2norm_in_kernel=True)
    o = mod.norm(o.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(bsz, seq, -1)
    return group_linear(o, (mod.out_proj,), defer=defer)[0], ((tail, rec) if want_state else None)


def _kv(mod, k, v):
    shape = k.shape[:-1]
    k = mod.k_norm(k.view(*shape, -1, mod.head_dim)).transpose(1, 2)
    v = v.view(*shape, -1, mod.head_dim).transpose(1, 2)
    return k, v


def _swiglu_eager(g, u):
    return F.silu(g) * u


_SWIGLU = None


def _mlp(mlp, x, need_out=True):
    """down(silu(gate(x)) * up(x)) như Qwen3_5MLP, với gate/up gộp (group_linear) và silu*mul compile thành một kernel
    mỗi chiều (MLP Unsloth compile bị ngắt ở mỗi GEMM MXFP8 nên silu và phép nhân thành hai kernel, lan ngược còn
    nhân dy với scaling 1.0). need_out=False: bỏ GEMM của down (xem group_linear)."""
    global _SWIGLU
    if type(mlp.act_fn).__name__ not in ("SiLU", "SiLUActivation"):  # transformers: SiLUActivation = F.silu
        STATS["mlp gọi thẳng"] = STATS.get("mlp gọi thẳng", 0) + 1
        return mlp(x)
    # gate/up/down đều LoRA + MXFP8 (gate/up ghép ở mxfp8.enable_mxfp8): cả MLP trong một autograd.Function, 2 kernel
    # Triton (fused_mlp.py). Chỉ import khi đã ghép (--no-mxfp8 không nạp mxfp8/torchao).
    if getattr(getattr(mlp.gate_proj, "base_layer", mlp.gate_proj), "_mx_gu", None) is not None:
        import fused_mlp
        y = fused_mlp.mlp(mlp, x, need_out)
        if y is not None:
            return y
    if _SWIGLU is None:
        _SWIGLU = torch.compile(_swiglu_eager, dynamic=True)
    g, u = group_linear(x, (mlp.gate_proj, mlp.up_proj))
    return group_linear(_SWIGLU(g, u), (mlp.down_proj,), need_out)[0]


def _recomputing():
    """Đang tính lại forward bên trong backward (checkpoint, kiểu reentrant của Unsloth lẫn non-reentrant của torch):
    khi đó giá trị đầu ra của đoạn checkpoint không được đọc, autograd chỉ cần đồ thị để lan ngược."""
    return torch._C._current_graph_task_id() != -1


def _attn_in(layer, x, ctx, *state):
    """Nửa đầu lớp attention: norm, chiếu q/k/v, RoPE, nối K/V prefix. state = (K, V) của prefix (đã norm + RoPE,
    2 head KV chưa lặp). Trả về (q, K, V, gate, k, v)."""
    mod = layer.self_attn
    h = layer.input_layernorm(x)
    shape = h.shape[:-1]
    qg, k, v = group_linear(h, (mod.q_proj, mod.k_proj, mod.v_proj))
    q, gate = torch.chunk(qg.view(*shape, -1, mod.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = mod.q_norm(q).transpose(1, 2)
    k, v = _kv(mod, k, v)
    q, k = apply_rotary_pos_emb(q, k, ctx[0], ctx[1])
    kk, vv = (k, v) if not state else (torch.cat([state[0], k], 2), torch.cat([state[1], v], 2))
    return q, kk, vv, gate, k, v


def _attn_core(mod, q, kk, vv, ctx):
    """Lõi attention. Khi cần gradient: forward vẫn của Flex, backward tự viết (attn_bwd.py: 5 matmul thay vì 7,
    đo 2026-10-23 phần riêng 2600 nhìn 9000: backward 26.0 -> 15.0 ms, lớp trên 700 hàng thưa 8.0 -> 4.2, prefix
    bước C 6400: 24.7 -> 17.2), gradient chỉ khác thứ tự cộng dQ. Không áp dụng được (batch > 1, head_dim lạ, đặt
    SHARED_PREFIX_ATTN_BWD=0, ctx không có qpos) thì dùng backward của Flex như cũ."""
    if (len(ctx) > 3 and torch.is_grad_enabled() and (q.requires_grad or kk.requires_grad or vv.requires_grad)
            and attn_bwd.supported(q, kk, vv)):
        STATS["attn bwd tự viết"] = STATS.get("attn bwd tự viết", 0) + 1
        return attn_bwd.AttnCore.apply(q, kk, vv, ctx[2], ctx[3], mod.scaling, _flex, _FLEX_OPTS)
    return _flex(q, kk, vv, block_mask=ctx[2], scale=mod.scaling, enable_gqa=True, kernel_options=_FLEX_OPTS)


def _attn_out(layer, x, o, gate, need_out=True):
    """Nửa sau lớp attention: cổng sigmoid, o_proj, residual, MLP."""
    o = o.transpose(1, 2).reshape(*gate.shape) * torch.sigmoid(gate)
    # nhánh LoRA của o_proj cộng trong kernel residual (residual-lora), khỏi addmm_ đọc + ghi lại [T, 2048]
    x = residual_add(x, group_linear(o, (layer.self_attn.o_proj,), defer=True)[0])
    return x + _mlp(layer.mlp, layer.post_attention_layernorm(x), need_out)


def _attn_out_ckpt(layer, x, o, gate):
    return _attn_out(layer, x, o, gate, not _recomputing())


def layer_forward(layer, x, ctx, state=None, want_state=False, need_out=True, save_inner=False):
    """need_out=False: giá trị đầu ra lớp không được đọc (chỉ làm gốc cho autograd.backward) -> bỏ GEMM của
    down_proj ở cuối lớp; gradient y hệt. save_inner: xem _gdn."""
    if layer.block_type == "linear_attention":
        m, st = _gdn(layer.linear_attn, layer.input_layernorm(x), state, want_state, defer=True, save_inner=save_inner)
        x = residual_add(x, m)  # + LoRA của out_proj trong cùng kernel (xem _attn_out)
        return x + _mlp(layer.mlp, layer.post_attention_layernorm(x), need_out), st
    q, kk, vv, gate, k, v = _attn_in(layer, x, ctx, *(state or ()))
    o = _attn_core(layer.self_attn, q, kk, vv, ctx)
    return _attn_out(layer, x, o, gate, need_out), ((k, v) if want_state else None)


def layer_state(layer, x, ctx):
    """Chỉ phần trạng thái mà đoạn sau cần của một lớp (dùng cho lớp trên cùng của prefix: đầu ra của nó
    không đi đâu cả). Lớp attention: chỉ K/V - bỏ q, attention, o_proj, MLP."""
    h = layer.input_layernorm(x)
    if layer.block_type == "linear_attention":
        return _gdn(layer.linear_attn, h, None, True, save_inner=True)[1]  # bước A (không grad) hoặc C (backward ngay sau)
    mod = layer.self_attn
    k, v = _kv(mod, *group_linear(h, (mod.k_proj, mod.v_proj)))
    _, k = apply_rotary_pos_emb(k, k, ctx[0], ctx[1])
    return k, v


def _layer_out(layer, x, ctx, *state):
    return layer_forward(layer, x, ctx, state or None, False)[0]


def _layer_out_ckpt(layer, x, ctx, *state):
    # save_inner cả ở lần forward đầu: checkpoint không giữ tensor lưu của lần đó (không tốn gì), và số tensor lưu
    # phải khớp giữa hai lần (determinism_check của checkpoint)
    return layer_forward(layer, x, ctx, state or None, False, not _recomputing(), save_inner=True)[0]


def _ctx_rows(x, ctx, start, kv_len, rows):
    """RoPE + block mask cho query chỉ ở các vị trí rows (chỉ số trong đoạn, tăng dần) của đoạn bắt đầu ở start:
    query thứ i thấy mọi token kv có chỉ số <= start + rows[i]."""
    qpos = (rows + start).to(torch.int32)

    def causal(b, h, qi, ki):
        return ki <= qpos[qi]
    return ctx[0][:, rows], ctx[1][:, rows], block_mask(qpos, rows.numel(), kv_len, causal), qpos


def _top_attn_in(layer, x, ctx, ctx_q, rows, *state):
    """_attn_in của lớp trên cùng: K/V ở mọi vị trí (query sau đọc chúng), q + cổng chỉ ở các vị trí rows."""
    mod = layer.self_attn
    h = layer.input_layernorm(x)
    k, v = _kv(mod, *group_linear(h, (mod.k_proj, mod.v_proj)))
    qg = group_linear(h[:, rows], (mod.q_proj,))[0]
    q, gate = torch.chunk(qg.view(1, rows.numel(), -1, mod.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(1, rows.numel(), -1)
    q = mod.q_norm(q).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, ctx_q[0], ctx_q[1])
    _, k = apply_rotary_pos_emb(k, k, ctx[0], ctx[1])
    kk, vv = (k, v) if not state else (torch.cat([state[0], k], 2), torch.cat([state[1], v], 2))
    return q, kk, vv, gate


def _top_attn_out(layer, x, rows, o, gate):
    return _attn_out(layer, x[:, rows], o, gate)


def _top_attn_out_ckpt(layer, x, rows, o, gate):
    return _attn_out(layer, x[:, rows], o, gate, not _recomputing())


def _top_suffix_layer(layer, x, ctx, ctx_q, rows, state, keep):
    """Lớp trên cùng của phần riêng khi nó là lớp attention: đầu ra chỉ cần ở các vị trí có nhãn (rows) - ở vị trí
    khác nó không đi đâu (sau nó chỉ còn norm cuối + CE), nên bỏ q / attention / o_proj / MLP ở đó là CHÍNH XÁC.
    K/V vẫn tính ở mọi vị trí. Trả về [1, len(rows), hidden]. Checkpoint như _suffix_layer."""
    if keep:
        q, kk, vv, gate = _top_attn_in(layer, x, ctx, ctx_q, rows, *state)
        return _top_attn_out(layer, x, rows, _attn_core(layer.self_attn, q, kk, vv, ctx_q), gate)
    q, kk, vv, gate = checkpoint(_top_attn_in, layer, x, ctx, ctx_q, rows, *state, use_reentrant=False)
    o = _attn_core(layer.self_attn, q, kk, vv, ctx_q)
    return checkpoint(_top_attn_out_ckpt, layer, x, rows, o, gate, use_reentrant=False)


def label_rows(lab, p):
    """Vị trí (trong phần riêng bắt đầu ở p) có nhãn cho token kế tiếp: hidden ở vị trí j dự đoán lab[p + 1 + j]."""
    return (lab[p + 1:] != -100).nonzero().squeeze(1)


def _suffix_layer(layer, x, ctx, state, keep):
    """Một lớp của phần riêng. keep: không checkpoint (giữ activation). Còn lại checkpoint cả lớp, riêng lớp
    attention thì để lõi attention ngoài checkpoint: output nhỏ (~12 KB/token) mà tính lại tốn ~8 ms/lớp."""
    if keep:
        return _layer_out(layer, x, ctx, *state)
    if layer.block_type == "linear_attention":
        return checkpoint(_layer_out_ckpt, layer, x, ctx, *state, use_reentrant=False)
    q, kk, vv, gate, _, _ = checkpoint(_attn_in, layer, x, ctx, *state, use_reentrant=False)
    o = _attn_core(layer.self_attn, q, kk, vv, ctx)
    return checkpoint(_attn_out_ckpt, layer, x, o, gate, use_reentrant=False)


class _FusedCE(torch.autograd.Function):
    """Tổng cross-entropy; gradient theo hidden tính luôn trong forward từng khúc (softmax - one_hot) @ W,
    nên không giữ logits 248K cột và không phải tính lại chúng khi backward."""

    @staticmethod
    def forward(ctx, h, w, tgt, chunk):
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        dh = torch.empty_like(h)
        dw = torch.zeros_like(w, dtype=torch.float32) if ctx.needs_input_grad[1] else None  # LoRA: lm_head đóng băng
        # Mỗi khúc đọc cả W (1 GB) hai lần (~5.3 ms dù khúc nhỏ) -> khúc đuôi < 64 hàng gộp vào khúc trước
        M = h.shape[0]
        starts = list(range(0, M, chunk))
        if len(starts) > 1 and M - starts[-1] < 64:
            starts.pop()
        for i, j in zip(starts, starts[1:] + [M]):
            t = tgt[i:j]
            if dw is None:  # phần elementwise gộp một kernel (fused_ce.py), ghi p đè lên logits
                logits = h[i:j] @ w.t()
                loss += ce_rows_(logits, t).sum()
                if logits.dtype == dh.dtype:
                    torch.mm(logits, w, out=dh[i:j])
                else:
                    dh[i:j] = logits @ w
                del logits
                continue
            logits = (h[i:j] @ w.t()).float()
            lse = torch.logsumexp(logits, dim=-1)
            loss += (lse - logits.gather(1, t[:, None]).squeeze(1)).sum()
            p = logits.sub_(lse[:, None]).exp_()
            p[torch.arange(t.numel(), device=t.device), t] -= 1
            dh[i:j] = p.to(w.dtype) @ w
            dw += p.t() @ h[i:j].float()
            del logits, p
        ctx.save_for_backward(dh, dw)
        return loss

    @staticmethod
    def backward(ctx, g):
        dh, dw = ctx.saved_tensors
        return dh * g.to(dh.dtype), (dw * g).to(dh.dtype) if dw is not None else None, None, None


# (W của 65,536 token giữ lại sau tỉa, bảng id cũ -> mới): tính softmax đúng trên từ vựng của model triển khai
# thay vì 248K token (CE nhanh ~3.8 lần). Đặt bằng set_vocab_keep() từ train.py (--vocab-keep).
_VOCAB = None


def set_vocab_keep(w_keep, old2new):
    global _VOCAB
    _VOCAB = (w_keep, old2new) if w_keep is not None else None


def _ce_sum(h, t, w, chunk):
    """Tổng cross-entropy; h [M, hidden] và nhãn t [M] chỉ gồm các vị trí có nhãn."""
    if _VOCAB is not None:
        w, t = _VOCAB[0], _VOCAB[1][t]
    return _FusedCE.apply(h.contiguous(), w, t, chunk)


_PIN = []
_COPY = []


def _copy_stream():
    if not _COPY:
        _COPY.append(torch.cuda.Stream())
    return _COPY[0]


def _pinned(i, n, like):
    """Bộ đệm RAM pinned (tái dùng giữa các bước) giữ đầu vào lớp i của prefix trong lúc chạy phần riêng."""
    while len(_PIN) <= i:
        _PIN.append(None)
    buf = _PIN[i]
    if buf is None or buf.shape[0] < n or buf.shape[1:] != like.shape[1:] or buf.dtype != like.dtype:
        buf = _PIN[i] = torch.empty((max(n, 8192), *like.shape[1:]), dtype=like.dtype, pin_memory=True)
    return buf[:n]


def shared_len(ids, labels):
    """Số token đầu chung của cả nhóm, cắt sao cho token đầu tiên sau prefix chưa có nhãn
    (nhãn đó được dự đoán từ vị trí cuối prefix, mà phần prefix không tính loss)."""
    p = min(len(x) for x in ids) - 1
    for x in ids[1:]:
        diff = (x[:p] != ids[0][:p]).nonzero()
        if len(diff):
            p = int(diff[0])
    for lab in labels:
        pos = (lab != -100).nonzero()
        if len(pos):
            p = min(p, int(pos[0]) - 1)
    return max(p, 0)


# Ước lượng VRAM cho keep_layers tự động (đo trên Qwen3.5-2B, LoRA r16, bf16): activation một lớp không checkpoint
# ~83 KB/token; mỗi lớp checkpoint giữ đầu vào 4 KB/token; lõi attention ngoài checkpoint ~12 KB/token + K/V nối 2 KB/token;
# backward cần thêm ~0.5 GB trên mức sau forward (một lớp tính lại, khúc CE, FlexAttention).
# Đỉnh nhắm tới mặc định 5.0 GiB (env SHARED_PREFIX_PEAK_GB), dưới trần 6.2 GB của train.py. Trước 2026-10-26: đỉnh 5.9
# dưới trần 7, không có expandable_segments, nvidia-smi báo thêm ~1.2 GB (CUDA context + cache phân mảnh): đỉnh 5.6 GiB
# -> ~6.8 GB, 5.9 GiB -> ~7.1 GB; 6.2 GiB (~7.5 GB) đã OOM (bước ~130: đã cấp 6.45 GiB + ~0.5
# GiB phân mảnh chạm trần 7 GiB). Từ 2026-10-26 train.py bật lại expandable_segments (phân mảnh còn ~0.15 GiB) và hạ
# trần vì màn hình chạy trên RTX giữ ~1 GB: đỉnh 5.0 thì bộ thử 400 hội thoại vẫn 11.8 s/bước (5.9: 11.6-12.4), loss y
# hệt; hội thoại dài nhất (10.752 token) đã giữ 0 lớp nên chạm sàn ~5.7 GiB bất kể đỉnh nhắm. WDDM tính cả tiến trình
# khác (llama-server -ngl 0 vẫn giữ ~0.85 GB VRAM; run_train.ps1 chờ GPU rảnh); tổng vượt 8 GB là tràn sang RAM, chậm
# 5-7 lần.
_ACT_PER_TOKEN = 90 * 1024
_FUSED_ACT = os.environ.get("SHARED_PREFIX_FUSED_ACT", "1") != "0"
_PEAK = min(float(os.environ.get("TRAIN_VRAM_GB", "6.2")),
            float(os.environ.get("SHARED_PREFIX_PEAK_GB", "5.0"))) * 2**30
attn_bwd.PEAK_BYTES = _PEAK  # bộ đệm dS của backward attention chỉ nới quá 128 MB (khi bật DS_MAX_MB) nếu còn trống dưới đỉnh này


def _auto_keep(layers, p, s):
    n_attn = sum(layer.block_type != "linear_attention" for layer in layers)
    fixed = torch.cuda.memory_allocated() + s * (len(layers) * 4096 + n_attn * 12288) + n_attn * (p + s) * 2048
    # + 16 KB/token: w, h, v_new mà lõi GDN gộp giữ từ lúc tính lại một lớp checkpoint tới backward của nó (save_inner)
    margin = 0.55 * 2**30 + s * (20 + (16 if fused_gdn.ENABLED else 0)) * 1024
    # MLP gộp (fused_mlp) không lưu h [T, 6144] bf16 cho backward: mỗi lớp giữ bớt đúng 12 KB/token (chưa đo lại trên
    # model thật; SHARED_PREFIX_FUSED_ACT=0 để dùng lại ước lượng cũ)
    fm = sys.modules.get("fused_mlp")
    act = _ACT_PER_TOKEN - (fm.SAVED_LESS if fm is not None and fm.ACTIVE[0] and _FUSED_ACT else 0)
    return max(0, min(len(layers), int((_PEAK - fixed - margin) // (s * act))))


def shared_prefix_step(model, group, amp, keep_layers=0, ce_chunk=256, share=True, p=None, n_items=None, rows=None):
    """Forward + backward cho một nhóm [(input_ids 1D, labels 1D), ...] cùng prefix; gradient cộng vào .grad.

    Loss = tổng cross-entropy mọi token có nhãn của nhóm / tổng số token có nhãn (như Trainer với
    num_items_in_batch khi gradient_accumulation = cỡ nhóm). amp: context manager autocast của trainer.
    keep_layers: số lớp đầu KHÔNG checkpoint ở phần riêng (nhanh hơn, tốn VRAM hơn); -1 = tự chọn theo VRAM trống
    và độ dài từng phần riêng. share=False: chạy từng hội thoại trọn vẹn (để kiểm tra).
    p, n_items: độ dài prefix chung và số token có nhãn, nếu đã tính sẵn trên CPU (_GroupDataset) - tính trên GPU
    ở đây là 1 + cỡ nhóm lần chờ GPU ngay đầu bước. rows: label_rows(lab, p) từng hội thoại trên CPU, tính sẵn cùng p.
    Trả về (loss, số token prefix dùng chung)."""
    new_step()  # bản bf16 của LoRA trong fused_linear tạo lại theo tham số của bước này
    text, head = model_parts(model)
    layers = list(text.layers)
    top = len(layers) - 1
    ids = [g[0] for g in group]
    labels = [g[1] for g in group]
    if p is None:
        p = shared_len(ids, labels) if share else 0
    elif not share:
        p = 0
    if n_items is None:
        n_items = sum(int((lab[1:] != -100).sum()) for lab in labels)
    train_emb = text.embed_tokens.weight.requires_grad  # LoRA: False
    # Lớp thấp nhất có tham số học (train.py --lora-from N: LoRA chỉ ở lớp >= N). Dưới nó không có gì để học nên gradient
    # đi xuống đó không ai dùng: bước B chạy các lớp đó không đồ thị, bước C dừng ở lớp low - gradient như cũ.
    low = 0 if train_emb else next((i for i, layer in enumerate(layers)
                                    if any(q.requires_grad for q in layer.parameters())), 0)
    _mem("bắt đầu")

    # A. prefix một lần, không lưu đồ thị. Đầu vào từng lớp gửi sang RAM pinned (26 MB/lớp, ~1 ms) để nhường VRAM
    # cho phần riêng; lớp trên cùng chỉ cần K/V.
    xs, leaves = [], [None] * len(layers)
    compute, copier = torch.cuda.current_stream(), _copy_stream()
    if p > 0:
        with torch.no_grad(), amp():
            x = text.embed_tokens(ids[0][:p][None])
            ctx_p = _ctx(text, x, 0, p)
            for i, layer in enumerate(layers):
                buf = None
                if i >= low:  # đầu vào lớp, cần lại ở bước C (bước C không xuống dưới lớp low)
                    buf = _pinned(i, p, x[0])
                    copier.wait_stream(compute)  # chép song song với lớp kế tiếp
                    with torch.cuda.stream(copier):
                        buf.copy_(x[0], non_blocking=True)
                    x.record_stream(copier)
                xs.append(buf)
                if i == top:
                    st = layer_state(layer, x, ctx_p)
                else:
                    x, st = layer_forward(layer, x, ctx_p, None, True)
                leaves[i] = tuple(t.detach().requires_grad_(i >= low) for t in st)
        del x
        _mem(f"sau prefix {p} token")

    # B. phần riêng của từng hội thoại, nối tiếp trạng thái prefix
    loss_total = torch.zeros((), device=ids[0].device)
    if rows is None or not share:
        rows = [label_rows(lab, p).cpu() for lab in labels]
    for x_ids, lab, r in zip(ids, labels, rows):
        if r.numel() == 0:
            continue  # không có nhãn nào: không đóng góp loss lẫn gradient
        # Token sau vị trí có nhãn cuối cùng không ảnh hưởng loss nào (nhân quả) -> bỏ
        s = int(r[-1]) + 1
        r = r.to(x_ids.device, non_blocking=True)
        s_ids, tgt = x_ids[p:p + s][None], lab[p + 1:][r]
        # keep: số lớp đầu (tính từ lớp low) KHÔNG checkpoint
        keep = _auto_keep(layers[low:], p, s) if keep_layers < 0 else keep_layers
        with amp():
            x = text.embed_tokens(s_ids)
            if not train_emb:
                x = x.detach()  # khỏi lan gradient xuống embedding đóng băng (lớp low bỏ được GEMM dX)
            ctx_s = _ctx(text, x, p, p + s)
            for i, layer in enumerate(layers):
                if i < low:
                    with torch.no_grad():
                        x = _layer_out(layer, x, ctx_s, *(leaves[i] or ()))
                elif i == top and layer.block_type != "linear_attention":
                    ctx_q = _ctx_rows(x, ctx_s, p, p + s, r)
                    x = _top_suffix_layer(layer, x, ctx_s, ctx_q, r, leaves[i] or (), i - low < keep)
                else:
                    x = _suffix_layer(layer, x, ctx_s, leaves[i] or (), i - low < keep)
                    if i == top:
                        x = x[:, r]
            h = text.norm(x)[0]
            loss = _ce_sum(h, tgt, head.weight, ce_chunk) / max(n_items, 1)
        _mem(f"forward phần riêng {s} token, {r.numel()} nhãn (giữ {keep} lớp)")
        loss.backward()
        loss_total += loss.detach()
        del x, h, loss
        # Lần cộng gradient đầu tiên autograd "mượn" luôn tensor gradient đến; với K/V prefix (trước 2026-10-21 cả
        # 3 token đuôi conv) đó là view của gradient cả đoạn -> giữ thừa tới ~0.5 GB. Chép ra bản gọn.
        for st in leaves:
            for t in st or ():
                if t.grad is not None and t.grad.untyped_storage().nbytes() > t.grad.numel() * t.grad.element_size():
                    t.grad = t.grad.clone()
        if _DEBUG:
            gs = [t.grad for st in leaves if st for t in st if t.grad is not None]
            print(f"    [mem] .grad của lá prefix: {sum(g.numel() * g.element_size() for g in gs) / 2**20:.0f} MiB dữ liệu,"
                  f" giữ {sum(g.untyped_storage().nbytes() for g in gs) / 2**20:.0f} MiB storage")
        _mem("backward phần riêng")

    # C. lan ngược qua prefix một lần (tính lại từng lớp có gradient)
    if p > 0:
        grad_x = None
        dev = ids[0].device

        def fetch(i):  # đầu vào lớp i từ RAM về GPU trên luồng chép, chạy song song lớp đang tính
            t = torch.empty(xs[i].shape, dtype=xs[i].dtype, device=dev)
            copier.wait_stream(compute)
            with torch.cuda.stream(copier):
                t.copy_(xs[i], non_blocking=True)
            return t
        nxt = fetch(top)
        for i in reversed(range(low, len(layers))):
            compute.wait_stream(copier)
            x_in = nxt[None].requires_grad_(i > low or train_emb)
            if i > low:
                nxt = fetch(i - 1)
            with torch.enable_grad(), amp():
                if i == top:
                    out, st = None, layer_state(layers[i], x_in, ctx_p)
                else:
                    out, st = layer_forward(layers[i], x_in, ctx_p, None, True, need_out=False, save_inner=True)
            outs = [t for t, leaf in zip(st, leaves[i]) if leaf.grad is not None]
            grads = [leaf.grad for leaf in leaves[i] if leaf.grad is not None]
            if grad_x is not None:
                outs.append(out)
                grads.append(grad_x)
            # lớp không có LoRA và đầu vào không cần gradient (lớp low): đầu ra không có grad_fn -> không có gì để lan ngược
            pairs = [(t, g) for t, g in zip(outs, grads) if t.requires_grad]
            if pairs:
                torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
            grad_x = x_in.grad if i > low or train_emb else None
            xs[i] = leaves[i] = None
            del out, st, x_in
        if train_emb and grad_x is not None:
            with torch.enable_grad(), amp():
                torch.autograd.backward(text.embed_tokens(ids[0][:p][None]), grad_x)
        _mem("lan ngược prefix")
    return float(loss_total), p


# ---------------------------------------------------------------- gắn vào Trainer

def make_groups(input_ids, group_size, seed, im_start_id):
    """Chia hội thoại thành nhóm group_size cái có cùng lượt system (system prompt + danh sách tool).
    Phần lẻ của các loại gom thành nhóm hỗn hợp (prefix chung ngắn hơn nhưng vẫn đúng)."""
    buckets = defaultdict(list)
    for i, ids in enumerate(input_ids):
        starts = [j for j, t in enumerate(ids[:20000]) if t == im_start_id][:2]
        buckets[tuple(ids[: starts[1]] if len(starts) > 1 else ids)].append(i)
    rng = random.Random(seed)
    groups, rest = [], []
    for key in sorted(buckets, key=lambda k: (len(k), k)):
        b = buckets[key]
        rng.shuffle(b)
        full = len(b) // group_size * group_size
        groups += [b[j:j + group_size] for j in range(0, full, group_size)]
        rest += b[full:]
    groups += [rest[j:j + group_size] for j in range(0, len(rest), group_size)]
    rng.shuffle(groups)
    return groups


class _GroupDataset(torch.utils.data.Dataset):
    """Một nhóm hội thoại mỗi phần tử. Đọc thẳng tensor từ arrow (with_format torch: ~2 ms thay vì ~30 ms đổi list
    Python sang tensor) và chuẩn bị sẵn nhóm kế tiếp trên một luồng nền trong lúc GPU chạy bước hiện tại."""

    def __init__(self, ds, groups):
        self.ds, self.groups = ds.with_format("torch", columns=["input_ids", "labels"]), groups
        self._pool = concurrent.futures.ThreadPoolExecutor(1)
        self._next = None  # (chỉ số, future) của nhóm đang chuẩn bị trước

    def __len__(self):
        return len(self.groups)

    def _load(self, i):
        rows = [self.ds[j] for j in self.groups[i]]
        ids = [r["input_ids"].pin_memory() for r in rows]
        labels = [r["labels"].pin_memory() for r in rows]
        # độ dài prefix chung, số token có nhãn, vị trí có nhãn tính ở đây (CPU, trong lúc GPU còn chạy bước trước)
        p = shared_len(ids, labels)
        return {"shared_group": [{"input_ids": a, "labels": b} for a, b in zip(ids, labels)],
                "prefix_len": p,
                "n_items": sum(int((lab[1:] != -100).sum()) for lab in labels),
                "rows": [label_rows(lab, p) for lab in labels]}

    def __getitem__(self, i):
        nxt, self._next = self._next, None
        out = nxt[1].result() if nxt is not None and nxt[0] == i else self._load(i)
        if i + 1 < len(self.groups):
            self._next = (i + 1, self._pool.submit(self._load, i + 1))
        return out


def enable_shared_prefix(trainer, group_size, im_start_id, keep_layers=-1):
    """Thay dataloader + training_step của trainer (đã qua train_on_responses_only): mỗi bước = một nhóm
    group_size hội thoại cùng prefix. Trainer phải được tạo với gradient_accumulation_steps=1, batch 1."""
    ds = trainer.train_dataset
    groups = make_groups(ds["input_ids"], group_size, trainer.args.seed, im_start_id)
    gds = _GroupDataset(ds, groups)
    sizes = defaultdict(int)
    for g in groups:
        sizes[len(g)] += 1
    print(f"Prefix dùng chung: {len(groups)} bước (cỡ nhóm: {dict(sorted(sizes.items()))})")

    def get_train_dataloader(self):
        # Thứ tự nhóm đã xáo cố định theo seed -> chạy tiếp từ checkpoint bỏ qua đúng các bước đã train
        return torch.utils.data.DataLoader(gds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0],
                                           num_workers=0, pin_memory=True)

    first = [True]
    last = [None]

    def training_step(self, model, inputs, num_items_in_batch=None):
        t_step = time.time()
        if _DEBUG:
            torch.cuda.synchronize()
            t0 = time.time()
            if last[0] is not None:
                print(f"    [time] ngoài training_step (optimizer, dataloader, log...): {t0 - last[0]:.2f}s")
        model.train()
        dev = self.args.device
        group = [(r["input_ids"].to(dev, non_blocking=True), r["labels"].to(dev, non_blocking=True))
                 for r in inputs["shared_group"]]
        loss, p = shared_prefix_step(model, group, self.accelerator.autocast, keep_layers=keep_layers,
                                     p=inputs["prefix_len"], n_items=inputs["n_items"], rows=inputs["rows"])
        if not loss < _LOSS_SKIP:  # cũng bắt NaN/Inf
            # Từng gặp: lỗi bộ nhớ của máy làm hỏng dữ liệu một bước, loss vọt lên ~1e9. Gradient bước đó là rác
            # (AdamW chuẩn hóa nên vẫn đẩy mọi tham số một bước cỡ LR) -> bỏ, optimizer bỏ qua tham số có grad None
            print(f"!! bước {self.state.global_step + 1}: loss {loss:.3g} vô lý (> {_LOSS_SKIP}) -> bỏ cập nhật bước này",
                  flush=True)
            model.zero_grad(set_to_none=True)
            loss = 0.0
        if _DEBUG:
            torch.cuda.synchronize()
            last[0] = time.time()
            print(f"    [time] training_step {last[0] - t0:.2f}s | prefix {p} | phần riêng {[len(g[0]) - p for g in group]}")
        elif _TIME:  # float(loss) cuối bước đã chờ GPU xong nên số đo vẫn đúng
            print(f"    [time] training_step {time.time() - t_step:.2f}s | phần riêng {[len(g[0]) - p for g in group]}")
        if first[0]:
            # Lần compile FlexAttention đầu tiên để lại ~0.4 GB tensor trong vòng tham chiếu, dọn một lần
            gc.collect()
            first[0] = False
        return torch.tensor(loss, device=group[0][0].device)

    trainer.get_train_dataloader = types.MethodType(get_train_dataloader, trainer)
    trainer.training_step = types.MethodType(training_step, trainer)
    return trainer
