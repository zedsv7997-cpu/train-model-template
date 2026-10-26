"""Phần elementwise của cross-entropy trên logits [M, V] bf16 (V = 248K): một kernel Triton mỗi hàng, đọc logits 2 lần,
ghi đè p = softmax - one_hot (bf16) lên chính bộ đệm logits.

Bản PyTorch cũ (logits.float(), logsumexp, gather, sub_, exp_, trừ 1, .to(bf16)) đọc/ghi ~3.3 GB cho mỗi khúc 256 hàng
(~26 lần kích thước logits bf16); ở đây ~0.38 GB. Phép tính như cũ trên fp32: lse = max + log(sum exp(x - max)) (cộng dồn
online theo khối thay vì hai lượt), p = exp(x - lse), trừ 1 ở nhãn, làm tròn bf16 một lần.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ce_rows(x_ptr, tgt_ptr, loss_ptr, V, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * stride
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    t = tl.load(tgt_ptr + row)
    xt = tl.load(base + t).to(tl.float32)
    tl.store(loss_ptr + row, lse - xt)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < V, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(offs == t, p - 1.0, p)
        tl.store(base + offs, p.to(x_ptr.dtype.element_ty), mask=offs < V)


def ce_rows_(logits, tgt):
    """logits [M, V] bf16 liền khối (bị ghi đè bằng softmax - one_hot), tgt [M] -> loss từng hàng [M] fp32."""
    M, V = logits.shape
    assert logits.is_contiguous() and tgt.numel() == M
    loss = torch.empty(M, dtype=torch.float32, device=logits.device)
    _ce_rows[(M,)](logits, tgt, loss, V, logits.stride(0), BLOCK=4096, num_warps=8)
    return loss
