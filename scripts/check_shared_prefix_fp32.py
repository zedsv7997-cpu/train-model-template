"""Kiểm tra thuật toán prefix dùng chung trên một Qwen3.5 tí hon fp32 (khởi tạo ngẫu nhiên, 3 lớp DeltaNet + 1 lớp
attention như model thật): loss và gradient MỌI tham số phải trùng với forward gốc của transformers chạy từng
chuỗi riêng (chỉ khác sai số fp32). Model thật bf16 có nhiễu nền lớn (chạy HF 2 lần đã lệch vài %), nên phép thử
chính xác làm ở đây.

  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\check_shared_prefix_fp32.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
import transformers.integrations.sdpa_attention as _sdpa  # noqa: E402
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

import shared_prefix as SP  # noqa: E402

_sdpa.use_gqa_in_sdpa = lambda *a, **k: False  # như train.py
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)


def exact_gdr(q, k, v, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
    """Gated delta rule tuần tự từng token bằng PyTorch fp32: gradient chính xác, không có TF32 / ranh giới chunk
    như kernel fla (fla tách chuỗi ở chỗ khác nhau sẽ lệch ~1e-3 do TF32, che mất lỗi thuật toán nếu có)."""
    dtype = q.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    bsz, seq, heads, dk = k.shape
    q = q * dk ** -0.5
    s = initial_state.float() if initial_state is not None else q.new_zeros(bsz, heads, dk, v.shape[-1])
    outs = []
    for t in range(seq):
        s = s * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (s * k[:, t][..., None]).sum(-2)) * beta[:, t][..., None]
        s = s + k[:, t][..., None] * delta[..., None, :]
        outs.append((s * q[:, t][..., None]).sum(-2))
    return torch.stack(outs, 1).to(dtype), (s if output_final_state else None)


import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: E402

_FLA = (M.torch_chunk_gated_delta_rule, SP.chunk_gated_delta_rule)


def use_kernel(exact):
    M.torch_chunk_gated_delta_rule, SP.chunk_gated_delta_rule = (exact_gdr, exact_gdr) if exact else _FLA


# Mặc định: kiểm tra thuật toán bằng kernel chính xác ở cả hai phía.
# --fla: đáp án vẫn tính bằng kernel chính xác, còn các đường được thử dùng kernel fla thật (kể cả HF gốc
#        với fla) -> thấy sai số do fla tự nó, so với sai số thêm vào khi tách prefix.
FLA = "--fla" in sys.argv
use_kernel(True)

cfg = Qwen3_5TextConfig(
    vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=8,
    layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
    linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
)
cfg._attn_implementation = "sdpa"
model = Qwen3_5ForCausalLM(cfg).cuda().float()
with torch.no_grad():  # trọng số norm khởi tạo 0 / A_log ngẫu nhiên: làm nhiễu nhẹ để mọi nhánh đều có gradient
    for n, p in model.named_parameters():
        if "norm" in n:
            p.add_(torch.randn_like(p) * 0.1)
# --lora: LoRA PEFT như train.py (cùng các lớp đích; r=4, alpha=8 -> scaling 2 để thử cả nhánh scaling != 1), B khởi tạo
# ngẫu nhiên: kiểm tra đường gộp fused_linear.group_linear. Không có cờ: full finetune (group_linear gọi thẳng module).
# --lora-from N (cùng --lora): LoRA chỉ ở lớp >= N như train.py --lora-from -> các lớp dưới chạy không đồ thị, bước C
# dừng ở lớp N; đáp án HF vẫn lan ngược qua mọi lớp.
LORA_FROM = int(sys.argv[sys.argv.index("--lora-from") + 1]) if "--lora-from" in sys.argv else 0
if "--lora" in sys.argv:
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "out_proj"],
        layers_to_transform=list(range(LORA_FROM, cfg.num_hidden_layers)) if LORA_FROM else None))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    print("LoRA ở các lớp", sorted({int(n.split(".layers.")[1].split(".")[0]) for n, _ in model.named_parameters()
                                    if "lora_A" in n}))
model.train()
params = [p for p in model.parameters() if p.requires_grad]

P, lens = 300, [460, 380, 529]
prefix = torch.randint(0, 1000, (P,))
group = []
for L in lens:
    ids = torch.cat([prefix, torch.randint(0, 1000, (L - P,))])
    lab = ids.clone()
    lab[: P + 5] = -100
    lab[torch.rand(L) < 0.4] = -100
    group.append((ids.cuda(), lab.cuda()))
n_items = sum(int((lab[1:] != -100).sum()) for _, lab in group)


names = [n for n, q in model.named_parameters() if q.requires_grad]


def take():
    g = [q.grad.clone() if q.grad is not None else torch.zeros_like(q) for q in params]
    for q in params:
        q.grad = None
    return g


def diff(g, ref):
    """Sai lệch tương đối toàn cục và của tham số tệ nhất."""
    flat, flat_ref = torch.cat([x.flatten() for x in g]), torch.cat([x.flatten() for x in ref])
    per = [((a - b).norm() / b.norm()).item() if b.norm() > 0 else (a.norm()).item() for a, b in zip(g, ref)]
    i = max(range(len(per)), key=per.__getitem__)
    if os.environ.get("VERBOSE") == "1":
        ab = [(a - b).norm().item() for a, b in zip(g, ref)]
        for j in sorted(range(len(per)), key=ab.__getitem__, reverse=True)[:12]:
            print(f"      {names[j]:45s} tương đối {per[j]:.2e} | tuyệt đối {ab[j]:.2e} | |ref| {ref[j].norm().item():.2e}")
    return ((flat - flat_ref).norm() / flat_ref.norm()).item(), per[i], names[i]


def hf_reference():
    loss = 0.0
    for ids, lab in group:
        out = model(input_ids=ids[None], labels=lab[None])
        k = int((lab[1:] != -100).sum())
        (out.loss * k / n_items).backward()
        loss += float(out.loss.detach()) * k / n_items
    return loss


ref_loss = hf_reference()
g_ref = take()
use_kernel(not FLA)

ok, base = True, None
amp = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
runs = [("riêng lẻ", lambda: SP.shared_prefix_step(model, group, amp, share=False)),
        ("prefix chung", lambda: SP.shared_prefix_step(model, group, amp)),
        ("prefix chung, keep_layers=2", lambda: SP.shared_prefix_step(model, group, amp, keep_layers=2))]
if FLA:
    runs.insert(0, ("HF gốc (fla)", lambda: (hf_reference(), 0)))
for name, fn in runs:
    loss, p = fn()
    rel, worst, wname = diff(take(), g_ref)
    print(f"{name:28s} prefix {p:3d} | loss {loss:.6f} vs {ref_loss:.6f} | sai lệch gradient {rel:.2e}"
          f" (tham số tệ nhất {worst:.2e}: {wname})")
    if FLA:  # kernel fla tự nó đã lệch so với đáp án chính xác: đường tách prefix không được lệch hơn rõ rệt
        base = base or rel
        ok &= abs(loss - ref_loss) < 1e-3 and rel < 3 * base
    else:
        ok &= abs(loss - ref_loss) < 1e-4 and rel < 1e-4 and worst < 1e-3
import fused_linear  # noqa: E402
print(f"group_linear: {fused_linear.STATS}")
print("ĐẠT" if ok else "KHÔNG ĐẠT")
sys.exit(0 if ok else 1)
