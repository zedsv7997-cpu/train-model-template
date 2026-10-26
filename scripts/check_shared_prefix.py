"""Kiểm tra shared_prefix.py: gradient LoRA khi dùng chung prefix phải khớp cách train từng hội thoại riêng,
và đo thời gian / VRAM một bước.

  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\check_shared_prefix.py --n 4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import train as T  # noqa: E402  (env, vá SDPA, trần VRAM, import unsloth trước)
import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

import shared_prefix as SP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seq", type=int, default=10752)
ap.add_argument("--keep-layers", type=int, nargs="*", default=[0])
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--no-hf", action="store_true", help="bỏ so sánh với forward gốc của transformers/Unsloth")
ap.add_argument("--no-sep", action="store_true", help="bỏ đường tự viết chạy từng hội thoại trọn vẹn")
ap.add_argument("--profile", action="store_true", help="profile kernel GPU của một bước prefix chung")
args = ap.parse_args()

model, processor = FastLanguageModel.from_pretrained(
    model_name=T.local_model_path("Qwen/Qwen3.5-2B"), max_seq_length=args.seq,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
tok = getattr(processor, "tokenizer", processor)
T.install_labeled_only_ce()
T.free_vision_tower(model)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", target_modules=T.ATTN_MLP + T.GDN,
    use_gradient_checkpointing="unsloth", random_state=3407)
model.train()
# LoRA B khởi tạo 0 -> gradient của A bằng 0; đặt B ngẫu nhiên nhỏ để kiểm tra cả hai
torch.manual_seed(0)
for n, p in model.named_parameters():
    if "lora_B" in n:
        p.data.normal_(0, 1e-3)
params = [p for p in model.parameters() if p.requires_grad]
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

# Nhãn giống train_on_responses_only: từ sau "<|im_start|>assistant\n" tới trước "<|im_start|>user\n"
rows = T.read_jsonl(args.train)
key = rows[0]["messages"][0]["content"]
rows = [r for r in rows if r["messages"][0]["content"] == key and len(r["tools"]) == len(rows[0]["tools"])][: args.n]
resp = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
inst = tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
group = []
for ex in rows:
    text, _, _ = T.fit_conversation(tok, ex, args.seq)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    lab, on, i = [-100] * len(ids), False, 0
    while i < len(ids):
        if ids[i:i + len(resp)] == resp:
            i += len(resp); on = True; continue
        if ids[i:i + len(inst)] == inst:
            on = False
        if on:
            lab[i] = ids[i]
        i += 1
    group.append((torch.tensor(ids).cuda(), torch.tensor(lab).cuda()))
p = SP.shared_len([g[0] for g in group], [g[1] for g in group])
print(f"{len(group)} hội thoại, dài {[len(g[0]) for g in group]}, prefix chung {p} token", flush=True)


def grads():
    out = torch.cat([q.grad.float().flatten() for q in params])
    for q in params:
        q.grad = None
    return out


def compare(name, a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / b.norm()).item()
    print(f"  {name}: cos={cos:.6f}  sai lệch tương đối={rel:.4%}", flush=True)


def run(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    loss = fn()
    torch.cuda.synchronize()
    out = loss, time.time() - t, torch.cuda.max_memory_allocated() / 2**30
    if os.environ.get("SHARED_PREFIX_DEBUG") == "1":
        import gc
        before = torch.cuda.memory_allocated()
        n = gc.collect()
        print(f"    [mem] gc.collect() dọn {n} object: {before / 2**30:.2f} -> {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return out


n_items = sum(int((lab[1:] != -100).sum()) for _, lab in group)
if not args.no_hf:
    def hf():
        tot = 0.0
        for ids, lab in group:
            with amp():
                out = model(input_ids=ids[None], labels=lab[None])
            k = int((lab[1:] != -100).sum())
            (out.loss * k / n_items).backward()
            tot += float(out.loss) * k / n_items
        return tot
    g_hf = None
    for r in range(args.rounds):
        loss, dt, mem = run(hf)
        g_hf0, g_hf = g_hf, grads()
        print(f"HF/Unsloth gốc   : loss {loss:.5f} | {dt:.2f}s | VRAM đỉnh {mem:.2f} GiB", flush=True)

g_sep = None
try:
    if args.no_sep:
        raise torch.OutOfMemoryError
    loss, dt, mem = run(lambda: SP.shared_prefix_step(model, group, amp, share=False)[0])
    g_sep = grads()
    print(f"Tự viết, riêng lẻ: loss {loss:.5f} | {dt:.2f}s | VRAM đỉnh {mem:.2f} GiB", flush=True)
except torch.OutOfMemoryError:
    print("Tự viết, riêng lẻ: hết VRAM (chuỗi trọn vẹn quá dài cho đường tự viết) - bỏ qua", flush=True)
    for q in params:
        q.grad = None
    torch.cuda.empty_cache()
for kl in args.keep_layers:
    for r in range(args.rounds):
        loss, dt, mem = run(lambda: SP.shared_prefix_step(model, group, amp, keep_layers=kl)[0])
        g_sh = grads()
        print(f"Prefix chung (keep_layers={kl}): loss {loss:.5f} | {dt:.2f}s | VRAM đỉnh {mem:.2f} GiB", flush=True)
if args.profile:
    from torch.profiler import ProfilerActivity, profile
    lora = next(m for n, m in model.named_modules() if n.endswith("mlp.gate_proj"))
    print("LoRA forward:", type(lora).__module__, type(lora).forward.__qualname__, getattr(lora, "scaling", None))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        SP.shared_prefix_step(model, group, amp, keep_layers=args.keep_layers[-1])
        torch.cuda.synchronize()
    grads()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=80))
    ops = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    ops.sort(key=lambda e: e.self_device_time_total, reverse=True)
    print("Theo phép aten (thời gian GPU tự thân):")
    for e in ops[:25]:
        print(f"  {e.key:40s} {e.self_device_time_total / 1e3:8.1f} ms  x{e.count}")
print("So gradient LoRA:")
if g_sep is not None:
    compare("chung vs riêng lẻ (cùng code)", g_sh, g_sep)
if not args.no_hf:
    if g_sep is not None:
        compare("riêng lẻ vs HF gốc", g_sep, g_hf)
    compare("chung vs HF gốc", g_sh, g_hf)
    if g_hf0 is not None:
        compare("HF gốc lần 1 vs lần 2 (nhiễu nền)", g_hf, g_hf0)
