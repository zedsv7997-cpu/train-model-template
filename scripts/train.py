"""Fine-tune Qwen3.5 (2B hoặc 0.8B) gọi tool bằng Unsloth bf16 LoRA, rồi lưu bản merge 16-bit để chuyển sang GGUF.

Unsloth khuyên KHÔNG dùng QLoRA 4-bit cho Qwen3.5 (sai số lượng tử hóa cao bất thường), nên train bf16 LoRA.
Mặc định: hội thoại nhiều lượt tới 10.752 token (system prompt + danh sách tool là phần đầu chung), mỗi bước một nhóm
16 hội thoại chung prefix (shared_prefix.py), LR 4e-4, GEMM trọng số gốc chạy MXFP8 (mxfp8.py). Công thức nhẹ hơn:
--grad-accum 8 --lr 2.8e-4 --no-mxfp8.

Dữ liệu: data/train.jsonl, mỗi dòng {"messages": [...], "tools": [...]} kiểu OpenAI (assistant có tool_calls).
Chạy trên Windows bằng venv %USERPROFILE%\\ml-env (thường qua scripts\\run_train.ps1):
  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\train.py --max-steps 5 --no-merge   # chạy thử
  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\train.py --out outputs\\qwen35-2b-x
  ... scripts\\train.py --model Qwen/Qwen3.5-0.8B --out outputs\\qwen35-0.8b-x          # bản đối chứng
  ... scripts\\train.py --init-lora outputs\\qwen35-2b-x\\lora --train data\\train-new.jsonl \\
      --lr 2e-4 --out outputs\\qwen35-2b-x2   # train nối tiếp: dữ liệu mới + một phần dữ liệu cũ
"""
import os

# Trên Windows, packing hybrid thử nghiệm của Unsloth làm Qwen3.5 dừng với lỗi
# "varlen boundary kernels were not invoked" -> tắt trước khi import unsloth
os.environ.setdefault("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "0")
# Cross-entropy chia khúc theo "nửa VRAM trống" đo từ driver ngay sau forward; lúc đó cache của PyTorch đang giữ
# gần hết nên đo ra ~0 và Unsloth dừng ("No or negligible GPU memory"). Đặt cố định ngân sách 0.5 GB cho bước này
# (chỉ đường --no-share-prefix dùng CE của Unsloth; prefix dùng chung có CE riêng ở fused_ce.py).
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "0.5")

import unsloth  # noqa: E402,F401  (phải import trước transformers/trl để Unsloth vá kịp)
import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset, concatenate_datasets  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
import transformers.integrations.sdpa_attention as _sdpa  # noqa: E402

from chat_tokens import encode, render  # noqa: E402

# Torch cho Windows không có FlashAttention. Khi transformers truyền enable_gqa=True (6 lớp attention của Qwen3.5
# dùng GQA), SDPA rơi về kernel "math" O(n²): chuỗi 8K tốn ~4.8GB activation, tràn VRAM sang RAM và chậm ~6 lần.
# Lặp K/V thủ công để SDPA dùng được kernel mem-efficient (đo: 8.96 -> 5.93 GiB, 104s -> 4.6s/chuỗi 8K),
# và tắt hẳn kernel math để nếu có rơi về đó thì báo lỗi thay vì âm thầm chậm.
_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
torch.backends.cuda.enable_math_sdp(False)
# PyTorch mặc định cho cuBLAS cộng dồn split-K của GEMM bf16 bằng bf16: GEMM mảnh của LoRA (N=16) và dh = p @ W của CE
# (cộng dồn qua 248K từ) lệch gấp 1.4-2 lần mức làm tròn bf16 lý tưởng (đo 2026-10-21: dh 3.45e-3 vs 1.69e-3). Tắt đi
# thì mọi GEMM bf16 đạt mức lý tưởng; tốc độ một bước không đổi (đo xen kẽ: 14.45 -> 14.31 s, trong nhiễu).
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# Unsloth gỡ expandable_segments trên Windows (torch cũ không hỗ trợ) và thay bằng roundup_power2_divisions; torch
# 2.14 trên Windows có (đo 2026-10-26). Không có nó, dưới trần VRAM bên dưới cache bị phân mảnh giữ ~1 GiB trống
# không dùng được (OOM ở 4.8 GiB đã cấp / 5.8 GiB trần); có thì còn ~0.15 GiB. TRAIN_EXPANDABLE_SEGMENTS=0 để tắt.
if torch.cuda.is_available() and os.environ.get("TRAIN_EXPANDABLE_SEGMENTS", "1") != "0":
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

# Khóa VRAM của tiến trình train (mặc định 6.2 GB / 8 GB). Windows (WDDM) không báo hết bộ nhớ mà âm thầm đẩy phần vượt
# sang RAM chia sẻ -> chậm 3-6 lần. Có trần thì bộ cấp phát của PyTorch tự dọn cache khi chạm ngưỡng và luôn ở
# trong VRAM thật; nếu thật sự không đủ thì báo OOM rõ ràng thay vì chậm âm thầm. Trần cũ 7 GB; hạ xuống 2026-10-26 vì
# màn hình chạy trên RTX giữ sẵn ~1 GB. Đo trên 48 hội thoại dài nhất (đều chạm 10.752 token, prefix ~6.4K, giữ 0 lớp ở
# shared_prefix): đã cấp đỉnh 5.71 GiB, cache 5.83 GiB, nvidia-smi ~6.0 GB cho tiến trình; 5.8 GiB thì OOM.
TRAIN_VRAM_GB = float(os.environ.get("TRAIN_VRAM_GB", "6.2"))
if torch.cuda.is_available() and TRAIN_VRAM_GB > 0:
    _total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, TRAIN_VRAM_GB * 2**30 / _total), 0)

def install_labeled_only_ce():
    """Chỉ tính lớp output (vocab 248K) ở các vị trí CÓ nhãn.

    Mỗi mẫu ~8-12K token nhưng chỉ ~7% là phần assistant được tính loss; phần còn lại (system prompt + tool,
    câu hỏi, kết quả tool) mang nhãn -100. Unsloth vẫn nhân hidden state của MỌI vị trí với lm_head 248K×2048
    (~1 GFLOP/token mỗi chiều) rồi mới bỏ qua -100 — chiếm ~20% thời gian train. Lọc trước rồi gọi hàm gốc với
    nhãn đã dịch sẵn (shift_labels=False): loss và gradient giữ nguyên, vị trí bị bỏ vốn có gradient 0.
    Phải gọi SAU from_pretrained vì module do Unsloth biên dịch giữ tham chiếu riêng tới hàm loss.
    """
    import sys
    import unsloth_zoo.fused_losses.cross_entropy_loss as ce

    orig = ce.unsloth_fused_ce_loss
    if getattr(orig, "_labeled_only", False):
        return

    def labeled_only(*args, hidden_states=None, labels=None, shift_labels=True, mask=None, **kw):
        if args or hidden_states is None or labels is None:
            return orig(*args, hidden_states=hidden_states, labels=labels, shift_labels=shift_labels, mask=mask, **kw)
        if shift_labels:
            hs, tgt = hidden_states[..., :-1, :], labels[..., 1:]
            if mask is not None:
                tgt = tgt.masked_fill(mask[..., 1:].to(tgt.device) == 0, -100)
        else:
            hs, tgt = hidden_states, labels
        keep = tgt != -100
        return orig(hidden_states=hs[keep].unsqueeze(0), labels=tgt[keep].unsqueeze(0), shift_labels=False, mask=None, **kw)

    labeled_only._labeled_only = True
    ce.unsloth_fused_ce_loss = labeled_only
    patched = [n for n, m in list(sys.modules.items()) if getattr(m, "unsloth_fused_ce_loss", None) is orig]
    for n in patched:
        sys.modules[n].unsloth_fused_ce_loss = labeled_only
    print(f"Loss chỉ tính ở vị trí có nhãn: đã vá {len(patched)} module")


INSTRUCTION_PART = "<|im_start|>user\n"   # cả kết quả tool cũng nằm trong lượt user (<tool_response>) -> không tính loss
RESPONSE_PART = "<|im_start|>assistant\n"
ATTN_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# q/k/v/o chỉ chạm 6 lớp attention đầy đủ; 18 lớp Gated DeltaNet dùng các tên này
GDN = ["in_proj_qkv", "in_proj_z", "out_proj"]


def read_jsonl(path):
    # Đọc bằng json thay vì datasets.load_dataset: arrow sẽ gộp schema của "arguments"/"tools"
    # giữa các dòng và chèn None vào tham số, làm hỏng hội thoại khi render
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def free_vision_tower(model):
    """Giải phóng tháp vision (~331M tham số, ~630 MiB bf16) khỏi GPU: dữ liệu chỉ có chữ nên nó không bao giờ chạy.

    Gán module = None không giải phóng được vì Unsloth còn giữ tham chiếu tới các Parameter; thay p.data bằng tensor
    rỗng trên CPU thì vùng nhớ cũ được trả lại. Merge sau train đọc trọng số gốc từ đĩa nên vision vẫn đủ trong bản merge.
    """
    before = torch.cuda.memory_allocated()
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "visual":
            for p in mod.parameters():
                p.data = torch.empty(0, dtype=p.dtype)
            for b in mod.buffers():
                b.data = torch.empty(0, dtype=b.dtype)
            break
    torch.cuda.empty_cache()
    print(f"Giải phóng tháp vision: {(before - torch.cuda.memory_allocated()) / 2**20:,.0f} MiB VRAM")


def local_model_path(name):
    """Tên model trên Hub -> thư mục bản đã tải trong cache HF (nếu có).

    Merge của Unsloth (save_pretrained_merged) hỏi Hub xem model gốc có tồn tại không; chạy offline
    (HF_HUB_OFFLINE=1) thì lỗi "could not reach the Hugging Face Hub". Dùng đường dẫn cục bộ thì nó đọc thẳng từ đĩa.
    """
    if os.path.isdir(name):
        return name
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name, local_files_only=True)
    except Exception:  # noqa: BLE001  (chưa tải -> để nguyên tên, from_pretrained sẽ tải)
        return name


def fit_conversation(tok, ex, max_len):
    """Render hội thoại; nếu dài hơn max_len thì bỏ dần các lượt cuối (cắt ngay trước một câu hỏi của user,
    nên mỗi lượt giữ lại vẫn trọn vẹn) thay vì bỏ cả hội thoại. Không bao giờ cắt giữa lời gọi tool.
    Trả về (text | None, số token, có bị cắt không)."""
    def ntok(t):
        return len(tok(t, add_special_tokens=False)["input_ids"])

    text = render(tok, ex)
    n = ntok(text)
    if n <= max_len:
        return text, n, False
    msgs = ex["messages"]
    cuts = [i for i, m in enumerate(msgs) if m["role"] == "user"][1:]  # giữ ít nhất lượt đầu
    for c in reversed(cuts):
        t = render(tok, {**ex, "messages": msgs[:c]})
        n = ntok(t)
        if n <= max_len:
            return t, n, True
    return None, n, False


@contextlib.contextmanager
def tokenizers_parallel():
    """Unsloth đặt TOKENIZERS_PARALLELISM=false lúc import (tránh cảnh báo khi fork) nên tokenizer Rust chạy một luồng.
    Windows không fork (dataset_num_proc=1) -> bật đa luồng trong lúc tokenize cả lô là an toàn; thư viện tokenizers
    đọc biến môi trường ở mỗi lần gọi."""
    old = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = old


def tokenize_conversations(tok, rows, max_len, chunk=1024):
    """fit_conversation cho cả tập train, tokenize MỘT lần, cả lô, đa luồng; trả về Dataset input_ids + attention_mask
    (SFTTrainer thấy input_ids thì không tokenize lại). Đường cũ tokenize hai lần, mỗi lần một luồng: đếm token trong
    fit_conversation (~3 phút với 15K hội thoại) rồi SFTTrainer tokenize cột "text" (~4:50). Token ra y hệt (kiểm tra 2026-10-23).
    Khúc giống nhau trong lô (khối system + tool) chỉ tokenize một lần (chat_tokens.encode, y hệt, kiểm tra 2026-10-25).
    Hội thoại dài hơn max_len: bỏ dần lượt cuối như fit_conversation, mỗi vòng thử một điểm cắt cho mọi hội thoại còn dài.
    Chia khúc để không giữ list token của cả tập (~134M số nguyên Python, ~5 GB) cùng lúc.
    Trả về (dataset, văn bản hội thoại đầu, số hội thoại bị cắt, số bị bỏ, tổng token)."""
    parts, first_text, cut, too_long, n_tok = [], None, 0, 0, 0
    with tokenizers_parallel():
        for s in range(0, len(rows), chunk):
            batch = rows[s:s + chunk]
            texts = [render(tok, ex) for ex in batch]
            ids = encode(tok, texts)
            # hội thoại còn dài -> các điểm cắt (trước một câu hỏi của user, giữ lượt đầu), thử từ cuối về
            pending = {i: [k for k, m in enumerate(batch[i]["messages"]) if m["role"] == "user"][1:]
                       for i in range(len(batch)) if len(ids[i]) > max_len}
            while pending:
                for i in [i for i, cs in pending.items() if not cs]:
                    ids[i] = None  # không điểm cắt nào vừa -> bỏ
                    del pending[i]
                idx = list(pending)
                cands = [render(tok, {**batch[i], "messages": batch[i]["messages"][:pending[i].pop()]}) for i in idx]
                for i, t, x in zip(idx, cands, encode(tok, cands)):
                    if len(x) <= max_len:
                        texts[i], ids[i] = t, x
                        cut += 1
                        del pending[i]
            keep = [x for x in ids if x is not None]
            too_long += len(ids) - len(keep)
            n_tok += sum(map(len, keep))
            if first_text is None:
                first_text = next((t for t, x in zip(texts, ids) if x is not None), None)
            if keep:
                parts.append(Dataset.from_dict({"input_ids": keep, "attention_mask": [[1] * len(x) for x in keep]}))
    return concatenate_datasets(parts), first_text, cut, too_long, n_tok


class Speed(TrainerCallback):
    """In tốc độ token/giây và VRAM đỉnh để ước lượng thời gian train."""

    def __init__(self, tokens_per_step):
        self.tps, self.t0, self.step0 = tokens_per_step, None, 0

    def on_step_begin(self, args, state, control, **kw):
        if self.t0 is None:
            self.t0, self.step0 = time.time(), state.global_step  # chạy tiếp từ checkpoint: chỉ đếm bước của lần này

    def on_log(self, args, state, control, logs=None, **kw):
        steps = state.global_step - self.step0
        if self.t0 and steps:
            dt = time.time() - self.t0
            left = (state.max_steps - state.global_step) * dt / steps
            print(f"  ~{self.tps * steps / dt:,.0f} tok/s | {dt / steps:.1f} s/bước"
                  f" | VRAM đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | còn khoảng {left / 3600:.1f} giờ")


class SlowGuard(TrainerCallback):
    """Dừng sớm (có lưu checkpoint) khi bước train chậm bất thường so với các bước đầu: dấu hiệu VRAM bị WDDM đẩy sang
    RAM vì tiến trình khác chiếm GPU (đo: 7 s -> ~60 s/bước). Đã tràn thì chỉ tiến trình mới mới hết, nên train.py
    thoát mã 3 và run_train.ps1 chờ GPU rảnh rồi chạy tiếp từ checkpoint. Tràn ngay từ đầu thì run_train.ps1 đã chặn."""

    def __init__(self, factor, patience=3, skip=2, ref_steps=5):
        self.factor, self.patience, self.skip, self.ref_steps = factor, patience, skip, ref_steps
        self.n, self.times, self.ref, self.slow, self.t = 0, [], None, 0, None
        self.tripped = False

    def on_step_begin(self, args, state, control, **kw):
        self.t = time.time()

    def on_step_end(self, args, state, control, **kw):
        dt = time.time() - self.t
        self.n += 1
        if self.factor <= 0 or self.n <= self.skip:  # vài bước đầu còn compile
            return
        if self.ref is None:
            self.times.append(dt)
            if len(self.times) >= self.ref_steps:
                self.ref = sorted(self.times)[len(self.times) // 2]
            return
        self.slow = self.slow + 1 if dt > self.factor * self.ref else 0
        if self.slow >= self.patience:
            print(f"!! {self.slow} bước liền chậm ({dt:.1f}s so với {self.ref:.1f}s lúc đầu) -> lưu checkpoint, dừng để chạy lại"
                  " (xem có tiến trình nào khác chiếm VRAM không)", flush=True)
            self.tripped = True
            control.should_save = True
            control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--test", default="data/test.jsonl", help="chỉ dùng cho bước thử sinh sau khi train")
    ap.add_argument("--out", default="outputs/qwen35-2b-lora")
    ap.add_argument("--max-seq-len", type=int, default=10752,
                    help="dài hơn thì VRAM 8GB đầy và tràn sang RAM chia sẻ (đo: ~12K token -> 7.87GB, chậm gấp đôi)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="> 0 để chạy thử nhanh")
    ap.add_argument("--lr", type=float, default=4e-4,
                    help="nhân căn 2 mỗi lần cỡ nhóm gấp đôi (nhóm 4: 2e-4; nhóm 8: 2.8e-4; nhóm 16: 4e-4)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="số hội thoại mỗi bước (= cỡ nhóm chung prefix). Từng thử 32 + LR 5.66e-4: kém hơn 16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-from", type=int, default=0, metavar="N",
                    help="chỉ gắn LoRA ở lớp >= N (0 = cả 24 lớp); prefix dùng chung chạy các lớp dưới N không đồ thị và "
                         "dừng lan ngược ở lớp N (gradient các lớp trên như cũ). Đổi công thức -> phải A/B")
    ap.add_argument("--init-lora", metavar="DIR",
                    help="train nối tiếp: bắt đầu từ adapter của một lượt trước (thư mục lora/, cùng --rank và lớp đích) thay vì "
                         "LoRA mới; optimizer + lịch LR vẫn bắt đầu lại. Dữ liệu thường là bộ mới + một phần bộ cũ")
    # LoRA chỉ ở 12 lớp trên cùng (thử 2026-10-20): bước nhanh hơn 33% nhưng model từ chối bừa, kém hẳn -> đã bỏ cờ
    ap.add_argument("--no-gdn", action="store_true", help="chỉ LoRA attention + MLP như hướng dẫn gốc của Unsloth")
    ap.add_argument("--no-merge", action="store_true", help="chỉ lưu adapter LoRA")
    ap.add_argument("--limit", type=int, default=0, help="chỉ dùng N mẫu đầu (để thử nhanh)")
    ap.add_argument("--full-lm-head", action="store_true",
                    help="tắt tối ưu: tính lm_head ở mọi vị trí như Unsloth gốc (chỉ có tác dụng với --no-share-prefix)")
    ap.add_argument("--no-samples", action="store_true", help="bỏ bước thử sinh 3 câu sau khi train (tiết kiệm vài phút)")
    ap.add_argument("--no-share-prefix", action="store_true",
                    help="tắt prefix dùng chung (shared_prefix.py): mỗi hội thoại chạy trọn vẹn qua forward gốc")
    # Mặc định bật từ 2026-10-20: A/B với bf16 cho điểm test không kém, train nhanh gấp đôi
    ap.add_argument("--mxfp8", action="store_true", default=True,
                    help="GEMM của trọng số gốc đóng băng chạy MXFP8 (mxfp8.py): bước nhanh ~25%% nhưng sai số mỗi GEMM "
                         "~3.7%% thay vì 0.17%% của bf16 (mặc định bật)")
    ap.add_argument("--no-mxfp8", dest="mxfp8", action="store_false", help="GEMM bf16")
    ap.add_argument("--vocab-keep", nargs="+", metavar="TEXT",
                    help="tính loss trên đúng từ vựng 65,536 token mà prune_vocab.py sẽ giữ (cùng danh sách --keep-texts "
                         "như bước tỉa prune_vocab.py) thay vì 248K token: CE nhanh ~3.8 lần, loss = loss của "
                         "model triển khai. Chỉ với prefix dùng chung")
    ap.add_argument("--vocab-total", type=int, default=65536)
    ap.add_argument("--counts", default="data/token_counts.json")
    args = ap.parse_args()
    out = Path(args.out)
    done_flag = out / "lora" / "TRAIN_DONE"
    args.model = local_model_path(args.model)

    if done_flag.exists():
        # Lần chạy trước đã train xong nhưng chết trước khi merge -> chỉ nạp LoRA đã lưu rồi merge
        print(f"Đã có {done_flag}: bỏ qua train, nạp LoRA để merge")
        cfg_path = out / "lora" / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not os.path.isdir(cfg.get("base_model_name_or_path", "")):
            cfg["base_model_name_or_path"] = local_model_path(cfg["base_model_name_or_path"])
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        model, processor = FastLanguageModel.from_pretrained(
            model_name=str(out / "lora"), max_seq_length=args.max_seq_len, load_in_4bit=False, load_in_16bit=True)
        tok = getattr(processor, "tokenizer", processor)
    else:
        model, processor, tok = train_lora(args, out)
        model.save_pretrained(str(out / "lora"))
        processor.save_pretrained(str(out / "lora"))
        done_flag.touch()

    if not args.no_samples:
        try:
            sample_generations(model, tok, args.test)
        except Exception as e:  # noqa: BLE001  (chỉ để xem bằng mắt, không được chặn bước merge)
            print("Bỏ qua phần thử sinh:", e)

    if not args.no_merge:
        # Lưu cùng processor để giữ đủ file cấu hình cho convert_hf_to_gguf.py
        model.save_pretrained_merged(str(out / "merged"), processor, save_method="merged_16bit")
        print(f"Đã lưu bản merge: {out / 'merged'}")


def sample_generations(model, tok, test_path):
    """Thử nhanh vài câu trong tập test để nhìn tận mắt định dạng lời gọi tool."""
    FastLanguageModel.for_inference(model)
    test = read_jsonl(test_path) if Path(test_path).exists() else []
    for ex in random.Random(0).sample(test, min(3, len(test))):
        prompt = render(tok, ex, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**ids, max_new_tokens=256, do_sample=False)
        last_user = next(m["content"] for m in reversed(ex["messages"]) if m["role"] == "user")
        print("-" * 70)
        print("USER :", last_user[:200])
        print("GOLD :", json.dumps(ex["messages"][-1].get("tool_calls") or ex["messages"][-1]["content"], ensure_ascii=False)[:300])
        print("MODEL:", tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)[:500])


def load_init_lora(model, path):
    """Chép trọng số LoRA đã train (adapter_model.safetensors do train.py lưu) vào adapter vừa tạo bằng get_peft_model.
    r / alpha phải trùng (đổi scaling = đổi hàm); đọc lại adapter sau khi nạp phải ra đúng từng bit file đã lưu, đủ mọi
    tensor - nên cũng bắt được lệch lớp đích hay --rank. Không so chuỗi target_modules: Unsloth tự sinh regex từ danh
    sách, đổi phiên bản có thể đổi chữ mà vẫn cùng các lớp."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(path)
    saved = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    cur = model.peft_config["default"]
    for k in ("r", "lora_alpha", "use_dora", "use_rslora"):
        if saved.get(k) != getattr(cur, k):
            raise SystemExit(f"--init-lora {path}: {k} = {saved.get(k)!r}, lượt này {getattr(cur, k)!r}")
    sd = load_file(str(path / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, sd, adapter_name="default")
    if res.unexpected_keys:
        raise SystemExit(f"--init-lora {path}: {len(res.unexpected_keys)} tensor không có trong model, vd {res.unexpected_keys[:3]}")
    now = get_peft_model_state_dict(model, adapter_name="default")
    bad = [k for k, t in sd.items() if k not in now or not torch.equal(now[k].detach().cpu(), t)]
    if bad or len(now) != len(sd):
        raise SystemExit(f"--init-lora {path}: {len(bad)} tensor nạp sai (vd {bad[:3]}), adapter có {len(now)} tensor, file {len(sd)}")
    print(f"Bắt đầu từ LoRA {path}: {len(sd)} tensor, {sum(t.numel() for t in sd.values()):,} tham số (đọc lại khớp từng bit)")


def setup_vocab_keep(model, dataset, args, out):
    """Loss trên từ vựng đã tỉa: W của lm_head chỉ gồm các hàng prune_vocab.py sẽ giữ (tính trước bằng cùng
    thuật toán + cùng --keep-texts), nhãn đổi sang id mới. Mọi nhãn trong tập train phải nằm trong tập giữ."""
    import prune_vocab
    import shared_prefix
    k = prune_vocab.compute_keep(args.model, args.counts, args.vocab_total, 2, args.vocab_keep)
    rows = torch.tensor(k.new_rows, dtype=torch.long)
    head = model.get_output_embeddings().weight
    old2new = torch.full((head.shape[0],), -1, dtype=torch.long)
    old2new[rows] = torch.arange(len(rows))
    labels = torch.unique(torch.cat([torch.as_tensor(r["labels"]) for r in dataset]))
    labels = labels[labels >= 0]
    missing = labels[old2new[labels] < 0]
    assert missing.numel() == 0, f"{missing.numel()} token nhãn không nằm trong từ vựng giữ lại, vd {missing[:10].tolist()}"
    with torch.no_grad():
        w_keep = head[rows.to(head.device)].contiguous()
    shared_prefix.set_vocab_keep(w_keep, old2new.to(head.device))
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab_keep.json").write_text(json.dumps(k.new_rows), encoding="utf-8")
    print(f"Loss trên {len(rows)} token giữ lại (W {tuple(w_keep.shape)}, {labels.numel()} token nhãn khác nhau đều có mặt)")


def train_lora(args, out):
    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    # Qwen3.5 là model đa phương thức nên Unsloth trả về processor; phần render text cần tokenizer bên trong
    tok = getattr(processor, "tokenizer", processor)
    if not args.full_lm_head:
        install_labeled_only_ce()
    free_vision_tower(model)

    targets = ATTN_MLP + ([] if args.no_gdn else GDN)
    n_layers = getattr(model.config, "text_config", model.config).num_hidden_layers
    if not 0 <= args.lora_from < n_layers:
        raise SystemExit(f"--lora-from {args.lora_from}: model có {n_layers} lớp")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        target_modules=targets,
        layers_to_transform=list(range(args.lora_from, n_layers)) if args.lora_from else None,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if args.lora_from:
        # Unsloth đổi danh sách lớp đích thành regex rồi thành tên module: kiểm tra lại đúng các lớp >= N có LoRA
        idx = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, m in model.named_modules()
                      if hasattr(m, "lora_A") and ".layers." in n})
        if idx != list(range(args.lora_from, n_layers)):
            raise SystemExit(f"--lora-from {args.lora_from}: LoRA gắn ở các lớp {idx}")
        print(f"LoRA chỉ ở lớp {args.lora_from}..{n_layers - 1}"
              f" ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} tham số học)")
    if args.init_lora:
        load_init_lora(model, args.init_lora)
    if args.mxfp8:
        import mxfp8
        mxfp8.enable_mxfp8(model)

    rows = read_jsonl(args.train)
    t0 = time.time()
    train_ds, sample, cut, too_long, n_tok = tokenize_conversations(
        tok, rows[: args.limit] if args.limit else rows, args.max_seq_len)
    print(f"Mẫu train: {len(train_ds)} (cắt bớt lượt cuối ở {cut} hội thoại, bỏ {too_long} mẫu vẫn dài hơn {args.max_seq_len} token),"
          f" tổng {n_tok:,} token, tokenize {time.time() - t0:.0f} s")
    del rows

    # Python 3.12 + transformers 5 trên Windows thỉnh thoảng crash "access violation" khi bộ dọn rác quét các object
    # của tokenizer/model (thoát với mã 5, không có traceback). Đưa các object đã khởi tạo ra khỏi diện quét.
    gc.collect()
    gc.freeze()

    if RESPONSE_PART not in sample:
        raise SystemExit(f"Chat template không có {RESPONSE_PART!r}, cần sửa INSTRUCTION_PART/RESPONSE_PART")
    print("=" * 30, "MẪU ĐÃ RENDER (phần cuối)", "=" * 30)
    print(sample[-1500:])

    # Prefix dùng chung: mỗi bước = một nhóm batch*grad_accum hội thoại cùng system prompt + tool, prefix ~6.4K token
    # tính một lần (gradient vẫn chính xác) -> Trainer thấy batch 1, không tích lũy gradient
    share = not args.no_share_prefix
    group_size = args.batch * args.grad_accum
    slow = SlowGuard(float(os.environ.get("TRAIN_SLOW_FACTOR", "2.5")))
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,  # đã tokenize (tokenize_conversations)
        args=SFTConfig(
            dataset_num_proc=1,  # Windows: nhiều tiến trình làm map dataset treo
            max_length=args.max_seq_len,
            packing=False,
            per_device_train_batch_size=1 if share else args.batch,
            gradient_accumulation_steps=1 if share else args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            warmup_steps=10,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.001,
            logging_steps=1 if 0 < args.max_steps <= 20 else 10,
            save_steps=25,  # crash ngẫu nhiên ở trên -> checkpoint dày để chạy lại mất ít
            save_total_limit=2,
            seed=3407,
            output_dir=str(out / "checkpoints"),
            report_to="none",
        ),
        callbacks=[Speed(n_tok / max(len(train_ds), 1) * args.batch * args.grad_accum), slow],
    )
    # Chỉ tính loss trên phần assistant (lời gọi tool + câu trả lời), không học thuộc system prompt / kết quả tool
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    if share:
        import shared_prefix
        shared_prefix.enable_shared_prefix(trainer, group_size, tok.convert_tokens_to_ids("<|im_start|>"))
        if args.vocab_keep:
            setup_vocab_keep(model, trainer.train_dataset, args, out)
    elif args.vocab_keep:
        raise SystemExit("--vocab-keep chỉ có với prefix dùng chung")
    # Chạy tiếp từ checkpoint mới nhất đã lưu XONG (Trainer ghi trainer_state.json sau cùng). resume_from_checkpoint=True
    # chọn thư mục số lớn nhất kể cả khi crash giữa lúc lưu -> mọi lần chạy lại đều chết như nhau.
    done = [p for p in (out / "checkpoints").glob("checkpoint-*")
            if p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").is_file()]
    resume = str(max(done, key=lambda p: int(p.name.split("-")[-1]))) if done and args.max_steps < 0 else None
    gc.freeze()
    trainer.train(resume_from_checkpoint=resume)
    if slow.tripped:
        raise SystemExit(3)  # run_train.ps1: chờ GPU rảnh, chạy tiếp từ checkpoint vừa lưu
    if args.mxfp8:
        mxfp8.disable_mxfp8(model)  # trả trọng số bf16 về GPU cho bước thử sinh + merge
    return model, processor, tok


if __name__ == "__main__":
    main()
