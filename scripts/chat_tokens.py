"""Render hội thoại bằng chat template và tokenize nhanh mà ra y hệt tokenizer (dùng chung: train.py, prune_vocab.py).

Hội thoại gọi tool nào cũng mở đầu bằng cùng một khối system + danh sách tool (vd ~6.400 token, ~72% số token của bộ
train), nên tokenize cả văn bản là tokenize lại khối đó hàng chục nghìn lần (đo 2026-10-24: train.py 167 s cho 22K hội thoại, prune_vocab.py
~6 phút một luồng). Tokenizer HF tách văn bản ở các added token (<|im_start|>, <tool_call>...) TRƯỚC khi chuẩn hoá NFC,
tách từ và BPE; với Qwen3.5 mọi added token có normalized / lstrip / rstrip / single_word = false, nên token của cả văn
bản = nối token của từng khúc cắt ngay trước mỗi <|im_start|>. encode() / token_ids() chỉ tokenize mỗi khúc khác nhau
một lần. chunking_ok() so một văn bản với đường thường; lệch (tokenizer khác) thì báo và tokenize cả văn bản như cũ.
"""
SEP = "<|im_start|>"


def render(tok, ex, add_generation_prompt=False):
    messages = ex["messages"][:-1] if add_generation_prompt else ex["messages"]
    return tok.apply_chat_template(
        messages,
        tools=ex.get("tools") or None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,  # Qwen3.5-2B mặc định non-thinking; gọi tool không cần suy luận dài
    )


def chunks(text):
    """Cắt ngay trước mỗi <|im_start|> (chỗ tokenizer vốn tự tách): "a<|im_start|>b" -> ["a", "<|im_start|>b"]."""
    parts = text.split(SEP)
    return ([parts[0]] if parts[0] else []) + [SEP + p for p in parts[1:]]


def _tok(tok, texts):
    return tok(texts, add_special_tokens=False)["input_ids"] if texts else []


def chunking_ok(tok, text):
    """Tokenize theo khúc có ra đúng token của cả văn bản không."""
    if [i for x in _tok(tok, chunks(text)) for i in x] == _tok(tok, [text])[0]:
        return True
    print("chat_tokens: tokenize theo khúc lệch đường thường -> tokenize cả văn bản", flush=True)
    return False


def encode(tok, texts):
    """= tok(texts, add_special_tokens=False)["input_ids"]; khúc giống nhau trong lô chỉ tokenize một lần."""
    if not texts or not chunking_ok(tok, texts[0]):
        return _tok(tok, texts)
    parts = [chunks(t) for t in texts]
    uniq = list(dict.fromkeys(c for p in parts for c in p))
    ids = dict(zip(uniq, _tok(tok, uniq)))
    out = []
    for p in parts:
        x = []
        for c in p:
            x += ids[c]
        out.append(x)
    return out


def token_ids(tok, texts, batch=2048):
    """Tập id có trong các văn bản (= hợp tok(t) của từng văn bản); mỗi khúc khác nhau của cả tập chỉ tokenize một lần.
    Trả về (tập id, số văn bản)."""
    found, seen, buf, n, plain = set(), set(), [], 0, None

    def flush():
        nonlocal plain
        if plain is None:
            plain = not chunking_ok(tok, buf[0])
        new = buf if plain else [c for c in dict.fromkeys(c for t in buf for c in chunks(t)) if c not in seen]
        if not plain:
            seen.update(new)
        for x in _tok(tok, new):
            found.update(x)

    for t in texts:
        buf.append(t)
        n += 1
        if len(buf) == batch:
            flush()
            buf = []
    if buf:
        flush()
    return found, n
