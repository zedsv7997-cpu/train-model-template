"""Các lớp tuyến tính dùng chung một đầu vào (gate/up; q/k/v; qkv/z/b/a của DeltaNet) tính trong MỘT autograd.Function.

Đường PEFT + Unsloth tính từng lớp LoRA riêng: GEMM gốc, x @ Aᵀ, addmm(kết quả, xA, Bᵀ). Khi lan ngược autograd cộng
dồn gradient của x từ từng lớp (mỗi lần cộng đọc 2 + ghi 1 tensor [T, K]), và MLP Unsloth compile (bị ngắt ở mỗi GEMM
MXFP8) còn nhân cả dy [T, 6144] với scaling = 1.0. Ở đây:
  - x đọc một lần cho mọi A của nhóm (và cả trọng số in_proj_b/a 16 cột của DeltaNet): một GEMM [T, K] x [K, R];
  - gradient vào x của mọi nhánh LoRA + b/a cộng thẳng vào dx của nhánh gốc trong epilogue của MỘT addmm;
  - dA của mọi lớp trong nhóm từ một GEMM; dB, d(xA) như cũ.
Toán học y hệt đường cũ (check_shared_prefix_fp32.py --lora: lệch 1.7e-6 so với transformers + PEFT); chỉ khác thứ tự
làm tròn (tổng các nhánh LoRA của dx cộng fp32 rồi làm tròn một lần thay vì làm tròn bf16 từng nhánh rồi cộng).
Cùng MLP gộp + CE gộp (shared_prefix.py) một bước nhóm 16 nhanh hơn ~16% (đo xen kẽ 2026-10-21: 17.1 -> 14.3 s).

Lớp không dùng được đường gộp (trọng số gốc cần gradient như full finetune, DoRA, dropout > 0, nhiều adapter) thì
group_linear gọi thẳng từng module như cũ.

MXFP8 + bf16 (2026-10-23, nhóm qkv/z/b/a, q/k/v, o_proj, out_proj - mxfp8.FUSED_BWD; MLP để mlp-fused):
  - lan ngược: mỗi dy đọc MỘT lần (mxfp8.lora_bwd: lượng tử MX + d(xa) + dB) thay vì ba, fp8 của cả nhóm ghi vào một
    bộ đệm ghép theo K -> dx = MỘT _scaled_mm với Wᵀ ghép (bỏ dx.add_). dB ra fp32 (không làm tròn bf16 nữa);
  - residual-lora: o_proj / out_proj trả Deferred (chưa cộng LoRA), residual_add cộng LoRA ngay trong kernel cộng
    residual (bỏ addmm_ đọc + ghi lại [T, 2048]); kết quả trùng từng bit đường cũ.
Không MX / fp32 / thiếu gradient của một đầu ra: đường cũ y nguyên (trùng từng bit). Đo: ~0.14 s/bước GPU.
"""
import torch
from torch import nn

_THIN = 64  # Linear thường đóng băng có <= chừng này đầu ra (in_proj_b/a: 16) tính chung GEMM với các A
STATS = {"gộp": 0, "gọi thẳng": 0}  # số lần mỗi đường chạy (để kiểm tra)


def _part(m):
    """(Linear gốc, A, B, scaling) của một LoRA Linear PEFT, (m, None, None, 1.0) với Linear thường đóng băng,
    None nếu không gộp được."""
    base = getattr(m, "base_layer", None)
    if base is None:
        if type(m) is nn.Linear and m.bias is None and not m.weight.requires_grad:
            return m, None, None, 1.0
        return None
    if base.bias is not None or base.weight.requires_grad or m.merged or m.disable_adapters:
        return None
    ads = [a for a in m.active_adapters if a in m.lora_A]
    if len(ads) != 1:
        return None
    a = ads[0]
    drop = m.lora_dropout[a]
    if (a in getattr(m, "lora_variant", {}) or m.use_dora.get(a, False) or m.lora_B[a].bias is not None
            or not (isinstance(drop, nn.Identity) or getattr(drop, "p", 1) == 0)):
        return None
    return base, m.lora_A[a].weight, m.lora_B[a].weight, m.scaling[a]


def _mx(base):
    return getattr(base, "_mx_w", None) is not None


# Bản bf16 của A/B LoRA (tham số fp32) và ma trận S ghép, tạo một lần mỗi bước thay vì ở mỗi lần gọi (~10K kernel chép
# nhỏ mỗi bước, tính cả lúc tính lại checkpoint). Optimizer bnb sửa tham số qua con trỏ (không tăng _version) nên không
# dựa vào _version: shared_prefix_step gọi new_step() ở đầu mỗi bước để bỏ cache cũ.
_CAST = {}


def new_step():
    _CAST.clear()


def _cast(key, make):
    t = _CAST.get(key)
    if t is None:
        t = _CAST[key] = make()
    return t


def _base_mm(base, x2, dt, q):
    """x2 @ Wᵀ của trọng số gốc: MXFP8 nếu mxfp8.enable_mxfp8 đã bật cho lớp này, không thì GEMM thường."""
    if _mx(base):
        return torch._scaled_mm(q[0], base._mx_w.t(), q[1], base._mx_ws, out_dtype=torch.bfloat16)
    return x2 @ base.weight.to(dt).t()


class _Group(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, parts, need_out, defer, *ws):
        ctx.set_materialize_grads(False)  # đầu ra không dùng (vd. q ở lớp prefix trên cùng) -> grad None, không phải 0
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        if x2.dtype != dt or not x2.is_contiguous():
            x2 = x2.to(dt).contiguous()
        # Hàng của ma trận mảnh S: A của mọi lớp LoRA (theo thứ tự) rồi trọng số các Linear thường mảnh
        rows, Bs, lora, thin = [], [], [], []
        off = 0
        for i, (base, A, B, s) in enumerate(parts):
            if A is not None:
                rows.append(A)
                Bs.append(_cast((id(B), dt), lambda B=B: B.detach().to(dt)))
                lora.append((i, off, A.shape[0], s))
                off += A.shape[0]
        for i, (base, A, B, s) in enumerate(parts):
            if A is None and not _mx(base) and base.weight.shape[0] <= _THIN:
                rows.append(base.weight)
                thin.append((i, off, base.weight.shape[0]))
                off += base.weight.shape[0]
        S = _cast((tuple(id(r) for r in rows), dt), lambda: torch.cat([r.detach().to(dt) for r in rows])) if rows else None
        xS = x2 @ S.t() if rows else None
        outs = [None] * len(parts)
        thin_ids = {i for i, _, _ in thin}
        if need_out:
            q = None
            if any(_mx(p[0]) for p in parts):
                import mxfp8
                q = mxfp8.mx_quant(x2)  # một lần cho cả nhóm
            for i, (base, A, B, s) in enumerate(parts):
                if i not in thin_ids:
                    outs[i] = _base_mm(base, x2, dt, q)
            for j, (i, o, r, s) in enumerate(lora):
                if defer is not None and _mx(parts[i][0]) and dt == torch.bfloat16:
                    defer[i] = (xS[:, o:o + r], Bs[j], s)  # residual_add cộng phần LoRA (xem đó)
                else:
                    outs[i].addmm_(xS[:, o:o + r], Bs[j].t(), alpha=s)
        else:  # chỉ cần đồ thị cho backward (x2, xS đã lưu), không ai đọc giá trị đầu ra
            STATS["bỏ đầu ra"] = STATS.get("bỏ đầu ra", 0) + 1
            for i, (base, A, B, s) in enumerate(parts):
                if i not in thin_ids:
                    outs[i] = x2.new_empty(x2.shape[0], base.out_features)
        for i, o, n in thin:
            outs[i] = xS[:, o:o + n].contiguous()
        ctx.save_for_backward(x2, xS, S, *Bs)
        ctx.meta = (parts, lora, thin, lead, K, dt, x.dtype)
        return tuple(y.view(*lead, y.shape[-1]) for y in outs)

    @staticmethod
    def backward(ctx, *dys):
        x2, xS, S, *Bs = ctx.saved_tensors
        parts, lora, thin, lead, K, dt, xdt = ctx.meta
        need_x = ctx.needs_input_grad[0]
        flat = [None if dy is None else dy.reshape(-1, dy.shape[-1]).to(dt) for dy in dys]
        plan = _fused_bwd_plan(parts, flat, dt, need_x)
        if plan is not None:
            return _fused_backward(ctx, x2, xS, S, Bs, flat, plan, need_x)
        dx = None
        if need_x:
            thin_ids = {i for i, _, _ in thin}
            for i, (base, A, B, s) in enumerate(parts):
                dy = flat[i]
                if dy is None or i in thin_ids:
                    continue
                if _mx(base):
                    import mxfp8
                    d = dy if dy.is_contiguous() else dy.contiguous()
                    qd, sd = mxfp8.mx_quant(d)
                    wt, wts = mxfp8.wt_of(base)
                    g = torch._scaled_mm(qd, wt, sd, wts, out_dtype=torch.bfloat16)
                    dx = g if dx is None else dx.add_(g)
                else:
                    w = base.weight.to(dt)
                    dx = dy @ w if dx is None else dx.addmm_(dy, w)
        # Nhánh mảnh: cột của d(xS) cho các lớp LoRA (s * dy @ B) và Linear mảnh (chính dy), cùng các hàng S tương ứng
        blocks, srows, grads = [], [], [None] * (2 * len(lora))
        act_lora = []
        for j, (i, o, r, s) in enumerate(lora):
            dy = flat[i]
            if dy is None:
                continue
            xa = xS[:, o:o + r]
            dB = dy.t() @ xa
            dxa = dy @ Bs[j]
            if s != 1:
                dB.mul_(s)
                dxa.mul_(s)
            grads[2 * j + 1] = dB
            blocks.append(dxa)
            srows.append((o, r))
            act_lora.append(j)
        n_act = sum(r for _, r in srows)
        if need_x:
            for i, o, n in thin:
                if flat[i] is not None:
                    blocks.append(flat[i])
                    srows.append((o, n))
        if blocks:
            dS = torch.cat(blocks, 1) if len(blocks) > 1 else blocks[0]
            if need_x:
                full = len(srows) == len(lora) + len(thin)  # mọi hàng của S đều dùng, đúng thứ tự
                Sa = S if full else torch.cat([S[o:o + w] for o, w in srows])
                dx = dS @ Sa if dx is None else dx.addmm_(dS, Sa)
            if act_lora:
                dA = dS[:, :n_act].t() @ x2
                c = 0
                for j in act_lora:
                    r = lora[j][2]
                    grads[2 * j] = dA[c:c + r]
                    c += r
        # A/B có thể là fp32 trong khi tính bf16: autograd tự ép gradient về dtype của tham số
        ws_in = ctx.needs_input_grad[4:]
        out = [None if (g is None or not ws_in[k]) else g for k, g in enumerate(grads)]
        dx = None if dx is None else dx.view(*lead, K).to(xdt)
        return (dx, None, None, None, *out)


def _cdiv(a, b):
    return -(-a // b)


def _fused_bwd_plan(parts, flat, dt, need_x):
    """Điều kiện của đường lan ngược gộp (mxfp8.lora_bwd + một GEMM dx cho cả nhóm): tính bf16, mọi đầu ra đều có
    gradient, các nhánh MX đều thuộc mxfp8.FUSED_BWD và (nếu cần dx) có Wᵀ ghép khớp đúng thứ tự nhóm. Trả
    (chỉ số nhánh MX, (mat2, scale) hoặc None), hoặc None -> đường cũ (không MX, fp32, full finetune, MLP, ...)."""
    if dt != torch.bfloat16 or any(dy is None for dy in flat):
        return None
    mxi = [i for i, p in enumerate(parts) if _mx(p[0])]
    if not mxi or not all(getattr(parts[i][0], "_mx_fbwd", False) for i in mxi):
        return None
    import mxfp8
    wt = mxfp8.group_wt([parts[i][0] for i in mxi]) if need_x else None
    if need_x and wt is None:
        return None
    return mxi, wt


def _fused_backward(ctx, x2, xS, S, Bs, flat, plan, need_x):
    """_Group.backward khi đi được mxfp8.lora_bwd: mỗi dy đọc MỘT lần (lượng tử MX vào bộ đệm ghép theo K + d(xa)
    + dB), dx của cả nhóm là MỘT _scaled_mm (cộng fp32 qua các nhánh thay vì làm tròn bf16 từng nhánh rồi add_).
    Cột dS = hàng S (d(xa) các LoRA theo thứ tự rồi dy của Linear mảnh) như đường cũ; dx.addmm_(dS, S) và
    dA = dSᵀ @ x2 giữ nguyên."""
    import mxfp8
    parts, lora, thin, lead, K, dt, xdt = ctx.meta
    mxi, wt = plan
    T, dev = x2.shape[0], x2.device
    lora_of = {i: (j, o, r, s) for j, (i, o, r, s) in enumerate(lora)}
    n_act = sum(r for _, _, r, _ in lora)
    n_cols = n_act + (sum(n for _, _, n in thin) if need_x else 0)
    dS = torch.empty((T, n_cols), dtype=dt, device=dev) if n_cols else None
    grads = [None] * (2 * len(lora))
    q = s_buf = None
    if need_x:
        n_mx = sum(flat[i].shape[1] for i in mxi)
        q = torch.empty((T, n_mx), dtype=mxfp8.F8, device=dev)
        s_buf = torch.empty(_cdiv(T, 128) * (n_mx // 128) * 512, dtype=torch.uint8, device=dev)
    tile = 0
    for i in mxi:
        dy = flat[i] if flat[i].stride(1) == 1 else flat[i].contiguous()
        if i in lora_of:
            j, o, r, s = lora_of[i]
            grads[2 * j + 1] = mxfp8.lora_bwd(dy, xS[:, o:o + r], Bs[j], s, q=q, s=s_buf, col_tile_off=tile,
                                              dxa_out=dS[:, o:o + r])
        else:
            mxfp8.lora_bwd(dy, q=q, s=s_buf, col_tile_off=tile)
        tile += dy.shape[1] // 128
    for j, (i, o, r, s) in enumerate(lora):  # LoRA trên Linear gốc không MX (hiếm): như đường cũ
        if i in mxi:
            continue
        dy = flat[i]
        dB = dy.t() @ xS[:, o:o + r]
        dS[:, o:o + r] = dy @ Bs[j]
        if s != 1:
            dB.mul_(s)
            dS[:, o:o + r].mul_(s)
        grads[2 * j + 1] = dB
    dx = None
    if need_x:
        dx = torch._scaled_mm(q, wt[0], s_buf.view(mxfp8.E8), wt[1], out_dtype=torch.bfloat16)
        del q, s_buf
        thin_ids = {i for i, _, _ in thin}
        for i, (base, A, B, s) in enumerate(parts):  # Linear bf16 thường (không MX, không mảnh)
            if i not in thin_ids and not _mx(base):
                dx.addmm_(flat[i], base.weight.to(dt))
        for i, o, n in thin:
            dS[:, o:o + n].copy_(flat[i])
        if dS is not None:
            dx.addmm_(dS, S)
    if lora:
        dA = dS[:, :n_act].t() @ x2
        for j, (i, o, r, s) in enumerate(lora):
            grads[2 * j] = dA[o:o + r]
    STATS["lan ngược gộp"] = STATS.get("lan ngược gộp", 0) + 1
    ws_in = ctx.needs_input_grad[4:]
    out = [None if (g is None or not ws_in[k]) else g for k, g in enumerate(grads)]
    dx = None if dx is None else dx.view(*lead, K).to(xdt)
    return (dx, None, None, None, *out)


# ---------------------------------------------------------------- residual-lora
class Deferred(tuple):
    """(y, xa, B, scaling): đầu ra MX bf16 của o_proj / out_proj CHƯA cộng nhánh LoRA; residual_add cộng nó."""


class _ResidualLoRA(torch.autograd.Function):
    """x + (y + s * xa @ Bᵀ) trong MỘT kernel: phép cộng residual vốn đọc y, x và ghi x mới, thêm tích hạng r ngay
    trong thanh ghi thì bỏ được addmm_ đọc + ghi lại cả [T, N] (o_proj / out_proj: ~39 ms/bước). xa, B đã tách khỏi
    đồ thị: gradient nhánh LoRA (dB, d(xa) -> dA, dx) vẫn do _Group.backward tính từ cùng dy; ở đây chỉ trả dy."""

    @staticmethod
    def forward(ctx, x, y, xa, B, s):
        import mxfp8
        N = x.shape[-1]
        return mxfp8.res_lora(x.reshape(-1, N), y.reshape(-1, N), xa, B, s).view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        return dy, dy, None, None, None


def residual_add(x, y):
    """x + y; y là Deferred (group_linear(..., defer=True)) thì cộng luôn nhánh LoRA trong kernel residual."""
    if not isinstance(y, Deferred):
        return x + y
    y, xa, B, s = y
    N = x.shape[-1]
    if (x.dtype == y.dtype == torch.bfloat16 and x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
            and N % 128 == 0 and xa.stride(1) == 1 and B.is_contiguous()):
        STATS["residual LoRA"] = STATS.get("residual LoRA", 0) + 1
        return _ResidualLoRA.apply(x, y, xa, B, s)
    return x + torch.addmm(y.reshape(-1, N), xa, B.t(), alpha=s).view(y.shape)  # dtype/hình khác: như cũ


def group_linear(x, mods, need_out=True, defer=False):
    """(m(x) for m in mods) - cùng kết quả, gộp như mô tả ở đầu file.
    need_out=False: chỉ dựng đồ thị cho backward, bỏ GEMM gốc + LoRA của các đầu ra (trả tensor chưa khởi tạo) -
    dùng khi tính lại lớp cuối của một đoạn checkpoint, nơi giá trị đầu ra không được đọc (gradient vẫn y hệt).
    defer=True: đầu ra MX bf16 có LoRA trả Deferred (chưa cộng LoRA) - chỉ dùng khi ngay sau là residual_add."""
    parts = tuple(_part(m) for m in mods)
    if any(p is None for p in parts):
        STATS["gọi thẳng"] += 1
        return tuple(m(x) for m in mods)
    STATS["gộp"] += 1
    ws = [w for p in parts if p[1] is not None for w in (p[1], p[2])]
    box = {} if defer and need_out else None
    outs = _Group.apply(x, parts, need_out, box, *ws)
    if not box:
        return outs
    return tuple(Deferred((y,) + box[i]) if i in box else y for i, y in enumerate(outs))
