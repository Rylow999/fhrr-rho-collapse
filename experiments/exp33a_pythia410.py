#!/usr/bin/env python3
"""Exp 33a — spectral pruning criterion on Pythia-410M (scale test).
lmin for ALL heads at square point (T=d_head=64); ablation importance on a
layer subset (6 of 24 layers, 96 heads) to fit CPU memory."""
import torch, numpy as np, json, time
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()
cfg = model.config
NH = cfg.num_attention_heads          # 16
D = cfg.hidden_size                   # 1024
DH = D // NH                          # 64
NL = cfg.num_hidden_layers            # 24
print(f"pythia-410m: d={D} layers={NL} heads={NH} d_head={DH} dtype={next(model.parameters()).dtype}", flush=True)

TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms "
        "and their role in modern language models across many architectures. "
        "The hard edge of random matrices governs the smallest eigenvalue.")
ids = tok(TEXT, return_tensors="pt").input_ids
if ids.shape[1] < 64:
    # pad by repeating
    reps = 64 // ids.shape[1] + 1
    ids = tok(TEXT * reps, return_tensors="pt").input_ids
ids = ids[:, :64]  # force square point: T = 64 = d_head
print("T =", ids.shape[1], flush=True)

# ---- 1) lmin for ALL heads (one forward with hooks) ----
keys_by_layer = {}
for i, blk in enumerate(model.layers):
    qkv = blk.attention.query_key_value
    def make_hook(idx, proj):
        def hook(mod, inp, out):
            hidden = inp[0].float()
            B, T, _ = hidden.shape
            qkv_out = hidden @ proj.weight.float().T + proj.bias.float()
            K = qkv_out[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
            keys_by_layer[idx] = K[0]
        return hook
    qkv.register_forward_hook(make_hook(i, qkv))
with torch.no_grad():
    base_out = model(ids)
for i in range(NL):
    model.layers[i].attention.query_key_value._forward_hooks.clear()

lmin_map = {}
for li, K in keys_by_layer.items():
    Kn = K.numpy()
    for h in range(NH):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmin_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
print(f"lmin computed for {len(lmin_map)} heads ({time.time()-t0:.0f}s)", flush=True)

# baseline logits (tied embedding head)
wte = model.embed_in.weight  # pythia: embed_in
with torch.no_grad():
    base_logits = base_out.last_hidden_state[0, -1].float() @ wte.T.float()

# ---- 2) ablation importance on 6 sampled layers ----
LAYER_SUBSET = [0, 4, 8, 12, 16, 20]
base_probs = F.softmax(base_logits, dim=-1)
importance = {}
for li in LAYER_SUBSET:
    blk = model.layers[li]
    def ablate_hook(mod, inp, out):
        ao = out[0].clone().float()
        B, T, Dd = ao.shape
        ao.view(B, T, NH, DH)[:, :, :, :] = ao.view(B, T, NH, DH)  # noop placeholder
        return (out[0],)  # unchanged - we zero per head below
    # simpler: zero the whole head slice at block output per head
    for h in range(NH):
        def ablate_h(mod, inp, out, hh=h):
            base = out[0] if isinstance(out, tuple) else out
            ao = base.clone().float()
            B, T, Dd = ao.shape
            ao.view(B, T, NH, DH)[:, :, hh, :] = 0
            ao = ao.to(base.dtype)
            return ao if not isinstance(out, tuple) else (ao,)
        hk = blk.register_forward_hook(ablate_h)
        with torch.no_grad():
            out_a = model(ids)
        blk._forward_hooks.clear()
        logits_a = out_a.last_hidden_state[0, -1].float() @ wte.T.float()
        kl = F.kl_div(F.log_softmax(logits_a, dim=-1), base_probs, reduction="sum").item()
        importance[f"{li},{h}"] = round(kl, 6)
    print(f"  layer {li} ablated ({time.time()-t0:.0f}s)", flush=True)

# ---- 3) correlation on the subset ----
def _rank(x):
    order = np.argsort(x); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x)); return ranks

lmins = [lmin_map[f"{li},{h}"] for li in LAYER_SUBSET for h in range(NH)]
imps = [importance[f"{li},{h}"] for li in LAYER_SUBSET for h in range(NH)]
la, ia = np.array(lmins), np.array(imps)
pearson_r = float(np.corrcoef(la, ia)[0, 1])
rng = np.random.default_rng(7)
count = sum(1 for _ in range(10000) if abs(np.corrcoef(la, rng.permutation(ia))[0, 1]) >= abs(pearson_r))
spearman_r = float(np.corrcoef(_rank(la), _rank(ia))[0, 1])
print(f"pearson = {pearson_r:.4f} (perm p={count/10000:.4f})", flush=True)
print(f"spearman = {spearman_r:.4f}", flush=True)

# bottom-12 vs top-12 importance: mean lmin
idx_sorted = np.argsort(ia)
b12, t12 = idx_sorted[:12], idx_sorted[-12:]
print(f"bottom12 importance: mean lmin = {la[b12].mean():.6f} | near-zero: {(la[b12] < 1e-4).sum()}/12")
print(f"top12 importance:    mean lmin = {la[t12].mean():.6f} | near-zero: {(la[t12] < 1e-4).sum()}/12")

out = {
 "experiment": "Exp 33a - spectral pruning on Pythia-410M (scale test)",
 "model": MODEL, "d_model": D, "layers": NL, "heads": NH, "d_head": DH, "T": 64,
 "lmin_map_all_heads": lmin_map,
 "importance_subset": importance,
 "layers_ablated": LAYER_SUBSET,
 "pearson": {"r": round(pearson_r, 4), "perm_p": round(count/10000, 4)},
 "spearman": round(spearman_r, 4),
 "bottom12_importance_mean_lmin": round(float(la[b12].mean()), 6),
 "top12_importance_mean_lmin": round(float(la[t12].mean()), 6),
 "near_zero_fraction": {"bottom12": f"{int((la[b12] < 1e-4).sum())}/12",
                         "top12": f"{int((la[t12] < 1e-4).sum())}/12"},
 "runtime_seconds": round(time.time() - t0, 1),
}
json.dump(out, open(r"C:\Users\lucia\workspace\exp33a_results.json", "w"), indent=2)
print("saved exp33a_results.json", f"({time.time()-t0:.0f}s total)")
