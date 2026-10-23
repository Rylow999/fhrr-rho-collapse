#!/usr/bin/env python3
"""Exp 36 — spectral pruning criterion on GPT-2 XL (1.5B, 48 layers, 25 heads).
Protocol: lmin for ALL heads at square point (T=64); ablation importance on a
layer subset (8 of 48 layers, 200 heads). fp16 to fit RAM.
Pre-registered: the GPT-2 family kept head-level lmin variance at 124M (Exp 32);
does the variance survive at 48 layers (the depth question of Exp 33a)?"""
import torch, numpy as np, json, time
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

t0 = time.time()
MODEL = "gpt2-xl"
tok = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2Model.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()
cfg = model.config
NH = cfg.n_head                     # 25
D = cfg.n_embd                      # 1600
DH = D // NH                        # 64
NL = cfg.n_layer                    # 48
print(f"gpt2-xl: d={D} layers={NL} heads={NH} d_head={DH}", flush=True)

TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms "
        "and their role in modern language models across many architectures. "
        "The hard edge of random matrices governs the smallest eigenvalue.")
ids = tok(TEXT, return_tensors="pt").input_ids
if ids.shape[1] < 64:
    ids = tok(TEXT * 3, return_tensors="pt").input_ids
ids = ids[:, :64]
print("T =", ids.shape[1], flush=True)

# ---- 1) lmin for ALL heads ----
keys_by_layer = {}
for i, blk in enumerate(model.h):
    cat = blk.attn.c_attn
    def make_hook(idx, proj):
        def hook(mod, inp, out):
            hidden = inp[0].float()
            B, T, _ = hidden.shape
            qkv = hidden @ proj.weight.float() + proj.bias.float()  # 5.x layout (in,out)
            K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
            keys_by_layer[idx] = K[0]
        return hook
    blk.attn.register_forward_hook(make_hook(i, cat))
with torch.no_grad():
    base_out = model(ids)
for i in range(NL):
    model.h[i].attn._forward_hooks.clear()
print(f"lmin forward done ({time.time()-t0:.0f}s)", flush=True)

lmin_map = {}
for li, K in keys_by_layer.items():
    Kn = K.numpy()
    for h in range(NH):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmin_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
lv = np.array(list(lmin_map.values()))
print(f"lmin range: [{lv.min():.2e}, {lv.max():.2e}] | collapsed(<1e-4): {(lv < 1e-4).sum()}/{len(lv)}", flush=True)
print(f"DEPTH QUESTION: does head-level variance survive at 48 layers?", flush=True)

# ---- 2) ablation importance on 8 sampled layers ----
LAYER_SUBSET = [0, 6, 12, 18, 24, 30, 36, 42]
wte = model.wte
with torch.no_grad():
    base_logits = base_out.last_hidden_state[0, -1].float() @ wte.weight.T.float()
base_probs = F.softmax(base_logits, dim=-1)
importance = {}
for li in LAYER_SUBSET:
    blk = model.h[li]
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
        logits_a = out_a.last_hidden_state[0, -1].float() @ wte.weight.T.float()
        importance[f"{li},{h}"] = float(F.kl_div(
            F.log_softmax(logits_a, dim=-1), base_probs, reduction="sum").item())
    print(f"  layer {li} ablated ({time.time()-t0:.0f}s)", flush=True)

# ---- 3) correlations ----
sub = [f"{li},{h}" for li in LAYER_SUBSET for h in range(NH)]
la = np.array([lmin_map[k] for k in sub])
ia = np.array([importance[k] for k in sub])
r = float(np.corrcoef(la, ia)[0, 1])
rng = np.random.default_rng(7)
pp = sum(1 for _ in range(10000)
         if abs(np.corrcoef(la, rng.permutation(ia))[0, 1]) >= abs(r)) / 10000
rho = float(np.corrcoef(np.argsort(np.argsort(la)).astype(float),
                        np.argsort(np.argsort(ia)).astype(float))[0, 1])
print(f"pearson(lmin, importance) = {r:.4f} (perm p={pp:.4f})", flush=True)
print(f"spearman = {rho:.4f}", flush=True)
idx_sorted = np.argsort(ia)
b12, t12 = idx_sorted[:12], idx_sorted[-12:]
print(f"bottom12 importance: mean lmin = {la[b12].mean():.6f} | near-zero: {(la[b12] < 1e-4).sum()}/12")
print(f"top12 importance:    mean lmin = {la[t12].mean():.6f} | near-zero: {(la[t12] < 1e-4).sum()}/12")

out = {
 "experiment": "Exp 36 - spectral pruning on GPT-2 XL (1.5B scale test)",
 "model": MODEL, "d_model": D, "layers": NL, "heads": NH, "d_head": DH, "T": 64,
 "lmin_range": [round(float(lv.min()), 8), round(float(lv.max()), 8)],
 "collapsed_fraction": f"{int((lv < 1e-4).sum())}/{len(lv)}",
 "lmin_map_all_heads": lmin_map,
 "importance_subset": importance,
 "layers_ablated": LAYER_SUBSET,
 "pearson": {"r": round(r, 4), "perm_p": round(pp, 4)},
 "spearman": round(rho, 4),
 "bottom12_importance_mean_lmin": round(float(la[b12].mean()), 6),
 "top12_importance_mean_lmin": round(float(la[t12].mean()), 6),
 "runtime_seconds": round(time.time() - t0, 1),
}
json.dump(out, open("/home/delorien/exp36_results.json", "w"), indent=2)
print("saved", f"({time.time()-t0:.0f}s total)")
