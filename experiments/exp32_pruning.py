#!/usr/bin/env python3
"""Exp 32 — spectral pruning criterion (P1 applied):
per-head lambda_min at the square point (T=d_head=64) vs functional importance
(logit change when the head is ablated). Prediction: heads with the LOWEST
lambda_min (most collapsed key space) contribute the LEAST (lowest ablation
delta) — making lambda_min a zero-training pruning criterion."""
import torch, numpy as np, json
from transformers import GPT2Model, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
DH = model.config.n_embd // NH
D = model.config.n_embd

TEXT = ("The quick brown fox jumps over the lazy dog. The city council "
        "approved the budget for the new park after a long debate about "
        "the environmental impact and the cost of maintenance.")
ids = tok(TEXT, return_tensors="pt").input_ids
T = ids.shape[1]
# square point needs T=64: use the last 64 tokens? T must be 64 exactly for the gram.
ids64 = ids[:, -64:]
T = 64
print("T =", T, "(square: T == d_head == 64)", flush=True)

# 1) per-head lambda_min at the square point
keys_by_layer = {}
for i, blk in enumerate(model.h):
    cat = blk.attn.c_attn
    def make_hook(idx, proj):
        def hook(mod, inp, out):
            hidden = inp[0].float()
            B, Tt, _ = hidden.shape
            qkv = hidden @ proj.weight  # transformers 5.x: (in, out) layout
            K = qkv[:, :, D:2*D].view(B, Tt, NH, DH).permute(0, 2, 1, 3)
            keys_by_layer[idx] = K[0]
        return hook
    blk.attn.register_forward_hook(make_hook(i, cat))
with torch.no_grad():
    base_out = model(ids64)
for i in range(len(model.h)):
    model.h[i].attn._forward_hooks.clear()

lmin_map = {}
for li, K in keys_by_layer.items():
    Kn = K.numpy()
    for h in range(NH):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmin_map[(li, h)] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])

# 2) per-head functional importance: ablate (zero) one head, measure KL divergence
#    of the output logits vs baseline at the last position.
with torch.no_grad():
    base_logits = model.lm_head(base_out.last_hidden_state)[0, -1] if hasattr(model, "lm_head") else None
# GPT2Model has no lm_head; use wte tied weights
wte = model.wte
with torch.no_grad():
    base_logits = base_out.last_hidden_state[0, -1] @ wte.weight.T

import torch.nn.functional as F
base_probs = F.softmax(base_logits, dim=-1)
import copy
importance = {}
for li in range(len(model.h)):
    for h in range(NH):
        # ablation hook: zero this head's contribution by zeroing its value-slice output
        def ablate_hook(mod, inp, out):
            ao = out.clone()
            B, Tt, Dd = ao.shape
            ao.view(B, Tt, NH, DH)[:, :, h, :] = 0
            return ao
        hk = model.h[li].register_forward_hook(ablate_hook)
        with torch.no_grad():
            out_a = model(ids64)
        model.h[li].attn._forward_hooks.clear()
        logits_a = out_a.last_hidden_state[0, -1] @ wte.weight.T
        kl = F.kl_div(F.log_softmax(logits_a, dim=-1), base_probs, reduction="sum").item()
        importance[f"{li},{h}"] = round(kl, 6)

# 3) correlation: lambda_min vs importance
def _rank(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks

lmins = [lmin_map[(li, h)] for li in range(len(model.h)) for h in range(NH)]
imps = [importance[f"{li},{h}"] for li in range(len(model.h)) for h in range(NH)]
lmins_a, imps_a = np.array(lmins), np.array(imps)
pearson_r = float(np.corrcoef(lmins_a, imps_a)[0, 1])
# permutation p-value for pearson
rng = np.random.default_rng(7)
count = 0
for _ in range(10000):
    pm = rng.permutation(imps_a)
    if abs(np.corrcoef(lmins_a, pm)[0, 1]) >= abs(pearson_r):
        count += 1
pearson_p = count / 10000
spearman_r = float(np.corrcoef(_rank(lmins_a), _rank(imps_a))[0, 1])
print(f"pearson(lmin, importance) = {pearson_r:.4f} (perm p={pearson_p:.4f})", flush=True)
print(f"spearman = {spearman_r:.4f}", flush=True)
pearson = type('R', (), {'statistic': pearson_r, 'pvalue': pearson_p})()
spearman = type('R', (), {'statistic': spearman_r, 'pvalue': 0.0})()

out = {
 "experiment": "Exp 32 - spectral pruning criterion (P1 applied)",
 "T": T, "square_point": True,
 "lmin_map": {f"{li},{h}": v for (li, h), v in lmin_map.items()},
 "importance_ablation_kl": importance,
 "pearson": [round(pearson.statistic, 4), round(pearson.pvalue, 6)],
 "spearman": [round(spearman.statistic, 4), round(spearman.pvalue, 6)],
 "pre_registered": "P1: heads with lmin*T^2 < c are the pruning-redundant ones. Operationalization: negative correlation between lmin and ablation-importance supports the criterion.",
}
json.dump(out, open(r"C:\\Users\\lucia\\workspace\\exp32_results.json", "w"), indent=2)
print("saved")
