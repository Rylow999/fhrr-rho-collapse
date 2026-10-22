#!/usr/bin/env python3
"""Exp 34 — sub-square spectral criterion (Pythia-410M).

PRE-REGISTRATION (before running, 2026-10-24):
  H1: at T = d_head/2 = 32, the head key-grams of Pythia-410M still show
      positive lambda_min variance across heads (NOT all collapsed).
  H2: per-head lmin at T=32 correlates with per-head lmin at T=64 (square):
      the collapse is a stable property of each head.
  H3: lmin at T=32 predicts ablation importance measured at T=64
      (negative correlation: lower lmin -> lower importance).
  Falsification: if H1 fails (all collapsed already at T=32) the sub-square
  criterion is dead for this model and depth.
"""
import torch, numpy as np, json, time
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

t0 = time.time()
MODEL = "EleutherAI/pythia-410m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()
cfg = model.config
NH, D = cfg.num_attention_heads, cfg.hidden_size
NL = cfg.num_hidden_layers
DH = D // NH

BASE = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms "
        "and their role in modern language models across many architectures. "
        "The hard edge of random matrices governs the smallest eigenvalue.")

def get_keys(T):
    ids = tok(BASE * 5, return_tensors="pt").input_ids
    if ids.shape[1] < T:
        ids = tok(BASE * 20, return_tensors="pt").input_ids
    ids = ids[:, :T]
    keys_by_layer = {}
    for i, blk in enumerate(model.layers):
        qkv = blk.attention.query_key_value
        def make_hook(idx, proj):
            def hook(mod, inp, out):
                hidden = inp[0].float()
                B, Tt, _ = hidden.shape
                qo = hidden @ proj.weight.float().T + proj.bias.float()
                K = qo[:, :, D:2*D].view(B, Tt, NH, DH).permute(0, 2, 1, 3)
                keys_by_layer[idx] = K[0]
            return hook
        qkv.register_forward_hook(make_hook(i, qkv))
    with torch.no_grad():
        base_out = model(ids)
    for i in range(NL):
        model.layers[i].attention.query_key_value._forward_hooks.clear()
    return keys_by_layer, base_out, ids

def lmin_map(keys_by_layer):
    m = {}
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
            m[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
    return m

# ---- H1: sub-square lmin variance ----
print("=== H1: lmin at T=32 (sub-square) ===", flush=True)
keys32, _, ids32 = get_keys(32)
lmin32 = lmin_map(keys32)
v32 = np.array(list(lmin32.values()))
print(f"T=32: lmin range [{v32.min():.2e}, {v32.max():.2e}] | "
      f"collapsed(<1e-4): {(v32 < 1e-4).sum()}/{len(v32)} | "
      f"std={v32.std():.2e}", flush=True)
h1_pass = bool((v32 < 1e-4).sum() < len(v32) * 0.5)  # not all collapsed
print(f"H1 (variance at sub-square): {'PASS' if h1_pass else 'FAIL'}", flush=True)

# ---- H2: sub-square lmin predicts square lmin ----
print("=== H2: lmin(T=32) vs lmin(T=64) per head ===", flush=True)
keys64, base_out64, ids64 = get_keys(64)
lmin64 = lmin_map(keys64)
common = sorted(set(lmin32) & set(lmin64))
a32 = np.array([lmin32[k] for k in common])
a64 = np.array([lmin64[k] for k in common])
r_h2 = float(np.corrcoef(a32, a64)[0, 1])
rng = np.random.default_rng(7)
perm_p = sum(1 for _ in range(10000)
             if abs(np.corrcoef(a32, rng.permutation(a64))[0, 1]) >= abs(r_h2)) / 10000
print(f"pearson(lmin32, lmin64) = {r_h2:.4f} (perm p={perm_p:.4f})", flush=True)
h2_pass = bool(perm_p < 0.05)
print(f"H2 (collapse is head-stable): {'PASS' if h2_pass else 'FAIL'}", flush=True)

# ---- H3: sub-square lmin predicts ablation importance at square ----
print("=== H3: lmin(T=32) vs ablation importance at T=64 ===", flush=True)
wte = model.embed_in.weight
with torch.no_grad():
    base_logits = base_out64.last_hidden_state[0, -1].float() @ wte.T.float()
base_probs = F.softmax(base_logits, dim=-1)
LAYER_SUBSET = [0, 4, 8, 12, 16, 20]
importance = {}
for li in LAYER_SUBSET:
    blk = model.layers[li]
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
            out_a = model(ids64)
        blk._forward_hooks.clear()
        logits_a = out_a.last_hidden_state[0, -1].float() @ wte.T.float()
        importance[f"{li},{h}"] = float(F.kl_div(
            F.log_softmax(logits_a, dim=-1), base_probs, reduction="sum").item())
    print(f"  layer {li} ablated ({time.time()-t0:.0f}s)", flush=True)

sub = [f"{li},{h}" for li in LAYER_SUBSET for h in range(NH)]
a_imp = np.array([importance[k] for k in sub])
a_lmin32 = np.array([lmin32[k] for k in sub])
r_h3 = float(np.corrcoef(a_lmin32, a_imp)[0, 1])
perm_p3 = sum(1 for _ in range(10000)
              if abs(np.corrcoef(a_lmin32, rng.permutation(a_imp))[0, 1]) >= abs(r_h3)) / 10000
rho_h3 = float(np.corrcoef(_rank(a_lmin32), _rank(a_imp))[0, 1]) if False else float(
    np.corrcoef(np.argsort(np.argsort(a_lmin32)).astype(float),
                np.argsort(np.argsort(a_imp)).astype(float))[0, 1])
print(f"pearson(lmin32, importance@64) = {r_h3:.4f} (perm p={perm_p3:.4f})", flush=True)
print(f"spearman = {rho_h3:.4f}", flush=True)
h3_pass = bool(perm_p3 < 0.05 and r_h3 > 0)
print(f"H3 (sub-square criterion): {'PASS' if h3_pass else 'FAIL'}", flush=True)

out = {
 "experiment": "Exp 34 - sub-square spectral criterion (Pythia-410M)",
 "preregistered": {"H1": "variance at T=32", "H2": "lmin32 ~ lmin64 per head",
                    "H3": "lmin32 predicts importance@64 (r>0, p<0.05)"},
 "H1": {"pass": h1_pass, "lmin_range": [round(float(v32.min()), 8), round(float(v32.max()), 8)],
        "collapsed_fraction": f"{int((v32 < 1e-4).sum())}/{len(v32)}"},
 "H2": {"pass": h2_pass, "pearson": round(r_h2, 4), "perm_p": round(perm_p, 4)},
 "H3": {"pass": h3_pass, "pearson": round(r_h3, 4), "perm_p": round(perm_p3, 4),
        "spearman": round(rho_h3, 4)},
 "lmin32_map": lmin32, "lmin64_map": lmin64, "importance_map": importance,
 "runtime_seconds": round(time.time() - t0, 1),
}
json.dump(out, open("/home/delorien/exp34_results.json", "w"), indent=2)
print("saved /home/delorien/exp34_results.json", f"({time.time()-t0:.0f}s)")
