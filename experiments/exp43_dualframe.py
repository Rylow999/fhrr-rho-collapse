#!/usr/bin/env python3
"""Exp 43 — Dual-frame correction on REAL transformer keys (GPT-2 124M).
Pre-registered (2026-10-24):
  D1: with real GPT-2 key-grams at square point, ambient inversion
      (M^-1 @ bundle) fails (accuracy ~ chance) — as predicted by the theory.
  D2: dual projection (C^T M^+ C @ bundle) recovers significantly better
      than ambient — the correction works on real trained keys.
  D3: pure (no correction) — the baseline for comparison.
  Falsification: if dual does NOT beat ambient on real keys, the paper's
  claim "memory layers must use dual decoding" is unsupported.

Setup: We extract the actual key vectors from GPT-2 at T=64 (=d_head).
We create a "memory task": bundle = sum(K_i * V_i) for random V_i.
Query: given the bundle and the key codebook C (the real GPT-2 keys),
recover V_target for a specific K_target. This is exactly what a
hyperdimensional memory layer (Dhayalkar 2025) would need to do.
"""
import torch, numpy as np, json, time
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

t0 = time.time()
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
D = model.config.n_embd
DH = D // NH
NL = model.config.n_layer
print(f"GPT-2 124M: d_head={DH}, {NL} layers, {NH} heads", flush=True)

TEXT64 = ("The quick brown fox jumps over the lazy dog. " * 12 +
    "The city council approved the budget for the new park after a long "
    "debate about the environmental impact and the cost of maintenance. "
    "Local residents supported the initiative but questioned the timeline.")
ids64 = tok(TEXT64, return_tensors="pt").input_ids[:, :64]

# Extract real keys per head
keys_by_layer = {}
for i, blk in enumerate(model.h):
    cat = blk.attn.c_attn
    def mk(idx, proj):
        def hook(mod, inp, out):
            hidden = inp[0].float()
            B, T, _ = hidden.shape
            qkv = hidden @ proj.weight.float() + proj.bias.float()
            K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
            keys_by_layer[idx] = K[0]
        return hook
    blk.attn.register_forward_hook(mk(i, cat))
with torch.no_grad():
    _ = model(ids64)
for i in range(NL):
    model.h[i].attn._forward_hooks.clear()

# For each head in a representative subset of layers: run the memory task
LAYERS_TO_TEST = [0, 3, 6, 9, 11]  # spread across depth
N_TRIALS = 50
results_by_decoder = {"ambient": [], "dual": [], "pure": []}

rng = np.random.default_rng(42)
for li in LAYERS_TO_TEST:
    K_np = keys_by_layer[li].numpy()  # (NH, T=64, DH=64)
    for h in range(NH):
        K = K_np[h]  # (64, 64) — the real GPT-2 keys for this head
        # Normalize keys (unit rows)
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
        T, d = Kn.shape  # T=64 (context), d=64 (d_head) — SQUARE POINT

        # The memory task: store T=64 (role, value) pairs
        # roles = the real keys; values = random vectors
        for trial in range(N_TRIALS):
            # Random values in the same dimension as the keys
            V = rng.normal(0, 1, (T, d))
            V = V / np.linalg.norm(V, axis=1, keepdims=True)

            # Bundle = sum of bound pairs (elementwise product binding)
            bundle = (Kn * V).sum(axis=0)

            # Query a random target
            target = int(rng.integers(0, T))

            # --- AMBIENT decoder: M^-1 @ bundle ---
            M = Kn @ Kn.T  # (T, T) key-gram
            try:
                M_inv = np.linalg.inv(M + 1e-10 * np.eye(T))
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            ambient_state = M_inv @ bundle  # (T,) in key space? No...
            # Actually: M^-1 @ bundle treats the bundle as a superposition in
            # key space. The unbinding is: M^-1 @ bundle gives coefficients
            # that weight each key. Then we compare against values.
            # Standard VSA decoding: est_value = (M_inv @ Kn) @ bundle @ ... 
            # No — let's be precise about what ambient vs dual means here.
            #
            # Ambient: the decoder works in the AMBIENT space (d-dimensional).
            # The "ambient" decoder for a memory layer is: M^-1 applied to
            # the key-space representation. In the VSA paper this is:
            #   f_ambient(bundle) = M^-1 @ (C @ bundle)  where C = key codebook
            # This is the WRONG placement (inverts in ambient space).
            #
            # Dual: the decoder projects through the codebook first:
            #   f_dual(bundle) = C^T @ M^+ @ C @ bundle
            # This is the RIGHT placement (inverts in coefficient space).
            #
            # Pure: no inversion at all:
            #   f_pure(bundle) = Kn[target] * bundle (direct elementwise unbind)
            
            # AMBIENT: M^-1 @ (C @ bundle) — the wrong placement
            C_bundle = Kn @ bundle  # (T,) — the bundle projected into key space
            ambient_coeffs = M_inv @ C_bundle  # (T,) — inverted coefficients
            # Recover value: the coefficients weight the keys; to get the value,
            # we need to solve: bundle ≈ sum(coeffs[i] * Kn[i] * V[i])
            # For the AMBIENT decoder, we compare the recovered state directly:
            ambient_state = Kn.T @ ambient_coeffs  # (d,) — back to ambient space
            ambient_est = ambient_state * Kn[target]
            
            # DUAL: C^T @ M^+ @ C @ bundle — the right placement
            M_pinv = np.linalg.pinv(M)
            dual_coeffs = M_pinv @ C_bundle  # (T,) — inverted in coefficient space
            dual_state = Kn.T @ dual_coeffs  # (d,) — back to ambient
            dual_est = dual_state * Kn[target]
            
            # PURE: direct unbind, no correction
            pure_est = Kn[target] * bundle
            
            # Score: argmax over stored values
            V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            
            ambient_pred = int(np.argmax(V_norm @ (ambient_est / (np.linalg.norm(ambient_est) + 1e-12))))
            dual_pred = int(np.argmax(V_norm @ (dual_est / (np.linalg.norm(dual_est) + 1e-12))))
            pure_pred = int(np.argmax(V_norm @ (pure_est / (np.linalg.norm(pure_est) + 1e-12))))
            
            results_by_decoder["ambient"].append(ambient_pred == target)
            results_by_decoder["dual"].append(dual_pred == target)
            results_by_decoder["pure"].append(pure_pred == target)

# Summarize
acc = {k: round(float(np.mean(v)), 4) for k, v in results_by_decoder.items()}
chance = 1.0 / 64  # 1/T
print(f"\nResults across {len(LAYERS_TO_TEST)} layers x {NH} heads x {N_TRIALS} trials:")
print(f"  chance level: {chance:.4f}")
for k, a in acc.items():
    print(f"  {k:10s}: {a:.4f} ({a/chance:.1f}x chance)")

# Paired comparison: dual vs ambient
dual_arr = np.array(results_by_decoder["dual"], dtype=float)
amb_arr = np.array(results_by_decoder["ambient"], dtype=float)
pure_arr = np.array(results_by_decoder["pure"], dtype=float)
n = len(dual_arr)
dual_mean = dual_arr.mean()
amb_mean = amb_arr.mean()
pure_mean = pure_arr.mean()
se = np.sqrt((dual_arr * (1-dual_arr) + amb_arr * (1-amb_arr)).sum()) / n
t_stat = (dual_mean - amb_mean) / (se + 1e-12)
print(f"\nDual vs Ambient: {dual_mean:.4f} vs {amb_mean:.4f} (t={t_stat:.2f})")
print(f"Dual vs Pure:    {dual_mean:.4f} vs {pure_mean:.4f}")

result = {
 "experiment": "Exp 43 — dual-frame correction on real GPT-2 key-grams",
 "date": "2026-10-24",
 "preregistered": {
   "D1": "ambient fails on real keys at square (as theory predicts)",
   "D2": "dual recovers significantly better than ambient",
   "D3": "pure is the no-correction baseline",
   "D4": "falsification: if dual doesn't beat ambient, the paper's memory-layer claim is unsupported",
 },
 "setup": {
   "model": "GPT-2 124M",
   "keys": "real key vectors from hooks (not random codebook)",
   "task": "memory: bundle=sum(K_i*V_i), query K_target, recover V_target",
   "T": 64, "d_head": 64, "square_point": True,
   "layers_tested": LAYERS_TO_TEST,
   "trials_per_head": N_TRIALS,
   "total_trials": len(results_by_decoder["dual"]),
 },
 "results": {
   "accuracy": acc,
   "chance_level": round(chance, 6),
   "dual_vs_ambient_t": round(float(t_stat), 3),
 },
 "verdict": "TO BE FILLED AFTER ANALYSIS",
 "runtime_s": round(time.time() - t0, 1),
}
json.dump(result, open("/home/delorien/exp43_results.json", "w"), indent=2)
print("\nsaved")
