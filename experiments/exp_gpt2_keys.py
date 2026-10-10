#!/usr/bin/env python3
# exp_gpt2_keys.py v2 — GPT-2 real con c_attn (QKV combinada)
import torch
import numpy as np
import json

from transformers import GPT2Model, GPT2Tokenizer

print("cargando GPT-2 124M...", flush=True)
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
D = model.config.n_embd
NL = model.config.n_layer
NH = model.config.n_head
DH = D // NH
print(f"d_model={D}, capas={NL}, heads={NH}, d_head={DH}", flush=True)

TEXTO = ("The Collatz conjecture states that every positive integer reaches "
         "one under the map that sends even numbers to half their value and "
         "odd numbers to three times plus one. The problem remains open "
         "despite extensive computational verification up to very large "
         "bounds. Researchers have studied statistical properties of the "
         "iterations, spectral methods, and probabilistic models. The "
         "conjecture is famous for its simplicity and the difficulty of "
         "finding a proof. Many approaches use the two adic structure of "
         "the accelerated map, where each odd step is followed by dividing "
         "out all factors of two. The orbit structure remains mysterious.")

ids = tok(TEXTO, return_tensors="pt")
T_total = ids["input_ids"].shape[1]
print(f"tokens: {T_total}", flush=True)

resultados = {"D": D, "n_layers": NL, "n_heads": NH, "d_head": DH, "T_total": T_total, "layers": []}

with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
hiddens = out.hidden_states

def keys_head(H_in, li, hi):
    """keys del head hi de la capa li: c_attn proyecta a 3D; K esta en [D, 2D)"""
    attn = model.h[li].attn
    qkv = attn.c_attn(torch.tensor(H_in, dtype=torch.float32)).detach().numpy()  # (T, 3D)
    K_all = qkv[:, D:2*D].reshape(T_total, NH, DH)
    return K_all[:, hi, :]

for li in range(NL):
    H_in = hiddens[li][0].numpy()
    layer_res = {"layer": li, "heads": []}
    for hi in range(NH):
        K = keys_head(H_in, li, hi)      # (T, 64)
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
        M = Kn @ Kn.T
        ev = np.linalg.eigvalsh(M)
        lmin = float(ev[0]); lmax = float(ev[-1])
        layer_res["heads"].append({
            "head": hi, "lmin": lmin, "lmax": lmax,
            "capacidad": float(DH * np.sqrt(max(lmin, 0))),
            "kappa": float(lmax / max(lmin, 1e-30)),
        })
    lm = [h["lmin"] for h in layer_res["heads"]]
    layer_res["lmin_med"] = float(np.median(lm))
    layer_res["capacidad_med"] = float(np.median([h["capacidad"] for h in layer_res["heads"]]))
    resultados["layers"].append(layer_res)
    print(f"  capa {li:2d}: lmin_med={layer_res['lmin_med']:.4f}  capacidad_med={layer_res['capacidad_med']:.1f} roles", flush=True)

# ---- M5: coherente vs shuffled ----
print("\n== M5: texto COHERENTE vs SHUFFLED ==", flush=True)
m5 = []
ids_rand = ids["input_ids"][:, torch.randperm(T_total)]
with torch.no_grad():
    out_r = model(input_ids=ids_rand, output_hidden_states=True)
for li in [0, 3, 6, 9, 11]:
    for tag, o in [("coherente", out), ("shuffled", out_r)]:
        H = o.hidden_states[li][0].numpy()
        lmins = []
        for hi in range(NH):
            K = keys_head(H, li, hi)
            Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
            M = Kn @ Kn.T
            lmins.append(np.linalg.eigvalsh(M)[0])
        m5.append({"layer": li, "tipo": tag, "lmin_med": float(np.median(lmins))})
        print(f"  capa {li:2d} {tag:10s}: lmin_med={np.median(lmins):.4f}", flush=True)
resultados["M5_coherente_vs_shuffled"] = m5

lmins_all = [h["lmin"] for L in resultados["layers"] for h in L["heads"]]
print(f"\nlmin GLOBAL: mediana={np.median(lmins_all):.4f} min={np.min(lmins_all):.2e}")
resultados["lmin_global_med"] = float(np.median(lmins_all))

with open("/home/delorien/fhrr-rho-collapse/experiments/data_gpt2.json", "w") as f:
    json.dump(resultados, f, indent=1)
print("GPT2_DONE", flush=True)
