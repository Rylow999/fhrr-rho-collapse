#!/usr/bin/env python3
# exp_gpt2_keys v3 — la medicion FINA: curva lmin(T) por sub-contexto + rango efectivo
# La pregunta discriminante: en T < d_head=64 (rango completo POSIBLE),
# siguen las keys entrenadas la ley del hard edge? Y cual es el rango
# efectivo del codebook en el cuadrado?
import torch
import numpy as np
import json

from transformers import GPT2Model, GPT2Tokenizer

print("cargando GPT-2...", flush=True)
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
D = model.config.n_embd; NL = model.config.n_layer; NH = model.config.n_head; DH = D // NH

# texto largo (300+ tokens) para tener sub-contextos amplios
TEXTO = ("The Collatz conjecture states that every positive integer reaches "
         "one under the map that sends even numbers to half their value and "
         "odd numbers to three times plus one. The problem remains open "
         "despite extensive computational verification. Researchers study "
         "statistical properties, spectral methods, and probabilistic models "
         "of the iterations. The conjecture is famous for its simplicity "
         "and the difficulty of finding a proof. Many approaches use the "
         "two adic structure of the accelerated map. The orbit structure "
         "remains mysterious. Similar questions arise in number theory "
         "and dynamical systems. The map has been analyzed with ergodic "
         "theory, random walks, and transfer operators. Computational "
         "experiments suggest the conjecture holds for all tested values. "
         "The mathematical community considers it one of the great open "
         "problems. Various generalizations have been proposed and studied "
         "extensively in the literature of discrete dynamics.")
ids = tok(TEXTO, return_tensors="pt")
T_total = ids["input_ids"].shape[1]
print(f"d={DH}, tokens={T_total}", flush=True)

with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
hiddens = out.hidden_states

def keys_head(H_in, li, hi):
    attn = model.h[li].attn
    qkv = attn.c_attn(torch.tensor(H_in, dtype=torch.float32)).detach().numpy()
    K_all = qkv[:, D:2*D].reshape(-1, NH, DH)
    return K_all[:, hi, :]

# ===== CURVA lmin(T): sub-contextos crecientes, capas 3/6/9 =====
Ts = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
print("\n== CURVA lmin(T) de keys entrenadas (mediana sobre heads de capas 3,6,9) ==", flush=True)
curva = []
for T in Ts:
    lmins_layers = []
    for li in [3, 6, 9]:
        H = hiddens[li][0].numpy()[:T]
        lmins = []
        for hi in range(NH):
            K = keys_head(H, li, hi)[:T]
            Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
            M = Kn @ Kn.T
            ev = np.linalg.eigvalsh(M)
            lmins.append(float(max(ev[0], 0.0)))
        lmins_layers.append(np.median(lmins))
    med = float(np.median(lmins_layers))
    curva.append({"T": T, "lmin_med": med, "T_over_d": T / DH})
    print(f"  T={T:3d} (T/d={T/DH:.2f}): lmin={med:.4f}", flush=True)

# ===== RANGO EFECTIVO: cuantos autovalores > 1e-3 en el cuadrado T=d =====
print("\n== RANGO EFECTIVO del codebook (T=64=d, umbral 1e-3) ==", flush=True)
rangos = []
for li in [3, 6, 9]:
    H = hiddens[li][0].numpy()[:DH]
    rr = []
    for hi in range(NH):
        K = keys_head(H, li, hi)[:DH]
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
        ev = np.linalg.eigvalsh(Kn @ Kn.T)
        rr.append(int((ev > 1e-3).sum()))
    rangos.append({"layer": li, "rango_med": float(np.median(rr))})
    print(f"  capa {li}: rango efectivo mediano = {np.median(rr):.0f} de {DH}", flush=True)

# ===== capacidad REAL: rango_efectivo * sqrt(lmin_positiva) =====
print("\n== CAPACIDAD REAL (rango_ef * sqrt(lmin^+)) ==", flush=True)
caps = []
for li in [3, 6, 9]:
    H = hiddens[li][0].numpy()
    cc = []
    for hi in range(NH):
        K = keys_head(H, li, hi)
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
        ev = np.linalg.eigvalsh(Kn @ Kn.T)[:T_total] if T_total <= DH else np.linalg.eigvalsh(Kn[:DH] @ Kn[:DH].T)
        pos = ev[ev > 1e-3]
        if len(pos) > 0:
            cc.append(len(pos) * np.sqrt(pos.min()))
    caps.append({"layer": li, "capacidad": float(np.median(cc))})
    print(f"  capa {li}: capacidad ~ {np.median(cc):.1f} roles", flush=True)

out = {"d_head": DH, "T_total": T_total, "curva_lmin_T": curva,
       "rango_efectivo": rangos, "capacidad_real": caps}
with open("/home/delorien/fhrr-rho-collapse/experiments/data_gpt2_fine.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nFINE_DONE", flush=True)
