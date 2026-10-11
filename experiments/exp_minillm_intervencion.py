#!/usr/bin/env python3
# exp_minillm_intervencion.py — MiniLLM baseline vs edge-reg (2026-10-10)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import time

torch.manual_seed(2026)
np.random.seed(2026)
DEVICE = "cpu"

# ---------------- hiperparametros ----------------
N_ROLES = 64          # vocabulario de roles
N_FILLERS = 64         # vocabulario de fillers
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS   # 32
MAX_PAIRS = 40         # maximo T del entrenamiento
T_EVAL = [6, 10, 16, 20, 24, 30, 36, 40]   # curva de evaluacion
BATCH = 32
STEPS = 1500
LR = 3e-4
ALPHA_EDGELIN = [0.0, 0.5, 2.0]   # baseline + dos dosis de la intervencion

# ---------------- tarea: pair-recall ----------------
# secuencia: [BOS] (r1 f1) (r2 f2) ... (rT fT) [SEP] (r_query) -> predecir f_query
# tokens: 0=BOS, 1=SEP, roles 2..65, fillers 66..129
VOCAB = 2 + N_ROLES + N_FILLERS   # 130

def make_batch(batch, T, rng):
    seqs = np.zeros((batch, 2*T + 3), dtype=np.int64)
    labels = np.zeros((batch,), dtype=np.int64)
    for b in range(batch):
        roles = rng.choice(N_ROLES, size=T, replace=False)
        fillers = rng.choice(N_FILLERS, size=T, replace=False)
        s = [0]
        for i in range(T):
            s += [2 + roles[i], 2 + N_ROLES + fillers[i]]
        qi = rng.integers(0, T)
        s += [1, 2 + roles[qi]]
        seqs[b] = s
        labels[b] = 2 + N_ROLES + fillers[qi]
    return (torch.tensor(seqs).to(DEVICE), torch.tensor(labels).to(DEVICE))

# ---------------- modelo ----------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(512, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T)[None].to(DEVICE))
        # capturar keys por head para la regularizacion
        keys_per_head = []
        for blk in self.blocks:
            h = blk.ln1(x)
            # extraer K de MultiheadAttention: in_proj_weight [3d, d] -> K = h @ Wk^T
            W = blk.attn.in_proj_weight   # (3*D, D)
            Wk = W[D_MODEL:2*D_MODEL]     # (D, D)
            K = h @ Wk.T                  # (B, T, D)
            K = K.view(B, T, N_HEADS, D_HEAD).permute(0, 2, 1, 3)  # (B, H, T, dh)
            keys_per_head.append(K)
            x = blk(x)
        logits = self.head(self.ln_f(x))
        return logits, keys_per_head

def edge_penalty(keys_per_head):
    """Penalizacion del edge: -log del autovalor MINIMO POSITIVO accesible.
    FIX (v2): cuando T > d_head, lmin=0 es inevitable (rango) — penalizarlo
    no aporta gradiente. Version robusta en todo T: penalizar el MENOR
    autovalor POSITIVO: ev[..., min(T,dh)-1] (el edge del rango util).
    Ademas un termino suave: -log(trace_low) = empujar TODOS los autovalores
    bajos, no solo el minimo (evita singularidad del gradiente).
    """
    pen = 0.0
    n = 0
    for K in keys_per_head:
        Kn = F.normalize(K, dim=-1)          # (B, H, T, dh)
        M = Kn @ Kn.transpose(-1, -2)        # (B, H, T, T)
        ev = torch.linalg.eigvalsh(M)       # (B, H, T) — no-negativos (PSD)
        T_here = ev.shape[-1]
        r = min(T_here, D_HEAD)
        # el r-esimo autovalor desde arriba (el edge del rango util):
        edge_ev = ev[..., r-1].clamp_min(1e-4)   # ev[r-1] >= 0, orden ascendente
        # y el minimo positivo (para T <= dh es ev[0]):
        if T_here <= D_HEAD:
            lmin = ev[..., 0].clamp_min(1e-4)
        else:
            lmin = edge_ev
        pen += (-torch.log(lmin)).mean()
        n += 1
    return pen / n

# ---------------- entrenamiento ----------------
def train(alpha, seed=2026):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniLLM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    hist = {"loss": [], "edge_pen": [], "lmin_med": []}
    t0 = time.time()
    for step in range(STEPS):
        T = int(rng.integers(4, MAX_PAIRS + 1))
        x, y = make_batch(BATCH, T, rng)
        logits, keys = model(x[:, :-1])
        # la respuesta esta en la ultima posicion: predecir el token final
        logit_last = logits[:, -1, :]
        loss = F.cross_entropy(logit_last, y)
        pen = edge_penalty(keys)
        total = loss + alpha * pen
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                lm = []
                for K in keys:
                    Kn = F.normalize(K, dim=-1)
                    ev = torch.linalg.eigvalsh(Kn @ Kn.transpose(-1, -2))
                    lm.append(ev[..., 0].median().item())
            hist["loss"].append(float(loss.detach()))
            hist["edge_pen"].append(float(pen.detach()))
            hist["lmin_med"].append(float(np.median(lm)))
            print(f"  alpha={alpha} step={step:4d}: loss={loss:.3f} pen={pen:.3f} lmin_med={np.median(lm):.4f}", flush=True)
    return model, hist, time.time() - t0

# ---------------- evaluacion: curva recall(T) ----------------
def evaluate(model, rng, n_trials=200):
    model.eval()
    curve = []
    with torch.no_grad():
        for T in T_EVAL:
            correct = 0
            for chunk in range(n_trials // BATCH + 1):
                x, y = make_batch(BATCH, T, rng)
                logits, _ = model(x[:, :-1])
                pred = logits[:, -1, :].argmax(dim=-1)
                correct += (pred == y).sum().item()
            acc = correct / ((n_trials // BATCH + 1) * BATCH)
            curve.append({"T": T, "acc": acc})
            print(f"    T={T:3d}: acc={acc:.3f}", flush=True)
    return curve

# ---------------- correr los 3 modelos ----------------
resultados = {"predicciones_preregistradas": {
    "PR1": "edge-reg sube lmin vs baseline (sanity)",
    "PR2": "edge-reg mejora recall en T grande",
    "PR3": "edge-reg no dana en T pequeno",
    "PR4": "curva recall(T) desplazada a la derecha"},
    "modelos": []}

for alpha in ALPHA_EDGELIN:
    print(f"\n===== ENTRENANDO alpha={alpha} =====", flush=True)
    model, hist, secs = train(alpha)
    rng_ev = np.random.default_rng(777)   # eval con datos NO vistos del mismo generador
    curve = evaluate(model, rng_ev)
    resultados["modelos"].append({
        "alpha": alpha, "steps": STEPS, "segundos": secs,
        "hist_loss_final": hist["loss"][-1], "lmin_final_med": hist["lmin_med"][-1],
        "curva_recall": curve,
        "hist_muestreo": {"cada": 100, "loss": hist["loss"], "lmin": hist["lmin_med"]}})
    print(f"  alpha={alpha}: lmin_final={hist['lmin_med'][-1]:.4f} loss_final={hist['loss'][-1]:.3f} ({secs:.0f}s)", flush=True)

with open("/home/delorien/fhrr-rho-collapse/experiments/data_minillm.json", "w") as f:
    json.dump(resultados, f, indent=1)
print("\nMINILLM_DONE", flush=True)
