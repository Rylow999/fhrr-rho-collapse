#!/usr/bin/env python3
# exp_init_ortogonal.py — via 2: edge sano desde el nacimiento
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import time

torch.manual_seed(2026)
np.random.seed(2026)

N_ROLES, N_FILLERS = 64, 64
D_MODEL, N_LAYERS, N_HEADS = 128, 4, 4
D_HEAD = D_MODEL // N_HEADS
MAX_PAIRS = 40
T_EVAL = [6, 10, 16, 20, 24, 30, 36, 40]
BATCH = 24
STEPS = 1500
LR = 3e-4
VOCAB = 2 + N_ROLES + N_FILLERS

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
    return (torch.tensor(seqs), torch.tensor(labels))

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
    def __init__(self, ortho_init=False):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(512, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)
        if ortho_init:
            self._ortho_keys()

    def _ortho_keys(self):
        """Inicializar Wk con bloque-diagonal ortogonal por head (QR).
        Cada head recibe filas ortogonales => K K^T ~ I en T <= d_head."""
        with torch.no_grad():
            for blk in self.blocks:
                W = blk.attn.in_proj_weight   # (3D, D)
                # construir Wk ortogonal POR HEAD: bloque QR de (d_head, D_MODEL->d_head)
                # cada head opera en su slice de d_head dims tras reshape
                Wk_new = torch.zeros(D_MODEL, D_MODEL)
                for h in range(N_HEADS):
                    q, _ = torch.linalg.qr(torch.randn(D_MODEL, D_HEAD))
                    Wk_new[h*D_HEAD:(h+1)*D_HEAD, :] = q.T   # filas ortonormales
                W[D_MODEL:2*D_MODEL] = Wk_new * math.sqrt(D_MODEL / D_HEAD)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T)[None])
        keys_per_head = []
        for blk in self.blocks:
            h = blk.ln1(x)
            W = blk.attn.in_proj_weight
            Wk = W[D_MODEL:2*D_MODEL]
            K = h @ Wk.T
            K = K.view(B, T, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
            keys_per_head.append(K)
            x = blk(x)
        return self.head(self.ln_f(x)), keys_per_head

def edge_penalty(keys):
    """FIX v2 (2026-10-11): indice correcto del edge.
    Para T <= d_head: lmin = ev[0] (autovalor minimo, rango completo).
    Para T > d_head: lmin = ev[d_head-1] (el edge del rango util).
    (El v1 usaba ev[min(T,dh)-1] que para T<dh daba el MAXIMO — bug
    detectado por el valor 1.5>1 imposible para lmin de gram unit-norm.)"""
    pen = 0.0
    for K in keys:
        Kn = F.normalize(K, dim=-1)
        M = Kn @ Kn.transpose(-1, -2)
        ev = torch.linalg.eigvalsh(M)
        T_here = ev.shape[-1]
        idx = 0 if T_here <= D_HEAD else D_HEAD - 1
        lmin = ev[..., idx].clamp_min(1e-4)
        pen += (-torch.log(lmin)).mean()
    return pen / len(keys)

def train(ortho, alpha, seed=2026):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniLLM(ortho_init=ortho)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    hist = {"loss": [], "lmin": []}
    t0 = time.time()
    for step in range(STEPS):
        T = int(rng.integers(4, MAX_PAIRS + 1))
        x, y = make_batch(BATCH, T, rng)
        logits, keys = model(x[:, :-1])
        loss = F.cross_entropy(logits[:, -1, :], y)
        total = loss + alpha * edge_penalty(keys)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            with torch.no_grad():
                lms = []
                for K in keys:
                    Kn = F.normalize(K, dim=-1)
                    ev = torch.linalg.eigvalsh(Kn @ Kn.transpose(-1, -2))
                    T_here = ev.shape[-1]
                    idx = 0 if T_here <= D_HEAD else D_HEAD - 1
                    lms.append(ev[..., idx].median().item())
            hist["loss"].append(float(loss.detach()))
            hist["lmin"].append(float(np.median(lms)))
            print(f"  orto={ortho} a={alpha} step={step:4d}: loss={loss:.3f} lmin={np.median(lms):.4f}", flush=True)
    return model, hist, time.time() - t0

def evaluate(model, n_trials=200):
    model.eval()
    rng = np.random.default_rng(777)
    curve = []
    with torch.no_grad():
        for T in T_EVAL:
            correct = 0
            n = 0
            for _ in range(n_trials // BATCH + 1):
                x, y = make_batch(BATCH, T, rng)
                logits, _ = model(x[:, :-1])
                correct += (logits[:, -1, :].argmax(-1) == y).sum().item()
                n += BATCH
            curve.append({"T": T, "acc": correct / n})
            print(f"    T={T:3d}: acc={correct/n:.3f}", flush=True)
    return curve

BRAZOS = [(False, 0.0, "baseline"), (True, 0.0, "init-ort"), (True, 0.1, "init-ort+reg0.1")]
res = {"predicciones": {"PR-O1": "init QR da lmin alto desde paso 0",
                         "PR-O2": "lmin se mantiene sin penalty",
                         "PR-O3": "task loss ~ baseline (no canibaliza)",
                         "PR-O4": "ganancia (si la hay) en T grande"},
       "brazos": []}
for ortho, alpha, name in BRAZOS:
    print(f"\n===== {name} =====", flush=True)
    model, hist, secs = train(ortho, alpha)
    curve = evaluate(model)
    res["brazos"].append({"name": name, "ortho": ortho, "alpha": alpha,
                           "lmin_final": hist["lmin"][-1], "loss_final": hist["loss"][-1],
                           "segundos": secs, "curva": curve,
                           "hist_loss": hist["loss"], "hist_lmin": hist["lmin"]})
    print(f"  {name}: lmin={hist['lmin'][-1]:.4f} loss={hist['loss'][-1]:.3f} ({secs:.0f}s)", flush=True)

with open("data_init_ort.json", "w") as f:
    json.dump(res, f, indent=1)
print("\nINITORT_DONE", flush=True)
