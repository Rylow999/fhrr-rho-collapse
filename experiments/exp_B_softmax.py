#!/usr/bin/env python3
# exp_B_softmax.py — Neural Computation: softmax como mecanismo causal de
# estabilizacion contra el colapso del punto cuadrado.
# 3 brazos, MISMA tarea (pair-recall), MISMO presupuesto:
#   1. resonator  (nuestro Exp 18 — control positivo: colapsa en rho=1 ambient)
#   2. attention-softmax (Exp 12b — control negativo: NO colapsa)
#   3. attention SIN softmax  (LA PREDICCION: colapsa como el resonator)
# Barrido rho = n_roles/d_head en la tarea.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, math, time

torch.manual_seed(2026)
np.random.seed(2026)
D_MODEL, N_LAYERS, N_HEADS = 128, 2, 4
D_HEAD = D_MODEL // N_HEADS    # 32

def make_batch(batch, T, rng):
    seqs = np.zeros((batch, 2*T+3), dtype=np.int64)
    labels = np.zeros((batch,), dtype=np.int64)
    for b in range(batch):
        roles = rng.choice(64, size=T, replace=False)
        fillers = rng.choice(64, size=T, replace=False)
        s = [0]
        for i in range(T):
            s += [2+roles[i], 2+64+fillers[i]]
        qi = rng.integers(0, T)
        s += [1, 2+roles[qi]]
        seqs[b] = s
        labels[b] = 2+64+fillers[qi]
    return torch.tensor(seqs), torch.tensor(labels)

class AttnBlock(nn.Module):
    def __init__(self, use_softmax):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3*D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4*D_MODEL), nn.GELU(), nn.Linear(4*D_MODEL, D_MODEL))
        self.use_softmax = use_softmax

    def forward(self, x):
        h = self.ln1(x)
        qkv = self.qkv(h)
        B, T, _ = qkv.shape
        q, k, v = qkv.split(D_MODEL, dim=-1)
        q = q.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_HEAD)
        if self.use_softmax:
            A = scores.softmax(dim=-1)
        else:
            # SIN softmax: pesos lineales (puede haber negativos — es la
            # condicion del brazo: Gram inversion sin normalizacion)
            A = scores / (scores.abs().max(dim=-1, keepdim=True).values + 1e-6)
        o = (A @ v).transpose(1, 2).reshape(B, T, D_MODEL)
        x = x + self.out(o)
        x = x + self.mlp(self.ln2(x))
        return x

class MiniModel(nn.Module):
    def __init__(self, use_softmax, vocab=2+64+64):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(512, D_MODEL)
        self.blocks = nn.ModuleList([AttnBlock(use_softmax) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)
    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T)[None])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))

def train_eval(use_softmax, seed=2026):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniModel(use_softmax)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # entrenar con T fijo = 20 (mismo para todos: control justo)
    T_TRAIN = 20
    for step in range(1200):
        x, y = make_batch(24, T_TRAIN, rng)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits[:, -1, :], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    # evaluar barrido T (rho = 2T/32 roles/dim)
    model.eval()
    curve = {}
    with torch.no_grad():
        for T in [4, 8, 12, 16, 20, 24, 28, 32]:
            rng_ev = np.random.default_rng(777)
            ok = n = 0
            for _ in range(6):
                x, y = make_batch(24, T, rng_ev)
                logits = model(x[:, :-1])
                ok += (logits[:, -1, :].argmax(-1) == y).sum().item()
                n += 24
            curve[T] = ok / n
    return curve

# brazo 1: resonator puro (de nuestro paper, Exp 18 pipeline)
def resonator_curve():
    """El pipeline del paper: bundle + cleanup vs rho. Control positivo."""
    rng = np.random.default_rng(42)
    curve = {}
    for T in [4, 8, 12, 16, 20, 24, 28, 32]:
        N = 512
        ok = n = 0
        for _ in range(100):
            C = rng.normal(0, 1, (T, N))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            fillers = rng.normal(0, 1, (T, N))
            bound = C * fillers
            bundle = bound.sum(axis=0)
            target = int(rng.integers(0, T))
            est = bundle * C[target]
            ok += int(np.argmax(est @ fillers.T) == target); n += 1
        curve[T] = ok / n
    return curve

res = {"predicciones": {
    "PR-B1": "sin softmax: colapsa en rho~1 como el resonator",
    "PR-B2": "con softmax: no colapsa (Exp 12b)"},
    "brazos": {}}

print("== brazo 1: resonator (control positivo, del paper) ==", flush=True)
res["brazos"]["resonator"] = resonator_curve()
print("  ", {k: round(v, 3) for k, v in res["brazos"]["resonator"].items()}, flush=True)

print("== brazo 2: attention CON softmax ==", flush=True)
t0 = time.time()
res["brazos"]["attn_softmax"] = train_eval(True)
print("  ", {k: round(v, 3) for k, v in res["brazos"]["attn_softmax"].items()}, f"({time.time()-t0:.0f}s)", flush=True)

print("== brazo 3: attention SIN softmax (LA PREDICCION) ==", flush=True)
t0 = time.time()
res["brazos"]["attn_linear"] = train_eval(False)
print("  ", {k: round(v, 3) for k, v in res["brazos"]["attn_linear"].items()}, f"({time.time()-t0:.0f}s)", flush=True)

with open("/home/delorien/fhrr-rho-collapse/experiments/data_expB_softmax.json", "w") as f:
    json.dump(res, f, indent=1)
print("EXPB_DONE", flush=True)
