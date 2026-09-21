#!/usr/bin/env python3
"""
EXP 19 — REPLICA INDEPENDIENTE de Exp 18 (Capa 0 del roadmap).

Restriccion clave: NO importa absolutamente NADA de exp_observer_taxonomy_v3.
Reimplementa desde cero un resonator HRR minimo:
  - codebooks gaussianos normalizados (n_roles=2 por bloque, 16 simbolos/rol)
  - bind = convolucion circular (FFT), unbind = convolucion con conjugado
  - bundle = suma de binds por bloque
  - resonator: estima, resta otros, unbind, cleanup
  - observers: pure / gram(M^-1 f ambient) / dual_same(C^T M^-1 C f)

Si gram~0.16 y dual_same~0.98 reproducen AQUI, la Frame-Duality Law no
depende del harness. Este es el experimento bloqueante del roadmap.

Salida: data/exp19_independent_replica.json/.txt
"""
import json
import numpy as np
import random
from pathlib import Path
import time

# ---------- implementacion desde cero (sin compartir codigo) ----------
def circ_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def circ_unbind(c, r):
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real


def build_block(seed, n_sym=16, n_roles=2, d=32):
    """Bloque cuadrado: n_cv = n_roles*n_sym = 32 codevectors en d=32.
    Cada rol tiene SU PROPIO codebook de 16 simbolos (como CFG6)."""
    rng = np.random.RandomState(seed)
    syms_per_role = []
    for r in range(n_roles):
        s = rng.randn(n_sym, d)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        syms_per_role.append(s)
    C = np.vstack(syms_per_role)   # (n_roles*n_sym) x d = 32 x 32
    roles = []
    for _ in range(n_roles):
        v = rng.randn(d)
        roles.append(v / np.linalg.norm(v))
    M = C @ C.T
    ev = np.linalg.eigvalsh(M)
    Minv = np.linalg.inv(M) if abs(ev[0]) > 1e-12 else None
    P_dual = C.T @ Minv @ C if Minv is not None else None
    return {"syms_per_role": syms_per_role, "C": C, "roles": roles, "M": M,
            "Minv": Minv, "P_dual": P_dual,
            "kappa": float(ev[-1] / ev[0]), "lmin": float(ev[0])}


def encode(block, fact):
    """fact: (i0, i1) indices de simbolo por rol."""
    s = np.zeros_like(block["roles"][0])
    for r, i in enumerate(fact):
        s = s + circ_bind(block["roles"][r], block["syms_per_role"][r][i])
    return s


def decode(block, s, mode, T=50, init_seed=0):
    rng = np.random.RandomState(init_seed)
    est = []
    for _ in block["roles"]:
        v = rng.randn(block["roles"][0].shape[0])
        est.append(v / np.linalg.norm(v))
    for _ in range(T):
        new = []
        for r in range(len(block["roles"])):
            others = s.copy()
            for o in range(len(block["roles"])):
                if o != r:
                    others = others - circ_bind(block["roles"][o], est[o])
            fj = circ_unbind(others, block["roles"][r])
            if mode == "gram" and block["Minv"] is not None and block["Minv"].shape[0] == fj.shape[0]:
                fj = block["Minv"] @ fj
            elif mode == "dual_same" and block["P_dual"] is not None:
                fj = block["P_dual"] @ fj
            sims = block["syms_per_role"][r] @ fj
            new.append(block["syms_per_role"][r][int(np.argmax(sims))])
        est = new
    out = []
    for r in range(len(block["roles"])):
        sims = block["syms_per_role"][r] @ est[r]
        out.append(int(np.argmax(sims)))
    return out


def main():
    t0 = time.time()
    N_SEEDS, N_FACTS, N_INIT = 60, 10, 5
    rng = random.Random(2026)
    acc = {m: [] for m in ("pure", "gram", "dual_same")}
    kap, lm = [], []
    for s in range(N_SEEDS):
        block = build_block(10_000 + s)
        kap.append(block["kappa"]); lm.append(block["lmin"])
        per_seed = {m: [] for m in acc}
        for _ in range(N_FACTS):
            fact = (rng.randrange(16), rng.randrange(16))
            sig = encode(block, fact)
            for init in range(N_INIT):
                for m in acc:
                    out = decode(block, sig, m, T=50, init_seed=5000 + init)
                    good = sum(int(a == b) for a, b in zip(out, fact)) / len(fact)
                    per_seed[m].append(good)
        for m in acc:
            acc[m].append(float(np.mean(per_seed[m])))
        if (s + 1) % 20 == 0:
            print(f"  seed {s+1}/{N_SEEDS}: " + " ".join(
                f"{m}={np.mean(acc[m]):.3f}" for m in acc))
    out = {}
    print("\nREPLICA INDEPENDIENTE (rho=1, T=50):")
    for m, a in acc.items():
        a = np.array(a)
        boots = [np.mean(np.random.RandomState(k).choice(a, len(a), True)) for k in range(500)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {"mean": float(a.mean()), "median": float(np.median(a)),
                  "boot95": [float(lo), float(hi)]}
        print(f"  {m:9s}: mean {a.mean():.3f} boot95 [{lo:.3f},{hi:.3f}]")
    d = np.array(acc["dual_same"]) - np.array(acc["gram"])
    out["paired"] = {"mean": float(d.mean()), "frac_pos": float((d > 0.01).mean())}
    out["kappa_med"] = float(np.median(kap)); out["lmin_med"] = float(np.median(lm))
    print(f"pareado dual_same-gram: {d.mean():+.3f} ({100*(d>0.01).mean():.0f}% positivo, N={N_SEEDS})")
    jp = Path(__file__).parent.parent / "data" / "exp19_independent_replica.json"
    jp.write_text(json.dumps(out, indent=1))
    print(f"Guardado: {jp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
