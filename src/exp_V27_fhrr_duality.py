#!/usr/bin/env python3
"""
EXP 27: test causal de Frame-Dual Stability en FHRR (el algebra originaria).

FHRR: codevectors unit-modulus complejos z_i = exp(i theta_i).
bind = multiplicacion elemento a elemento (compleja).
Gram: M_ij = <c_i, c_j>_C = c_i^* c_j (Hermitiana: M = C^H C no C C^T).

La forma dual: C^H M^{-1} C  (con C^H = conjugado transpuesto).
En cuadrado invertible: C^H (C C^H)^{-1} C = I (mismo algebra, compleja).

Comparamos: pure / gram amb / dual_same amb en el punto cuadrado.
Mismo harness que Exp 19 (independiente), adaptado a complejos.

Salida: data/exp27_fhrr_duality.json
"""
import json
import numpy as np
import random
from pathlib import Path

def fhrr_bundle(seed, n_sym=16, n_roles=2, d=32):
    rng = np.random.RandomState(seed)
    syms = []
    for _ in range(n_roles):
        ph = rng.uniform(0, 2 * np.pi, size=(n_sym, d))
        syms.append(np.exp(1j * ph))    # fases puras, |-|=1
    C = np.vstack(syms)                # (n_roles*n_sym, d) = (32, 32)
    roles = []
    for _ in range(n_roles):
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, size=d))
        roles.append(v)
    M = C @ C.conj().T                 # Gram Hermitiana
    ev = np.linalg.eigvalsh(M)
    Minv = np.linalg.inv(M) if abs(ev[0]) > 1e-12 else None
    P_dual = C.conj().T @ Minv @ C if Minv is not None else None
    return {"syms": syms, "C": C, "roles": roles, "M": M,
            "Minv": Minv, "P_dual": P_dual,
            "kappa": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
            "lmin": float(ev[0])}


def encode(b, fact):
    s = np.zeros(b["C"].shape[1], dtype=complex)
    for r, i in enumerate(fact):
        # bind complejo = producto elemento a elemento (rotacion de fases)
        s = s + b["roles"][r] * b["syms"][r][i]
    return s


def decode(b, s, mode, T=50, init_seed=0):
    rng = np.random.RandomState(init_seed)
    est = [np.exp(1j * rng.uniform(0, 2 * np.pi, b["C"].shape[1])) for _ in b["roles"]]
    for _ in range(T):
        new = []
        for r in range(len(b["roles"])):
            others = s.copy()
            for o in range(len(b["roles"])):
                if o != r:
                    others = others - b["roles"][o] * est[o]
            # unbind complejo = multiplicar por conjugado del rol
            fj = np.conj(b["roles"][r]) * others
            if mode == "gram" and b["Minv"] is not None:
                fj = b["Minv"] @ fj
            elif mode == "dual_same" and b["P_dual"] is not None:
                fj = b["P_dual"] @ fj
            # cleanup: argmax del producto interno Hermitiano
            sims = np.abs(b["syms"][r] @ np.conj(fj))
            new.append(b["syms"][r][int(np.argmax(sims))])
        est = new
    out = []
    for r in range(len(b["roles"])):
        sims = np.abs(b["syms"][r] @ np.conj(est[r]))
        out.append(int(np.argmax(sims)))
    return out


def main():
    rng = random.Random(42)
    N_SEEDS, N_FACTS, N_INIT = 60, 10, 5
    acc = {m: [] for m in ("pure", "gram", "dual_same")}
    ks, lmins = [], []
    print("EXP 27: Frame-Dual Stability en FHRR (punto cuadrado, d=32)")
    for s in range(N_SEEDS):
        b = fhrr_bundle(7000 + s)
        ks.append(b["kappa"]); lmins.append(b["lmin"])
        per = {m: [] for m in acc}
        for _ in range(N_FACTS):
            fact = (rng.randrange(16), rng.randrange(16))
            sig = encode(b, fact)
            for init in range(N_INIT):
                for m in acc:
                    out = decode(b, sig, m, T=50, init_seed=55 + init)
                    per[m].append(sum(int(a == f) for a, f in zip(out, fact)) / 2)
        for m in acc:
            acc[m].append(float(np.mean(per[m])))
        if (s + 1) % 20 == 0:
            print(f"  seed {s+1}: pure={np.mean(acc['pure']):.3f} "
                  f"gram={np.mean(acc['gram']):.3f} dual={np.mean(acc['dual_same']):.3f}")
    for m in acc:
        a = np.array(acc[m])
        print(f"  {m:9s}: mean {a.mean():.3f}  median {np.median(a):.3f}")
    d = np.array(acc["dual_same"]) - np.array(acc["gram"])
    out = {m: {"mean": float(np.mean(acc[m]))} for m in acc}
    out["paired"] = {"mean": float(d.mean()), "frac_pos": float((d > 0.01).mean())}
    out["kappa_med"] = float(np.median(ks))
    out["lmin_med"] = float(np.median(lmins))
    print(f"pareado dual_same - gram: {d.mean():+.3f}, {100*(d>0.01).mean():.0f}% positivo")
    Path("../data/exp27_fhrr_duality.json").write_text(json.dumps(out, indent=1))
    print("Guardado: data/exp27_fhrr_duality.json")


if __name__ == "__main__":
    main()
