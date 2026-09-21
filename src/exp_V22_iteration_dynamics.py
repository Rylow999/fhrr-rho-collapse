#!/usr/bin/env python3
"""
EXP 22 (Capa 2): dinamica del error en iteraciones.

Mide e_t = ||f_t - f_true|| en cada paso t = 0..50 para ambos wirings,
en rho=1. Fit e_{t+1} ~ a * e_t por observer. Predice a>1 para gram
despues de la primer aplicacion, a<=1 para dual.

Salida: data/exp22_iteration_dynamics.json
"""
import json
import numpy as np
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_V19_independent_replica import circ_bind, circ_unbind


def block( seed):
    rng = np.random.RandomState(seed)
    n_sym, n_roles, d = 16, 2, 32
    syms = []
    for r in range(n_roles):
        s = rng.randn(n_sym, d); s /= np.linalg.norm(s, axis=1, keepdims=True); syms.append(s)
    C = np.vstack(syms)
    roles = []
    for _ in range(n_roles):
        v = rng.randn(d); roles.append(v / np.linalg.norm(v))
    M = C @ C.T
    Minv = np.linalg.inv(M)
    P_dual = C.T @ Minv @ C
    return {"syms": syms, "roles": roles, "M": M, "Minv": Minv, "P_dual": P_dual}


def run_trajectory(b, fact, mode, T=50):
    s = sum(circ_bind(b["roles"][r], b["syms"][r][fact[r]]) for r in range(2))
    rng = np.random.RandomState(0)
    est = []
    for _ in range(2):
        v = rng.randn(32); est.append(v / np.linalg.norm(v))
    truths = [b["syms"][r][fact[r]] for r in range(2)]
    errors = []
    for t in range(T):
        errs = []
        new = []
        for r in range(2):
            others = s.copy()
            for o in range(2):
                if o != r:
                    others -= circ_bind(b["roles"][o], est[o])
            fj = circ_unbind(others, b["roles"][r])
            if mode == "gram":
                fj = b["Minv"] @ fj
            elif mode == "dual_same":
                fj = b["P_dual"] @ fj
            sims = b["syms"][r] @ fj
            new.append(b["syms"][r][int(np.argmax(sims))])
            errs.append(float(np.linalg.norm(fj - truths[r]) / np.linalg.norm(truths[r])))
        errors.append(float(np.mean(errs)))
        est = new
    return errors


def main():
    T = 50
    traj_gram, traj_dual = [], []
    for s in range(30):
        b = block(2000 + s)
        fact = (random.Random(s).randrange(16), random.Random(s + 99).randrange(16))
        traj_gram.append(run_trajectory(b, fact, "gram", T))
        traj_dual.append(run_trajectory(b, fact, "dual_same", T))
    G = np.array(traj_gram); D = np.array(traj_dual)
    # growth factor: a in log e_{t+1} = log a + log e_t
    def growth(E):
        # usar tramo t=0..10 (pre-saturacion)
        l = np.log(np.maximum(E[:, :11], 1e-16))
        x = np.arange(11)
        a_lin, _ = [], None
        slopes = []
        for row in l:
            coef = np.polyfit(x, row, 1)
            slopes.append(coef[0])
        return float(np.median(slopes))
    g = growth(G); d = growth(D)
    print(f"EXP 22: dinamica del error (n=30 seeds, T=50)")
    print(f"  gram:      e_0={G[:,0].mean():.2f}  e_10={G[:,10].mean():.2f}  e_49={G[:,49].mean():.2f}  growth a={np.exp(g):.3f}")
    print(f"  dual_same: e_0={D[:,0].mean():.2f}  e_10={D[:,10].mean():.2f}  e_49={D[:,49].mean():.2f}  growth a={np.exp(d):.3f}")
    out = {"gram": {"e0": float(G[:,0].mean()), "e10": float(G[:,10].mean()),
                    "e49": float(G[:,49].mean()), "growth_a": float(np.exp(g))},
           "dual_same": {"e0": float(D[:,0].mean()), "e10": float(D[:,10].mean()),
                         "e49": float(D[:,49].mean()), "growth_a": float(np.exp(d))},
           "traj_gram_mean": G.mean(axis=0).tolist(),
           "traj_dual_mean": D.mean(axis=0).tolist()}
    Path("../data/exp22_iteration_dynamics.json").write_text(json.dumps(out, indent=1))
    print("Guardado: data/exp22_iteration_dynamics.json")


if __name__ == "__main__":
    main()
