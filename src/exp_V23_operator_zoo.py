#!/usr/bin/env python3
"""
EXP 23 — Zoo de operadores bajo ambos wirings (Capa F del roadmap final).

Para cada T in {M^-1, M^+, Tikhonov(lambda), TruncSVD(k=n-2)}:
  - ambiente:   y = T f
  - dual:       y = C^T T C f
Medir: norma de operador exacta, ganancia de perturbacion, y error de
reconstruccion ||A-I|| (A = operador completo).
Todo sobre C cuadrada (n=d=32) mal condicionada + otra bien condicionada.
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_V20_ensembles_rho import make_C


def operators(M):
    out = {}
    w, U = np.linalg.eigh(M)
    out["inv"] = np.linalg.inv(M)
    out["pinv"] = np.linalg.pinv(M, rcond=1e-10)
    out["tikh_1e-3"] = np.linalg.solve(M + 1e-3 * np.eye(len(M)), np.eye(len(M)))
    out["tikh_1e-1"] = np.linalg.solve(M + 1e-1 * np.eye(len(M)), np.eye(len(M)))
    D = np.where(w > np.sort(w)[2], 1.0 / w, 0.0)  # truncar 2 autovalores menores
    out["trunc_svd"] = (U * D) @ U.T
    return out


def measure(T, C):
    """Metricas del operador T bajo ambos wirings."""
    Amb = T
    Dua = C.T @ T @ C
    # ganancia de perturbacion con vectores random
    rng = np.random.RandomState(0)
    n = Amb.shape[0]
    ga, gd = [], []
    for _ in range(300):
        f = rng.randn(n); f /= np.linalg.norm(f)
        eps = 1e-5 * rng.randn(n)
        ga.append(np.linalg.norm(Amb @ (f + eps) - Amb @ f) / np.linalg.norm(eps))
        gd.append(np.linalg.norm(Dua @ (f + eps) - Dua @ f) / np.linalg.norm(eps))
    return {
        "norm_amb": float(np.linalg.norm(Amb, 2)),
        "norm_dual": float(np.linalg.norm(Dua, 2)),
        "gain_amb_med": float(np.median(ga)),
        "gain_dual_med": float(np.median(gd)),
        "err_recon_amb": float(np.linalg.norm(Amb - C @ np.linalg.inv(C @ C.T) @ C, 'fro')),
        "err_dual_minus_I": float(np.linalg.norm(Dua - np.eye(n), 'fro')),
    }


def main():
    ress = {}
    for name, seed in (("ill_conditioned", 11), ("well_conditioned", 22)):
        C = make_C("gauss", 32, 32, seed=seed)
        if name == "ill_conditioned":
            C[1] = C[0] + 1e-5 * np.random.RandomState(1).randn(32) / np.sqrt(32)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
        M = C @ C.T
        kappa = float(np.linalg.cond(M))
        ress[name] = {"kappa_gram": kappa, "per_operator": {}}
        print(f"\nEXP 23 — {name}  (kappa Gram = {kappa:.2e})")
        print(f"  {'op':12s} {'||amb||':>12s} {'||dual||':>12s} {'gainAmb':>10s} {'gainDual':>10s} {'||dual-I||':>10s}")
        for oname, T in operators(M).items():
            r = measure(T, C)
            ress[name]["per_operator"][oname] = r
            print(f"  {oname:12s} {r['norm_amb']:12.3e} {r['norm_dual']:12.3f} "
                  f"{r['gain_amb_med']:10.2e} {r['gain_dual_med']:10.3f} {r['err_dual_minus_I']:10.2e}")
    Path("../data/exp23_operator_zoo.json").write_text(json.dumps(ress, indent=1))
    print("\nGuardado: data/exp23_operator_zoo.json")


if __name__ == "__main__":
    main()
