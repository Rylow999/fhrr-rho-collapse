#!/usr/bin/env python3
"""
EXP 24 (reescrito): frame bounds controlados EXACTAMENTE.

Construccion: C = Q D  con Q ortogonal (n x n) y D = diag(sigma_1..sigma_n)
un operador diagonal. Entonces M = C C^T = Q D D^T Q^T y los frame bounds
son alpha = sigma_min^2, beta = sigma_max^2, exactos.

Se mide, como funcion de kappa(C) = sigma_max/sigma_min controlado:
  - ||C^T M^+ C||_2 (debe ser 1 para todo kappa: proyector sobre row(C))
  - gain ambiente ||M^+||
  - gain del dual en perturbacion.
La proposicion predice: norm_dual = 1 SIEMPRE (proyector sobre R^n=C row),
incluso para kappa(C) arbitrario.
"""
import json
import numpy as np
from pathlib import Path

def main():
    n = 32
    rng = np.random.RandomState(0)
    out = []
    for kappa_C in [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]:
        Q, _ = np.linalg.qr(rng.randn(n, n))
        sig = np.geomspace(1.0, kappa_C, n)
        C = Q @ np.diag(sig)
        M = C @ C.T
        alpha, beta = float(np.min(sig**2)), float(np.max(sig**2))
        Mpinv = np.linalg.pinv(M, rcond=1e-13)
        Dual = C.T @ Mpinv @ C
        r2 = []
        for _ in range(200):
            f = rng.randn(n); f /= np.linalg.norm(f)
            e = 1e-6 * rng.randn(n)
            # ganancia dual del proyector
            r2.append(np.linalg.norm(Dual @ (f + e) - Dual @ f) / np.linalg.norm(e))
        out.append({
            "kappa_C": float(kappa_C), "alpha": alpha, "beta": beta,
            "norm_dual": float(np.linalg.norm(Dual, 2)),
            "gain_dual_med": float(np.median(r2)),
        })
        print(f"kappa(C)={kappa_C:8.0e}  alpha={alpha:.2e}  beta={beta:.2e}  "
              f"||dual||={out[-1]['norm_dual']:.3f}  gain_dual={out[-1]['gain_dual_med']:.3f}")
    Path("../data/exp24_frame_bounds.json").write_text(json.dumps(out, indent=1))
    print("Guardado: data/exp24_frame_bounds.json")


if __name__ == "__main__":
    main()
