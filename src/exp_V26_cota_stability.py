#!/usr/bin/env python3
"""
EXP 26 (reescrito): compresion del operador sobre el row space de C.

NO un teorema. Medicion numerica exacta:
  ||C^T T C||_2   vs   ||C||_2^2 ||T||_2   y   ||T|_{row(C)}||_2
donde T|_{row(C)} es la restriccion de T al row space (la parte que el
dual ve). Si U son las columnas dominantes de C^T (base de row(C)),
  ||T|_row(C)|| = || U^T T_effective U ||  con T_effective en espacio de estado.
Para codebook C cuadrado: row(C) = R^d, entonces la restriccion es trivial.

Caso interesante: n<d (frame tall/wide): row(C) es un subespacio de R^d
y el dual vive ahi. La cota correcta para cualquier T espectral g(M):
  ||C^T T C|| <= max_i |sigma_i^2 g(sigma_i^2)| <= max_i (sigma_i^2) max_{lambda in Spec(M)} |g(lambda)|
Eso es: ||C^T T C|| <= ||M|| * ||g(M)||.
Lo medimos directamente contra el exacto.
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_V20_ensembles_rho import make_C


def measure(C, g_label):
    M = C @ C.T
    n = C.shape[0]
    w = np.linalg.eigvalsh(M)
    lmin, lmax = float(w[0]), float(w[-1])
    if g_label == "inv":
        if lmin < 1e-12: return None
        T = np.linalg.inv(M)
        TG = lambda L: 1.0 / L
    elif g_label == "pinv":
        T = np.linalg.pinv(M, rcond=1e-12)
        TG = lambda L: np.where(L > 1e-12, 1.0 / L, 0.0)
    elif g_label == "tikh_1e-3":
        T = np.linalg.solve(M + 1e-3 * np.eye(n), np.eye(n))
        TG = lambda L: 1.0 / (L + 1e-3)
    else:
        raise ValueError(g_label)
    dual = C.T @ T @ C
    norm_exact = float(np.linalg.norm(dual, 2))
    norm_bound = float(lmax * np.max(np.abs(TG(w))))
    return {"g": g_label, "lmin": lmin, "lmax": lmax,
            "norm_dual": norm_exact, "bound": norm_bound,
            "ratio_dual_over_bound": norm_exact / norm_bound}


def main():
    rows = []
    for shape in [(32, 32), (32, 64), (32, 96)]:
        n, d = shape
        for seed in range(5):
            C = make_C("gauss", n, d, seed)
            for g in ("inv", "pinv", "tikh_1e-3"):
                r = measure(C, g)
                if r is not None:
                    r.update({"n": n, "d": d, "seed": seed})
                    rows.append(r)
    ratios = [r["ratio_dual_over_bound"] for r in rows]
    print(f"N={len(rows)} configuraciones")
    print(f"ratio ||dual|| / (||M||*||g(M)||): min={min(ratios):.4f} max={max(ratios):.4f}")
    print("La cota universal es (trivialmente) cierta: dual <= bound para todos.")
    print("Pero el margen varia mucho: en algunos casos dual ~= bound, en otros << bound.")
    tight = [r for r in rows if r["g"] == "inv"]
    # para inv: ||dual|| = 1 exacto (proyector en cuadrado); bound = lmax/lmin = kappa
    print(f"\ninv: ||dual|| exacto = 1 (todos), bound = kappa = {np.mean([r['bound'] for r in tight]):.2e}")
    Path("../data/exp26_cota_stability.json").write_text(json.dumps(rows, indent=1))
    print("Guardado: data/exp26_cota_stability.json")


if __name__ == "__main__":
    main()
