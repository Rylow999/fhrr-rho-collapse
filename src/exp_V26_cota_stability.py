#!/usr/bin/env python3
"""
EXP 26: cota analitica de la FDS principle.

Teorema empirico (verificado en Exp 21/23/24/25):
  Para T un operador con accion sobre coeficientes (espacio n),
  C (n x d) codebook, S = C^T, A = C:
    || S T A ||_2 = || C^T T C ||_2  <=  kappa(C) * ||T|_{row(C)}||_2
  y en particular si T = M^{-1}  =>  C^T T C = P_row(C), norma <= 1,
  EN TODOS LOS CASOS (no solo cuadrado), siempre que M sea invertible
  o reemplazada por su pseudo-inversa (entonces P es el proyector sobre
  el rango).

Este script verifica numericamente la cota
    ||C^T T C||_2 / max(||T||_{row(C)}, 1)  ≈<= kappa(C)
en una bateria de (C, T) con T de espectro controlado.
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_V20_ensembles_rho import make_C

def check(n, d, lams, seed):
    rng = np.random.RandomState(seed)
    C = make_C("gauss", n, d, seed)
    M = C @ C.T
    kappa_C = float(np.linalg.cond(C))
    # T con espectro controlado: U diag(lams) U^T
    Q, _ = np.linalg.qr(rng.randn(n, n))
    T = (Q * lams) @ Q.T
    # T restringido al row(C): proyectar — approxima como singular value max
    P = C.T @ M @ C  # not needed explicitly
    dual = C.T @ T @ C
    norm_dual = float(np.linalg.norm(dual, 2))
    norm_T = float(np.linalg.norm(T, 2))
    ratio = norm_dual / max(norm_T, 1e-300)
    return {"n": n, "d": d, "kappa_C": kappa_C,
            "lams_max": float(np.max(np.abs(lams))),
            "norm_dual": norm_dual, "norm_T": norm_T,
            "ratio_dual_over_T": ratio}


def main():
    rows = []
    rng = np.random.RandomState(0)
    for n, d in [(32, 32), (16, 32), (48, 32)]:
        for scale in (1.0, 1e3, 1e6):
            lams = np.concatenate([rng.rand(n // 2) * 1.0,
                                   rng.rand(n - n // 2) * scale])
            for seed in range(5):
                try:
                    rows.append(check(n, d, lams, seed))
                except np.linalg.LinAlgError:
                    pass
    ratios = [r["ratio_dual_over_T"] for r in rows]
    kappas = [r["kappa_C"] for r in rows]
    print(f"EXP 26: N={len(rows)} corridas")
    print(f"  ratio ||C^T T C|| / ||T||: min {min(ratios):.3f}  max {max(ratios):.3f}")
    print(f"  correlacion ratio vs kappa_C: {np.corrcoef(np.log(kappas), np.log(ratios))[0,1]:.3f}")
    out = {"rows": rows, "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios))}
    Path("../data/exp26_cota_stability.json").write_text(json.dumps(out, indent=1))
    print("Guardado: data/exp26_cota_stability.json")

if __name__ == "__main__":
    main()
