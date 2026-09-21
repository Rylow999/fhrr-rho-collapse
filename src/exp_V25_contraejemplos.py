#!/usr/bin/env python3
"""
EXP 25 (Capa J) — Contraejemplos deliberados con matrices CONSTRUIDAS.

El objetivo es encontrar casos donde la envoltura dual NO salva. Si ninguno
la rompe, el principio es mas fuerte. Si alguno la rompe, aprendemos la
frontera exacta.

Casos:
  1. Identidad estricta (C = I): should be perfectly stable.
  2. C_cuadrada con columnas cuasi-colineales (epsilon -> 0).
  3. C tall (n > d): frame over-complete, M n x n singular. dual usa pinv.
  4. C wide (n < d): well-cond, M^-1 segura; dual es proyector propio.
  5. C con filas en escalas muy distintas (1, 1e-3): ill-scaled.
  6. C = bloque diagonal de dos escalas.
"""
import json
import numpy as np
from pathlib import Path

def measure(C, eps=1e-6, n_test=200):
    """C es (n x d). Ambient solo si n == d (M cuadrado aplicable a f in R^d)."""
    n, d = C.shape
    M = C @ C.T
    w = np.linalg.eigvalsh(M)
    alpha, beta = float(w[0]), float(w[-1])
    Mpinv = np.linalg.pinv(M, rcond=1e-12)
    Minv = np.linalg.inv(M) if alpha > 1e-14 else None
    da = C.T @ (Minv if Minv is not None else Mpinv) @ C
    rng = np.random.RandomState(0)
    ga, gd = [], []
    ambient_ok = (Minv is not None and n == d)
    n_state = C.shape[1]   # espacio de ESTADO (ambiente + dual viven aca)
    for _ in range(n_test):
        f = rng.randn(n_state); f /= np.linalg.norm(f)
        e = eps * rng.randn(n_state)
        if ambient_ok:
            ga.append(np.linalg.norm(Minv @ (f + e) - Minv @ f) / np.linalg.norm(e))
        gd.append(np.linalg.norm(da @ (f + e) - da @ f) / np.linalg.norm(e))
    return {"n": n, "d": d, "alpha": alpha, "beta": beta, "kappa": beta / alpha,
            "ambiente_aplicable": bool(ambient_ok),
            "norm_dual": float(np.linalg.norm(da, 2)),
            "gain_amb": float(np.median(ga)) if ga else None,
            "gain_dual": float(np.median(gd))}

def main():
    out = {}
    # 1. Identidad
    C = np.eye(32)
    out["identity"] = measure(C)
    # 2. Cuasi-colineal (epsilon chico)
    for eps in [1.0, 1e-1, 1e-3, 1e-6]:
        C = np.eye(32); C[1] = C[0] + eps * np.eye(32)[1]
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        out[f"quasi_colinear_eps={eps}"] = measure(C)
    # 3. Tall (n > d): mas filas que columnas → M singular
    C = np.random.RandomState(0).randn(40, 32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    out["tall_n40_d32"] = measure(C)
    # 4. Wide (n < d)
    C = np.random.RandomState(1).randn(24, 48)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    out["wide_n24_d48"] = measure(C)
    # 5. Escalas muy distintas: SIN normalizar filas
    C = np.eye(32); C[16:] *= 1e-3
    out["ill_scaled"] = measure(C)
    # 6. Bloque diagonal (sin renorm)
    C = np.block([[np.eye(16), np.zeros((16, 16))],
                  [np.zeros((16, 16)), 1e-2 * np.eye(16)]])
    out["block_diag"] = measure(C)

    print(f"  {'caso':28s} {'n xd':>10s} {'alpha':>10s} {'kappa':>10s} {'||dual||':>9s} {'gain_amb':>12s} {'gain_dual':>10s}")
    for k, r in out.items():
        ga = f"{r['gain_amb']:.2e}" if r['gain_amb'] is not None else "  n/a (n<d)"
        print(f"  {k:28s} {r['n']}x{r['d']:>6d} {r['alpha']:10.2e} {r['kappa']:10.2e} "
              f"{r['norm_dual']:9.3f} {ga:>12s} {r['gain_dual']:10.3f}")
    Path("../data/exp25_contraejemplos.json").write_text(json.dumps(out, indent=1))
    print("\nGuardado: data/exp25_contraejemplos.json")


if __name__ == "__main__":
    main()
