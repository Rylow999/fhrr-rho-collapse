#!/usr/bin/env python3
"""
EXP 24 (Capa I): frame bounds (alpha, beta) vs estabilidad del operador compuesto.

Para matrices C con diferentes frame bounds (alpha = lambda_min(AA^T),
beta = lambda_max(AA^T)) medir:
  - || C^T (CC^T)^+ C ||_2       (debe ser <= 1 siempre: proyector)
  - ganancia de perturbacion del dual
  - `tightness` = beta/alpha
Prediccion: la ganancia dual no depende de alpha/beta (es proyector); la
ganancia ambient SI depende de alpha (= 1/alpha). Este desacople es el
contenido no trivial del principio.

Construir familias controlando tightness via mezcla de lineas mitad
bien condicionada + mitad degenerada:
  C = [random rows (bien) ; escala * random rows (mal condicionadas)]
"""
import json
import numpy as np
from pathlib import Path


def make_frame(n, d, tightness_target, seed):
    """Construye C (n x d) con control sobre kappa_frame = beta/alpha."""
    rng = np.random.RandomState(seed)
    # mezclar filas: la mitad en un subespacio de dim reducida
    C = rng.randn(n, d)
    # cuanto mas tightness_target, mas colapso de lambda_min
    k = tightness_target
    # Las filas 2..int(n/2) se convierten en combos casi-lineales de la 1
    for i in range(1, max(2, int(n * min(k - 1, 0.9) / (k - 1 + 1e-9)))):
        C[i] = C[0] + 1e-3 * rng.randn(d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def main():
    n, d = 32, 32
    rows = []
    print("EXP 24: frame bounds vs estabilidad (n=d=32)")
    print(f"  {'tight':>6s} {'alpha':>10s} {'beta':>8s} {'kappa':>8s} {'||dual||':>9s} {'gain_amb':>10s} {'gain_dual':>9s}")
    for tk in [1, 1.5, 2, 3, 5, 10, 100]:
        for seed in range(3):
            C = make_frame(n, d, tk, seed)
            M = C @ C.T
            w = np.linalg.eigvalsh(M)
            alpha, beta = float(w[0]), float(w[-1])
            Dual = C.T @ np.linalg.pinv(M, rcond=1e-12) @ C
            Amb = np.linalg.pinv(M, rcond=1e-12)  # (No inv exacta si sing)
            rng = np.random.RandomState(0)
            ga, gd = [], []
            for _ in range(100):
                f = rng.randn(d); f /= np.linalg.norm(f)
                e = 1e-6 * rng.randn(d)
                ga.append(np.linalg.norm(Amb @ (f + e) - Amb @ f) / np.linalg.norm(e))
                gd.append(np.linalg.norm(Dual @ (f + e) - Dual @ f) / np.linalg.norm(e))
            rows.append({"tight": tk, "seed": seed, "alpha": alpha, "beta": beta,
                         "kappa_frame": beta / alpha,
                         "norm_dual": float(np.linalg.norm(Dual, 2)),
                         "gain_amb_med": float(np.median(ga)),
                         "gain_dual_med": float(np.median(gd))})
    for r in rows[::3]:
        print(f"  {r['tight']:6.1f} {r['alpha']:10.2e} {r['beta']:8.2f} "
              f"{r['kappa_frame']:8.2f} {r['norm_dual']:9.3f} {r['gain_amb_med']:10.2e} {r['gain_dual_med']:9.3f}")

    # Consolidacion por tightness
    import collections
    by_t = collections.defaultdict(list)
    for r in rows: by_t[r['tight']].append(r)
    print("\nPor tightness (mediana):")
    for t, rs in sorted(by_t.items()):
        print(f"  t={t}: alpha={np.median([r['alpha'] for r in rs]):.2e} "
              f"kappa={np.median([r['kappa_frame'] for r in rs]):.2e} "
              f"gain_amb={np.median([r['gain_amb_med'] for r in rs]):.2e} "
              f"gain_dual={np.median([r['gain_dual_med'] for r in rs]):.3f}")
    Path("../data/exp24_frame_bounds.json").write_text(json.dumps(rows, indent=1))
    print("\nGuardado: data/exp24_frame_bounds.json")


if __name__ == "__main__":
    main()
