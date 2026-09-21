#!/usr/bin/env python3
"""
EXP 21 (Capa 3) — De-VSA-ificacion: pipeline puramente lineal, sin resonator.

Experimento: y = Cx + eps (measurement); decodificar x por:
  ambient : xhat = M^{-1} (C y)... NO: M^{-1} opera sobre espacio n.
  En decoding VSA el "amplificado" es f en espacio d; la pregunta es si
  la mala inversion es intrinsica al OPERADOR, no al decoder.

Test puro de operador: tomo f ANY vector en R^d. Comparo:
  ambient : A_amb f = M^{-1} f          (solo valido si n == d)
  dual    : A_dual f = C^T M^{-1} C f   = f si C cuadrado invertible
Agrego ruido eps a f y mido ||A(f+eps) - A f|| / ||eps||.

Prediccion FDL: ambient ~ 1/lambda_min, dual ~ 1.

NO HAY RESONATOR. NO HAY DECODE. Si el efecto replica aca, la "causa" es
el placement del operador, no el algoritmo VSA.
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_V20_ensembles_rho import make_C

def main():
    rows = []
    print("EXP 21: amplificacion lineal pura (sin resonator), n=d=32")
    for seed in range(30):
        C = make_C("gauss", 32, 32, seed=seed)
        M = C @ C.T
        w = np.linalg.eigvalsh(M)
        lmin = float(w[0])
        Minv = np.linalg.inv(M)
        A_amb = Minv                      # en rho=1, aplicable (n=d=32)
        A_dual = C.T @ Minv @ C           # = I
        rng = np.random.RandomState(seed)
        ratios_amb, ratios_dual = [], []
        for _ in range(500):
            f = rng.randn(32); f /= np.linalg.norm(f)
            eps = 1e-6 * rng.randn(32)
            noisy = f + eps
            da = np.linalg.norm(A_amb @ noisy - A_amb @ f) / np.linalg.norm(eps)
            dd = np.linalg.norm(A_dual @ noisy - A_dual @ f) / np.linalg.norm(eps)
            ratios_amb.append(da); ratios_dual.append(dd)
        rows.append({"seed": seed, "lambda_min": lmin,
                     "gain_amb_med": float(np.median(ratios_amb)),
                     "gain_amb_p99": float(np.percentile(ratios_amb, 99)),
                     "gain_dual_med": float(np.median(ratios_dual)),
                     "gain_dual_p99": float(np.percentile(ratios_dual, 99)),
                     "inv_lmin": 1.0 / lmin})
    med_amb = float(np.median([r["gain_amb_med"] for r in rows]))
    med_dual = float(np.median([r["gain_dual_med"] for r in rows]))
    p99_amb = float(np.median([r["gain_amb_p99"] for r in rows]))
    p99_dual = float(np.median([r["gain_dual_p99"] for r in rows]))
    print(f"  ganancia ambient 1/lambda_min: mediana {med_amb:.2e}  p99 {p99_amb:.2e}")
    print(f"  ganancia dual:                 mediana {med_dual:.3f}  p99 {p99_dual:.3f}")
    print(f"  ratio (teorico): 1/lambda_min vs 1")
    out = {"med_amb": med_amb, "med_dual": med_dual,
           "p99_amb": p99_amb, "p99_dual": p99_dual}
    Path("../data/exp21_linear_amp.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print("Guardado: data/exp21_linear_amp.json")

if __name__ == "__main__":
    main()
