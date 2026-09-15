#!/usr/bin/env python3
"""
Diagnóstico de condicionamiento: autovalores del Gram matrix por zona de rho.
Explica la doble disociación en rho=1 (pinv no trunca el modo pequeño
y por tanto amplifica el ruido).
"""
import numpy as np
from base_fhrr import *
from exp_F2 import FastBundle
from exp_H2 import CFG6

casos = [
    ("n=6 K=1 N=128 (rho=0.75)", 6, 1, 128, CFG6),
    ("n=6 K=3 N=120 (rho=0.80)", 6, 3, 120, CFG6),
    ("n=6 K=3 N=96  (rho=1.00, CUADRADO)", 6, 3, 96, CFG6),
    ("n=6 K=3 N=72  (rho=1.33)", 6, 3, 72,  CFG6),
    ("n=6 K=2 N=64  (rho=1.50)", 6, 2, 64,  CFG6),
]

print(f"{'caso':<38} | {'BLK':>4} | {'rank':>5} | {'cond':>12} | {'lambda_min':>12} | {'rcond_pinv=1e-10 trunca?':>10}")

for desc, n, K, N, cfg in casos:
    t = FastBundle(7, K, N, n, cfg)
    M = t.np_M[0]
    rank = np.linalg.matrix_rank(M)
    cond = np.linalg.cond(M)
    eigvals = np.linalg.eigvalsh((M + M.conj().T) / 2)  # M deberia ser Hermitiana (Gram)
    lam_min = eigvals.min()
    lam_max = eigvals.max()
    trunca = "SI (elimina modos)" if (lam_min / lam_max) < 1e-10 else "NO (usa el modo chico -> amplifica ruido)"
    print(f"{desc:<38} | {t.BLK:>4} | {rank:>5} | {cond:>12.3e} | {lam_min:>12.3e} | {trunca:>10}")
