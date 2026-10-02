#!/usr/bin/env python3
"""Exp 28 - Hard-edge convergence at the square point (rho = n/d = 1).

Post Exp 16: the fitted exponent of lambda_min(n) was -1.6 with few seeds.
This closes the convergence question with many seeds and larger n, and
directly tests the Chen-Liu-Zhou (arXiv:1002.3975) prediction that a
per-realization global constraint (our row normalization) does not alter
the local hard-edge law: unit-norm Gram vs pure Wishart, same law.

Arm U: rows ~ N(0,1) normalized to unit norm (the decoder's codebook).
Arm W: C_ij ~ N(0, 1/n) i.i.d. (literal Wishart, the Exp 16 reference).
M = C C^T. The hard-edge variable is lambda_min * n^2.
"""
import json, time
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "exp28_hardedge_convergence.json"
NS = [(32, 1200), (64, 1000), (128, 700), (256, 400), (512, 200), (1024, 80)]

def lmin_scaled(n, seeds, rng, unit_norm):
    out = np.empty(seeds)
    for s in range(seeds):
        C = rng.standard_normal((n, n))
        if unit_norm:
            C /= np.linalg.norm(C, axis=1, keepdims=True)
        else:
            C /= np.sqrt(n)
        ev = np.linalg.eigvalsh(C @ C.T)
        out[s] = ev[0] * n * n
    return out

def block(res_list, n, seeds, scaled, dt):
    q = np.quantile(scaled, [0.25, 0.5, 0.75])
    res_list.append({
        "n": n, "seeds": seeds, "seconds": round(dt, 1),
        "lmin_n2_q25": float(q[0]), "lmin_n2_med": float(q[1]),
        "lmin_n2_q75": float(q[2]), "lmin_n2_mean": float(scaled.mean()),
        "lambda_min_med": float(q[1] / (n * n)),
    })

unit, wish, ks_all = [], [], {}
for n, seeds in NS:
    t0 = time.time()
    su = lmin_scaled(n, seeds, np.random.default_rng(281 + n), True)
    sw = lmin_scaled(n, seeds, np.random.default_rng(937 + n), False)
    dt = time.time() - t0
    block(unit, n, seeds, su, 0.0)
    block(wish, n, seeds, sw, round(dt, 1))
    a, b = np.sort(su), np.sort(sw)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    ks_all[str(n)] = float(np.max(np.abs(cdf_a - cdf_b)))
    print(f"n={n}: U med={unit[-1]['lmin_n2_med']:.4f}  W med={wish[-1]['lmin_n2_med']:.4f}  KS={ks_all[str(n)]:.3f}  ({dt:.0f}s)", flush=True)
    OUT.write_text(json.dumps({"unit": unit, "wishart": wish, "ks": ks_all}, indent=1))

# exponent fit on unit-arm medians
ns = np.array([r["n"] for r in unit], float)
lm = np.array([r["lambda_min_med"] for r in unit])
fit_all = float(np.polyfit(np.log(ns), np.log(lm), 1)[0])
fit_tail = float(np.polyfit(np.log(ns[-3:]), np.log(lm[-3:]), 1)[0])
out = {"unit": unit, "wishart": wish, "ks": ks_all,
       "fit_exponent_all": fit_all, "fit_exponent_tail": fit_tail}
OUT.write_text(json.dumps(out, indent=1))
print(f"FIT: all={fit_all:.3f}  tail(n=256..1024)={fit_tail:.3f}", flush=True)
print("EXP28_DONE", flush=True)
