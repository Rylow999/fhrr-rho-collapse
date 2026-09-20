#!/usr/bin/env python3
"""Figura Exp 16: escala en n — espectro Wishart cuadrada vs prediccion 4n^2."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

d = json.load(open(Path(__file__).parent.parent / "data" / "exp16_scaling_n.json"))
s = d["spectra"]
ns = [r["n"] for r in s]
kap = [r["kappa_med"] for r in s]
kap_lo = [r["kappa_iqr"][0] for r in s]
kap_hi = [r["kappa_iqr"][1] for r in s]
pred = [r["pred_4n2"] for r in s]
lmin = [r["lambda_min_med"] for r in s]
lmax = [r["lambda_max_med"] for r in s]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# panel A: kappa vs 4n^2
ax1.fill_between(ns, kap_lo, kap_hi, alpha=0.25, color="tab:red", label="IQR (12 seeds)")
ax1.plot(ns, kap, "o-", color="tab:red", label="measured $\\kappa$ (median)")
ax1.plot(ns, pred, "k--", label="$4n^2$ (Edelman scaling)")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("n (codevectors per block, $\\rho=1$)")
ax1.set_ylabel("$\\kappa(M)$")
ax1.set_title("(a) Condition number at the square point")
ax1.legend(); ax1.grid(alpha=0.3, which="both")

# panel B: lambda_min y lambda_max
ax2.plot(ns, lmin, "s-", color="tab:blue", label="$\\lambda_{\\min}$ (median)")
ax2.plot(ns, [1.0/(n*n) for n in ns], "b:", label="$1/n^2$ scale")
ax2.plot(ns, lmax, "o-", color="tab:green", label="$\\lambda_{\\max}$ (median)")
ax2.axhline(4.0, color="g", ls="--", alpha=0.6, label="MP upper edge $=4$")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("n"); ax2.set_ylabel("eigenvalue")
ax2.set_title("(b) Soft edge vs hard edge")
ax2.legend(); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Scaling in $n$ at $\\rho=1$: the square Wishart anchors the $\\kappa$-law (Exp.~16)")
fig.tight_layout()
out = Path(__file__).parent.parent / "figures" / "fig_exp16_scaling.png"
fig.savefig(out, dpi=150)
print("Guardada:", out)
