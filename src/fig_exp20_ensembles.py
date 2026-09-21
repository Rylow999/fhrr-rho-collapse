#!/usr/bin/env python3
"""Figura Exp 20: ensembles y barrido rho."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

d = json.load(open(Path(__file__).parent.parent / "data" / "exp20_ensembles_rho.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

# panel A: ensembles
fams = ["gauss", "rademacher", "sphere", "toeplitz", "tight", "near_dup"]
labels = ["Gaussian", "Rademacher", "Sphere", "Toeplitz", "Tight\n(DFT)", "Near-dup\n(control)"]
pure = [d["ensembles"][f].get("pure", np.nan) for f in fams]
gram = [d["ensembles"][f].get("gram", np.nan) for f in fams]
dual = [d["ensembles"][f].get("dual_same", np.nan) for f in fams]
x = np.arange(len(fams))
w = 0.28
ax1.bar(x - w, pure, w, label="pure", color="tab:green", alpha=0.85)
ax1.bar(x, gram, w, label="gram ($M^{-1}$ ambient)", color="tab:red", alpha=0.85)
ax1.bar(x + w, dual, w, label="dual-same ($C^{\\top}M^{-1}C$)", color="tab:blue", alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.axhline(1.0, ls=":", color="gray"); ax1.set_ylim(0, 1.1)
ax1.set_ylabel("accuracy at $\\rho=1$")
ax1.set_title("(a) Ensemble transfer (independent implementation)")
ax1.legend(); ax1.grid(alpha=0.3, axis="y")

# panel B: rho sweep
rhos = [r["rho"] for r in d["rho_sweep"]]
g = [r["gram"] for r in d["rho_sweep"]]
du = [r["dual_same"] for r in d["rho_sweep"]]
pu = [r["pure"] for r in d["rho_sweep"]]
lmin = [r["lmin"] if r["lmin"] > 0 else np.nan for r in d["rho_sweep"]]
ax2.plot(rhos, pu, "o-", color="tab:green", label="pure")
ax2.plot(rhos, g, "s-", color="tab:red", label="gram (ambient $M^{-1}$)")
ax2.plot(rhos, du, "^-", color="tab:blue", label="dual-same")
ax2.axvline(1.0, color="k", ls="--", alpha=0.5, label="$\\rho=1$ (hard edge)")
ax2.set_xscale("log"); ax2.set_xlabel("$\\rho = n/d$")
ax2.set_ylabel("accuracy"); ax2.set_ylim(-0.05, 1.05)
ax2.set_title("(b) $\\rho$ sweep (Gaussian ensemble)")
ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle("Frame-duality across codebook ensembles and aspect ratios (Exp. 20)")
fig.tight_layout()
out = Path(__file__).parent.parent / "figures" / "fig_exp20_ensembles.png"
fig.savefig(out, dpi=150)
print("Guardada:", out)
