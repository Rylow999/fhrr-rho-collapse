#!/usr/bin/env python3
"""Figura Exp 18: normas de operador — la imagen del mecanismo."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

d = json.load(open(Path(__file__).parent.parent / "data" / "exp18_summary.json"))
on = json.load(open(Path(__file__).parent.parent / "data" / "exp18_operator_norms.json"))

kappas = [r["kappa"] for r in on]
gram_max = [r["gram_max"] for r in on]
dual_max = [r["dual_same_max"] for r in on]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# panel izq: normas de operador
ax1.scatter(kappas, gram_max, s=8, alpha=0.4, color="tab:red",
            label=r"$\|M^{-1}\|_2$ (ambient, asymmetric)")
ax1.scatter(kappas, dual_max, s=8, alpha=0.4, color="tab:blue",
            label=r"$\|C^\top M^{-1}C\|_2$ (dual, same resolvent)")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.axhline(1.0, color="tab:blue", ls="--", alpha=0.5)
ax1.set_xlabel(r"$\kappa(M)$")
ax1.set_ylabel(r"operator norm (max gain over 200 test vectors)")
ax1.set_title("(a) Same resolvent, different wiring")
ax1.legend(); ax1.grid(alpha=0.3, which="both")

# panel der: accuracy en rho=1 con las 5 condiciones
sq = {r["mode"]: r for r in d["square"]}
modes = ["pure", "gram", "pinv", "dual", "dual_same"]
labels = ["pure (no inv.)", "gram $M^{-1}f$", "pinv $M^{+}f$",
          "dual $C^{\\top}M^{+}Cf$", "dual-same $C^{\\top}M^{-1}Cf$"]
vals = [sq[m]["mean"] for m in modes]
errs = [sq[m]["ic95"] for m in modes]
colors = ["tab:green", "tab:red", "tab:red", "tab:blue", "tab:blue"]
bars = ax2.bar(range(5), vals, yerr=errs, color=colors, alpha=0.85, capsize=4)
ax2.set_xticks(range(5))
ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.axhline(1.0, ls=":", color="gray", alpha=0.5)
ax2.set_ylabel("accuracy at $\\rho=1$ (200 seeds $\\times$ 10 facts)")
ax2.set_title("(b) Causal test: only the wiring changes")
ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Frame-Duality Law: the collapse is in the operator wiring, not the resolvent (Exp.~18)")
fig.tight_layout()
out = Path(__file__).parent.parent / "figures" / "fig_exp18_frame_duality.png"
fig.savefig(out, dpi=150)
print("Guardada:", out)
