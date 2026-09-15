#!/usr/bin/env python3
"""
Regenera fig3_double_dissociation.png a partir de data/out_H2.txt (caso
CUADRADO, rho=1.00) y data/out_diag_H2_cond.txt (autovalores del Gram).

La fig3 shippeada es PRE-auditoria y no coincide con los datos actuales:
mostraba pinv=0.767 vs gram=0.800 en rho=1.00 (una diferencia chica, sin
disociacion visible). Los datos actuales dan pinv=0.550 vs gram=0.550
(empatados entre si, y muy por debajo de pure=1.000) -- ESA es la doble
disociacion real que describe el README: pinv arregla rho<1 pero no rho=1,
y en rho=1 termina empatado con el gram sin corregir, no separado de el.
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
h2 = (ROOT / "data" / "out_H2.txt").read_text()
cond = (ROOT / "data" / "out_diag_H2_cond.txt").read_text()

# --- Panel izquierdo: accuracy en el caso cuadrado (rho=1.00), T=100 ---
m = re.search(
    r"--- n=6 K=3 N=96 \(CUADRADO.*?---\n(.*?)\n\n", h2, flags=re.S)
body = m.group(1)
acc = {}
for d in ["gram", "gradient", "pure", "pinv"]:
    mm = re.search(rf"^\s*{d} \|\s*([\d.]+) \|\s*([\d.]+)\s*$", body, flags=re.M)
    acc[d] = float(mm.group(2))

# --- Panel derecho: condicionamiento del Gram matrix vs rho ---
rows = re.findall(
    r"^n=6 K=(\d) N=(\d+)\s+\(rho=([\d.]+)[^)]*\)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.eE+-]+)\s*\|",
    cond, flags=re.M)
rhos_c = [float(r[2]) for r in rows]
conds = [float(r[3]) for r in rows]

order = ["gram", "pinv", "gradient", "pure"]
disp = {"gram": "gram", "pinv": "pinv", "gradient": "gradient", "pure": "pure"}
colors = {"gram": "#c0392b", "gradient": "#8e44ad", "pure": "#27ae60", "pinv": "#2980b9"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

ax = axes[0]
labels = [disp[d] for d in order][::-1]
vals = [acc[d] for d in order][::-1]
cols = [colors[d] for d in order][::-1]
bars = ax.barh(labels, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=12, fontweight="bold")
ax.axvline(0.85, color="black", linestyle="--", linewidth=1.3)
ax.set_xlim(0, 1.12)
ax.set_xlabel(r"Accuracy at $\rho$ = 1.00 (T = 100)", fontsize=12)
ax.set_title("Double dissociation at the square boundary\n(regenerated from current data/out_H2.txt)",
              fontsize=12.5, fontweight="bold")

ax2 = axes[1]
ax2.axvspan(0.4, 1.0, color="#e74c3c", alpha=0.10, zorder=0)
ax2.axvspan(1.0, 1.6, color="#2ecc71", alpha=0.10, zorder=0)
ax2.axvline(1.0, color="#e67e22", linestyle="--", linewidth=2, zorder=1)
ax2.plot(rhos_c, conds, "o-", color="#2c3e50", markersize=8, linewidth=2)
ax2.set_yscale("log")
ax2.set_xlim(0.6, 1.6)
ax2.set_xlabel(r"$\rho$", fontsize=13)
ax2.set_ylabel("Condition number of Gram matrix", fontsize=11)
ax2.set_title("Spectral conditioning across rho\n(from data/out_diag_H2_cond.txt)",
               fontsize=12.5, fontweight="bold")
for x, y in zip(rhos_c, conds):
    ax2.annotate(f"{y:.1e}", (x, y), xytext=(0, 10), textcoords="offset points",
                 ha="center", fontsize=8.5)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "fig3_double_dissociation.png", dpi=200, bbox_inches="tight")
print("wrote fig3_double_dissociation.png")
print("acc @ rho=1.00:", acc)
print("cond vs rho:", list(zip(rhos_c, conds)))
