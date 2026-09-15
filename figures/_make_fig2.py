#!/usr/bin/env python3
"""
Regenera fig2_accuracy_vs_rho.png a partir de data/out_H2.txt (columna T=100).

La fig2 shippeada en el repo es PRE-auditoria: no hay script que la generara
(ver changelog del README), y sus valores en rho=1.00 (gram~0.80, pinv~0.77)
NO coinciden con data/out_H2.txt actual (gram=0.550, pinv=0.550). Ese es
justamente el resultado central del paper (doble disociacion en rho=1), asi
que la figura vieja lo contradice visualmente.

Este script parsea out_H2.txt y usa exactamente los 6 casos que arman la
tabla "H2: Double dissociation at rho = 1" del README (uno por cada rho),
sin inventar puntos adicionales.
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

DATA = Path(__file__).parent.parent / "data" / "out_H2.txt"

# Casos oficiales (uno por rho) tal como los usa la tabla del README.
# n=6 K=2 N=128 (tambien rho=0.75) y n=3 K=3 N=126 (control, sin Gram) se
# excluyen a proposito: son puntos redundantes/de control, no parte del
# sweep principal de rho.
WANTED = [
    "n=4 K=1 N=128",
    "n=6 K=1 N=128",
    "n=6 K=3 N=120",
    "n=6 K=3 N=96 (CUADRADO, replica Prueba B)",
    "n=6 K=3 N=72",
    "n=6 K=2 N=64",
]

text = DATA.read_text()
blocks = re.split(r"^--- (.+?) \| BLK=.*? ---\n", text, flags=re.M)
# blocks = [preamble, header1, body1, header2, body2, ...]
bodies = {blocks[i]: blocks[i + 1] for i in range(1, len(blocks), 2)}

rho_of = {
    "n=4 K=1 N=128": 0.50,
    "n=6 K=1 N=128": 0.75,
    "n=6 K=3 N=120": 0.80,
    "n=6 K=3 N=96 (CUADRADO, replica Prueba B)": 1.00,
    "n=6 K=3 N=72": 1.33,
    "n=6 K=2 N=64": 1.50,
}

decoders = ["gram", "gradient", "pure", "pinv"]
rhos, series = [], {d: [] for d in decoders}
for label in WANTED:
    body = bodies[label]
    rhos.append(rho_of[label])
    for d in decoders:
        m = re.search(rf"^\s*{d} \|\s*([\d.]+) \|\s*([\d.]+)\s*$", body, flags=re.M)
        series[d].append(float(m.group(2)))  # columna T=100

colors = {"gram": "#c0392b", "gradient": "#8e44ad", "pure": "#27ae60", "pinv": "#2980b9"}
markers = {"gram": "o", "gradient": "s", "pure": "^", "pinv": "D"}
labels = {"gram": "gram (mat_inv)", "gradient": "gradient", "pure": "pure", "pinv": "pinv"}

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axvspan(0.0, 1.0, color="#e74c3c", alpha=0.10, zorder=0)
ax.axvspan(1.0, 1.7, color="#2ecc71", alpha=0.10, zorder=0)
ax.axvline(1.0, color="#e67e22", linestyle="--", linewidth=2, zorder=1)

for d in decoders:
    ax.plot(rhos, series[d], marker=markers[d], color=colors[d], linewidth=2,
             markersize=8, label=labels[d])

ax.set_xlim(0.42, 1.62)
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel(r"$\rho$", fontsize=13)
ax.set_ylabel("Accuracy (T = 100)", fontsize=12)
ax.set_title("Decoder accuracy across the rho phase diagram\n(regenerated from current data/out_H2.txt)",
              fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, ncol=2, frameon=True)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "fig2_accuracy_vs_rho.png", dpi=200, bbox_inches="tight")
print("wrote fig2_accuracy_vs_rho.png")
for d in decoders:
    print(d, list(zip(rhos, series[d])))
