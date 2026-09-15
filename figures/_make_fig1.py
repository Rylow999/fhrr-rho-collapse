#!/usr/bin/env python3
"""
Regenera fig1_phase_diagram.png con layout corregido (auditoria 2026-09).
Problemas del original: titulo cortado a la izquierda, exceso de espacio en
blanco, y labels de rho en el eje x superpuestos/ilegibles.
No habia script de figuras en el repo -- este es nuevo, escrito para esta
correccion, usando unicamente los valores de rho respaldados por data/out_F2.txt
y data/out_H2.txt (no se inventan puntos adicionales).
"""
import matplotlib.pyplot as plt

# rho values con respaldo real en data/out_F2.txt (n=2,3,4,6 a K=1,N=128)
# y data/out_H2.txt (grid de n=6 a distintos K,N).
rho_points = [0.25, 0.38, 0.50, 0.75, 0.80, 1.00, 1.33, 1.50]

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.axvspan(0.0, 1.0, color="#e74c3c", alpha=0.12, zorder=0)
ax.axvspan(1.0, 2.1, color="#2ecc71", alpha=0.12, zorder=0)
ax.axvline(1.0, color="#e67e22", linestyle="--", linewidth=2.5, zorder=1,
           label="Regime II (rho = 1)")

ax.scatter(rho_points, [1] * len(rho_points), s=90, color="#2c3e50",
           zorder=3, label="Experimental sweeps")

for x in rho_points:
    ax.annotate(f"{x:.2f}", (x, 1), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=9, rotation=45)

ax.text(0.03, 1.55,
        "Rank-deficient Gram\nmat_inv yields garbage\nAccuracy ~ 1/|V|",
        color="#c0392b", fontsize=11, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#c0392b"))
ax.text(1.05, 1.75,
        "Overcomplete frame\nStable dual\nAll decoders work",
        color="#27ae60", fontsize=11, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#27ae60"))
ax.text(1.02, 1.35,
        "Square Gram\nDual degenerate\npinv amplifies noise",
        color="#d35400", fontsize=11, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#d35400"))

ax.set_xlim(0.0, 2.1)
ax.set_ylim(0.3, 2.1)
ax.set_yticks([])
ax.set_xlabel(r"$\rho = \dfrac{\mathrm{distinct\ codevectors}}{\mathrm{block\ dimension}}$",
              fontsize=13, labelpad=12)
ax.set_title("Phase diagram of compositional decoding in FHRR",
             fontsize=15, fontweight="bold", pad=14)

handles = [
    plt.Rectangle((0, 0), 1, 1, color="#e74c3c", alpha=0.12),
    plt.Rectangle((0, 0), 1, 1, color="#2ecc71", alpha=0.12),
    plt.Line2D([0], [0], color="#e67e22", linestyle="--", linewidth=2.5),
    plt.Line2D([0], [0], marker="o", color="#2c3e50", linestyle="",
               markersize=9),
]
labels = ["Regime I (rho < 1)", "Regime III (rho > 1)",
          "Regime II (rho = 1)", "Experimental sweeps"]
ax.legend(handles, labels, loc="lower left", frameon=True, fontsize=10)

fig.tight_layout()
fig.savefig("fig1_phase_diagram.png", dpi=200, bbox_inches="tight")
print("wrote fig1_phase_diagram.png")
