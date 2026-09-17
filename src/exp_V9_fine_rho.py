#!/usr/bin/env python3
"""
Exp 9: Barrido fino alrededor de rho=1.00 (HRR real, arquitectura V3).

Pregunta: la anti-resonancia en rho=1 (gram cae a ~0.05-0.17) es un punto
matemático exacto o tiene una región de transición de ancho finito?

Método: mantener n=6, K=3 (3 roles por bloque), codebooks de 16 símbolos.
Variando N de a pasos finos cerca de N=96 (donde BLK=32 y n_cv=32 -> rho=1):

    rho = 32/BLK, BLK = N/3

Grid fino: BLK en {26,27,28,29,30,31,32,33,34,35,36,38,40,44}
-> N = 3*BLK en {78,...,132}; rho de 1.23 a 0.73.

Decoders: gram, pure, pinv (gradient omitido: ya sabemos que colapsa).
10 seeds distintos por punto (bundle seed 1000+i) para error bars serios,
20 facts por punto.

Salidas:
  data/out_V9_fine_rho.json
  figures/fig_V9_fine_rho.png
"""
import numpy as np
import random
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy

def main():
    t0 = time.time()
    BLKs = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 44]
    n_seeds = 10
    n_facts = 20
    modes = ["gram", "pure", "pinv"]

    print("=" * 70)
    print("EXP 9: barrido fino alrededor de rho=1.00 (HRR real)")
    print(f"  BLKs: {BLKs}")
    print(f"  {n_seeds} seeds x {n_facts} facts x {len(modes)} modos x {len(BLKs)} BLKs")
    print("=" * 70)

    results = []
    for BLK in BLKs:
        N = 3 * BLK
        rho = 32 / BLK
        for mode in modes:
            accs_seed = []
            for s in range(n_seeds):
                rng = random.Random(42 + s)
                b = BundleV3(1000 + s, 3, N)
                accs = []
                for _ in range(n_facts):
                    f = make_fact(rng)
                    c = b.encode_fact(f)
                    dec = b.decode_fact(c, T=100, mode=mode, cleanup_method="argmax")
                    accs.append(accuracy(dec, f))
                accs_seed.append(float(np.mean(accs)))
            results.append({
                "BLK": BLK, "N": N, "rho": round(rho, 4), "mode": mode,
                "acc_mean": float(np.mean(accs_seed)),
                "acc_std": float(np.std(accs_seed)),
                "acc_sem": float(np.std(accs_seed) / np.sqrt(n_seeds)),
                "per_seed": [round(a, 3) for a in accs_seed],
            })
            print(f"  rho={rho:.3f} (BLK={BLK:>2}) {mode:>8}: "
                  f"{np.mean(accs_seed):.3f} ± {np.std(accs_seed)/np.sqrt(n_seeds):.3f} (sem)")

    out_path = Path(__file__).parent.parent / "data" / "out_V9_fine_rho.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {out_path}")

    # ---------- figura ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"gram": "#c0392b", "pure": "#27ae60", "pinv": "#2980b9"}
    for mode in modes:
        sub = sorted([r for r in results if r["mode"] == mode], key=lambda r: r["rho"])
        xs = [r["rho"] for r in sub]
        ys = [r["acc_mean"] for r in sub]
        es = [r["acc_sem"] for r in sub]
        ax.plot(xs, ys, marker="o", ms=5, lw=2, color=colors[mode], label=mode)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)], color=colors[mode], alpha=0.15)
    ax.axvline(1.0, ls="--", color="k", alpha=0.5)
    ax.annotate("ρ = 1", xy=(1.0, 1.02), fontsize=11, ha="center")
    ax.set_xlabel("ρ  (n_codevectors / BLK)")
    ax.set_ylabel("Accuracy (10 seeds × 20 facts)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Exp 9 — Fine sweep around ρ=1: ¿punto exacto o región de transición?")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_V9_fine_rho.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
