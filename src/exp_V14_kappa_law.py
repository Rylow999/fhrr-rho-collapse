#!/usr/bin/env python3
"""
Exp 14: ley fundamentada — el parametro fisico que gobierna la transicion.

Hipotesis: el colapso del decoder Gram no depende de rho directamente,
sino de la condicion de la matriz de Gram:
    kappa(M) = lambda_max / lambda_min   (condition number)

El colapso ocurre cuando 1/kappa cruza la precision de la pseudoinversa
(o el threshold de truncamiento de pinv, ~rcond).

Protocolo:
  - Mismo grid que Exp 9 (barrido fino alrededor de rho=1)
  - Por cada punto: medir kappa(M) del bloque, accuracy de gram de verdad
  - Correlacionar: log(kappa) vs accuracy
  - MOSTRAR que existe un umbral critico kappa* tal que:
      kappa < kappa* -> accuracy ~1
      kappa > kappa* -> accuracy colapsa
  - Esto eleva la ley rho de "fenomeno empirico" a "ley con parametro
    fisico mesurable" — la condicion de Gram

Bonus: verificar que rho=1 es donde kappa explota (singularidad).
"""
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy

def main():
    print("=" * 72)
    print("EXP 14: ley fundamentada — kappa(M) como parametro fisico del colapso")
    print("=" * 72)

    # Grid fino como Exp 9 (BLK varía, n_cv fijo en 32)
    results = []
    BLKs = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40]
    n_seeds = 6
    n_facts = 12

    for BLK in BLKs:
        N = 3 * BLK
        kappas, accs_gram, accs_pure = [], [], []
        for s in range(n_seeds):
            b = BundleV3(1000 + s, 3, N)
            # kappa de cada bloque multi-rol
            for bidx in range(b.K):
                if bidx in b.np_M:
                    M = b.np_M[bidx]
                    w = np.linalg.eigvalsh(M)
                    lam_max = abs(w).max()
                    lam_min = abs(w).min()
                    if lam_min > 1e-14:
                        kappas.append(lam_max / lam_min)
                    else:
                        kappas.append(1e16)  # singular
            # Accuracy gram vs pure sobre el bundle completo
            rng_seeded = __import__('random').Random(42 + s)
            for _ in range(n_facts):
                f = make_fact(rng_seeded)
                c = b.encode_fact(f)
                dg = b.decode_fact(c, T=50, mode="gram", cleanup_method="argmax")
                dp = b.decode_fact(c, T=50, mode="pure", cleanup_method="argmax")
                accs_gram.append(accuracy(dg, f))
                accs_pure.append(accuracy(dp, f))

        kappa_med = float(np.median(kappas))
        acc_g = float(np.mean(accs_gram))
        acc_p = float(np.mean(accs_pure))
        rho = 32 / BLK
        results.append({
            "BLK": BLK, "rho": rho, "kappa_med": kappa_med,
            "log_kappa": float(np.log10(kappa_med)),
            "acc_gram": acc_g, "acc_pure": acc_p,
        })
        print(f"  BLK={BLK:2d} rho={rho:.3f} kappa=1e{np.log10(kappa_med):5.1f} "
              f"gram={acc_g:.3f} pure={acc_p:.3f}")

    # Guardar
    out = Path(__file__).parent.parent / "data" / "exp14_kappa_law.json"
    out.write_text(json.dumps(results, indent=1))

    # Figura: accuracy vs log10(kappa) — la ley real
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [r["log_kappa"] for r in results]
    ax.scatter(xs, [r["acc_gram"] for r in results], c=["r"]*len(xs), s=70, zorder=3, label="gram")
    ax.plot(xs, [r["acc_pure"] for r in results], "g--", lw=2, label="pure", zorder=2)
    ax.axvline(16, ls=":", color="gray", alpha=0.5)
    ax.axvline(3, ls=":", color="k", alpha=0.3)
    ax.annotate("singular (rho=1.00)", xy=(16, 0.15), fontsize=9, color="gray")
    ax.set_xlabel("log10(kappa) — condition number de la Gram por bloque")
    ax.set_ylabel("accuracy")
    ax.set_title("Ley fundamentada: accuracy del decoder Gram vs kappa(M)\n"
                 "El colapso ocurre cuando kappa cruza el umbral de precision")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig_path = Path(__file__).parent.parent / "figures" / "fig_exp14_kappa_law.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigura: {fig_path}")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
