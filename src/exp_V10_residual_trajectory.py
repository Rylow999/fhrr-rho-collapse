#!/usr/bin/env python3
"""
Exp 10: Trayectoria del residuo — cómo converge (o no) cada decoder.

En rho=1.00 (caso cuadrado), registramos la norma L2 del residuo
    r(t) = || seg - sum_i bind(role_i, est_i(t)) ||
en cada iteración del resonator, para gram / pure / pinv.

Hipótesis:
  - pure:   decaimiento ~1/sqrt(t), residuo -> ~0
  - gram:   residuo se mantiene alto o diverge (anti-resonancia)
  - pinv:   decae hasta un piso > 0 (se estanca)

Esto es la pista numérica para la Linea B (teoría analítica): el operador de
iteración tiene (o no) autovalores con |lambda|>1 cerca de rho=1.

Salidas:
  data/out_V10_residuals.json
  figures/fig_V10_residuals.png
"""
import numpy as np
import random
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy, hrr_bind, hrr_unbind


def resonator_traj(bundle, c, mode, T=200):
    """Resonator paso a paso, registrando residuo. Devuelve (accuracy, res_list)."""
    BLK = bundle.BLK
    # replicar decode_fact pero a mano para registrar residuo
    out = {}
    res_traj = []
    for b in range(bundle.K):
        seg = c[b * BLK:(b + 1) * BLK]
        roles = bundle.br[b]
        if len(roles) == 1:
            _, rname = roles[0]
            out[rname] = bundle.cleanup(seg, rname, "argmax")
            continue

        R = np.array(bundle.roles[b])
        r_ = len(roles)
        if mode == "gram":
            Minv = bundle.np_Minv.get(b)
        elif mode == "pinv":
            Minv = bundle.np_pinv.get(b)
        else:
            Minv = None

        est = []
        for _ in range(r_):
            v = bundle.np_rng.randn(BLK); est.append(v / np.linalg.norm(v))

        blk_res = []
        for t in range(T):
            recon = sum(hrr_bind(R[o], est[o]) for o in range(r_))
            blk_res.append(float(np.linalg.norm(seg - recon)))

            new = [None] * r_
            for idx in range(r_):
                others = seg.copy()
                for o in range(r_):
                    if o != idx:
                        others = others - hrr_bind(R[o], est[o])
                fj = hrr_unbind(others, R[idx])
                if Minv is not None and Minv.shape[0] == BLK:
                    fj = Minv @ fj
                name = bundle.cleanup(fj, roles[idx][1], "argmax")
                new[idx] = bundle.sym[name]
            est = new

        out.update({rname: bundle.cleanup(est[i], rname, "argmax")
                    for i, (j, rname) in enumerate(roles)})
        res_traj.append(blk_res)

    return out, res_traj


def main():
    t0 = time.time()
    rng = random.Random(42)

    # Caso cuadrado (rho=1.00) + dos controles (rho=0.80, rho=1.33)
    cases = [
        ("rho=0.80", 3, 120),
        ("rho=1.00", 3, 96),
        ("rho=1.33", 3, 72),
    ]
    modes = ["gram", "pure", "pinv"]
    n_facts = 10
    T = 200

    print("=" * 70)
    print("EXP 10: trayectoria del residuo en el resonator")
    print("=" * 70)

    results = []
    for case_label, K, N in cases:
        for mode in modes:
            all_trajs = []
            accs = []
            for rep in range(n_facts):
                b = BundleV3(1000 + rep, K, N)
                f = make_fact(rng)
                c = b.encode_fact(f)
                dec, trajs = resonator_traj(b, c, mode, T=T)
                accs.append(accuracy(dec, f))
                # promediar residuo sobre bloques multi-rol
                if trajs:
                    m = np.mean(np.array(trajs), axis=0)
                    all_trajs.append(m)
            mean_traj = np.mean(all_trajs, axis=0).tolist()
            results.append({
                "case": case_label, "K": K, "N": N, "mode": mode,
                "acc_mean": float(np.mean(accs)),
                "residual_final": float(np.mean([tr[-1] for tr in all_trajs])),
                "residual_traj": [round(x, 4) for x in mean_traj],
            })
            print(f"  {case_label} {mode:>6}: acc={np.mean(accs):.3f}  "
                  f"res(t=0)={mean_traj[0]:.3f}  res(T)={mean_traj[-1]:.4f}")

    out_path = Path(__file__).parent.parent / "data" / "out_V10_residuals.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {out_path}")

    # ---------- figura: 3 paneles (uno por caso), escala log Y
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    colors = {"gram": "#c0392b", "pure": "#27ae60", "pinv": "#2980b9"}
    for ax, (case_label, K, N) in zip(axes, cases):
        for r in results:
            if r["case"] != case_label:
                continue
            tr = r["residual_traj"]
            ax.plot(range(len(tr)), tr, color=colors[r["mode"]], lw=2,
                    label=f"{r['mode']} (acc={r['acc_mean']:.2f})")
        ax.set_title(case_label)
        ax.set_xlabel("iteración t")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("||residuo||_2  (log)")
    fig.suptitle("Exp 10 — Trayectoria del residuo por decoder (promedio de 10 facts)")
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_V10_residuals.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
