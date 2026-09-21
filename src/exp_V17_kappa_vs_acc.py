#!/usr/bin/env python3
"""
Exp 17: test per-seed kappa vs accuracy en el punto cuadrado (rho=1).

Verifica la objecion del revisor: la banda kappa [1e3, 1e4] predice que
colapsan solo los bloques con kappa en banda. Correr gram en rho=1 con
muchos seeds, medir kappa del bloque y accuracy por seed. Mirar:
  - rango de kappa (hay seeds con kappa << 1e4?)
  - correlacion kappa vs accuracy
  - si hay seeds con kappa BAJO que aun asi colapsan (refuta banda)
ADEMAS: corrector en espacio de coeficientes (gram_fix) implementado BIEN:
    coef_hat = M^{-1} (C c)  -> decodificar directamente por coeficientes
que en el punto cuadrado es (con M = C C^T): C^T M^{-1} C = I sobre la
imagen -> debe recuperar ~1.0 si el mecanismo es el amplificado 1/lambda_min
de la aplicacion ASIMETRICA fj <- M^{-1} fj (loop cerrado del resonator).

Salida: data/exp17_kappa_vs_acc.json + .txt
"""
import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy

N_SEEDS = 40
N_FACTS = 10


def block_kappa(b, blk):
    M = b.np_M[blk]
    w = np.linalg.eigvalsh(M)
    lmin, lmax = abs(w.min()), abs(w.max())
    return float(lmax / lmin) if lmin > 0 else 1e16


def coef_decode(b, blk, seg, roles):
    """Decoder en espacio de coeficientes (dimension n_cv), valido para
    cualquier BLK vs n_cv. Minv es n_cv x n_cv; C_blk es n_cv x BLK."""
    names = b.block_cb_names[blk]
    C = np.array([b.sym[s] for s in names])           # n_cv x BLK
    Minv = b.np_Minv[blk]
    if Minv is None:
        return None
    coef = Minv @ (C @ seg)
    # map coeficientes por rol
    out = {}
    offset = {rn: [] for _, rn in roles}
    for i, nm in enumerate(names):
        for _, rn in roles:
            if nm in b.codebooks[rn]:
                offset[rn].append(i)
    for _, rn in roles:
        idxs = offset[rn]
        cand = coef[idxs]
        # top-1 sobre coeficientes de ESTE rol
        cbn = [nm for nm in names if nm in b.codebooks[rn]]
        best = cbn[int(np.argmax(cand))]
        out[rn] = best
    return out


def main():
    rng = random.Random(42)
    rows = []
    print("=" * 70)
    print("EXP 17: kappa vs accuracy per-seed en rho=1 (n=32/cv, BLK=32)")
    print("=" * 70)
    for s in range(N_SEEDS):
        b = BundleV3(1000 + s, 3, 96)  # BLK=32, square
        kappas = {blk: block_kappa(b, blk) for blk in b.np_M}
        acc_gram, acc_pure, acc_coef = [], [], []
        for _ in range(N_FACTS):
            f = make_fact(rng)
            c = b.encode_fact(f)
            dg = b.decode_fact(c, T=100, mode="gram")
            dp = b.decode_fact(c, T=100, mode="pure")
            acc_gram.append(accuracy(dg, f))
            acc_pure.append(accuracy(dp, f))
            # coef decode por bloque (sin iterar — single shot)
            fc = {}
            for blk, roles in b.br.items():
                seg = c[blk * b.BLK:(blk + 1) * b.BLK]
                if len(roles) == 1:
                    _, rn = roles[0]
                    fc[rn] = b.cleanup(seg, rn)
                else:
                    dc = coef_decode(b, blk, seg, roles)
                    if dc:
                        fc.update(dc)
            acc_coef.append(accuracy(fc, f))
        rows.append({
            "seed": s,
            "kappa_blocks": kappas,
            "kappa_max": max(kappas.values()),
            "kappa_min": min(kappas.values()),
            "acc_gram": float(np.mean(acc_gram)),
            "acc_pure": float(np.mean(acc_pure)),
            "acc_coef": float(np.mean(acc_coef)),
        })
        if s % 8 == 0 or s == N_SEEDS - 1:
            print(f"  seed {s:3d}: kappa_max={rows[-1]['kappa_max']:.2e} "
                  f"gram={rows[-1]['acc_gram']:.3f} pure={rows[-1]['acc_pure']:.3f} "
                  f"coef={rows[-1]['acc_coef']:.3f}")

    ks = np.array([r["kappa_max"] for r in rows])
    ag = np.array([r["acc_gram"] for r in rows])
    # correlacion log-kappa vs acc
    corr = float(np.corrcoef(np.log10(ks), ag)[0, 1])
    # seeds con kappa bajo la banda que colapsan
    under = [(r["seed"], r["kappa_max"], r["acc_gram"]) for r in rows if r["kappa_max"] < 1e3]
    inband = [(r["seed"], r["kappa_max"], r["acc_gram"]) for r in rows if 1e3 <= r["kappa_max"] < 1e4]
    above = [(r["seed"], r["kappa_max"], r["acc_gram"]) for r in rows if r["kappa_max"] >= 1e4]

    print(f"\ncorr(log10 kappa_max, acc_gram) = {corr:.3f}")
    print(f"seeds kappa<1e3 : {len(under):2d}  acc_gram media = "
          f"{np.mean([a for _,_,a in under]) if under else float('nan'):.3f}")
    print(f"seeds 1e3<=k<1e4: {len(inband):2d}  acc_gram media = "
          f"{np.mean([a for _,_,a in inband]) if inband else float('nan'):.3f}")
    print(f"seeds kappa>=1e4: {len(above):2d}  acc_gram media = "
          f"{np.mean([a for _,_,a in above]) if above else float('nan'):.3f}")
    print("\nseeds con kappa<1e3 que colapsan (acc_gram<0.5):")
    for s_, k_, a_ in under:
        if a_ < 0.5:
            print(f"  seed {s_}: kappa={k_:.2e} acc_gram={a_:.3f}")
    print(f"\nacc_coef (decoder en coeficientes, single-shot): media "
          f"{np.mean([r['acc_coef'] for r in rows]):.4f}")

    out = {"n_seeds": N_SEEDS, "n_facts": N_FACTS,
           "corr_logk_acc": corr,
           "under_band": under, "in_band": inband, "above_band": above,
           "acc_coef_mean": float(np.mean([r["acc_coef"] for r in rows])),
           "acc_gram_mean": float(ag.mean()), "rows": rows}
    jp = Path(__file__).parent.parent / "data" / "exp17_kappa_vs_acc.json"
    jp.write_text(json.dumps(out, indent=1))
    print(f"\nGuardado: {jp}")


if __name__ == "__main__":
    main()
