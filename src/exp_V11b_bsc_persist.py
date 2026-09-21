#!/usr/bin/env python3
"""
Exp 11b (persistido): ley de rho en BSC — data dura para el paper.

Corre la matriz completa: 5 casos de rho x 4 decoders x 8 seeds x 25 facts,
midiendo ademas el kappa de la Gram de cada bloque (mediana por caso).
Salida: data/exp11_bsc_rho.json  +  data/exp11_bsc_rho.txt
"""
import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_V11_bsc_binary import BSCBundle, make_fact, accuracy

CASES = [
    ("rho=0.75 K=1 N=128", 6, 1, 128),
    ("rho=0.80 K=3 N=120", 6, 3, 120),
    ("rho=1.00 K=3 N=96 CUADRADO", 6, 3, 96),
    ("rho=1.33 K=3 N=72", 6, 3, 72),
    ("rho=1.50 K=2 N=64", 6, 2, 64),
]
MODES = ["gram", "pure", "pinv", "gradient"]
N_SEEDS = 8
N_FACTS = 25
# n_cv por bloque: K bloques reparten 6 roles; cada rol aporta 16 simbolos
# compartidos (en CFG6 los codebooks comparten simbolos entre roles).
# Medido en la corrida: K=3 -> 32 codevectors/bloque -> rho = 32/BLK.
# K=2 -> 48/bloque (3 roles) -> rho = 48/BLK. K=1 -> 96 = 96/128 = 0.75.
N_CV = {1: 96, 2: 48, 3: 32}

def kappa_blocks(b):
    ks = []
    for blk in b.br:
        if len(b.br[blk]) <= 1:
            continue
        ck = []
        for _, rn in b.br[blk]:
            ck += list(b.codebooks[rn].keys())
        ck = sorted(set(ck))
        C = np.array([b.sym[s] for s in ck])
        M = C @ C.T
        w = np.linalg.eigvalsh(M)
        lmin, lmax = abs(w.min()), abs(w.max())
        if lmin > 1e-14:
            ks.append(lmax / lmin)
        else:
            ks.append(1e16)
    return ks

def main():
    rng = random.Random(42)
    out = []
    print("=" * 74)
    print("EXP 11b: BSC binaria — grid rho x decoder (PERSISTIDO)")
    print("=" * 74)
    for label, n, K, N in CASES:
        rho = N_CV[K] / (N // K)
        row = {"label": label, "K": K, "N": N, "BLK": N // K,
               "n_cv_per_block": N_CV[K], "rho": round(rho, 4),
               "n_seeds": N_SEEDS, "n_facts": N_FACTS}
        kappas = []
        for mode in MODES:
            accs = []
            for s in range(N_SEEDS):
                b = BSCBundle(1000 + s, K, N)
                if mode == "gram" and s == 0:
                    kappas = kappa_blocks(b)
                for _ in range(N_FACTS):
                    f = make_fact(rng)
                    c = b.encode_fact(f)
                    dec = b.decode_fact(c, T=100, mode=mode)
                    accs.append(accuracy(dec, f))
            row[f"acc_{mode}"] = round(float(np.mean(accs)), 4)
            row[f"acc_{mode}_std"] = round(float(np.std(accs)), 4)
        if not kappas:  # medir kappa una vez aunque gram no corra primero
            b = BSCBundle(1000, K, N)
            kappas = kappa_blocks(b)
        row["kappa_med"] = float(np.median(kappas)) if kappas else None
        row["kappa_all"] = [round(k, 1) for k in kappas]
        out.append(row)
        print(f"{label:32s} rho={rho:5.3f} kappa_med={row['kappa_med']!s:>12} | "
              + " ".join(f"{m}={row[f'acc_{m}']:.3f}" for m in MODES))

    jpath = Path(__file__).parent.parent / "data" / "exp11_bsc_rho.json"
    jpath.write_text(json.dumps(out, indent=1))
    tpath = jpath.with_suffix(".txt")
    with open(tpath, "w") as fh:
        fh.write("EXP 11b: BSC binaria — grid rho x decoder\n")
        fh.write(f"n_seeds={N_SEEDS}, n_facts={N_FACTS}, T=100, seed base 1000+s, rng facts seed 42\n\n")
        for r in out:
            fh.write(f"--- {r['label']} (BLK={r['BLK']}) ---\n")
            fh.write(f"  kappa mediana por bloque: {r['kappa_med']}\n")
            for m in MODES:
                fh.write(f"  {m:>9}: {r[f'acc_{m}']:.4f} ± {r[f'acc_{m}_std']:.4f}\n")
            fh.write("\n")
    print(f"\nGuardado: {jpath} y {tpath}")

if __name__ == "__main__":
    main()
