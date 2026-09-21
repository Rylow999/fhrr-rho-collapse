#!/usr/bin/env python3
"""
Exp 18b: replicacion de la Frame-Duality Law en BSC (algebra binaria {±1}).

Misma intervencion que Exp 18, pero sobre BSC (bind = multiplicacion
elemento a elemento, codevectors en {+-1}^BLK). Prueba si la relocacion
del resolvente rescata tambien en una algebra continua-distinta.

Condiciones: pure / gram / pinv / dual / dual_same, mismo C, mismo M,
mismo M^-1, mismos facts, mismos estados iniciales.

Salida: data/exp18_bsc_summary.json
"""
import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_V11_bsc_binary import BSCBundle, make_fact, accuracy, cleanup_bin

N_SEEDS = 100
N_FACTS = 10
T_LONG = 50
MODES = ["pure", "gram", "pinv", "dual", "dual_same"]


class BSCDual(BSCBundle):
    def __init__(self, seed, K, N):
        super().__init__(seed, K, N)
        self._amb = {}
        for b in range(self.K):
            roles = self.br[b]
            if len(roles) <= 1:
                continue
            ck = []
            for _, rn in roles:
                ck += list(self.codebooks[rn].keys())
            ck = sorted(set(ck))
            names = ck
            C = np.array([self.sym[s] for s in names])  # n_cv x BLK
            Minv = self.Minv.get(b)
            Mpinv = self.pinv.get(b)
            self._amb[b] = {
                "C": C,
                "dual_same": (C.T @ Minv @ C) if Minv is not None else None,
                "dual": C.T @ Mpinv @ C,
            }

    def decode_observer(self, c, T, mode, initial):
        out = {}
        for blk, roles in self.br.items():
            seg = c[blk * self.BLK:(blk + 1) * self.BLK]
            if len(roles) == 1:
                _, rn = roles[0]
                out[rn] = cleanup_bin(seg, self.cb_names[rn], self.cb_arr[rn])
                continue
            est = [v.copy() for v in initial[blk]]
            Minv = self.Minv.get(blk)
            Mpinv = self.pinv.get(blk)
            amb_d = self._amb[blk]["dual"]
            amb_ds = self._amb[blk]["dual_same"]
            R = self.roles[blk]
            r = len(roles)
            for _ in range(T):
                new = [None] * r
                for idx in range(r):
                    others = seg.copy()
                    for o in range(r):
                        if o != idx:
                            others = others - R[o] * est[o]   # BSC bind = *
                    fj = R[idx] * others                     # unbind (auto-inv)
                    if mode == "pure":
                        pass
                    elif mode == "gram" and Minv is not None and Minv.shape[0] == fj.shape[0]:
                        fj = Minv @ fj
                    elif mode == "pinv" and Mpinv.shape[0] == fj.shape[0]:
                        fj = Mpinv @ fj
                    elif mode == "dual":
                        fj = amb_d @ fj
                    elif mode == "dual_same" and amb_ds is not None:
                        fj = amb_ds @ fj
                    rn = roles[idx][1]
                    cb = self.cb_arr[rn]
                    sims = cb @ fj / (np.linalg.norm(cb, axis=1) * max(np.linalg.norm(fj), 1e-12))
                    new[idx] = self.sym[self.cb_names[rn][int(np.argmax(sims))]]
                est = new
            for idx, (_, rn) in enumerate(roles):
                cb = self.cb_arr[rn]
                sims = cb @ est[idx] / (np.linalg.norm(cb, axis=1) * max(np.linalg.norm(est[idx]), 1e-12))
                out[rn] = self.cb_names[rn][int(np.argmax(sims))]
        return out


def initial_estimates(b, seed):
    rng = np.random.RandomState(seed)
    init = {}
    for blk, roles in b.br.items():
        init[blk] = [rng.choice([-1.0, 1.0], size=b.BLK) for _ in roles]
    return init


def main():
    rng = random.Random(42)
    acc = {m: [] for m in MODES}
    print("=" * 70)
    print("EXP 18b: replicacion Frame-Duality en BSC (rho=1, BLK=32)")
    print("=" * 70)
    for s in range(N_SEEDS):
        b = BSCDual(1000 + s, 3, 96)
        init = initial_estimates(b, 777)
        for _ in range(N_FACTS):
            f = make_fact(rng)
            c = b.encode_fact(f)
            for m in MODES:
                acc[m].append(accuracy(b.decode_observer(c, T_LONG, m, init), f))
        if (s + 1) % 50 == 0:
            print(f"  ... {s+1}/{N_SEEDS}: " + " ".join(f"{m}={np.mean(acc[m]):.3f}" for m in MODES))

    out = {}
    for m in MODES:
        a = np.array(acc[m])
        out[m] = {"mean": float(a.mean()), "median": float(np.median(a)),
                  "std": float(a.std()), "ic95": float(1.96*a.std()/np.sqrt(len(a)))}
    dg = np.array(acc["dual_same"]) - np.array(acc["gram"])
    out["paired_dual_same_minus_gram"] = {"mean": float(dg.mean()),
                                          "median": float(np.median(dg)),
                                          "frac_pos": float((dg > 0.01).mean())}
    print("\nRESUMEN BSC rho=1:")
    for m in MODES:
        print(f"  {m:9s}: {out[m]['mean']:.3f} ± {out[m]['ic95']:.3f}")
    print(f"\npareado: {out['paired_dual_same_minus_gram']}")

    Path("../data/exp18_bsc_summary.json").write_text(json.dumps(out, indent=1))
    print("Guardado: data/exp18_bsc_summary.json")


if __name__ == "__main__":
    main()
