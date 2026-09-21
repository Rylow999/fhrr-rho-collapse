#!/usr/bin/env python3
"""
Exp 18 (v2, HRR correcto): relocacion del resolvente — prueba causal.

El bug de v1: implemente el unbind como multiplicacion elemento a elemento,
cuando BundleV3 es HRR real (convolucion circular via FFT). Todos los
observadores daban chance por eso. Esta version subclasea BundleV3 y
monkey-patchea _decode_multi con la correccion inyectada.

Cinco observadores, mismo C, mismo M, mismo M^{-1}, mismos estados
iniciales, mismos facts:
  O0 pure      : f' = f
  O1 gram      : f' = M^{-1} f              (asimetrico, ambient)
  O2 pinv      : f' = M^{+} f               (ambient)
  O3 dual      : f' = C^T M^{+} C f         (dual, pseudo)
  O4 dual_same : f' = C^T M^{-1} C f        (dual, MISMO M^{-1}) <- causal
"""
import json
import random
import numpy as np
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import (
    BundleV3, make_fact, accuracy, hrr_bind, hrr_unbind, CFG6)

N_SEEDS_SQ = 200      # rho=1 (BLK=32, n_cv=32)
N_SEEDS_GRID = 30
N_FACTS = 10
T_LONG, T_SHORT = 50, 1
Q_VEC = 200


class DualBundle(BundleV3):
    """BundleV3 con correcciones ambientales dual re-derivadas por bloque."""

    def __init__(self, seed, K, N, cleanup_method="argmax"):
        super().__init__(seed, K, N, role_config=CFG6)
        self._amb = {}
        for b in range(self.K):
            roles = self.br[b]
            if len(roles) <= 1:
                continue
            names = self.block_cb_names[b]
            C = np.array([self.sym[s] for s in names])   # n_cv x BLK
            Minv = self.np_Minv[b]
            Mpinv = self.np_pinv[b]
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
                out[rn] = self.cleanup(seg, rn)
                continue
            est = [v.copy() for v in initial[blk]]
            Minv = self.np_Minv.get(blk)
            Mpinv = self.np_pinv.get(blk)
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
                            others = others - hrr_bind(R[o], est[o])
                    fj = hrr_unbind(others, R[idx])
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
                    sims = self.cb_vec[rn] @ fj
                    norms = np.linalg.norm(self.cb_vec[rn], axis=1) * max(np.linalg.norm(fj), 1e-12)
                    name = self.cb_names[rn][int(np.argmax(sims / norms))]
                    new[idx] = self.sym[name]
                est = new
            for idx, (_, rn) in enumerate(roles):
                out[rn] = self.cleanup(est[idx], rn)
        return out


def initial_estimates(b, seed):
    rng = np.random.RandomState(seed)
    init = {}
    for blk, roles in b.br.items():
        vecs = []
        for _ in roles:
            v = rng.randn(b.BLK)
            vecs.append(v / np.linalg.norm(v))
        init[blk] = vecs
    return init


def main():
    t0 = time.time()
    rng = random.Random(42)
    MODES = ["pure", "gram", "pinv", "dual", "dual_same"]
    out = {"square": [], "grid": [], "opnorms": [], "softedge": [],
           "identity": [], "corr": {}}

    print("=" * 74)
    print("EXP 18 v2 (HRR correcto): relocacion del resolvente")
    print("=" * 74)

    # ---- [1a] algebraica + ganancia
    print(f"\n[1a] Identidad + normas de operador (N={N_SEEDS_SQ} seeds, BLK=32)")
    for s in range(N_SEEDS_SQ):
        b = DualBundle(1000 + s, 3, 96)
        for blk, amb in b._amb.items():
            C = amb["C"]
            Minv = b.np_Minv[blk]
            if Minv is None:
                continue
            P = amb["dual_same"]
            errF = float(np.linalg.norm(P - np.eye(b.BLK), "fro"))
            out["identity"].append(errF)
            # operadores
            rngq = np.random.RandomState(7)
            Q = rngq.randn(Q_VEC, b.BLK)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            def g(A):
                if A is None or A.shape[0] != b.BLK:
                    return None
                v = np.linalg.norm(Q @ A.T, axis=1)
                return float(np.median(v)), float(np.max(v))
            w = np.linalg.eigvalsh(b.np_M[blk])
            out["opnorms"].append({
                "seed": s, "blk": blk,
                "kappa": float(w[-1] / w[0]),
                "gram_med": g(Minv)[0], "gram_max": g(Minv)[1],
                "dual_same_med": g(amb["dual_same"])[0], "dual_same_max": g(amb["dual_same"])[1],
            })
            _, U = np.linalg.eigh(b.np_M[blk])
            u_min, u_max = U[:, 0], U[:, -1]
            out["softedge"].append({
                "seed": s, "blk": blk,
                "gram_umin": float(np.linalg.norm(Minv @ u_min)),
                "same_umin": float(np.linalg.norm(amb["dual_same"] @ u_min)),
            })
        if (s + 1) % 100 == 0:
            print(f"  ... {s+1}/{N_SEEDS_SQ} ({time.time()-t0:.0f}s)")

    ident = np.array(out["identity"])
    print(f"  ||C^T M^-1 C - I||_F: mediana {np.median(ident):.2e}, max {ident.max():.2e}")
    gm = np.array([r["gram_max"] for r in out["opnorms"]])
    dm = np.array([r["dual_same_max"] for r in out["opnorms"]])
    print(f"  ||M^-1||_2 (max gain): mediana {np.median(gm):.3e}")
    print(f"  ||C^T M^-1 C||_2 (max gain): mediana {np.median(dm):.3f}")
    su = np.array([r["gram_umin"] for r in out["softedge"]])
    sd = np.array([r["same_umin"] for r in out["softedge"]])
    print(f"  ||M^-1 u_min||: mediana {np.median(su):.3e} (1/lambda_min)")
    print(f"  ||C^T M^-1 C u_min||: mediana {np.median(sd):.3f}")

    # ---- [1b] decoding rho=1
    print(f"\n[1b] Decoding rho=1 (N={N_SEEDS_SQ} seeds x {N_FACTS} facts)")
    acc = {m: [] for m in MODES}
    accT1 = {m: [] for m in MODES}
    for s in range(N_SEEDS_SQ):
        b = DualBundle(1000 + s, 3, 96)
        init = initial_estimates(b, 777)
        for _ in range(N_FACTS):
            f = make_fact(rng)
            c = b.encode_fact(f)
            for m in MODES:
                acc[m].append(accuracy(b.decode_observer(c, T_LONG, m, init), f))
                if s < 50:
                    accT1[m].append(accuracy(b.decode_observer(c, T_SHORT, m, init), f))
        if (s + 1) % 100 == 0:
            print(f"  ... {s+1}/{N_SEEDS_SQ}: " +
                  " ".join(f"{m}={np.mean(acc[m]):.3f}" for m in MODES))

    print("\nRESUMEN rho=1 (media ± IC95 [T1]):")
    for m in MODES:
        a = np.array(acc[m])
        ic = 1.96 * a.std() / np.sqrt(len(a))
        t1m = float(np.mean(accT1[m])) if accT1[m] else None
        out["square"].append({"mode": m, "mean": float(a.mean()),
                              "median": float(np.median(a)), "std": float(a.std()),
                              "ic95": float(ic), "T1_mean": t1m})
        print(f"  {m:9s}: {a.mean():.3f} ± {ic:.3f}   [T=1: {t1m:.3f}]")

    dg = np.array(acc["dual_same"]) - np.array(acc["gram"])
    dp = np.array(acc["dual"]) - np.array(acc["pinv"])
    out["paired"] = {
        "dual_same_minus_gram": {"mean": float(dg.mean()),
                                 "median": float(np.median(dg)),
                                 "frac_pos": float((dg > 0.01).mean())},
        "dual_minus_pinv": {"mean": float(dp.mean()),
                            "median": float(np.median(dp))},
    }
    print(f"\nPareado dual_same - gram: {dg.mean():+.3f} (mediana "
          f"{np.median(dg):+.3f}, {100*(dg>0.01).mean():.0f}% positivo)")

    # ---- [2] grid rho
    print("\n[2] Grid rho")
    for K, N, rho in [(3, 120, 0.800), (3, 96, 1.000), (3, 72, 1.333), (1, 128, 0.750)]:
        accm = {m: [] for m in MODES}
        for s in range(N_SEEDS_GRID):
            b = DualBundle(1000 + s, K, N)
            init = initial_estimates(b, 555)
            for _ in range(N_FACTS):
                f = make_fact(rng); c = b.encode_fact(f)
                for m in MODES:
                    accm[m].append(accuracy(b.decode_observer(c, T_LONG, m, init), f))
        row = {"K": K, "N": N, "rho": rho}
        row.update({m: round(float(np.mean(accm[m])), 4) for m in MODES})
        out["grid"].append(row)
        print(f"  rho={rho:.3f}: " + " ".join(f"{m}={row[m]:.3f}" for m in MODES))

    # ---- guardar
    base = Path(__file__).parent.parent / "data"
    (base / "exp18_summary.json").write_text(json.dumps({
        "square": out["square"], "grid": out["grid"], "paired": out["paired"],
        "identity_median_errF": float(np.median(ident)),
        "identity_max_errF": float(ident.max()),
        "norm_Minv_med": float(np.median(gm)),
        "norm_dual_same_med": float(np.median(dm)),
        "gram_umin_med": float(np.median(su)),
        "dual_same_umin_med": float(np.median(sd)),
    }, indent=1))
    (base / "exp18_operator_norms.json").write_text(json.dumps(out["opnorms"], indent=1))
    (base / "exp18_spectral_alignment.json").write_text(json.dumps(out["softedge"], indent=1))
    print(f"\nGuardado. ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
