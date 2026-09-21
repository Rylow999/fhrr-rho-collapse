#!/usr/bin/env python3
"""
EXP 20 — CAPA 1: generalizacion del ensemble + barrido continuo de rho.

Usa la implementacion independiente de Exp 19 (no BundleV3).

Parte A (ensembles): para cada familia F de C, correr el punto cuadrado
ρ=1 y medir pure / gram / dual_same. Familias:
  gauss : N(0,1/d) iid filas (baseline)
  rademacher : +-1/sqrt(d) iid
  sphere : uniforme en la esfera
  toeplitz : vecinos correlacionados (rho_toep = 0.7)
  tight  : frame tight (DFT rows)
  near_dup : 16 vectores + pequenas perturbaciones

Parte B (barrido rho): gaussiano, mismo protocolo, rho en
{0.5,0.75,0.9,0.95,0.99,1.0,1.05,1.25,1.5,2.0}. En cada uno medir:
acc gram ambient vs gram dual, lambda_min, kappa, ||C^T M+ C - I||_F.
Identifica cual variable predice el colapso.

Salida: data/exp20_ensembles_rho.json/.txt
"""
import json
import numpy as np
import random
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).parent))
from exp_V19_independent_replica import circ_bind, circ_unbind, encode, decode

def make_C(family, n, d, seed):
    rng = np.random.RandomState(seed)
    if family == "gauss":
        C = rng.randn(n, d) / np.sqrt(d)
    elif family == "rademacher":
        C = rng.choice([-1.0, 1.0], size=(n, d)) / np.sqrt(d)
    elif family == "sphere":
        C = rng.randn(n, d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    elif family == "toeplitz":
        base = rng.randn(d) / np.sqrt(d)
        rows = []
        for _ in range(n):
            shift = rng.randint(d)
            noise = 0.3 * rng.randn(d) / np.sqrt(d)
            rows.append(np.roll(base, shift) + noise)
        C = np.array(rows)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    elif family == "tight":
        k = np.arange(n)[:, None]; j = np.arange(d)[None, :]
        C = np.exp(-2j * np.pi * k * j / d).real
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    elif family == "near_dup":
        Cbase = rng.randn(1, d) / np.sqrt(d)
        C = np.tile(Cbase, (n, 1)) + 0.02 * rng.randn(n, d) / np.sqrt(d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    else:
        raise ValueError(family)
    return C


def build_block_C(C, n_roles, seed):
    """C es (n_cv, d) ya construida; los roles son isotropicos."""
    rng = np.random.RandomState(seed)
    n, d = C.shape
    roles = []
    for _ in range(n_roles):
        v = rng.randn(d)
        roles.append(v / np.linalg.norm(v))
    M = C @ C.T
    ev = np.linalg.eigvalsh(M)
    Minv = np.linalg.inv(M) if abs(ev[0]) > 1e-12 else None
    Mpinv = np.linalg.pinv(M, rcond=1e-10)
    return {"C": C, "roles": roles, "M": M, "Minv": Minv, "Mpinv": Mpinv,
            "P_dual_same": (C.T @ Minv @ C) if Minv is not None else None,
            "P_dual": C.T @ Mpinv @ C,
            "kappa": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
            "lmin": float(ev[0]), "d": d}


def decode_C(block, syms_per_role, s, mode, T=50, init_seed=0):
    rng = np.random.RandomState(init_seed)
    est = []
    d = block["d"]
    for _ in block["roles"]:
        v = rng.randn(d); est.append(v / np.linalg.norm(v))
    for _ in range(T):
        new = []
        for r in range(len(block["roles"])):
            others = s.copy()
            for o in range(len(block["roles"])):
                if o != r:
                    others = others - circ_bind(block["roles"][o], est[o])
            fj = circ_unbind(others, block["roles"][r])
            if mode == "gram":
                A = block["Minv"]
                if A is not None and A.shape[0] == fj.shape[0]:
                    fj = A @ fj
            elif mode == "dual_same":
                A = block["P_dual_same"]
                if A is not None: fj = A @ fj
            sims = syms_per_role[r] @ fj
            new.append(syms_per_role[r][int(np.argmax(sims))])
        est = new
    return [int(np.argmax(syms_per_role[r] @ est[r])) for r in range(len(block["roles"]))]


def run_config(C, n_sym=16, n_seeds=20, n_facts=6, n_init=3):
    n, d = C.shape
    n_roles = n // n_sym
    rng = random.Random(11)
    accs = {"pure": [], "gram": [], "dual_same": []}
    for s in range(n_seeds):
        Cs = C.copy()
        block = build_block_C(Cs, n_roles, seed=1000 + s)
        syms_role = [Cs[r * n_sym:(r + 1) * n_sym] for r in range(n_roles)]
        per = {m: [] for m in accs}
        for _ in range(n_facts):
            fact = tuple(rng.randrange(n_sym) for _ in range(n_roles))
            sig = np.zeros(d)
            for r, i in enumerate(fact):
                sig += circ_bind(block["roles"][r], syms_role[r][i])
            for init in range(n_init):
                for m in accs:
                    out = decode_C(block, syms_role, sig, m, T=50, init_seed=77 + init)
                    per[m].append(sum(int(a == f) for a, f in zip(out, fact)) / n_roles)
        for m in accs:
            accs[m].append(float(np.mean(per[m])))
    return {m: float(np.mean(a)) for m, a in accs.items()}, \
           {"kappa": float(np.median([build_block_C(C, n_roles, 1000 + s)["kappa"] for s in range(min(5, n_seeds))])),
            "lmin": float(np.median([build_block_C(C, n_roles, 1000 + s)["lmin"] for s in range(min(5, n_seeds))]))}


def main():
    t0 = time.time()
    out = {"ensembles": {}, "rho_sweep": []}

    print("=" * 72)
    print("EXP 20A: ensembles de C en rho=1 (n=32, d=32)")
    print("=" * 72)
    FAMS = ["gauss", "rademacher", "sphere", "toeplitz", "tight", "near_dup"]
    for fam in FAMS:
        try:
            C = make_C(fam, 32, 32, seed=777)
            accs, meta = run_config(C, n_seeds=20, n_facts=6, n_init=3)
            out["ensembles"][fam] = {**accs, **meta}
            print(f"  {fam:10s} kappa={meta['kappa']:.2e} lmin={meta['lmin']:.2e} | "
                  f"pure={accs['pure']:.3f} gram={accs['gram']:.3f} dual_same={accs['dual_same']:.3f}")
        except Exception as e:
            print(f"  {fam}: FALLO {e}")
            out["ensembles"][fam] = {"error": str(e)}

    print("\n" + "=" * 72)
    print("EXP 20B: barrido rho (gaussiano, n_cv=32, d variable)")
    print("=" * 72)
    for rho in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0, 1.05, 1.25, 1.5, 2.0]:
        d = int(round(32 / rho))
        C = make_C("gauss", 32, d, seed=99)
        accs, meta = run_config(C, n_seeds=15, n_facts=5, n_init=2)
        row = {"rho": rho, "d": d, **accs, **meta}
        out["rho_sweep"].append(row)
        print(f"  rho={rho:5.2f} d={d:3d} kappa={meta['kappa']:.2e} lmin={meta['lmin']:.2e} | "
              f"pure={accs['pure']:.3f} gram={accs['gram']:.3f} dual_same={accs['dual_same']:.3f}")

    jp = Path(__file__).parent.parent / "data" / "exp20_ensembles_rho.json"
    jp.write_text(json.dumps(out, indent=1))
    print(f"\nGuardado: {jp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
