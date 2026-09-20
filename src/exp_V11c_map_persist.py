#!/usr/bin/env python3
"""
Exp 11c (persistido): ley de rho en MAP con permutacion (Gayler).
Cierra la cobertura de algebras del paper (FHRR, HRR, BSC, MAP).

MAP: bind(a, rol) = Pi_rol(a) (permutacion publica por rol) seguido de
bundle aditivo. Decoding MIMO: para un bloque con roles {r1, r2}, el
segmento es  s = Pi_1(x1) + Pi_2(x2). El resonator estima x's y ressitra
en el espacio permutado.

Gram de decodificacion del bloque: NO sobre los codevectors sino sobre la
MATRIZ DE MEZCLA — para gram/pinv modelamos el canal como
    s = P X,   P = [P_1 C_1 | P_2 C_2]  (BLK x (n1+n2))
donde C_i es la matriz de codevectors del rol i y P_i = matriz de
permutacion. Decodificar por "Gram" = least squares sobre P:
    x_hat = (P^T P)^{-1} P^T s  -> cleanup por rol.
Esa M = P^T P es la Gram del bloque; kappa(M) es el numero de condicion
fisico relevante. Como las P_i son permutaciones (isometrías),
kappa(M) = kappa(Gram de los codevectors apilados con permutaciones de
columnas) — exactamente el analogo de la Gram de BSC pero con mezcla de
dos codebooks distintos en el mismo bloque (n_cv = 16+16 = 32).

Salida: data/exp11c_map_rho.json + .txt
"""
import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import CFG6


def cleanup(cand, names, cb_arr):
    sims = cb_arr @ cand / (np.linalg.norm(cb_arr, axis=1) * max(np.linalg.norm(cand), 1e-12))
    return names[int(np.argmax(sims))]


class MAPBlock:
    """Un bloque MAP con roles mezclados por permutaciones distintas."""

    def __init__(self, rng, roles, syms, cb_names, BLK):
        self.rng = rng
        self.roles = roles
        self.BLK = BLK
        self.perm = {}
        self.P = {}
        for rn in roles:
            p = np.arange(BLK)
            rng.shuffle(p)
            self.perm[rn] = p
            Pm = np.zeros((BLK, BLK))
            Pm[np.arange(BLK), p] = 1.0  # (Pm v)[i] = v[p[i]]
            self.P[rn] = Pm
        self.sym = syms
        self.cb_names = cb_names
        self.cb_arr = {rn: np.array([syms[s] for s in cb_names[rn]]) for rn in roles}
        # matriz de mezcla del bloque: [P_r C_r, ...]
        blocks = [self.P[rn] @ self.cb_arr[rn].T for rn in roles]
        self.A = np.concatenate(blocks, axis=1)  # BLK x n_cv
        self.n_cv = self.A.shape[1]
        # limites de cada rol dentro de A (para extraer el vector por rol)
        self.offsets = {}
        off = 0
        for rn in roles:
            self.offsets[rn] = (off, off + len(cb_names[rn]))
            off += len(cb_names[rn])
        self.M = self.A.T @ self.A
        try:
            self.Minv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            self.Minv = None
        self.Mpinv = np.linalg.pinv(self.M, rcond=1e-10)

    def bind_role(self, rn, s):
        return self.P[rn] @ self.sym[s]

    def encode(self, fact):
        return sum(self.bind_role(rn, fact[rn]) for rn in self.roles)

    def gram_decode(self, seg, pinv=False):
        W = self.Mpinv if pinv else self.Minv
        if W is None:
            return None
        coef = W @ self.A.T @ seg
        out = {}
        for rn in self.roles:
            a, b = self.offsets[rn]
            idx = int(np.argmax(coef[a:b]))
            out[rn] = self.cb_names[rn][idx]
        return out

    def resonator(self, seg, T=100, gram_mode=None):
        """Resonator: estima en el espacio permutado, cleanup por rol."""
        est = {rn: self.rng.choice([-1.0, 1.0], size=self.BLK) for rn in self.roles}
        for _ in range(T):
            new = {}
            for rn in self.roles:
                others = seg.copy()
                for rn2 in self.roles:
                    if rn2 != rn:
                        others -= est[rn2]
                # candidato: des-permutar al espacio del codebook
                cand = self.P[rn].T @ others
                if gram_mode == "gram":
                    # aproximar coeficientes por least squares SOLO con este rol
                    Cr = self.cb_arr[rn].T
                    G = Cr.T @ Cr
                    try:
                        cf = np.linalg.solve(G, Cr.T @ cand)
                    except np.linalg.LinAlgError:
                        cf = np.linalg.lstsq(Cr, cand, rcond=None)[0]
                    cand = Cr @ cf
                elif gram_mode == "pinv":
                    Cr = self.cb_arr[rn].T
                    cf = np.linalg.pinv(Cr.T @ Cr, rcond=1e-10) @ Cr.T @ cand
                    cand = Cr @ cf
                name = cleanup(cand, self.cb_names[rn], self.cb_arr[rn])
                new[rn] = self.bind_role(rn, name)
            est = new
        return {rn: cleanup(self.P[rn].T @ est[rn], self.cb_names[rn], self.cb_arr[rn])
                for rn in self.roles}


class MAPBundle:
    def __init__(self, seed, K, N, role_config=CFG6):
        self.np_rng = np.random.RandomState(seed)
        self.K = K
        self.N = N
        self.BLK = N // K
        self.role_names = role_config["names"]
        allsyms = sorted({s for syms in role_config["codebooks"].values() for s in syms})
        self.sym = {s: self.np_rng.choice([-1.0, 1.0], size=self.BLK) for s in allsyms}
        self.codebooks = role_config["codebooks"]
        self.cb_names = {r: list(role_config["codebooks"][r]) for r in self.role_names}

        self.br = {}
        for j, rn in enumerate(self.role_names):
            self.br.setdefault(j % K, []).append(rn)
        self.blocks = {b: MAPBlock(self.np_rng, rns, self.sym, self.cb_names, self.BLK)
                       for b, rns in self.br.items()}

    def encode_fact(self, fact):
        segs = [None] * self.K
        for b, blk in self.blocks.items():
            segs[b] = blk.encode(fact)
        return np.concatenate(segs)

    def decode_fact(self, c, T, mode):
        out = {}
        for b, blk in self.blocks.items():
            seg = c[b * self.BLK:(b + 1) * self.BLK]
            if mode in ("gram", "pinv"):
                r = blk.gram_decode(seg, pinv=(mode == "pinv"))
                if r is None:
                    r = blk.resonator(seg, T=T)
            elif mode == "gradient":
                # descenso continuo sobre coeficientes
                coef = np.linalg.lstsq(blk.A, seg, rcond=None)[0]
                for _ in range(T):
                    grad = blk.A.T @ (seg - blk.A @ coef)
                    coef = coef + 0.05 * grad
                r = {}
                for rn in blk.roles:
                    a, bb = blk.offsets[rn]
                    r[rn] = blk.cb_names[rn][int(np.argmax(coef[a:bb]))]
            else:
                r = blk.resonator(seg, T=T)
            out.update(r)
        return out


def make_fact(rng, cfg=CFG6):
    return {r: rng.choice(cfg["codebooks"][r]) for r in cfg["names"]}


def accuracy(dec, fact):
    return sum(1 for r in fact if dec.get(r) == fact[r]) / len(fact)


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


def main():
    rng = random.Random(42)
    out = []
    print("=" * 74)
    print("EXP 11c: MAP (bind = permutacion por rol) — grid rho x decoder")
    print("=" * 74)
    for label, n, K, N in CASES:
        BLK = N // K
        n_cv = (6 // K) * 16
        rho = n_cv / BLK
        row = {"label": label, "K": K, "N": N, "BLK": BLK,
               "n_cv_per_block": n_cv, "rho": round(rho, 4),
               "n_seeds": N_SEEDS, "n_facts": N_FACTS}
        kappas = []
        for s in range(N_SEEDS):
            b = MAPBundle(1000 + s, K, N)
            if s == 0:
                for blk in b.blocks.values():
                    if blk.n_cv > 0:
                        w = np.linalg.eigvalsh(blk.M)
                        lmin, lmax = abs(w.min()), abs(w.max())
                        kappas.append(lmax / lmin if lmin > 1e-14 else 1e16)
            for mode in MODES:
                pass
        # accuracy
        for mode in MODES:
            accs = []
            for s in range(N_SEEDS):
                b = MAPBundle(1000 + s, K, N)
                for _ in range(N_FACTS):
                    f = make_fact(rng)
                    c = b.encode_fact(f)
                    dec = b.decode_fact(c, T=100, mode=mode)
                    accs.append(accuracy(dec, f))
            row[f"acc_{mode}"] = round(float(np.mean(accs)), 4)
            row[f"acc_{mode}_std"] = round(float(np.std(accs)), 4)
        row["kappa_med"] = float(np.median(kappas)) if kappas else None
        out.append(row)
        print(f"{label:32s} rho={rho:5.3f} kappa_med={row['kappa_med']!s:>12} | "
              + " ".join(f"{m}={row[f'acc_{m}']:.3f}" for m in MODES))

    jpath = Path(__file__).parent.parent / "data" / "exp11c_map_rho.json"
    jpath.write_text(json.dumps(out, indent=1))
    with open(jpath.with_suffix(".txt"), "w") as fh:
        fh.write("EXP 11c: MAP — grid rho x decoder\n")
        fh.write(f"n_seeds={N_SEEDS}, n_facts={N_FACTS}, T=100\n\n")
        for r in out:
            fh.write(f"--- {r['label']} (BLK={r['BLK']}) ---\n")
            fh.write(f"  kappa mediana: {r['kappa_med']}\n")
            for m in MODES:
                fh.write(f"  {m:>9}: {r[f'acc_{m}']:.4f} ± {r[f'acc_{m}_std']:.4f}\n")
            fh.write("\n")
    print(f"\nGuardado: {jpath}")


if __name__ == "__main__":
    main()
