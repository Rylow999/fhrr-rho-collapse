#!/usr/bin/env python3
"""
Exp 11: ley de rho en VSA BINARIAS (BSC y MAP).

Pregunta: la transicion en rho=1 (singularidad del Gram para gram/pinv,
robustez del resonator puro) existe tambien cuando el algebra es binaria?

Algebras:
  BSC (Binary Spatter Codes, Kanerva): vectores en {±1}^N,
    bind = multiplicacion elemento a elemento, bundle = signo de la suma
    (o suma sin signo para "soft" bundling). Unbind = bind (auto-inverso).
  MAP (Multiply-Add-Permute, Gayler): vectores reales en hiperesfera,
    bind = multiplicacion elemento a elemento + permutacion.
    Aca usamos la variante real bipolar para comparabilidad con BSC.

Gram analogo: M[i][j] = <c_i, c_j> sobre los codevectors del bloque.
Decoders: gram (inv), pinv, pure (resonator), gradient.

Grid: mismo que V3 — n=6 roles, CFG6, casos de rho {0.75, 0.80, 1.00, 1.33, 1.50}.
"""
import numpy as np
import random

from exp_observer_taxonomy_v3 import CFG6  # codebooks de palabras

# ================================================================ ops binarias
def bsc_bind(a, b):
    return a * b

def bsc_unbind(c, r):
    return c * r  # auto-inverso

def cleanup_bin(cand, names, cb_arr):
    sims = cb_arr @ cand / (np.linalg.norm(cb_arr, axis=1) * max(np.linalg.norm(cand), 1e-12))
    return names[int(np.argmax(sims))]

class BSCBundle:
    """Roles repartidos en bloques como V3, simbolos en {±1}^BLK."""
    def __init__(self, seed, K, N, role_config=CFG6):
        self.np_rng = np.random.RandomState(seed)
        self.K = K
        self.N = N
        self.BLK = N // K
        self.role_names = role_config["names"]
        self.cfg = role_config

        self.sym = {}
        allsyms = sorted({s for syms in role_config["codebooks"].values() for s in syms})
        for s in allsyms:
            self.sym[s] = self.np_rng.choice([-1.0, 1.0], size=self.BLK)

        self.codebooks = {r: {s: self.sym[s] for s in syms}
                          for r, syms in role_config["codebooks"].items()}
        self.cb_names = {r: list(cb) for r, cb in self.codebooks.items()}
        self.cb_arr = {r: np.array([cb[n] for n in self.cb_names[r]])
                       for r, cb in self.codebooks.items()}

        self.br = {}
        for j, rn in enumerate(self.role_names):
            self.br.setdefault(j % K, []).append((j, rn))
        self.roles = []
        for b in range(self.K):
            blk = []
            for _ in range(len(self.br[b])):
                v = self.np_rng.choice([-1.0, 1.0], size=self.BLK)
                blk.append(v)
            self.roles.append(blk)

        # Gram por bloque (sobre codevectors del bloque)
        self.Minv, self.pinv = {}, {}
        for b in range(self.K):
            if len(self.br[b]) <= 1:
                continue
            ck = []
            for _, rn in self.br[b]:
                ck += list(self.codebooks[rn].keys())
            ck = sorted(set(ck))
            C = np.array([self.sym[s] for s in ck])
            M = C @ C.T
            try:
                self.Minv[b] = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                self.Minv[b] = None
            self.pinv[b] = np.linalg.pinv(M, rcond=1e-10)

    def encode_fact(self, fact):
        segs = [None] * self.K
        for b, roles in self.br.items():
            if len(roles) == 1:
                _, rn = roles[0]
                segs[b] = self.sym[fact[rn]]
            else:
                acc = np.zeros(self.BLK)
                for idx, (j, rn) in enumerate(roles):
                    acc += bsc_bind(self.roles[b][idx], self.sym[fact[rn]])
                segs[b] = acc  # soft bundle (sin signo) para que el resonador itere
        return np.concatenate(segs)

    def decode_fact(self, c, T=50, mode="pure"):
        out = {}
        for b, roles in self.br.items():
            seg = c[b * self.BLK:(b + 1) * self.BLK]
            if len(roles) == 1:
                _, rn = roles[0]
                out[rn] = cleanup_bin(seg, self.cb_names[rn], self.cb_arr[rn])
                continue
            r_ = len(roles)
            if mode == "gram":
                Minv = self.Minv.get(b)
            elif mode == "pinv":
                Minv = self.pinv.get(b)
            else:
                Minv = None
            est = [self.np_rng.choice([-1.0, 1.0], size=self.BLK) for _ in roles]
            for _ in range(T):
                new = [None] * r_
                for idx in range(r_):
                    others = seg.copy()
                    for o in range(r_):
                        if o != idx:
                            others -= bsc_bind(self.roles[b][o], est[o])
                    fj = bsc_unbind(others, self.roles[b][idx])
                    if mode == "gradient":
                        resid = seg - sum(bsc_bind(self.roles[b][o], est[o]) for o in range(r_))
                        grad = bsc_unbind(resid, self.roles[b][idx])
                        est[idx] = est[idx] + 0.1 * grad
                        nm = np.linalg.norm(est[idx])
                        if nm > 0: est[idx] = est[idx] / nm
                        name = cleanup_bin(est[idx], self.cb_names[roles[idx][1]], self.cb_arr[roles[idx][1]])
                    else:
                        if Minv is not None and Minv.shape[0] == self.BLK:
                            fj = Minv @ fj
                        name = cleanup_bin(fj, self.cb_names[roles[idx][1]], self.cb_arr[roles[idx][1]])
                    new[idx] = self.sym[name]
                est = new
            for idx, (_, rn) in enumerate(roles):
                out[rn] = cleanup_bin(est[idx], self.cb_names[rn], self.cb_arr[rn])
        return out


def make_fact(rng, cfg=CFG6):
    return {r: rng.choice(cfg["codebooks"][r]) for r in cfg["names"]}

def accuracy(dec, fact):
    return sum(1 for r in fact if dec.get(r) == fact[r]) / len(fact)


def main():
    rng = random.Random(42)
    cases = [
        ("rho=0.75 K=1 N=128", 6, 1, 128),
        ("rho=0.80 K=3 N=120", 6, 3, 120),
        ("rho=1.00 K=3 N=96 CUADRADO", 6, 3, 96),
        ("rho=1.33 K=3 N=72", 6, 3, 72),
        ("rho=1.50 K=2 N=64", 6, 2, 64),
        ("control n=3 K=3 N=126", 3, 3, 126),
    ]
    modes = ["gram", "pure", "pinv", "gradient"]
    print("=" * 72)
    print("EXP 11: ley de rho en VSA BINARIA (BSC {±1}, bind multiplicativo)")
    print("=" * 72)
    for label, n, K, N in cases:
        print(f"\n--- {label} ---")
        for mode in modes:
            accs = []
            for _ in range(15):
                b = BSCBundle(7, K, N)
                f = make_fact(rng)
                c = b.encode_fact(f)
                dec = b.decode_fact(c, T=100, mode=mode)
                accs.append(accuracy(dec, f))
            print(f"  {mode:>9}: acc={np.mean(accs):.3f} ±{np.std(accs):.3f}")

if __name__ == "__main__":
    main()
