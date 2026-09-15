#!/usr/bin/env python3
"""
F2: F corregido + diagnosticado + vectorizado (numpy).
[FIX]  ROLE_CONFIGS no tenia clave 2 -> el original crashea (KeyError: 2).
[DIAG] Imprime rank(M)/cond(M): para K=1, M es exactamente singular.
[VAR]  Tres modos: original (Minv de mat_inv, replica el bug) / pure / pinv.

NOTA (auditoria 2026-09): el comentario original de este archivo decia que
TIME (VM) tenia 14 simbolos unicos ("tarde" x3). Eso ya no es cierto para el
base_fhrr.py de este repo (VM tiene 16 unicos, sin duplicados) -> para n=6 el
diagnostico de abajo da vec_dist=96, rho=0.75 (no 78/0.61 como en versiones
previas de data/resultados_reales_completos.txt, que quedaron desactualizadas
respecto al codigo). No afecta el resultado central del paper: exp_H2.py usa
sus propios codebooks VM2/VT, ya limpios desde el vamos.
"""
from base_fhrr import *
import numpy as np

ROLE_CONFIGS[2] = {"names": ["SUJ", "ROL"], "codebooks": {"SUJ": VS, "ROL": VR}}

class FastBundle(BlockBundle):
    """Mismo constructor/seeds que BlockBundle; decode vectorizado.
    El Minv 'original' es EXACTAMENTE el de mat_inv (numerica intacta)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.np_roles = [np.array(self.roles[b]) for b in range(self.K)]
        self.np_cb, self.cb_names = {}, {}
        for rn, cb in self.codebooks.items():
            self.cb_names[rn] = list(cb.keys())
            self.np_cb[rn] = np.array([list(cb[s]) for s in self.cb_names[rn]])
        self.np_sym = {s: np.array(list(v)) for s, v in self.sym.items()}
        self.np_M = [np.array(self.M[b]) if self.M[b] is not None else None
                     for b in range(self.K)]
        self.np_Minv = [np.array(self.Minv[b]) if self.Minv[b] is not None else None
                        for b in range(self.K)]
        self.np_pinv = [np.linalg.pinv(self.np_M[b], rcond=1e-10)
                        if self.np_M[b] is not None else None for b in range(self.K)]

    def cleanup(self, fj, rname):
        CB = self.np_cb[rname]
        nf = np.linalg.norm(fj)
        if nf == 0:
            return self.cb_names[rname][0], 0.0
        sims = (CB @ np.conj(fj)).real / (np.linalg.norm(CB, axis=1) * nf)
        k = int(np.argmax(sims))
        return self.cb_names[rname][k], float(sims[k])

    def decode_fact(self, c, mem=None, mode="original", T=None):
        T = self.T if T is None else T
        c = np.asarray(c); out = {}
        for b in range(self.K):
            seg = c[b * self.BLK:(b + 1) * self.BLK]
            roles = self.br[b]
            if len(roles) == 1:
                _, rname = roles[0]
                name, _ = self.cleanup(seg, rname)
                out[rname] = ("SYM", name)
            elif mode == "gradient":
                out.update(self._decode_multi_gradient(seg, b, roles, T))
            else:
                out.update(self._decode_multi(seg, b, roles, mode, T))
        return out

    def _decode_multi_gradient(self, seg, b, roles, T, lr=0.1):
        R = self.np_roles[b]; r = len(roles)
        est = [np.array(rnd_phase(self.rng, self.BLK)) for _ in range(r)]
        for _ in range(T):
            residual = seg - sum(R[o] * est[o] for o in range(r))
            new = [None] * r
            for idx in range(r):
                grad = residual * np.conj(R[idx])
                updated = est[idx] + lr * grad
                nrm = np.linalg.norm(updated)
                if nrm > 0:
                    updated = updated / nrm
                name, _ = self.cleanup(updated, roles[idx][1])
                new[idx] = self.np_sym[name]
            est = new
        out = {}
        for idx, (j, rname) in enumerate(roles):
            name, _ = self.cleanup(est[idx], rname)
            out[rname] = ("SYM", name)
        return out

    def _decode_multi(self, seg, b, roles, mode, T):
        R = self.np_roles[b]; r = len(roles)
        Minv = {"original": self.np_Minv[b], "pure": None,
                "pinv": self.np_pinv[b]}[mode]
        est = [np.array(rnd_phase(self.rng, self.BLK)) for _ in range(r)]
        for _ in range(T):
            new = [None] * r
            for idx in range(r):
                others = seg - sum(R[o] * est[o] for o in range(r) if o != idx)
                fj = others * np.conj(R[idx])
                if Minv is not None:
                    fj = Minv @ fj
                name, _ = self.cleanup(fj, roles[idx][1])
                new[idx] = self.np_sym[name]
            est = new
        out = {}
        for idx, (j, rname) in enumerate(roles):
            name, _ = self.cleanup(est[idx], rname)
            out[rname] = ("SYM", name)
        return out

def main():
    rng = random.Random(42)
    casos = [{"n": 2, "K": 1, "N": 128}, {"n": 3, "K": 1, "N": 128},
             {"n": 4, "K": 1, "N": 128}, {"n": 6, "K": 1, "N": 128}]
    Ts = [25, 50, 100, 200, 500]
    modes = ["original", "pure", "pinv"]

    print("=" * 78)
    print("F2: variar T, tres modos de decode (original/pure/pinv), K=1")
    print("=" * 78)

    bundles = {c["n"]: FastBundle(7, c["K"], c["N"], c["n"],
                                  ROLE_CONFIGS[c["n"]]) for c in casos}

    print("\nDIAGNOSTICO bloque 0 (K=1 => un bloque con todos los roles):")
    print(f"{'n':>2} | {'vec_dist':>8} | {'BLK':>4} | {'rho':>5} | {'rank':>5} | {'cond':>9}")
    for c in casos:
        t = bundles[c["n"]]; M = t.np_M[0]
        nv = len({s for _, rn in t.br[0] for s in t.codebooks[rn]})
        print(f"{c['n']:>2} | {nv:>8} | {t.BLK:>4} | {nv/t.BLK:>5.2f} | "
              f"{np.linalg.matrix_rank(M):>5} | {np.linalg.cond(M):>9.2e}")
    print("Teoria: rank = vec_dist < BLK => M EXACTAMENTE singular (Minv = basura).")

    for c in casos:
        t = bundles[c["n"]]; cfg = ROLE_CONFIGS[c["n"]]
        print(f"\n--- n_roles={c['n']}, K=1, N=128 (BLK={t.BLK}) ---")
        print(f"{'T':>5} | " + " | ".join(f"{m:>9}" for m in modes))
        for T in Ts:
            row = [f"{T:>5}"]
            for m in modes:
                accs = []
                for _ in range(10):
                    f = make_fact_flat(rng, c["n"], cfg)
                    cf, mem = t.encode_fact(f)
                    dec = t.decode_fact(cf, mem, mode=m, T=T)
                    ok, tot = t.fact_accuracy(dec, f)
                    accs.append(ok / tot)
                row.append(f"{sum(accs)/len(accs):>9.3f}")
            print(" | ".join(row))

    print("\nLECTURA: original plano ~1/16 => no es convergencia lenta, es Minv basura.")
    print("pure/pinv altos ya con T=25 => no hay techo de capacidad en n<=6.")
    print("Si ORIGINAL da >=0.85 en n=2, la auditoria esta equivocada (falsacion).")

if __name__ == "__main__":
    main()
