#!/usr/bin/env python3
"""
Exp 16 (persistido): escala en n del punto cuadrado (rho = 1) — el test
directo de la prediccion RMT del paper.

Prediccion (Sec. RMT, Eq. kappa-scaling): para la Gram Wishart cuadrada
n x n con entradas de varianza 1/n,
    E-log escala:  kappa ~ 4 n^2   (lambda_max -> 4, lambda_min ~ 1/n^2? no:
    lambda_min ~ sigma^2/n con sigma^2 = 1/n -> 1/n^2, lambda_max -> 4.)
Mediciones: distribucion de kappa (mediana, IQR), lambda_min, lambda_max,
y accuracy del decoder gram (resonator con correccion) vs pure en cada n.

Ademas: test del decalaje de ESCALA del colapso: el decoder gram debe caer
en todos los n (la banda critica [1e3, 1e4] es SUPERADA para n grande:
kappa~4n^2 = 1e4 en n=50, 4e4 en n=100...). OJO: esto predice que el
colapso en rho=1 exacto DEGRADA con n (el punto cuadrado pasa la banda
critica y entra en la zona extendida). Verificar: accuracy gram vs n.

n in {32, 64, 128, 256, 512}, n_seeds=12 por n (kappa + autovalores),
mas accuracy gram/pure con T=100 en resonator cerrado en los mismos seeds.

Salida: data/exp16_scaling_n.json + .txt + figures/fig_exp16_scaling.png
"""
import json
import random
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import CFG6

NS = [32, 64, 128, 256, 512]
N_SEEDS_KAPPA = 12
N_SEEDS_ACC = 6
N_FACTS = 20


def grm(n, seed):
    """Gram Wishart cuadrada: C (n x n) entradas N(0, 1/n), M = C C^T."""
    rng = np.random.RandomState(seed)
    C = rng.standard_normal((n, n)) / np.sqrt(n)
    return C @ C.T


def measure_spectrum(n, n_seeds):
    rows = []
    for s in range(n_seeds):
        M = grm(n, 5000 + s)
        w = np.linalg.eigvalsh(M)
        lmin, lmax = float(w[0]), float(w[-1])
        rows.append({"seed": s, "lambda_min": lmin, "lambda_max": lmax,
                     "kappa": lmax / lmin})
    ks = np.array([r["kappa"] for r in rows])
    lmins = np.array([r["lambda_min"] for r in rows])
    lmaxs = np.array([r["lambda_max"] for r in rows])
    return {
        "n": n,
        "kappa_med": float(np.median(ks)),
        "kappa_iqr": [float(np.percentile(ks, 25)), float(np.percentile(ks, 75))],
        "kappa_mean": float(ks.mean()),
        "lambda_min_med": float(np.median(lmins)),
        "lambda_max_med": float(np.median(lmaxs)),
        "pred_4n2": 4.0 * n * n,
        "pred_lmin": 1.0 / (n * n),  # sigma^2/n con sigma^2=1/n
        "ratio_kappa_4n2": float(np.median(ks) / (4.0 * n * n)),
    }


class SquareBundle:
    """Bloque cuadrado: un mini-sistema con n_roles roles, n_sym simbolos
    por rol, codevectors gaussianos de dimension BLK = n_cv (cuadrado),
    resonator con correccion Gram opcional (misma dinamica que Exp 14).
    n_cv = n_roles * n_sym. Ajustamos n_roles=4, n_sym = n//4."""

    def __init__(self, seed, n_cv, n_roles=4):
        rng = np.random.RandomState(seed)
        self.n_cv = n_cv
        self.n_roles = n_roles
        self.n_sym = n_cv // n_roles
        self.BLK = n_cv  # cuadrado
        self.codebooks = {
            r: rng.standard_normal((self.n_sym, self.BLK)) / np.sqrt(self.BLK)
            for r in range(n_roles)
        }
        # normalizar filas a norma 1
        for r in range(n_roles):
            norms = np.linalg.norm(self.codebooks[r], axis=1, keepdims=True)
            self.codebooks[r] /= norms
        self.roles_vec = []
        for r in range(n_roles):
            v = rng.standard_normal(self.BLK)
            self.roles_vec.append(v / np.linalg.norm(v))
        # Gram del bloque: todos los codevectors activos? En Exp 14 la Gram
        # del bloque es sobre los codevectors del bloque (n_cv x n_cv).
        C = np.concatenate([self.codebooks[r] for r in range(n_roles)])
        self.C = C
        M = C @ C.T
        self.kappa = float(np.linalg.cond(M))
        try:
            self.Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            self.Minv = None

    def encode(self, fact):
        """fact: lista de indices de simbolo por rol."""
        acc = np.zeros(self.BLK)
        for r, si in enumerate(fact):
            acc += self.roles_vec[r] * self.codebooks[r][si]
        return acc

    def cleanup(self, cand, r):
        sims = self.codebooks[r] @ cand
        return int(np.argmax(sims))

    def decode(self, c, T=100, gram=False):
        est = []
        rng = np.random.RandomState(999)
        for r in range(self.n_roles):
            v = rng.standard_normal(self.BLK)
            est.append(v / np.linalg.norm(v))
        for _ in range(T):
            new = []
            for r in range(self.n_roles):
                others = c.copy()
                for o in range(self.n_roles):
                    if o != r:
                        others -= self.roles_vec[o] * est[o]
                fj = self.roles_vec[r] * others  # unbind (elemento a elemento)
                if gram and self.Minv is not None:
                    # correccion proyectiva: resolver en el espacio de coeficientes
                    # del codebook completo (aprox Exp 14: amplifica 1/lambda_min)
                    coef = self.C @ fj
                    coef = coef * (1.0 / max(1e-12, self.kappa**0))  # neutral
                    # ruido amplificado: mezclar por Minv via coef
                    coef2 = self.Minv @ coef
                    # reconstruccion con ruido amplificado:
                    fj = self.C.T @ coef2 / self.n_cv
                    fj = fj / max(np.linalg.norm(fj), 1e-12)
                si = self.cleanup(fj, r)
                new.append(self.codebooks[r][si])
            est = new
        return [self.cleanup(est[r], r) for r in range(self.n_roles)]


def main():
    t0 = time.time()
    rng = random.Random(42)
    out = {"spectra": [], "accuracy": []}

    print("=" * 74)
    print("EXP 16: escala en n del punto cuadrado (test de kappa ~ 4n^2)")
    print("=" * 74)

    print("\n[1] Espectro de la Gram cuadrada (mediana sobre seeds)")
    for n in NS:
        s = measure_spectrum(n, N_SEEDS_KAPPA)
        out["spectra"].append(s)
        print(f"  n={n:4d}: kappa_med={s['kappa_med']:.3e} (pred 4n^2={s['pred_4n2']:.3e}, "
              f"ratio={s['ratio_kappa_4n2']:.2f}) "
              f"lmin={s['lambda_min_med']:.2e} (pred {s['pred_lmin']:.2e}) "
              f"lmax={s['lambda_max_med']:.3f}")

    print("\n[2] Accuracy del decoder en el punto cuadrado vs n")
    for n in NS:
        n_roles = 4
        row = {"n": n, "n_roles": n_roles, "n_facts": N_FACTS, "n_seeds": N_SEEDS_ACC}
        for mode in ("pure", "gram"):
            accs, ks = [], []
            for s in range(N_SEEDS_ACC):
                b = SquareBundle(1000 + s, n, n_roles)
                ks.append(b.kappa)
                for _ in range(N_FACTS):
                    fact = [rng.randrange(b.n_sym) for _ in range(n_roles)]
                    c = b.encode(fact)
                    dec = b.decode(c, T=60, gram=(mode == "gram"))
                    accs.append(sum(int(d == f) for d, f in zip(dec, fact)) / n_roles)
            row[f"acc_{mode}"] = round(float(np.mean(accs)), 4)
            row[f"acc_{mode}_std"] = round(float(np.std(accs)), 4)
            row["kappa_med"] = float(np.median(ks))
        out["accuracy"].append(row)
        print(f"  n={n:4d}: kappa_med={row['kappa_med']:.2e} "
              f"pure={row['acc_pure']:.3f} gram={row['acc_gram']:.3f}")

    jpath = Path(__file__).parent.parent / "data" / "exp16_scaling_n.json"
    jpath.write_text(json.dumps(out, indent=1))
    with open(jpath.with_suffix(".txt"), "w") as fh:
        fh.write("EXP 16: escala en n del punto cuadrado\n")
        fh.write(f"kappa: {N_SEEDS_KAPPA} seeds; accuracy: {N_SEEDS_ACC} seeds x {N_FACTS} facts, T=60\n\n")
        fh.write("n | kappa_med | 4n^2 | ratio | lambda_min_med | pred 1/n^2 | lambda_max_med\n")
        for s in out["spectra"]:
            fh.write(f"{s['n']:4d} {s['kappa_med']:.4e} {s['pred_4n2']:.4e} "
                     f"{s['ratio_kappa_4n2']:6.3f} {s['lambda_min_med']:.4e} "
                     f"{s['pred_lmin']:.4e} {s['lambda_max_med']:.4f}\n")
        fh.write("\nn | pure | gram | kappa_med\n")
        for r in out["accuracy"]:
            fh.write(f"{r['n']:4d} {r['acc_pure']:.4f} {r['acc_gram']:.4f} {r['kappa_med']:.4e}\n")
    print(f"\nGuardado: {jpath} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
