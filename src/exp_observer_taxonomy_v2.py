#!/usr/bin/env python3
"""
Exp 8 v2: Taxonomía de observadores con DECODERS REALES.

Basado en exp_V3_hrr_real.py (HRRRealBundle) que SÍ implementa:
  - gram:    Minv real = inv(M) donde M[i][j] = <c_i, c_j> sobre codevectors del bloque
  - pinv:    np.linalg.pinv(M, rcond=1e-10)
  - pure:    resonator sin corrección (Minv=None)
  - gradient: descenso de gradiente continuo

Grid:
  - mode:  ['gram', 'pure', 'pinv', 'gradient']
  - T:     [10, 25, 50, 100, 200]
  - cleanup: ['argmax', 'softmax', 'top_k']
  - scheme: ['flat', 'nested']  (nested = roles agrupados en bloques jerárquicos)
  - ρ:     [0.50, 0.75, 1.00, 1.33, 1.50]  (via N con CFG6)

Total: 4 × 5 × 3 × 2 × 5 = 600 configs × 10 facts = 12k decodificaciones
Tiempo estimado: ~20 min
"""
import numpy as np
import random
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================ Ops HRR
def hrr_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def hrr_unbind(c, r):
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real

# ================================================================ Codebooks
VS  = ["lobo","zorro","ave","venado","oso","aguila","serpiente","conejo",
       "tigre","leon","pantera","halcon","coyote","jaguar","puma","lobo_marino"]
VR  = ["come","corre","esta_en","persigue","caza","observa","sigue","evita",
       "protege","ataca","huye","marca","ronda","acecha","vigila","explora"]
VO  = ["manzana","pasto","rio","arbol","roca","cueva","nido","madriguera",
       "lago","montaña","bosque","desierto","playa","isla","glaciar","volcan"]
VL  = ["rapido","lento","grande","pequeño","cerca","lejos","alto","bajo",
       "pesado","liviano","joven","viejo","fuerte","debil","salvaje","domestico"]
VM2 = ["mañana","tarde","noche","hoy","ayer","siempre","nunca","ahora",
       "antes","despues","pronto","anochecer","temprano","mediodia",
       "frecuente","raro"]
VT  = ["norte","sur","este","oeste","cima","valle","sombra","claro",
       "denso","humedo","seco","pedregoso","frondoso","abierto",
       "empinado","pantano"]

CFG6 = {"names": ["SUJ","ROL","OBJ","LOC","TIME","TERR"],
        "codebooks": {"SUJ":VS,"ROL":VR,"OBJ":VO,"LOC":VL,"TIME":VM2,"TERR":VT}}

# ================================================================ Bundle con decoders reales
class ObserverBundleV2:
    """
    HRR real con decoders completos (gram/pinv/pure/gradient).
    Arquitectura K-bloques como V3 (FastBundle). scheme='nested' agrupa roles
    en jerarquía (nivel 1: SUJ+ROL+OBJ, nivel 2: LOC+TIME, nivel 3: TERR),
    cada nivel con su propio bloque -> Gram por nivel (chico, bien condicionado).
    """

    def __init__(self, seed, N, scheme):
        assert scheme in ("flat", "nested")
        self.np_rng = np.random.RandomState(seed)
        self.N = N
        self.scheme = scheme
        self.codebooks = CFG6["codebooks"]
        self.role_names = CFG6["names"]

        # Símbolos: gaussian normalizados
        self.sym = {}
        for syms in self.codebooks.values():
            for s in syms:
                if s not in self.sym:
                    v = self.np_rng.randn(N)
                    self.sym[s] = v / np.linalg.norm(v)

        # Roles
        def rv():
            v = self.np_rng.randn(N); return v / np.linalg.norm(v)
        self.role_vec = {r: rv() for r in self.role_names}

        # Niveles para nested
        self.lvl_vec = {1: rv(), 2: rv(), 3: rv()}
        self.nivel_roles = [["SUJ","ROL","OBJ"], ["LOC","TIME"], ["TERR"]]

        # Codebooks arrays para cleanup
        self.cb_names = {r: list(cb) for r, cb in self.codebooks.items()}
        self.cb = {r: np.array([self.sym[s] for s in self.cb_names[r]])
                   for r in self.role_names}

        # Gram y Minv por bloque. Gram se calcula sobre los CODEVECTORS (los 16×roles
        # símbolos que aparecen en ese bloque), pero para aplicarlo a un vector N-dim
        # necesitamos el OPERADOR N×N: Minv_op = C^T @ inv(G) @ C  (pseudo-inversa
        # inducida), donde C = matriz (n_codevectors × N) de los codevectors.
        # Esto es el "Gram operator" estándar de frame theory en el espacio ambiente.
        self.M = {}
        self.Minv = {}   # operador N×N
        self.pinv = {}   # operador N×N
        if scheme == "flat":
            block_roles = self.role_names
            ck_names = []
            C_rows = []
            for r in block_roles:
                for s in self.codebooks[r]:
                    ck_names.append(s)
                    C_rows.append(self.sym[s])
            C = np.array(C_rows)                       # (n_cv, N)
            Mm = C @ C.T                                # (n_cv, n_cv) Gram
            self.M[0] = Mm
            self.Minv[0] = self._gram_operator(C, Mm, kind="inv")
            self.pinv[0]  = self._gram_operator(C, Mm, kind="pinv")
        else:  # nested
            for lvl, roles in enumerate(self.nivel_roles):
                ck_names = []
                C_rows = []
                for r in roles:
                    for s in self.codebooks[r]:
                        ck_names.append(s)
                        C_rows.append(self.sym[s])
                C = np.array(C_rows)
                Mm = C @ C.T
                self.M[lvl] = Mm
                self.Minv[lvl] = self._gram_operator(C, Mm, kind="inv")
                self.pinv[lvl]  = self._gram_operator(C, Mm, kind="pinv")

    @staticmethod
    def _gram_operator(C, M, kind):
        """Devuelve el operador N×N inducido por la inversa de la Gram de los codevectors.
        kind='inv' -> C^T inv(M) C (None si M singular)
        kind='pinv' -> C^T pinv(M) C"""
        N = C.shape[1]
        if kind == "inv":
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                return None
        else:
            Minv = np.linalg.pinv(M, rcond=1e-10)
        return C.T @ Minv @ C

    # --------------------------------------------------------- encode
    def encode(self, fact):
        if self.scheme == "flat":
            acc = np.zeros(self.N)
            for r in self.role_names:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        # nested: cada nivel se codifica como subvector
        def pack(roles):
            acc = np.zeros(self.N)
            for r in roles:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        n1 = pack(self.nivel_roles[0])
        n2 = hrr_bind(self.lvl_vec[2], pack(self.nivel_roles[1]))
        n3 = hrr_bind(self.lvl_vec[3], pack(self.nivel_roles[2]))
        return hrr_bind(self.lvl_vec[1], n1) + n2 + n3

    # --------------------------------------------------------- cleanup
    def cleanup(self, v, rname, method):
        CB = self.cb[rname]
        n = np.linalg.norm(v)
        if n == 0:
            return self.cb_names[rname][0]
        sims = (CB @ v) / (np.linalg.norm(CB, axis=1) * n)
        if method == "argmax":
            return self.cb_names[rname][int(np.argmax(sims))]
        elif method == "softmax":
            t = 1.0
            probs = np.exp(sims / t)
            probs = probs / probs.sum()
            return self.cb_names[rname][int(self.np_rng.choice(len(probs), p=probs))]
        else:  # top_k
            k = min(3, len(sims))
            top_idx = np.argpartition(sims, -k)[-k:]
            top_vec = np.mean([self.sym[self.cb_names[rname][i]] for i in top_idx], axis=0)
            top_vec = top_vec / np.linalg.norm(top_vec)
            sims2 = (CB @ top_vec) / (np.linalg.norm(CB, axis=1) * np.linalg.norm(top_vec))
            return self.cb_names[rname][int(np.argmax(sims2))]

    # --------------------------------------------------------- decode
    def _resonator(self, seg, roles, T, mode, cleanup_method, block_idx):
        """Resonator iterativo. roles = lista de nombres de rol en este bloque."""
        R = [self.role_vec[r] for r in roles]
        r_count = len(roles)

        # Minv según modo (solo si block tiene Gram y >1 rol)
        Minv = None
        if r_count > 1:
            if mode == "gram":
                Minv = self.Minv.get(block_idx)
            elif mode == "pinv":
                Minv = self.pinv.get(block_idx)

        est = []
        for _ in roles:
            v = self.np_rng.randn(self.N); est.append(v / np.linalg.norm(v))

        for _ in range(T):
            new = [None] * r_count
            for idx in range(r_count):
                others = seg.copy()
                for j in range(r_count):
                    if j != idx:
                        others = others - hrr_bind(R[j], est[j])
                fj = hrr_unbind(others, R[idx])

                # Aplicar Minv si corresponde (operador N×N)
                if Minv is not None:
                    fj = Minv @ fj

                if mode == "gradient":
                    # gradient step
                    residual = seg - sum(hrr_bind(R[j], est[j]) for j in range(r_count))
                    grad = hrr_unbind(residual, R[idx])
                    updated = est[idx] + 0.1 * grad
                    nrm = np.linalg.norm(updated)
                    if nrm > 0: updated = updated / nrm
                    name = self.cleanup(updated, roles[idx], cleanup_method)
                    new[idx] = self.sym[name]
                else:
                    name = self.cleanup(fj, roles[idx], cleanup_method)
                    new[idx] = self.sym[name]
            est = new

        return {r: self.cleanup(est[i], r, cleanup_method) for i, r in enumerate(roles)}

    def decode(self, c, T=50, mode="pure", cleanup_method="argmax"):
        if self.scheme == "flat":
            return self._resonator(c, self.role_names, T, mode, cleanup_method, 0)
        # nested: decodificar nivel por nivel
        out = {}
        for lvl, roles in enumerate(self.nivel_roles):
            if lvl == 0:
                seg = hrr_unbind(c, self.lvl_vec[1])
            elif lvl == 1:
                seg = hrr_unbind(c, self.lvl_vec[2])
            else:
                seg = hrr_unbind(c, self.lvl_vec[3])
            out.update(self._resonator(seg, roles, T, mode, cleanup_method, lvl))
        return out

# ================================================================ Fact & accuracy
def make_fact(rng):
    return {r: rng.choice(CFG6["codebooks"][r]) for r in CFG6["names"]}

def accuracy(dec, fact):
    return sum(1 for r in fact if dec.get(r) == fact[r]) / len(fact)

# ================================================================ Main
def main():
    t0 = time.time()
    rng = random.Random(42)

    # Grid de ρ
    rho_grid = [
        ("rho=0.50", 192),
        ("rho=0.75", 128),
        ("rho=1.00", 96),
        ("rho=1.33", 72),
        ("rho=1.50", 64),
    ]

    # Observadores: 4 modos × 5 T × 3 cleanup = 60 × 2 esquemas = 120
    observers = []
    for mode in ("gram", "pure", "pinv", "gradient"):
        for T in (10, 25, 50, 100, 200):
            for clean in ("argmax", "softmax", "top_k"):
                observers.append({"mode": mode, "num_iterations": T, "cleanup_method": clean})

    print("=" * 78)
    print("EXP 8 v2: TAXONOMÍA CON DECODERS REALES")
    print(f"  Modos: gram/pure/pinv/gradient (implementación real)")
    print(f"  Configs: {len(observers)} × 2 esquemas = {len(observers)*2}")
    print(f"  Grid ρ: {len(rho_grid)} puntos")
    print(f"  Hechos por punto: 10")
    print(f"  Total: {len(observers) * len(rho_grid) * 2 * 10} decodificaciones")
    print("=" * 78)

    results = []
    total = len(observers) * len(rho_grid) * 2
    done = 0

    for scheme in ("flat", "nested"):
        for rho_label, N in rho_grid:
            for obs_config in observers:
                done += 1
                tag = f"{scheme}|mode={obs_config['mode']}|T={obs_config['num_iterations']}|clean={obs_config['cleanup_method']}"

                b = ObserverBundleV2(7, N, scheme)
                accs = []
                for _ in range(10):
                    f = make_fact(rng)
                    c = b.encode(f)
                    dec = b.decode(c, T=obs_config["num_iterations"],
                                   mode=obs_config["mode"],
                                   cleanup_method=obs_config["cleanup_method"])
                    accs.append(accuracy(dec, f))

                m = float(np.mean(accs))
                s = float(np.std(accs))
                results.append({
                    "scheme": scheme,
                    "rho_label": rho_label,
                    "N": N,
                    **obs_config,
                    "accuracy_mean": m,
                    "accuracy_std": s
                })

                if done % 50 == 0 or done == total:
                    pct = 100 * done / total
                    print(f"  [{pct:5.1f}%] {tag} @ {rho_label}: acc={m:.3f} ±{s:.3f}")

    # Guardar
    out_path = Path(__file__).parent.parent / "data" / "observer_taxonomy_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out_path}")

    # Heatmap
    rho_labels = [r[0] for r in rho_grid]
    obs_labels = sorted(set(
        f"mode={r['mode']}|T={r['num_iterations']}|clean={r['cleanup_method']}"
        for r in results if r["scheme"] == "flat"
    ))

    Z = np.zeros((len(obs_labels), len(rho_labels)))
    for i, ol in enumerate(obs_labels):
        for j, rl in enumerate(rho_labels):
            matches = [r for r in results if r["scheme"] == "flat" and
                       f"mode={r['mode']}|T={r['num_iterations']}|clean={r['cleanup_method']}" == ol
                       and r["rho_label"] == rl]
            if matches:
                Z[i, j] = matches[0]["accuracy_mean"]

    fig, ax = plt.subplots(figsize=(14, 18))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(rho_labels)))
    ax.set_xticklabels(rho_labels, fontsize=9)
    ax.set_yticks(range(len(obs_labels)))
    ax.set_yticklabels(obs_labels, fontsize=7)
    ax.set_title("Taxonomía de Observadores v2: Accuracy vs ρ (flat) — decoders reales")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_observer_taxonomy_v2.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")

    # Resumen por modo
    print("\n" + "=" * 78)
    print("RESUMEN POR MODO (flat, promedio sobre T/cleanup):")
    for mode in ("gram", "pure", "pinv", "gradient"):
        subset = [r for r in results if r["scheme"] == "flat" and r["mode"] == mode]
        by_rho = {}
        for rl in rho_labels:
            vals = [r["accuracy_mean"] for r in subset if r["rho_label"] == rl]
            by_rho[rl] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"\n  mode={mode}:")
        for rl, (mu, sd) in by_rho.items():
            print(f"    {rl:>10}: {mu:.3f} ±{sd:.3f}")

    # Top en ρ=1.00
    rho1 = [r for r in results if r["scheme"] == "flat" and r["rho_label"] == "rho=1.00"]
    rho1.sort(key=lambda x: -x["accuracy_mean"])
    print("\n  TOP 10 en ρ=1.00:")
    for r in rho1[:10]:
        print(f"    {r['accuracy_mean']:.3f}  mode={r['mode']} T={r['num_iterations']} clean={r['cleanup_method']}")

    # Guardar log
    log_path = Path(__file__).parent.parent / "data" / "out_V8_v2.txt"
    import sys
    with open(log_path, "w") as f:
        sys.stdout = f
        print("=" * 78)
        print("EXP 8 v2 COMPLETO")
        print(f"Total configs: {len(results)}")
        print(f"Tiempo: {time.time()-t0:.1f}s")
        print("\nResumen por modo (flat):")
        for mode in ("gram", "pure", "pinv", "gradient"):
            subset = [r for r in results if r["scheme"] == "flat" and r["mode"] == mode]
            for rl in rho_labels:
                vals = [r["accuracy_mean"] for r in subset if r["rho_label"] == rl]
                print(f"  {mode} @ {rl}: {np.mean(vals):.3f} ±{np.std(vals):.3f}")
        print("\nTOP 10 en ρ=1.00:")
        for r in rho1[:10]:
            print(f"  {r['accuracy_mean']:.3f}  {r['mode']}/T={r['num_iterations']}/{r['cleanup_method']}")
    sys.stdout = sys.__stdout__

    print(f"\nLog: {log_path}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()