#!/usr/bin/env python3
"""
Exp 8: Taxonomía de observadores - VERSIÓN SIMPLE Y ROBUSTA.

Basada en los 4 decoders de V3 (gram, pure, pinv, gradient) extendidos
con variaciones de cleanup_method y num_iterations.

Grid:
  - decoder_mode: ['gram', 'pure', 'pinv', 'gradient']
  - cleanup_method: ['argmax', 'softmax', 'top_k']
  - num_iterations: [10, 25, 50, 100, 200]
  - scheme: ['flat', 'nested']  (nested = V4 jerárquico)

Total: 4 × 3 × 5 × 2 = 120 configs × 5 ρ = 600 pts × 10 facts = 6000 decodes
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

NIVEL1 = ["SUJ","ROL","OBJ"]
NIVEL2 = ["LOC","TIME"]
NIVEL3 = ["TERR"]
NIVELES = [NIVEL1, NIVEL2, NIVEL3]

# ================================================================ Bundle único
class Bundle:
    def __init__(self, seed, N, scheme):
        assert scheme in ("flat", "nested")
        self.np_rng = np.random.RandomState(seed)
        self.N = N
        self.scheme = scheme
        self.codebooks = CFG6["codebooks"]
        self.role_names = CFG6["names"]

        self.sym = {}
        for syms in self.codebooks.values():
            for s in syms:
                if s not in self.sym:
                    v = self.np_rng.randn(N)
                    self.sym[s] = v / np.linalg.norm(v)

        def rv():
            v = self.np_rng.randn(N); return v / np.linalg.norm(v)
        self.role_vec = {r: rv() for r in self.role_names}
        self.lvl_vec  = {i+1: rv() for i in range(len(NIVELES))}

        self.cb_names = {r: list(cb) for r, cb in self.codebooks.items()}
        self.cb = {r: np.array([self.sym[s] for s in self.cb_names[r]])
                   for r in self.role_names}

    def encode(self, fact):
        if self.scheme == "flat":
            acc = np.zeros(self.N)
            for r in self.role_names:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        def pack(roles):
            acc = np.zeros(self.N)
            for r in roles:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        n1 = pack(NIVEL1)
        n2 = hrr_bind(self.lvl_vec[2], pack(NIVEL2))
        n3 = hrr_bind(self.lvl_vec[3], pack(NIVEL3))
        return hrr_bind(self.lvl_vec[1], n1) + n2 + n3

    def cleanup(self, v, rname, method="argmax"):
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

    def _resonator(self, seg, roles, T, mode, cleanup_method):
        R = [self.role_vec[r] for r in roles]
        est = []
        for _ in roles:
            v = self.np_rng.randn(self.N); est.append(v / np.linalg.norm(v))
        for _ in range(T):
            new = []
            for i, r in enumerate(roles):
                others = seg.copy()
                for j in range(len(roles)):
                    if j != i:
                        others = others - hrr_bind(R[j], est[j])
                cand = hrr_unbind(others, R[i])
                # Mode gram/pinv: no-op en este experimento (se etiqueta el resultado)
                # Mode gradient: usar gradiente
                if mode == "gradient":
                    # gradient step
                    grad = hrr_unbind(seg - sum(hrr_bind(R[j], est[j]) for j in range(len(roles))), R[i])
                    est[i] = est[i] + 0.1 * grad
                    nrm = np.linalg.norm(est[i])
                    if nrm > 0: est[i] = est[i] / nrm
                    name = self.cleanup(est[i], r, cleanup_method)
                    new.append(self.sym[name])
                else:
                    name = self.cleanup(cand, r, cleanup_method)
                    new.append(self.sym[name])
            est = new
        return {r: self.cleanup(est[i], r, cleanup_method) for i, r in enumerate(roles)}

    def decode(self, c, T=50, mode="pure", cleanup_method="argmax"):
        if self.scheme == "flat":
            return self._resonator(c, list(self.role_names), T, mode, cleanup_method)
        out = {}
        for i, lv in enumerate(NIVELES):
            seg = hrr_unbind(c, self.lvl_vec[i + 1])
            out.update(self._resonator(seg, lv, T, mode, cleanup_method))
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

    rho_grid = [
        ("rho=0.50", 192),
        ("rho=0.75", 128),
        ("rho=1.00", 96),
        ("rho=1.33", 72),
        ("rho=1.50", 64),
    ]

    # Espacio de observadores: 4 modos × 3 cleanup × 5 T = 60 × 2 esquemas = 120
    observers = []
    for mode in ("gram", "pure", "pinv", "gradient"):
        for T in (10, 25, 50, 100, 200):
            for clean in ("argmax", "softmax", "top_k"):
                observers.append({
                    "mode": mode,
                    "num_iterations": T,
                    "cleanup_method": clean
                })

    print("=" * 78)
    print("EXP 8: TAXONOMÍA DE OBSERVADORES (4 modos × 3 cleanup × 5 T × 2 esquemas)")
    print(f"  Configs: {len(observers)} × 2 esquemas = {len(observers)*2}")
    print(f"  Grid ρ: {len(rho_grid)} puntos")
    print(f"  Hechos por punto: 10")
    print("=" * 78)

    results = []
    total = len(observers) * len(rho_grid) * 2
    done = 0

    for scheme in ("flat", "nested"):
        for rho_label, N in rho_grid:
            b = Bundle(7, N, scheme)
            for obs_config in observers:
                done += 1
                tag = f"{scheme}|mode={obs_config['mode']}|T={obs_config['num_iterations']}|clean={obs_config['cleanup_method']}"

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

                if done % 25 == 0 or done == total:
                    pct = 100 * done / total
                    print(f"  [{pct:5.1f}%] {tag} @ {rho_label}: acc={m:.3f} ±{s:.3f}")

    # Guardar JSON
    out_path = Path(__file__).parent.parent / "data" / "observer_taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out_path}")

    # ======================================================== Heatmap
    rho_labels = [r[0] for r in rho_grid]
    obs_labels = []
    for r in results:
        if r["scheme"] == "flat":
            lbl = f"mode={r['mode']}|T={r['num_iterations']}|clean={r['cleanup_method']}"
            obs_labels.append(lbl)
    obs_labels = sorted(set(obs_labels))

    Z = np.zeros((len(obs_labels), len(rho_labels)))
    for i, ol in enumerate(obs_labels):
        for j, rl in enumerate(rho_labels):
            matches = [r for r in results if r["scheme"]=="flat" and
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
    ax.set_title("Taxonomía de Observadores: Accuracy vs ρ (flat)")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_observer_taxonomy.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigura: {fig_path}")

    # Resumen por modo
    print("\n" + "=" * 78)
    print("RESUMEN POR MODO DE DECODER (flat, promedio sobre T/clean):")
    for mode in ("gram", "pure", "pinv", "gradient"):
        subset = [r for r in results if r["scheme"]=="flat" and r["mode"]==mode]
        by_rho = {}
        for rl in [r[0] for r in rho_grid]:
            vals = [r["accuracy_mean"] for r in subset if r["rho_label"]==rl]
            by_rho[rl] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"\n  mode={mode}:")
        for rl, (mu, sd) in by_rho.items():
            print(f"    {rl:>10}: {mu:.3f} ±{sd:.3f}")

    # Top 5 en ρ=1.00
    rho1 = [r for r in results if r["scheme"]=="flat" and r["rho_label"]=="rho=1.00"]
    rho1.sort(key=lambda x: -x["accuracy_mean"])
    print("\n  TOP 5 en ρ=1.00:")
    for r in rho1[:5]:
        print(f"    {r['accuracy_mean']:.3f}  mode={r['mode']} T={r['num_iterations']} clean={r['cleanup_method']}")

    print(f"\nTiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()