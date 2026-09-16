#!/usr/bin/env python3
"""
V5: Capacidad M_max vs dimensión D en HRR real (test del escalamiento √D).

Pandora exp_SGM_0029 predijo M_max ∝ sqrt(D) para anidamiento. Aquí lo
testeamos para decoding plano (sin anidamiento): número máximo de roles que
el resonator puro puede recuperar con accuracy >= 0.95.

Protocolo:
  - Para cada D ∈ {64, 128, 256, 512, 1024}:
    * Generar vocabulario compartido de 16 símbolos por rol
    * Para cada R ∈ {1, 2, 4, 8, 16, 32} roles (hasta que D lo permita):
      * Generar fact aleatorio, codificar, decodificar con resonator puro
      * Medir accuracy
  - Reportar M_max(D) = máximo R con acc >= 0.95
  - Ajuste log-log: log M_max = a + b log D → b debería ser ~0.5 (√D)

Esto conecta con la "binary collapse" de FHRR: si M_max ~ √D, entonces
el rho_crítico para resonator decoding es ρ* = M_max/D ~ 1/√D → 0 cuando
D crece. Es decir: el resonator puro no tiene región de fallo (rho<1 siempre)
— contradice la RM.

Salida: data + fig6_accuracy_vs_D.png
"""
import numpy as np
import random, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def hrr_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def hrr_unbind(c, r):
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real

def resonator_decode(seg, role_vecs, codebooks, sym, np_rng, T=80):
    """Pure resonator: sin Gram. Iterativo, cleanup por rol."""
    N = len(seg)
    k = len(role_vecs)
    est = []
    for _ in range(k):
        v = np_rng.randn(N); est.append(v / np.linalg.norm(v))
    cbs = [np.array([sym[s] for s in cb]) for cb in codebooks]
    cb_names = [list(cb) for cb in codebooks]
    for _ in range(T):
        new = []
        for i in range(k):
            others = seg.copy()
            for j in range(k):
                if j != i:
                    others = others - hrr_bind(role_vecs[j], est[j])
            cand = hrr_unbind(others, role_vecs[i])
            CB = cbs[i]
            n = np.linalg.norm(cand)
            if n == 0:
                new.append(sym[cb_names[i][0]]); continue
            sims = (CB @ cand) / (np.linalg.norm(CB, axis=1) * n)
            idx = int(np.argmax(sims))
            new.append(sym[cb_names[i][idx]])
        est = new
    out = []
    for i in range(k):
        CB = cbs[i]
        n = np.linalg.norm(est[i])
        if n == 0:
            out.append(cb_names[i][0]); continue
        sims = (CB @ est[i]) / (np.linalg.norm(CB, axis=1) * n)
        out.append(cb_names[i][int(np.argmax(sims))])
    return out

def run_config(D, R, n_facts=10, seed=42):
    """Accuracy promedio over n_facts hechos para D dimensiones y R roles."""
    np_rng = np.random.RandomState(seed)
    rng = random.Random(seed)
    # 16 símbolos por rol
    codebooks = [[f"s{r}_{k}" for k in range(16)] for r in range(R)]
    role_vecs = []
    sym = {}
    for cb in codebooks:
        for s in cb:
            if s not in sym:
                v = np_rng.randn(D); sym[s] = v / np.linalg.norm(v)
    for _ in range(R):
        v = np_rng.randn(D); role_vecs.append(v / np.linalg.norm(v))
    accs = []
    for _ in range(n_facts):
        fact = [rng.choice(cb) for cb in codebooks]
        seg = np.zeros(D)
        for i, s in enumerate(fact):
            seg += hrr_bind(role_vecs[i], sym[s])
        dec = resonator_decode(seg, role_vecs, codebooks, sym, np_rng, T=60)
        accs.append(sum(1 for a, b in zip(fact, dec) if a == b) / R)
    return float(np.mean(accs))

def main():
    t0 = time.time()
    Ds = [64, 128, 256, 512]
    print("=" * 78)
    print("V5: M_max vs D en HRR real — resonator puro (test escalamiento √D)")
    print("=" * 78)

    # grid de roles por D (acotado para que termine)
    grid = {
        64:  [1, 2, 4, 8, 16],
        128: [1, 2, 4, 8, 16, 32],
        256: [4, 8, 16, 32, 64],
        512: [8, 16, 32, 64, 128],
    }
    results = {}
    for D in Ds:
        results[D] = {}
        print(f"\n--- D={D} ---")
        for R in grid[D]:
            acc = run_config(D, R, n_facts=10)
            results[D][R] = acc
            print(f"  R={R:>3}: acc={acc:.3f}")

    # M_max por D con umbral 0.95
    print("\n--- M_max(D) (acc>=0.95) ---")
    M_max = {}
    for D, rdict in results.items():
        mm = 0
        for R, acc in rdict.items():
            if acc >= 0.95:
                mm = R
        M_max[D] = mm
        print(f"  D={D:>5}: M_max={mm}")

    # Ajuste log-log
    xs = [D for D in Ds if M_max[D] > 0]
    ys = [M_max[D] for D in xs]
    print(f"\nDatos para ajuste (log-log):")
    for D, m in zip(xs, ys):
        print(f"  D={D}, M_max={m}, logD={np.log(D):.3f}, logM={np.log(m):.3f}")
    if len(xs) >= 3:
        coef = np.polyfit(np.log(xs), np.log(ys), 1)
        print(f"\nAjuste: log M = {coef[0]:.3f} log D + {coef[1]:.3f}")
        print(f"  → M_max ∝ D^{coef[0]:.3f}  (√D sería 0.500)")

    # Figura
    fig, ax = plt.subplots(figsize=(8, 5))
    for D in Ds:
        rdict = results[D]
        Rs = sorted(rdict.keys())
        accs = [rdict[R] for R in Rs]
        ax.plot(Rs, accs, marker="o", label=f"D={D}")
    ax.axhline(0.95, ls="--", color="k", alpha=0.5, label="threshold 0.95")
    ax.set_xscale("log"); ax.set_xlabel("R (número de roles simultáneos)")
    ax.set_ylabel("accuracy (resonator puro)")
    ax.set_title("HRR real: decoding vs número de roles, parametrizado por D")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/fig_V5_scaling.png", dpi=150)
    print(f"\nFigura: figures/fig_V5_scaling.png")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
