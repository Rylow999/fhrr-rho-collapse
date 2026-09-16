#!/usr/bin/env python3
"""
Exp 8 v3: Taxonomía de observadores con DECODERS REALES (arquitectura V3 exacta).

Arquitectura heredada de exp_V3_hrr_real.py (que reproduce anti-resonancia):
  - K bloques de dimensión BLK = N // K
  - Símbolos gaussianos en R^BLK (NO en R^N)
  - Roles repartidos por bloques (j % K)
  - Gram por bloque: M[i][j] = <c_i, c_j> sobre codevectors del bloque (n_cv_b × n_cv_b)
  - decode_multi aplica Minv SOLO si Minv.shape[0] == BLK (colapso n_cv<BLK nunca ocurre:
    la singularidad es debida a que n_cv puede EXCEDER BLK en rho>1 quedando cuadrada en rho=1)

Modos reales:
  - gram:     Minv = np.linalg.inv(M) (colapsa si M singular o mal condicionada)
  - pinv:     pinv(M, rcond=1e-10) (trunca modos < 1e-10, falla en cuadrado)
  - pure:     resonator sin corrección
  - gradient: gradiente descendiente con lr=0.1

Grid:
  mode × T × cleanup × K × N → ρ efectivo = n_cv_por_bloque / BLK
Salidas:
  data/observer_taxonomy_v3.json
  figures/fig_observer_taxonomy_v3.png  (heatmap por modo: accuracy vs rho)
  data/out_V8_v3.txt
"""
import numpy as np
import random
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ================================================================ Bundle estilo V3
class BundleV3:
    """Réplica de HRRRealBundle con cleanup parametrizable + modos extendidos."""

    def __init__(self, seed, K, N, role_config=CFG6):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.K = K
        self.N = N
        self.BLK = N // K
        self.role_names = role_config["names"]
        self.codebooks_by_role = role_config["codebooks"]

        # Símbolos en R^BLK
        self.sym = {}
        all_syms = sorted({s for syms in self.codebooks_by_role.values() for s in syms})
        for s in all_syms:
            v = self.np_rng.randn(self.BLK)
            self.sym[s] = v / np.linalg.norm(v)

        # Codebooks restringidos (por rol)
        self.codebooks = {r: {s: self.sym[s] for s in syms}
                          for r, syms in self.codebooks_by_role.items()}

        # Repartición de roles por bloque (j % K)
        self.br = {}
        for j, rname in enumerate(self.role_names):
            b = j % self.K
            self.br.setdefault(b, []).append((j, rname))

        # Vectores-rol por bloque
        self.roles = []
        for b in range(self.K):
            blk_roles = []
            for _ in range(len(self.br[b])):
                v = self.np_rng.randn(self.BLK)
                blk_roles.append(v / np.linalg.norm(v))
            self.roles.append(blk_roles)

        # Gram por bloque (si hay >1 rol) — sobre codevectors DEL BLOQUE
        self.np_M = {}
        self.np_Minv = {}
        self.np_pinv = {}
        self.block_cb_names = {}
        for b in range(self.K):
            block_roles = self.br[b]
            if len(block_roles) <= 1:
                continue
            ck_names = []
            for _, rname in block_roles:
                ck_names += list(self.codebooks[rname].keys())
            # Evitar duplicados (símbolos que aparecen en dos roles del bloque)
            ck_unique = sorted(set(ck_names))
            self.block_cb_names[b] = ck_unique
            Mm = np.array([[np.dot(self.sym[a], self.sym[c])
                            for c in ck_unique] for a in ck_unique])
            self.np_M[b] = Mm
            try:
                self.np_Minv[b] = np.linalg.inv(Mm)
            except np.linalg.LinAlgError:
                self.np_Minv[b] = None
            self.np_pinv[b] = np.linalg.pinv(Mm, rcond=1e-10)

        # Cache arrays para cleanup rápido
        self.cb_names = {r: list(cb) for r, cb in self.codebooks.items()}
        self.cb_vec = {r: np.array([self.sym[s] for s in self.cb_names[r]])
                       for r in self.role_names}

    # --------------------------------------------------------- encode
    def encode_fact(self, fact):
        BLK = self.BLK
        segs = [None] * self.K
        for b, roles in self.br.items():
            if len(roles) == 1:
                j, rname = roles[0]
                segs[b] = self.sym[fact[rname]]
            else:
                acc = np.zeros(BLK)
                for idx, (j, rname) in enumerate(roles):
                    bound = hrr_bind(self.roles[b][idx], self.sym[fact[rname]])
                    acc += bound
                segs[b] = acc
        return np.concatenate(segs)

    # --------------------------------------------------------- cleanup
    def cleanup(self, v, rname, method="argmax"):
        CB = self.cb_vec[rname]
        n = np.linalg.norm(v)
        if n == 0:
            return self.cb_names[rname][0]
        sims = (CB @ v) / (np.linalg.norm(CB, axis=1) * n)
        if method == "argmax":
            return self.cb_names[rname][int(np.argmax(sims))]
        elif method == "softmax":
            t = 1.0
            p = np.exp(sims / t); p = p / p.sum()
            return self.cb_names[rname][int(self.np_rng.choice(len(p), p=p))]
        else:  # top_k
            k = min(3, len(sims))
            top_idx = np.argpartition(sims, -k)[-k:]
            top_vec = np.mean([self.sym[self.cb_names[rname][i]] for i in top_idx], axis=0)
            top_vec = top_vec / np.linalg.norm(top_vec)
            sims2 = (CB @ top_vec) / (np.linalg.norm(CB, axis=1) * np.linalg.norm(top_vec))
            return self.cb_names[rname][int(np.argmax(sims2))]

    # --------------------------------------------------------- decode (V3 exacto)
    def _decode_multi(self, seg, b, roles, mode, T, cleanup_method):
        R = np.array(self.roles[b])
        r = len(roles)
        if mode == "gram":
            Minv = self.np_Minv.get(b)
        elif mode == "pinv":
            Minv = self.np_pinv.get(b)
        else:
            Minv = None

        est = []
        for _ in range(r):
            v = self.np_rng.randn(self.BLK); est.append(v / np.linalg.norm(v))

        for _ in range(T):
            new = [None] * r
            for idx in range(r):
                others = seg.copy()
                for o in range(r):
                    if o != idx:
                        others = others - hrr_bind(R[o], est[o])
                fj = hrr_unbind(others, R[idx])

                if mode == "gradient":
                    # gradiente seguido por proyección cleanup
                    residual = seg - sum(hrr_bind(R[o], est[o]) for o in range(r))
                    grad = hrr_unbind(residual, R[idx])
                    updated = est[idx] + 0.1 * grad
                    nrm = np.linalg.norm(updated)
                    if nrm > 0: updated = updated / nrm
                    name = self.cleanup(updated, roles[idx][1], cleanup_method)
                    new[idx] = self.sym[name]
                    continue

                # APLICAR Minv (idéntico a V3): solo si dims coinciden
                # Esto reproduce el "no-op" cuando Minv es n_cv×n_cv pero BLK != n_cv
                if Minv is not None and Minv.shape[0] == self.BLK:
                    fj = Minv @ fj

                name = self.cleanup(fj, roles[idx][1], cleanup_method)
                new[idx] = self.sym[name]
            est = new

        return {rname: self.cleanup(est[i], rname, cleanup_method)
                for i, (j, rname) in enumerate(roles)}

    def decode_fact(self, c, T=50, mode="pure", cleanup_method="argmax"):
        BLK = self.BLK
        out = {}
        for b in range(self.K):
            seg = c[b * BLK:(b + 1) * BLK]
            roles = self.br[b]
            if len(roles) == 1:
                _, rname = roles[0]
                out[rname] = self.cleanup(seg, rname, cleanup_method)
            else:
                out.update(self._decode_multi(seg, b, roles, mode, T, cleanup_method))
        return out

    # --------------------------------------------------------- rho efectivo
    def rho_per_block(self):
        rho = {}
        for b in range(self.K):
            if b in self.block_cb_names:
                rho[b] = len(self.block_cb_names[b]) / self.BLK
            else:
                rho[b] = 1.0 / self.BLK  # 1 solo codevector efectivo en bloque de 1 rol
        return rho

# ================================================================ Fact & accuracy
def make_fact(rng, role_config=CFG6):
    return {r: rng.choice(role_config["codebooks"][r]) for r in role_config["names"]}

def accuracy(dec, fact):
    return sum(1 for r in fact if dec.get(r) == fact[r]) / len(fact)

# ================================================================ Main
def main():
    t0 = time.time()
    rng = random.Random(42)

    # Casos V3 (F2/H2-style): cubren rho < 1, = 1, > 1
    cases = [
        ("n=6 K=1 N=128 (rho=0.75)", 6, 1, 128),
        ("n=6 K=2 N=128 (rho=0.75)", 6, 2, 128),
        ("n=6 K=3 N=120 (rho=0.80)", 6, 3, 120),
        ("n=6 K=3 N=96  (rho=1.00, CUADRADO)", 6, 3, 96),
        ("n=6 K=3 N=72  (rho=1.33)", 6, 3, 72),
        ("n=6 K=2 N=64  (rho=1.50)", 6, 2, 64),
        ("n=3 K=3 N=126 (control)", 3, 3, 126),
    ]

    modes = ["gram", "pure", "pinv", "gradient"]
    T_list = [25, 100]
    cleanups = ["argmax", "softmax", "top_k"]

    print("=" * 78)
    print("EXP 8 v3: TAXONOMÍA CON DECODERS REALES (arquitectura V3)")
    print(f"  Casos: {len(cases)}  ×  modos: {len(modes)}  ×  T: {len(T_list)}  ×  cleanup: {len(cleanups)}")
    print(f"  Hechos por punto: 10")
    total_pts = len(cases) * len(modes) * len(T_list) * len(cleanups)
    print(f"  Total puntos: {total_pts}")
    print("=" * 78)

    results = []
    done = 0
    for case_label, n, K, N in cases:
        for mode in modes:
            for T in T_list:
                for cleanup in cleanups:
                    done += 1
                    bundle = BundleV3(7, K, N)
                    rho = bundle.rho_per_block()
                    accs = []
                    for _ in range(10):
                        f = make_fact(rng)
                        c = bundle.encode_fact(f)
                        dec = bundle.decode_fact(c, T=T, mode=mode, cleanup_method=cleanup)
                        accs.append(accuracy(dec, f))
                    m = float(np.mean(accs)); s = float(np.std(accs))
                    results.append({
                        "case": case_label, "n": n, "K": K, "N": N,
                        "BLK": bundle.BLK,
                        "rho_blocks": {str(k): round(v, 3) for k, v in rho.items()},
                        "mode": mode, "T": T, "cleanup": cleanup,
                        "accuracy_mean": m, "accuracy_std": s,
                    })
                    if done % 24 == 0 or done == total_pts:
                        pct = 100 * done / total_pts
                        print(f"  [{pct:5.1f}%] {case_label} | {mode} T={T} {cleanup}: {m:.3f}")

    # Guardar
    out_path = Path(__file__).parent.parent / "data" / "observer_taxonomy_v3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado JSON: {out_path}")

    # ---- Figura: grid de subplots por caso, accuracy por modo
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    axes = axes.ravel()
    for i, (case_label, n, K, N) in enumerate(cases):
        ax = axes[i]
        sub = [r for r in results if r["case"] == case_label]
        width = 0.2
        # solo argmax para la figura (la más informativa)
        sub_a = [r for r in sub if r["cleanup"] == "argmax"]
        for j, mode in enumerate(modes):
            subsm = [r for r in sub_a if r["mode"] == mode and r["T"] == 100]
            if not subsm: continue
            ax.bar(j, subsm[0]["accuracy_mean"],
                   yerr=subsm[0]["accuracy_std"], width=0.7,
                   label=mode, alpha=0.85)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=20, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", color="k", alpha=0.3)
        ax.set_title(case_label.split("(")[-1].rstrip(")"), fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    axes[0].set_ylabel("Accuracy"); axes[len(cases)//2].set_ylabel("Accuracy")
    fig.suptitle("Taxonomía real: modo × ρ (T=100, cleanup=argmax)", fontsize=13)
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_observer_taxonomy_v3.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")

    # ---- Resumen texto
    lines = ["=" * 78, "EXP 8 v3 COMPLETO"]
    for case_label, n, K, N in cases:
        lines.append(f"\n--- {case_label} ---")
        for mode in modes:
            sub = [r for r in results
                   if r["case"] == case_label and r["mode"] == mode
                   and r["T"] == 100 and r["cleanup"] == "argmax"]
            if sub:
                lines.append(f"  {mode:>8} T=100 argmax: acc={sub[0]['accuracy_mean']:.3f} ±{sub[0]['accuracy_std']:.3f}")
    lines.append(f"\nTiempo total: {time.time()-t0:.1f}s")
    text = "\n".join(lines)
    log_path = Path(__file__).parent.parent / "data" / "out_V8_v3.txt"
    with open(log_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nLog: {log_path}")

if __name__ == "__main__":
    main()
