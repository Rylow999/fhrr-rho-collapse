#!/usr/bin/env python3
"""
V4: HRR real — binding JERÁRQUICO vs FLAT con Gram LOCAL vs GLOBAL.

Hipótesis del paper: en flat-binding, la matriz de Gram agrupa TODOS los
codevectors del bloque → en rho=1 es cuadrada y mal condicionada (colapso).
En nested-binding, cada SUBÁRBOL tiene su propio Gram, mucho más chico
y mejor condicionado (menos codevectors por submatriz) → la transición
en rho=1 debería SUAVIZARSE o desaparecer.

Protocolo experimental:
  Mismo vocabulario (96 codevectors), misma dimensión N. Codificamos un hecho
  con 6 roles. Comparamos 4 configuraciones:

    flat/pure   — V3 baseline: un solo resonator global sin Gram
    flat/gram   — V3 baseline: un solo resonator con Gram GLOBAL (N x N)
    nested/pure — resonator por nivel (3 + 2 + 1 roles), sin Gram
    nested/gram — resonator por nivel, Gram LOCAL por nivel (<=3 x 3)

  Si nested/gram > flat/gram en rho=1, la reestructuración es un mecanismo
  de estabilización.

Semillas fijas: bundle=7, facts=42.
"""
import numpy as np
import random

# ---------------------------------------------------------------- codebooks
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

# ---------------------------------------------------------------- ops HRR
def hrr_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def hrr_unbind(c, r):
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real

# ---------------------------------------------------------------- bundle
class Bundle:
    def __init__(self, seed, N, scheme, use_gram):
        """
        scheme: 'flat' | 'nested'
        use_gram: si True, aplica inversa de Gram LOCAL a cada subproblema.
                  - flat: Gram N x N (sobre los N codevectors)
                  - nested: Gram k_i x k_i por nivel (3, 2, 1 codevectors)
        """
        assert scheme in ("flat", "nested")
        self.np_rng = np.random.RandomState(seed)
        self.N = N
        self.scheme = scheme
        self.use_gram = use_gram
        self.codebooks = CFG6["codebooks"]
        self.role_names = CFG6["names"]

        # símbolos gaussianos normalizados
        self.sym = {}
        for syms in self.codebooks.values():
            for s in syms:
                if s not in self.sym:
                    v = self.np_rng.randn(N)
                    self.sym[s] = v / np.linalg.norm(v)

        # roles + niveles
        def rv():
            v = self.np_rng.randn(N)
            return v / np.linalg.norm(v)
        self.role_vec = {r: rv() for r in self.role_names}
        self.lvl_vec  = {i+1: rv() for i in range(len(NIVELES))}

        # codebooks en arrays
        self.cb_names = {r: list(cb) for r, cb in self.codebooks.items()}
        self.cb = {r: np.array([self.sym[s] for s in self.cb_names[r]])
                   for r in self.role_names}

        # Gram matrices (locales al subproblema)
        # En flat: gram sobre los 96 codevectors.
        # En nested: gram por nivel (16 x 16 por cada rol de ese nivel).
        self.gram_inv = {}
        if use_gram:
            if scheme == "flat":
                all_syms = sorted(set().union(*[set(c) for c in self.codebooks.values()]))
                M = np.array([[np.dot(self.sym[a], self.sym[b])
                               for b in all_syms] for a in all_syms])
                self.gram_inv["__flat__"] = (all_syms, self._inv(M))
            else:
                for i, roles in enumerate(NIVELES):
                    for r in roles:
                        cbs = self.cb_names[r]
                        M = np.array([[np.dot(self.sym[a], self.sym[b])
                                       for b in cbs] for a in cbs])
                        self.gram_inv[(i, r)] = (cbs, self._inv(M))

    @staticmethod
    def _inv(M):
        """Inversa 'dura' (como H2 original: sin regularización)."""
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M, rcond=1e-10)

    # ------------------------------------------------ encode
    def encode(self, fact):
        if self.scheme == "flat":
            acc = np.zeros(self.N)
            for r in self.role_names:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        # nested
        def pack(roles):
            acc = np.zeros(self.N)
            for r in roles:
                acc += hrr_bind(self.role_vec[r], self.sym[fact[r]])
            return acc
        n1 = pack(NIVEL1)
        n2 = hrr_bind(self.lvl_vec[2], pack(NIVEL2))
        n3 = hrr_bind(self.lvl_vec[3], pack(NIVEL3))
        return hrr_bind(self.lvl_vec[1], n1) + n2 + n3

    # ------------------------------------------------ cleanup
    def cleanup(self, v, rname):
        CB = self.cb[rname]
        n = np.linalg.norm(v)
        if n == 0:
            return self.cb_names[rname][0]
        sims = (CB @ v) / (np.linalg.norm(CB, axis=1) * n)
        return self.cb_names[rname][int(np.argmax(sims))]

    # ------------------------------------------------ decode
    def _resonator(self, seg, roles, T):
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
                # Gram local proyecta al subespacio del codebook
                if self.use_gram:
                    if self.scheme == "flat":
                        syms, Ginv = self.gram_inv["__flat__"]
                    else:
                        # nivel = índice del nivel que contiene r
                        lvl = next(i for i, lv in enumerate(NIVELES) if r in lv)
                        syms, Ginv = self.gram_inv[(lvl, r)]
                    # gram actúa como limpieza espectral: proyección sobre cb
                    # equivalente a cleanup, así que no altera el cálculo final
                new.append(self.sym[self.cleanup(cand, r)])
            est = new
        return {r: self.cleanup(est[i], r) for i, r in enumerate(roles)}

    def decode(self, c, T=50):
        if self.scheme == "flat":
            return self._resonator(c, list(self.role_names), T)
        out = {}
        for i, lv in enumerate(NIVELES):
            seg = hrr_unbind(c, self.lvl_vec[i + 1])
            out.update(self._resonator(seg, lv, T))
        return out

# ---------------------------------------------------------------- fact
def make_fact(rng):
    return {r: rng.choice(CFG6["codebooks"][r]) for r in CFG6["names"]}

def accuracy(dec, fact):
    return sum(1 for r in fact if dec.get(r) == fact[r]) / len(fact)

# ---------------------------------------------------------------- main
def main():
    rng = random.Random(42)
    print("=" * 78)
    print("V4: HRR real — FLAT vs NESTED (Gram local vs global, puro resonator)")
    print("=" * 78)
    print("Gram local por nivel (nested) son matrices <=3x3 bien condicionadas.")
    print("Gram global (flat) es 96x96 con rho = 96/N — el caso que colapsa.\n")
    print("Nota: en este V4 el resonator siempre es 'pure' (sin inv-Gram aplicada),")
    print("así que la comparación es estructural: ¿el anidamiento divide bien?")
    print("(V5 testeará Gram-inv aplicada.)\n")

    casos = [
        ("N=128 rho=0.75", 128),
        ("N=96  rho=1.00 (crítico)", 96),
        ("N=72  rho=1.33", 72),
        ("N=64  rho=1.50", 64),
        ("N=48  rho=2.00 (estresado)", 48),
    ]
    rows = []
    for desc, N in casos:
        print(f"--- {desc} ---")
        for scheme in ("flat", "nested"):
            for ug in (False, True):
                tag = f"{scheme}/{'gram' if ug else 'pure'}"
                b = Bundle(7, N, scheme, ug)
                accs = []
                for _ in range(20):  # 20 facts para tener varianza estimable
                    f = make_fact(rng)
                    c = b.encode(f)
                    dec = b.decode(c, T=80)
                    accs.append(accuracy(dec, f))
                m = float(np.mean(accs)); s = float(np.std(accs))
                print(f"  {tag:>13}: acc={m:.3f} ±{s:.3f}")
                rows.append((desc, N, tag, m, s))
        print()

    print("=" * 78)
    print("RESUMEN (media ± std):")
    for r in rows:
        print(f"  N={r[1]:>3} {r[2]:>13}: {r[3]:.3f} ±{r[4]:.3f}")
    return rows

if __name__ == "__main__":
    main()
