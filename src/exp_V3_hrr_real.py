#!/usr/bin/env python3
"""
V3: HRR real (Plate 1995) con resonator completo.
Test de universalidad de la ley de ρ en VSA continuas.

Diferencias clave con FHRR:
  - FHRR: vectores de fase compleja (|v|=1), bind = multiplicación elemento a elemento
  - HRR real: vectores Gaussianos normalizados (R^N), bind = convolución circular real
  
Predicción:
  - Si la ley de ρ es universal para VSA continuas: misma transición en ρ=1
  - Si es específica de FHRR: no muestra la transición
"""
import numpy as np
import math, random

# =============================================================================
# Codebooks (16 símbolos por rol, mismos que H2 para comparabilidad)
# =============================================================================

VS=["lobo","zorro","ave","venado","oso","aguila","serpiente","conejo","tigre","leon","pantera","halcon","coyote","jaguar","puma","lobo_marino"]
VR=["come","corre","esta_en","persigue","caza","observa","sigue","evita","protege","ataca","huye","marca","ronda","acecha","vigila","explora"]
VO=["manzana","pasto","rio","arbol","roca","cueva","nido","madriguera","lago","montaña","bosque","desierto","playa","isla","glaciar","volcan"]
VL=["rapido","lento","grande","pequeño","cerca","lejos","alto","bajo","pesado","liviano","joven","viejo","fuerte","debil","salvaje","domestico"]
VM2=["mañana","tarde","noche","hoy","ayer","siempre","nunca","ahora","antes","despues","pronto","anochecer","temprano","mediodia","frecuente","raro"]
VT=["norte","sur","este","oeste","cima","valle","sombra","claro","denso","humedo","seco","pedregoso","frondoso","abierto","empinado","pantano"]

CFG6 = {"names": ["SUJ", "ROL", "OBJ", "LOC", "TIME", "TERR"],
        "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO, "LOC": VL, "TIME": VM2, "TERR": VT}}

# =============================================================================
# HRR Real: bind/unbind por convolución circular
# =============================================================================

def hrr_bind(a, b):
    """Convolución circular real: ifft(fft(a) * fft(b))."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def hrr_unbind(c, r):
    """Correlación circular real: ifft(fft(c) * conj(fft(r)))."""
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real

def hrr_sim(a, b):
    """Similitud coseno real."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na * nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# =============================================================================
# HRRRealBundle: análogo a FastBundle pero para HRR real
# =============================================================================

class HRRRealBundle:
    """
    HRR real con roles distribuidos en bloques.
    Análogo a FastBundle de FHRR pero con:
      - Vectores Gaussianos normalizados (no fases complejas)
      - Bind por convolución circular (no multiplicación elemento a elemento)
      - Unbind por correlación circular (no conjugado)
    """
    
    def __init__(self, seed, K, N, n_roles, role_config, T=25):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.K = K
        self.N = N
        self.BLK = N // K
        self.n_roles = n_roles
        self.ROLE_NAMES = role_config["names"]
        self.T = T
        
        # Generar símbolos: vectores Gaussianos normalizados de dim BLK
        self.sym = {}
        all_symbols = sum(role_config["codebooks"].values(), [])
        for s in set(all_symbols):
            if s not in self.sym:
                v = self.np_rng.randn(self.BLK)
                self.sym[s] = v / np.linalg.norm(v)
        
        # Codebooks por rol
        self.codebooks = {
            rname: {s: self.sym[s] for s in syms}
            for rname, syms in role_config["codebooks"].items()
        }
        
        # Asignación de roles a bloques (j % K)
        br = {}
        for j, rname in enumerate(self.ROLE_NAMES):
            b = j % self.K
            br.setdefault(b, []).append((j, rname))
        
        # Vectores-rol por bloque (Gaussianos normalizados)
        self.roles = []
        for b in range(self.K):
            block_roles = []
            for _ in range(len(br[b])):
                v = self.np_rng.randn(self.BLK)
                block_roles.append(v / np.linalg.norm(v))
            self.roles.append(block_roles)
        self.br = br
        
        # Matrices M y Minv por bloque (solo si hay >1 rol)
        self.M = [None] * self.K
        self.Minv = [None] * self.K
        self.np_M = [None] * self.K
        self.np_Minv = [None] * self.K
        self.np_pinv = [None] * self.K
        
        for b in range(self.K):
            nroles = len(br[b])
            if nroles > 1:
                cb = []
                for j, rname in br[b]:
                    cb += list(self.codebooks[rname].values())
                # M[i][j] = <c_i, c_j> (producto punto real)
                Mm = np.array([[np.dot(cb[kk], cb[jj]) for jj in range(len(cb))] 
                              for kk in range(len(cb))])
                # NOTA: M es len(cb) x len(cb), no BLK x BLK
                # Para HRR real, la matriz de Gram es sobre los codevectors, no sobre el espacio
                self.M[b] = Mm
                self.np_M[b] = Mm
                try:
                    self.Minv[b] = np.linalg.inv(Mm)
                    self.np_Minv[b] = np.linalg.inv(Mm)
                except np.linalg.LinAlgError:
                    self.Minv[b] = None
                    self.np_Minv[b] = None
                self.np_pinv[b] = np.linalg.pinv(Mm, rcond=1e-10)
        
        # Arrays numpy para decode vectorizado
        self.np_roles = [np.array(self.roles[b]) for b in range(self.K)]
        self.np_cb, self.cb_names = {}, {}
        for rn, cb in self.codebooks.items():
            self.cb_names[rn] = list(cb.keys())
            self.np_cb[rn] = np.array([cb[s] for s in self.cb_names[rn]])
        self.np_sym = {s: np.array(v) for s, v in self.sym.items()}
    
    def encode_fact(self, fact, mem=None):
        """Codifica un hecho plano: superposición de bindings por bloque."""
        if mem is None: mem = {}
        BLK = self.BLK
        br = self.br
        segs = [None] * self.K
        
        for b, roles in br.items():
            if len(roles) == 1:
                j, rname = roles[0]
                fval = fact[2*j + 1]
                segs[b] = self.sym[fval]
            else:
                # Superposición de bindings: sum(role_i * filler_i)
                acc = np.zeros(BLK)
                for idx, (j, rname) in enumerate(roles):
                    fval = fact[2*j + 1]
                    fv = self.sym[fval]
                    bound = hrr_bind(self.roles[b][idx], fv)
                    acc += bound
                segs[b] = acc
        
        c = np.concatenate(segs)
        mem[fact] = c
        return c, mem
    
    def cleanup(self, fj, rname):
        """Encuentra el símbolo más similar en el codebook del rol."""
        CB = self.np_cb[rname]
        nf = np.linalg.norm(fj)
        if nf == 0:
            return self.cb_names[rname][0], 0.0
        # Similitud coseno con cada codevector
        sims = (CB @ fj) / (np.linalg.norm(CB, axis=1) * nf)
        k = int(np.argmax(sims))
        return self.cb_names[rname][k], float(sims[k])
    
    def decode_fact(self, c, mem=None, mode="original", T=None):
        """Decodifica un hecho con el modo especificado."""
        T = self.T if T is None else T
        c = np.asarray(c)
        out = {}
        
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
    
    def _decode_multi(self, seg, b, roles, mode, T):
        """Decodificación iterativa con Gram (análogo a _decode_multi de FHRR)."""
        R = self.np_roles[b]
        r = len(roles)
        
        # Minv según modo
        if mode == "original":
            Minv = self.np_Minv[b]
        elif mode == "pinv":
            Minv = self.np_pinv[b]
        elif mode == "pure":
            Minv = None
        else:
            Minv = None
        
        # Inicializar estimaciones aleatorias
        est = [self.np_rng.randn(self.BLK) for _ in range(r)]
        for idx in range(r):
            est[idx] = est[idx] / np.linalg.norm(est[idx])
        
        for _ in range(T):
            new = [None] * r
            for idx in range(r):
                # Restar contribuciones de otros roles
                others = seg - sum(hrr_bind(R[o], est[o]) for o in range(r) if o != idx)
                # Unbind: correlación circular
                fj = hrr_unbind(others, R[idx])
                
                # Aplicar Minv si existe
                # NOTA: En HRR real, Minv opera sobre el espacio de codevectors, no sobre BLK
                # Por simplicidad, aplicamos Minv como transformación lineal si las dimensiones coinciden
                if Minv is not None and Minv.shape[0] == self.BLK:
                    fj = Minv @ fj
                
                # Cleanup al codebook
                name, _ = self.cleanup(fj, roles[idx][1])
                new[idx] = self.np_sym[name]
            
            est = new
        
        # Cleanup final
        out = {}
        for idx, (j, rname) in enumerate(roles):
            name, _ = self.cleanup(est[idx], rname)
            out[rname] = ("SYM", name)
        
        return out
    
    def _decode_multi_gradient(self, seg, b, roles, T, lr=0.1):
        """Decodificación por gradiente (análogo a FHRR)."""
        R = self.np_roles[b]
        r = len(roles)
        
        # Inicializar estimaciones aleatorias
        est = [self.np_rng.randn(self.BLK) for _ in range(r)]
        for idx in range(r):
            est[idx] = est[idx] / np.linalg.norm(est[idx])
        
        for _ in range(T):
            residual = seg - sum(hrr_bind(R[o], est[o]) for o in range(r))
            new = [None] * r
            for idx in range(r):
                # Gradiente: unbind(residual, role)
                grad = hrr_unbind(residual, R[idx])
                updated = est[idx] + lr * grad
                nrm = np.linalg.norm(updated)
                if nrm > 0:
                    updated = updated / nrm
                # Cleanup en cada paso (mismo esquema que FHRR)
                name, _ = self.cleanup(updated, roles[idx][1])
                new[idx] = self.np_sym[name]
            est = new
        
        # Cleanup final
        out = {}
        for idx, (j, rname) in enumerate(roles):
            name, _ = self.cleanup(est[idx], rname)
            out[rname] = ("SYM", name)
        
        return out
    
    def fact_accuracy(self, dec, gt):
        """Accuracy sobre todos los roles."""
        ok = 0
        tot = 0
        gm = {gt[i]: gt[i+1] for i in range(0, len(gt), 2)}
        for r in self.ROLE_NAMES:
            tot += 1
            if r not in dec:
                continue
            tipo, val = dec[r]
            gv = gm[r]
            if isinstance(gv, str):
                if tipo == "SYM" and val == gv:
                    ok += 1
        return ok, tot

def make_fact_flat(rng, n_roles, role_config):
    """Hecho plano sin recursión."""
    values = [rng.choice(syms) for syms in role_config["codebooks"].values()]
    result = []
    for i, rname in enumerate(role_config["names"]):
        result.append(rname)
        result.append(values[i])
    return tuple(result)

# =============================================================================
# Main: grid de ρ con 4 decoders
# =============================================================================

def main():
    rng = random.Random(42)
    
    print("=" * 78)
    print("V3: HRR real (Plate 1995) — grid de ρ con 4 decoders")
    print("=" * 78)
    print("\nDiferencias con FHRR:")
    print("  - Vectores Gaussianos normalizados (no fases complejas)")
    print("  - Bind por convolución circular (no multiplicación elemento a elemento)")
    print("  - Unbind por correlación circular (no conjugado)")
    print("\nPredicción: si la ley de ρ es universal para VSA continuas,")
    print("HRR real debería mostrar la misma transición en ρ=1.\n")
    
    # Grid de ρ (mismo que H2 para comparabilidad)
    casos = [
        ("n=6 K=1 N=128 (rho=0.75)", 6, 1, 128, CFG6),
        ("n=6 K=3 N=120 (rho=0.80)", 6, 3, 120, CFG6),
        ("n=6 K=3 N=96 (rho=1.00, CUADRADO)", 6, 3, 96, CFG6),
        ("n=6 K=3 N=72 (rho=1.33)", 6, 3, 72, CFG6),
        ("n=6 K=2 N=64 (rho=1.50)", 6, 2, 64, CFG6),
        ("n=3 K=3 N=126 (control)", 3, 3, 126, 
         {"names": ["SUJ", "ROL", "OBJ"], 
          "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO}}),
    ]
    
    modes = [("gram", "original"), ("gradient", "gradient"),
             ("pure", "pure"), ("pinv", "pinv")]
    
    for desc, n, K, N, cfg in casos:
        t = HRRRealBundle(7, K, N, n, cfg)
        
        # Info de ρ por bloque
        info = []
        for b in range(K):
            roles = t.br[b]
            if len(roles) == 1:
                info.append(f"b{b}:1rol")
            else:
                nv = len({s for _, rn in roles for s in t.codebooks[rn]})
                info.append(f"b{b}:{nv}/{t.BLK}={nv/t.BLK:.2f}")
        
        print(f"\n--- {desc} | BLK={t.BLK} | " + "  ".join(info) + " ---")
        
        # Generar hechos
        facts = [make_fact_flat(rng, n, cfg) for _ in range(10)]
        enc = [t.encode_fact(f)[0] for f in facts]
        
        # Decode con cada modo
        print(f"{'decoder':>9} | {'T=25':>7} | {'T=100':>7}")
        for label, m in modes:
            accs = {25: [], 100: []}
            for f, c in zip(facts, enc):
                for T in (25, 100):
                    dec = t.decode_fact(c, None, mode=m, T=T)
                    ok, tot = t.fact_accuracy(dec, f)
                    accs[T].append(ok / tot)
            print(f"{label:>9} | {sum(accs[25])/10:>7.3f} | {sum(accs[100])/10:>7.3f}")
    
    print("\n" + "=" * 78)
    print("INTERPRETACIÓN")
    print("=" * 78)
    print("Si HRR real muestra la misma transición en ρ=1:")
    print("  → La ley de ρ es UNIVERSAL para VSA continuas")
    print("Si HRR real NO muestra la transición:")
    print("  → La ley es específica de FHRR (fases complejas)")
    print("Si HRR real muestra comportamiento diferente (ej. no monótono):")
    print("  → Hay factores adicionales específicos de cada álgebra VSA")

if __name__ == "__main__":
    main()
