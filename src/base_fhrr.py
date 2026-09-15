#!/usr/bin/env python3
"""
Base compartida: implementación FHRR basada en run_decode_0059h.py (Luciano, 2026)
Versión original (sin fixes de auditoría) — preservar reproducibilidad de F2/H2.
"""
import cmath, math, random

def rnd_phase(rng, dim):
    return [cmath.exp(1j*rng.random()*2*math.pi) for _ in range(dim)]

def bind(a,b):
    return [a[i]*b[i] for i in range(len(a))]

def unbind(c,r):
    return [c[i]*r[i].conjugate() for i in range(len(c))]

def vsub(a,b):
    return [a[i]-b[i] for i in range(len(a))]

def bundle(vecs):
    return [sum(v[i] for v in vecs) for i in range(len(vecs[0]))]

def sim(a,b):
    na=math.sqrt(sum(abs(x)**2 for x in a))
    nb=math.sqrt(sum(abs(x)**2 for x in b))
    if na*nb==0: return 0.0
    return sum((a[i]*b[i].conjugate()).real for i in range(len(a)))/(na*nb)

def mat_inv(M):
    n=len(M)
    A=[list(M[i])+[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv=max(range(col,n), key=lambda r: abs(A[r][col]))
        A[col],A[piv]=A[piv],A[col]
        pv=A[col][col]; A[col]=[x/pv for x in A[col]]
        for r in range(n):
            if r!=col:
                f=A[r][col]; A[r]=[A[r][k]-f*A[col][k] for k in range(2*n)]
    return [[A[i][n+j] for j in range(n)] for i in range(n)]

def mat_vec(M,v):
    return [sum(M[i][k]*v[k] for k in range(len(v))) for i in range(len(M))]

def proj_fhr(vec, dim):
    n=len(vec); out=[0j]*dim; seg=n/dim
    for k in range(dim):
        a=int(k*seg); b=int((k+1)*seg); b=max(b,a+1)
        re=sum(v.real for v in vec[a:b])/(b-a)
        im=sum(v.imag for v in vec[a:b])/(b-a)
        ang=cmath.phase(re+1j*im)
        out[k]=cmath.exp(1j*ang)
    return out

VS=["lobo","zorro","ave","venado","oso","aguila","serpiente","conejo","tigre","leon","pantera","halcon","coyote","jaguar","puma","lobo_marino"]
VR=["come","corre","esta_en","persigue","caza","observa","sigue","evita","protege","ataca","huye","marca","ronda","acecha","vigila","explora"]
VO=["manzana","pasto","rio","arbol","roca","cueva","nido","madriguera","lago","montaña","bosque","desierto","playa","isla","glaciar","volcan"]
VL=["rapido","lento","grande","pequeño","cerca","lejos","alto","bajo","pesado","liviano","joven","viejo","fuerte","debil","salvaje","domestico"]
VM=["mañana","tarde","noche","hoy","ayer","siempre","nunca","ahora","antes","despues","pronto","temprano","anochecer","medianoche","frecuente","raro"]
VL_TERR=["norte","sur","este","oeste","cima","valle","sombra","claro","denso","humedo","seco","pedregoso","frondoso","abierto","empinado","pantano"]

ROLE_CONFIGS = {
    3: {"names": ["SUJ","ROL","OBJ"], "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO}},
    4: {"names": ["SUJ","ROL","OBJ","LOC"], "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO, "LOC": VL}},
    6: {"names": ["SUJ","ROL","OBJ","LOC","TIME","TERR"], "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO, "LOC": VL, "TIME": VM, "TERR": VL_TERR}},
}

class BlockBundle:
    def __init__(self, seed, K, N, n_roles, role_config, T=25):
        self.rng=random.Random(seed); self.K=K; self.N=N; self.BLK=N//K
        self.n_roles=n_roles; self.ROLE_NAMES=role_config["names"]; self.T=T
        self.sym={}
        # FIX (auditoria): iterar sobre un set() de strings tiene orden dependiente
        # del hash de Python, que se randomiza por proceso (PYTHONHASHSEED) salvo que
        # se fije explicitamente. Eso hacia que cada corrida consumiera self.rng en un
        # orden distinto -> resultados numericos distintos entre corridas pese a la
        # seed fija, rompiendo la promesa de reproducibilidad bit-a-bit del README.
        # sorted() fuerza un orden determinista y estable entre corridas y maquinas.
        for s in sorted(set(sum(role_config["codebooks"].values(), []))):
            if s not in self.sym: self.sym[s]=rnd_phase(self.rng, self.BLK)
        self.codebooks={rname:{s:self.sym[s] for s in syms} for rname,syms in role_config["codebooks"].items()}
        br={}
        for j, rname in enumerate(self.ROLE_NAMES):
            b=j%self.K; br.setdefault(b,[]).append((j,rname))
        self.roles=[[rnd_phase(self.rng, self.BLK) for _ in range(len(br[b]))] for b in range(self.K)]
        self.br=br
        self.M=[None]*self.K; self.Minv=[None]*self.K
        for b in range(self.K):
            nroles=len(br[b])
            if nroles>1:
                cb=[]
                for j, rname in br[b]:
                    cb+=list(self.codebooks[rname].values())
                Mm=[[sum(cb[kk][i]*cb[kk][jj].conjugate() for kk in range(len(cb))) for jj in range(self.BLK)] for i in range(self.BLK)]
                self.M[b]=Mm
                try: self.Minv[b]=mat_inv(Mm)
                except: self.Minv[b]=None

    def encode_fact(self, fact, mem=None):
        if mem is None: mem={}
        BLK=self.BLK; br=self.br; segs=[None]*self.K
        for b, roles in br.items():
            if len(roles)==1:
                j,rname=roles[0]; fval=fact[2*j+1]
                segs[b]=self.sym[fval]
            else:
                parts=[]
                for idx,(j,rname) in enumerate(roles):
                    fval=fact[2*j+1]
                    fv=self.sym[fval]
                    parts.append(bind(self.roles[b][idx], fv))
                segs[b]=bundle(parts)
        c=[]
        for b in range(self.K): c+=segs[b]
        mem[fact]=c
        return c, mem

    def cleanup_sym(self, vec, rname):
        best=None; bd=-2
        for name,cw in self.codebooks[rname].items():
            d=sim(vec, cw)
            if d>bd: bd=d; best=name
        return best, bd

    def decode_block_multi(self, seg, b, roles, mem):
        T=self.T; est=[rnd_phase(self.rng, len(seg)) for _ in roles]
        for _ in range(T):
            new=[None]*len(roles)
            for idx,(j,rname) in enumerate(roles):
                others=vsub(seg, bundle([bind(self.roles[b][o], est[o]) for o in range(len(roles)) if o!=idx]))
                fj=unbind(others, self.roles[b][idx])
                if self.Minv[b] is not None:
                    try: fj=mat_vec(self.Minv[b], fj)
                    except: pass
                cw,_=self.cleanup_sym(fj, rname); new[idx]=self.sym[cw]
            est=new
        out={}
        for idx,(j,rname) in enumerate(roles):
            fj=est[idx]
            sym_name,sd=self.cleanup_sym(fj, rname)
            out[rname]=("SYM", sym_name)
        return out

    def decode_block_single(self, seg, j, rname, mem):
        sym_name,_=self.cleanup_sym(seg, rname)
        return ("SYM", sym_name)

    def decode_fact(self, c, mem):
        BLK=self.BLK; out={}
        for b, roles in self.br.items():
            seg=c[b*BLK:(b+1)*BLK]
            if len(roles)==1:
                j,rname=roles[0]
                out[rname]=self.decode_block_single(seg, j, rname, mem)
            else:
                out.update(self.decode_block_multi(seg, b, roles, mem))
        return out

    def fact_accuracy(self, dec, gt):
        ok=0; tot=0; gm={gt[i]:gt[i+1] for i in range(0,len(gt),2)}
        for r in self.ROLE_NAMES:
            tot+=1
            if r not in dec: continue
            tipo,val=dec[r]; gv=gm[r]
            if isinstance(gv,str):
                if tipo=="SYM" and val==gv: ok+=1
        return ok,tot

def make_fact_flat(rng, n_roles, role_config):
    values=[rng.choice(syms) for syms in role_config["codebooks"].values()]
    result=[]
    for i, rname in enumerate(role_config["names"]):
        result.append(rname); result.append(values[i])
    return tuple(result)

def make_fact(rng, depth, n_roles, role_config):
    if depth<=1: return make_fact_flat(rng, n_roles, role_config)
    result=[]
    values=[rng.choice(syms) for syms in role_config["codebooks"].values()]
    for i, rname in enumerate(role_config["names"]):
        result.append(rname)
        if i==n_roles-1:
            result.append(make_fact(rng, depth-1, n_roles, role_config))
        else:
            result.append(values[i])
    return tuple(result)
