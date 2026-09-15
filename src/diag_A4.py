#!/usr/bin/env python3
"""
Diagnóstico A4: verificación directa sobre run_decode_0059h.py.
Requiere run_decode_0059h.py en el mismo directorio (archivo original del
experimento, no incluido en el repo: contiene VS/VR/VO de 4 símbolos).
"""
import sys
import numpy as np
try:
    import run_decode_0059h as m
except ModuleNotFoundError:
    print("Este diagnostico requiere run_decode_0059h.py, el script original del")
    print("experimento que motivo esta auditoria. No esta incluido en este repo")
    print("(fue superado por base_fhrr.py / base_fhrr_corregida.py) y por lo tanto")
    print("este paso NO es reproducible desde un clon limpio.")
    print("La salida ya capturada de esa corrida esta preservada en")
    print("data/out_diag_A4.txt -- usar esa como evidencia archivada.")
    sys.exit(0)

print("=== A4: verificacion directa sobre run_decode_0059h.py real ===")
print("VS,VR,VO reales en 0059h (OJO: NO son los de 16 simbolos de base_fhrr.py):")
print("VS:", m.VS, "len=", len(m.VS))
print("VR:", m.VR, "len=", len(m.VR))
print("VO:", m.VO, "len=", len(m.VO))
print()
print(f"{'K':>2} | {'N':>4} | {'BLK':>4} | {'vec_dist':>8} | {'rho':>5} | {'rank(M0)':>8} | {'cond(M0)':>10}")

for K in [1, 2, 3]:
    for N in [64, 128, 192]:
        t = m.BlockBundle(7, K, N)
        BLK = t.BLK
        roles0 = t.br.get(0, [])
        if len(roles0) > 1 and t.Minv[0] is not None:
            # reconstruir M explicitamente para diagnosticar (0059h no guarda self.M, solo self.Minv)
            cb = []
            for j, rname in roles0:
                cb += list(t.codebooks[rname].values())
            Mm = [[sum(cb[kk][i]*cb[kk][jj].conjugate() for kk in range(len(cb))) for jj in range(BLK)] for i in range(BLK)]
            M = np.array(Mm)
            vec_dist = len(set(tuple(v) for v in cb))
            rank = np.linalg.matrix_rank(M)
            cond = np.linalg.cond(M)
            print(f"{K:>2} | {N:>4} | {BLK:>4} | {vec_dist:>8} | {vec_dist/BLK:>5.2f} | {rank:>8} | {cond:>10.3e}")
        else:
            print(f"{K:>2} | {N:>4} | {BLK:>4} | {'(K=nroles, sin M)':>8} |")
