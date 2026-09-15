#!/usr/bin/env python3
"""
H2: gram vs gradient vs pure vs pinv + grid de rho (layout fijo n=6,
seis codebooks DISTINTOS definidos localmente: VM2 y VT, para no depender de
si ROLE_CONFIGS[6] en base_fhrr.py esta o no al dia -- ver NOTA en exp_F2.py).
Importa FastBundle de exp_F2 (mismo directorio).
"""
from base_fhrr import *
from exp_F2 import FastBundle
import numpy as np

VM2 = ["mañana", "tarde", "noche", "hoy", "ayer", "siempre", "nunca", "ahora",
       "antes", "despues", "pronto", "anochecer", "temprano", "mediodia",
       "frecuente", "raro"]                # 16 unicos
VT = ["norte", "sur", "este", "oeste", "cima", "valle", "sombra", "claro",
      "denso", "humedo", "seco", "pedregoso", "frondoso", "abierto",
      "empinado", "pantano"]               # 6to codebook

CFG6 = {"names": ["SUJ", "ROL", "OBJ", "LOC", "TIME", "TERR"],
        "codebooks": {"SUJ": VS, "ROL": VR, "OBJ": VO, "LOC": VL,
                      "TIME": VM2, "TERR": VT}}

def main():
    rng = random.Random(42)
    casos = [
        ("n=4 K=1 N=128", 4, 1, 128, ROLE_CONFIGS[4]),
        ("n=6 K=1 N=128", 6, 1, 128, CFG6),
        ("n=6 K=2 N=128", 6, 2, 128, CFG6),
        ("n=6 K=3 N=120", 6, 3, 120, CFG6),
        ("n=6 K=3 N=96 (CUADRADO, replica Prueba B)", 6, 3, 96, CFG6),
        ("n=6 K=3 N=72",  6, 3, 72,  CFG6),
        ("n=6 K=2 N=64",  6, 2, 64,  CFG6),
        ("n=3 K=3 N=126 (control)", 3, 3, 126, ROLE_CONFIGS[3]),
    ]
    modes = [("gram", "original"), ("gradient", "gradient"),
             ("pure", "pure"), ("pinv", "pinv")]

    print("=" * 78)
    print("H2: gram vs gradient vs pure vs pinv  +  grid de rho")
    print("=" * 78)

    for desc, n, K, N, cfg in casos:
        t = FastBundle(7, K, N, n, cfg)
        info = []
        for b in range(K):
            roles = t.br[b]
            if len(roles) == 1:
                info.append(f"b{b}:1rol")
            else:
                nv = len({s for _, rn in roles for s in t.codebooks[rn]})
                info.append(f"b{b}:{nv}/{t.BLK}={nv/t.BLK:.2f}")
        print(f"\n--- {desc} | BLK={t.BLK} | " + "  ".join(info) + " ---")

        facts = [make_fact_flat(rng, n, cfg) for _ in range(10)]
        enc = [t.encode_fact(f)[0] for f in facts]

        print(f"{'decoder':>9} | {'T=25':>7} | {'T=100':>7}")
        for label, m in modes:
            accs = {25: [], 100: []}
            for f, c in zip(facts, enc):
                for T in (25, 100):
                    dec = t.decode_fact(c, None, mode=m, T=T)
                    ok, tot = t.fact_accuracy(dec, f)
                    accs[T].append(ok / tot)
            print(f"{label:>9} | {sum(accs[25])/10:>7.3f} | {sum(accs[100])/10:>7.3f}")

    print("\nPredicho: gram colapsa para rho<=1 (basura si rho<1, dual degenerado")
    print("si rho=1); pure/gradient funcionan en todo el grid; pinv arregla rho<1")
    print("pero NO el caso cuadrado. Esa doble disociacion es el resultado clave.")

if __name__ == "__main__":
    main()
