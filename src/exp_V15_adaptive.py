#!/usr/bin/env python3
"""
Exp 15: cross-adaptive decoder testing.

La ley rho dice: el decoder optimo depende del estado del sistema.
Si kappa(M) es bajo, pinv funciona. Si kappa es alto, pure funciona.
Si kappa es critico (~1e4), gram falla y pure es la unica opcion.

Implementamos un router: medir kappa (estimada) y elegir decoder.
El "observador adaptivo" es el que lee el contexto y llama al experto.
"""
import numpy as np
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy

def adaptive_decode(bundle, bundle_obj, T=50):
    """Router: estima kappa del bloque y elige decoder en consecuencia."""
    K = bundle_obj.K
    out = {}
    for b in range(K):
        seg = bundle[b * bundle_obj.BLK:(b + 1) * bundle_obj.BLK]
        roles = bundle_obj.br[b]
        if len(roles) == 1:
            rn = roles[0][1]
            CB = bundle_obj.cb_vec[bundle_obj.cb_names.index(rn)] if hasattr(bundle_obj, 'cb_vec') else None
            # fallback simple
            out[rn] = ("SYM", bundle_obj.cleanup(seg, rn, "argmax"))
            continue
        # Estimar kappa del bloque antes de decodificar
        M = bundle_obj.np_M.get(b)
        if M is not None:
            w = np.linalg.eigvalsh(M)
            lam_max = np.abs(w).max()
            lam_min = max(np.abs(w).min(), 1e-16)
            kappa = lam_max / lam_min
        else:
            kappa = 1.0

        if kappa < 1e5:      # bien condicionado: pinv es seguro
            mode = "pinv"
        else:                # kappa >= 1e5 (incluye zona critica 1e4-1e16): pure
            mode = "pure"

        # decode con el modo elegido
        out.update(bundle_obj._decode_multi(seg, b, roles, mode, T, "argmax"))
    return out

print("EXP 15: cross-adaptive decoder")
rng = random.Random(42)
results = []
for N, BLK in [(96,32), (96,32), (128,64), (128,64)]:
    pass
# evaluar en el grid estandar
cases = [
    ("rho=0.75", 1, 128),
    ("rho=1.00", 3, 96),
    ("rho=1.33", 3, 72),
    ("rho=1.50", 2, 64),
]
for label, K, N in cases:
    accs_adap = []
    accs_pure = []
    accs_pinv = []
    accs_gram = []
    for _ in range(15):
        b = BundleV3(7, K, N)
        f = make_fact(rng)
        c = b.encode_fact(f)
        try: accs_adap.append(accuracy(adaptive_decode(c, b), f))
        except: pass
        accs_pure.append(accuracy(b.decode_fact(c, T=50, mode="pure"), f))
        accs_gram.append(accuracy(b.decode_fact(c, T=50, mode="gram"), f))
        accs_pinv.append(accuracy(b.decode_fact(c, T=50, mode="pinv"), f))
    print(f"{label}: adap={np.mean(accs_adap):.3f} gram={np.mean(accs_gram):.3f} "
          f"pinv={np.mean(accs_pinv):.3f} pure={np.mean(accs_pure):.3f}")
    results.append({"case": label, "adaptive": float(np.mean(accs_adap)),
                    "gram": float(np.mean(accs_gram)), "pinv": float(np.mean(accs_pinv)),
                    "pure": float(np.mean(accs_pure))})

out = Path(__file__).parent.parent / "data" / "exp15_cross_adaptive.json"
out.write_text(__import__('json').dumps(results, indent=1))
print(f"\nGuardado: {out}")
