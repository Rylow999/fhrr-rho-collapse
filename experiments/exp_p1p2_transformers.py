#!/usr/bin/env python3
# exp_p1_p2_transformers.py v2 — M3 corregido (dimensional, fiel al paper)
# ambient M^-1 @ bundle SOLO en punto cuadrado T=d (como el paper declara);
# dual C^T M^+ C @ bundle en todo T. Unbinding elementwise para product-binding.
import json
import math
import numpy as np

SEED = 2026
D = 64
N_ROLES_LIST = [8, 16, 32, 48, 64, 96, 128]
N_TRIALS = 300

def run_m3():
    print("== M3: unbinding AMBIENT vs DUAL (corregido) ==", flush=True)
    m3 = []
    for T in N_ROLES_LIST:
        accs = {"ambient": None, "dual": 0, "pure": 0}
        n_amb = 0
        amb_ok = 0
        for trial in range(N_TRIALS):
            r = np.random.default_rng(3000 + trial * 7 + T)
            C = r.normal(0, 1, (T, D))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            fillers = r.normal(0, 1, (T, D))
            bound = C * fillers
            bundle = bound.sum(axis=0)
            target = int(r.integers(0, T))
            M = C @ C.T          # (T,T) gram de roles
            # --- PURE: unbind directo con el rol y limpiar contra fillers
            est = bundle * C[target]
            accs["pure"] += int(np.argmax(est @ fillers.T) == target)
            # --- DUAL: proyeccion dual C^T M^+ C sobre el bundle
            Mp = np.linalg.pinv(M)
            dual_state = C.T @ Mp @ C @ bundle
            est_d = dual_state * C[target]
            accs["dual"] += int(np.argmax(est_d @ fillers.T) == target)
            # --- AMBIENT: M^-1 @ bundle (solo dimensional en cuadrado)
            if T == D:
                try:
                    Minv = np.linalg.inv(M)
                    amb_state = Minv @ bundle
                    est_a = amb_state * C[target]
                    amb_ok += int(np.argmax(est_a @ fillers.T) == target)
                    n_amb += 1
                except np.linalg.LinAlgError:
                    pass
        row = {"T": T, "pure": accs["pure"]/N_TRIALS, "dual": accs["dual"]/N_TRIALS,
               "ambient": (amb_ok/n_amb) if n_amb else None, "chance": 1/T,
               "n_ambient": n_amb}
        m3.append(row)
        amb_s = f"  ambient={row['ambient']:.3f}" if row["ambient"] is not None else "  ambient=N/A (no cuadrado)"
        print(f"  T={T:4d}: pure={row['pure']:.3f}  dual={row['dual']:.3f}{amb_s}  (chance={1/T:.3f})", flush=True)
    return m3

m3 = run_m3()

# verificacion extra en el cuadrado: lambda_min y el factor de blow-up
print("\n== Cuadrado T=d=64: blow-up 1/lmin = ? ==", flush=True)
r = np.random.default_rng(99)
C = r.normal(0, 1, (64, 64))
C /= np.linalg.norm(C, axis=1, keepdims=True)
M = C @ C.T
ev = np.linalg.eigvalsh(M)
print(f"  lmin={ev[0]:.4e}  1/lmin={1/ev[0]:.3e}  (el amplificador ambient)")

out = {"D": D, "M3_ambient_vs_dual": m3, "lmin_cuadrado": float(ev[0]),
       "blowup_cuadrado": float(1/ev[0]), "trials": N_TRIALS}
with open("data_p1p2.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nP1P2_DONE", flush=True)
