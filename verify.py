#!/usr/bin/env python3
"""Verifica numeros del paper contra data/. Portable (relativo al script)."""
import json, sys
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
ok = fail = 0
def chk(name, claim, actual, tol):
    global ok, fail
    if abs(claim - actual) <= tol: ok += 1
    else: fail += 1; print(f"  FAIL {name}: claim {claim} vs data {actual:.6g}")

d18 = json.load(open(D / "exp18_summary.json"))
sq = {r["mode"]: r for r in d18["square"]}
for m, c in [("pure",0.985),("gram",0.160),("dual_same",0.985)]:
    chk(f"Exp18 {m}", c, sq[m]["mean"], 0.003)
chk("Exp18 paired", 0.824, d18["paired"]["dual_same_minus_gram"]["mean"], 0.003)
chk("Exp18 sign", 1.0, d18["paired"]["dual_same_minus_gram"]["frac_pos"], 0.001)
chk("Exp18 norm Minv", 3473, d18["norm_Minv_med"], 5)

d19 = json.load(open(D / "exp19_independent_replica.json"))
chk("Exp19 pure", 0.982, d19["pure"]["mean"], 0.002)
chk("Exp19 gram", 0.159, d19["gram"]["mean"], 0.002)
chk("Exp19 dual_same", 0.982, d19["dual_same"]["mean"], 0.002)

b = json.load(open(D / "exp18_bsc_summary.json"))
chk("BSC gram", 0.192, b["gram"]["mean"], 0.002)
chk("BSC dual_same", 0.995, b["dual_same"]["mean"], 0.002)

e20 = json.load(open(D / "exp20_ensembles_rho.json"))
chk("Exp20A gauss", 0.160, e20["ensembles"]["gauss"]["gram"], 0.005)
chk("Exp20A orthog", 0.986, e20["ensembles"]["orthogonal"]["gram"], 0.005)
sw = {r["rho"]: r for r in e20["rho_sweep"]}
chk("Exp20B rho=1 gram", 0.197, sw[1.0]["gram"], 0.01)
chk("Exp20B rho=1 dual", 0.973, sw[1.0]["dual_same"], 0.01)

e21 = json.load(open(D / "exp21_linear_amp.json"))["summary"]
chk("Exp21 amb", 362, e21["med_amb"], 10)
chk("Exp21 dual", 1.0, e21["med_dual"], 0.001)

e22 = json.load(open(D / "exp22_iteration_dynamics.json"))
chk("Exp22 gram e0", 6758, e22["gram"]["e0"], 200)
chk("Exp22 dual e0", 1.66, e22["dual_same"]["e0"], 0.05)

e23 = json.load(open(D / "exp23_operator_zoo.json"))
chk("Exp23 inv dual (well)", 1.0, e23["well_conditioned"]["per_operator"]["inv"]["gain_dual_med"], 0.001)

e25 = json.load(open(D / "exp25_contraejemplos.json"))
for k, r in e25.items():
    if k != "wide_n24_d48": chk(f"Exp25 {k} dual", 1.0, r["norm_dual"], 0.001)

e27 = json.load(open(D / "exp27_fhrr_duality.json"))
chk("Exp27 pure", 1.000, e27["pure"]["mean"], 0.001)
chk("Exp27 gram", 0.131, e27["gram"]["mean"], 0.01)
chk("Exp27 dual_same", 1.000, e27["dual_same"]["mean"], 0.001)

total = ok + fail
print(f"\n{ok}/{total} checks OK")
sys.exit(0 if fail == 0 else 1)
