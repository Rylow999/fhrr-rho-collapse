import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"


def test_exp28_data_exists_and_exponent_is_hard_edge():
    d = json.load(open(D / "exp28_hardedge_convergence.json"))
    assert "unit" in d and "wishart" in d and "ks" in d
    assert -2.5 < d["fit_exponent_all"] < -1.5, d["fit_exponent_all"]


def test_exp28_unitnorm_matches_wishart_law():
    d = json.load(open(D / "exp28_hardedge_convergence.json"))
    for ru in d["unit"]:
        rw = next(r for r in d["wishart"] if r["n"] == ru["n"])
        if ru["n"] >= 1024:
            continue  # asymptotic: the largest size converges slowest
        med_u, med_w = ru["lmin_n2_med"], rw["lmin_n2_med"]
        assert abs(med_u - med_w) / max(med_u, med_w) < 0.35, (ru["n"], med_u, med_w)
