#!/usr/bin/env python3
"""
Tests del claim central del paper: dual placement como canonical dual-frame
projector y el efecto causal persistido de Exp. 18/27.

Las pruebas algebraicas son deterministas. Las pruebas sobre resultados
persistidos se ejecutan solo cuando data/ esta presente en el checkout.
"""
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


def square_frame(n, seed):
    """C (n x n) con filas Gaussianas normalizadas."""
    rng = np.random.RandomState(seed)
    C = rng.randn(n, n)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


class TestDualProjectorIdentity:
    def test_square_real_projector_is_identity(self):
        for seed in range(5):
            C = square_frame(32, seed)
            M = C @ C.T
            P = C.T @ np.linalg.inv(M) @ C
            assert np.linalg.norm(P - np.eye(32), ord="fro") < 1e-8
            assert abs(np.linalg.norm(P, 2) - 1.0) < 1e-8

    def test_ambient_inverse_norm_is_one_over_lmin(self):
        C = square_frame(32, 0)
        M = C @ C.T
        lmin = np.linalg.eigvalsh(M)[0]
        assert np.linalg.norm(np.linalg.inv(M), 2) == pytest.approx(1.0 / lmin)
        assert 1.0 / lmin > 100

    def test_square_complex_projector_is_identity(self):
        rng = np.random.RandomState(3)
        C = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(32, 32)))
        M = C @ C.conj().T
        P = C.conj().T @ np.linalg.inv(M) @ C
        assert np.linalg.norm(P - np.eye(32), ord="fro") < 1e-8
        assert abs(np.linalg.norm(P, 2) - 1.0) < 1e-8

    def test_wide_frame_projector_is_bounded_by_one(self):
        rng = np.random.RandomState(5)
        C = rng.randn(16, 48)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        P = C.T @ np.linalg.inv(C @ C.T) @ C
        assert abs(np.linalg.norm(P, 2) - 1.0) < 1e-8
        assert np.linalg.norm(P - np.eye(48), ord="fro") > 1.0

    def test_hard_edge_kappa_has_quadratic_order_growth(self):
        """
        Check only the robust asymptotic statement supported here:
        kappa grows with approximately quadratic order in n.

        The smallest Wishart eigenvalue has a heavy-tailed finite-size
        distribution, so a unit test must not require a factor-of-four ratio
        between two tiny Monte Carlo samples.
        """
        ns = np.array([32, 64, 128, 256], dtype=float)
        medians = []
        for n in ns.astype(int):
            ks = []
            for seed in range(8):
                C = square_frame(n, seed)
                w = np.linalg.eigvalsh(C @ C.T)
                ks.append(float(w[-1] / w[0]))
            medians.append(float(np.median(ks)))

        slope = float(np.polyfit(np.log(ns), np.log(medians), 1)[0])
        assert 1.0 < slope < 4.0, f"log-log slope={slope:.3f}, medians={medians}"
        assert medians[2] > medians[0], f"no visible growth: {medians}"


@pytest.mark.skipif(not (DATA / "exp18_summary.json").exists(),
                    reason="data/exp18_summary.json no presente")
class TestExp18PersistedEffect:
    def test_paired_difference_reproduces(self):
        with open(DATA / "exp18_summary.json", encoding="utf-8") as fh:
            d18 = json.load(fh)
        sq = {r["mode"]: r for r in d18["square"]}
        assert sq["gram"]["mean"] < 0.20
        assert sq["dual_same"]["mean"] > 0.95
        p = d18["paired"]["dual_same_minus_gram"]
        assert p["mean"] == pytest.approx(0.824, abs=0.01)
        assert p["frac_pos"] > 0.99

    def test_operator_norms(self):
        with open(DATA / "exp18_summary.json", encoding="utf-8") as fh:
            d18 = json.load(fh)
        assert d18["norm_Minv_med"] > 1e3


@pytest.mark.skipif(not (DATA / "exp27_fhrr_duality.json").exists(),
                    reason="data/exp27_fhrr_duality.json no presente")
class TestExp27PersistedEffect:
    def test_fhrr_causal_effect_reproduces(self):
        with open(DATA / "exp27_fhrr_duality.json", encoding="utf-8") as fh:
            e27 = json.load(fh)
        assert e27["gram"]["mean"] < 0.20
        assert e27["dual_same"]["mean"] > 0.95
        assert e27["paired"]["mean"] == pytest.approx(0.868, abs=0.05)
        assert e27["paired"]["frac_pos"] > 0.99
