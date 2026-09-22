# Operator placement, not representation

**Author:** Luciano Benjamín Nieto
**Location:** General Alvear, Mendoza, Argentina
**License:** MIT

> The "binary collapse" of resonator VSA decoding at high superposition is not a
> limit of the representation — it is a decoder wiring bug. At the square point
> ρ = n/d = 1 the Gram matrix sits at the square-Wishart hard edge
> (λ_min ~ n⁻², κ ~ 4n²), and applying M⁻¹ to the ambient state blows up by
> 1/λ_min. The same resolvent in coefficient space (CᵀM⁻¹C) is the canonical
> dual-frame projector — norm exactly 1 — and decodes perfectly.
> Exp 18 demonstrates this by a controlled intervention: same codebook, same
> Gram, same resolvent, only the placement changes.

---

## Summary

Code, data, and figures for the paper
*"Operator placement, not representation: a resonator-decoder failure mode
at the square-Wishart hard edge"*.

Three regimes by ρ = (distinct codevectors per block) / (block dimensionality):

| Regime | rho | Gram matrix M | Decoding behaviour |
|--------|-----|---------------|--------------------|
| **I. Over-complete** | > 1 (n > d) | rank-deficient | Gram-inverse singular; pinv/pure stable |
| **II. Square** | = 1 (n = d) | hard-edge conditioned | ambient resolvent collapses; dual does not |
| **III. Under-complete** | < 1 (n < d) | well-conditioned | stable |

---

## Key results

### 1. The collapse at ρ=1 is a single-point failure of ambient Gram decoding
Fine sweep (Exp 9, 14 ρ values, 10 seeds × 20 facts): ambient `gram` drops to
0.148 at ρ=1.000; `pure` holds 0.99. (Implementation note: the ambient Gram
correction is dimensionally applicable only at the square point in our harness —
the cross-decoder comparison is measured at ρ=1.)

### 2. The square-Wishart anchoring
Gram M = C Cᵀ with unit-norm codevectors is a scaled square Wishart. At ρ=1,
λ_min ~ n⁻² (hard edge) and κ ~ 4n² (median κ measured vs 4n² within the
heavy-tailed Wishart spread). Verified across n ∈ [32, 512] (Exp 16,
`data/exp16_scaling_n.json`).

### 3. The trigger is NOT a κ-band
Per-seed (Exp 17, 40 seeds at ρ=1): every seed collapses regardless of κ
(κ_min spans [2.3e2, 1.9e5]), corr(log κ, acc) = −0.26. The "critical band"
reading is refuted; the trigger is the hard-edge geometry + ambient placement.

### 4. The Frame-Dual Stability Observation (Exp 18, replicated)
Defined at the square point; the causal claim is placement-specific:

| Observer | Operator | Accuracy at ρ=1 [boot95] |
|----------|----------|--------------------------|
| pure | f | 0.985 [0.982, 0.987] |
| gram (ambient) | M⁻¹f | 0.160 [0.152, 0.169] |
| pinv | M⁺f | 0.161 [0.152, 0.169] |
| dual (pinv) | CᵀM⁺Cf | 0.985 [0.982, 0.987] |
| **dual-same (same M⁻¹)** | **CᵀM⁻¹Cf** | **0.985 [0.982, 0.987]** |

Per-seed stats: N=200 codebooks × 10 facts, bootstrap 95%. Paired per-seed
difference +0.824, positive in 200/200 seeds (p ≤ 2⁻²⁰⁰, exact). Operator
norms: ‖M⁻¹‖₂ = 1/λ_min (median 3.5×10³); ‖CᵀM⁻¹C‖₂ = 1.000;
‖CᵀM⁻¹C − I‖_F median 6×10⁻¹³.

**Independent replication (Exp 19):** pure 0.982, gram 0.159, dual 0.982,
paired +0.823, 100% of seeds, no shared code.

**Ensembles (Exp 20A):** Gaussian, Rademacher, sphere, Toeplitz replicate.
Orthogonal frame at square → no collapse (M=I). Near-duplicate pathological
frame: all decoders fail (control).

**ρ sweep (Exp 20B):** ambient collapses only at ρ=1.

**Linear-only (Exp 21):** gain ambient 362× vs dual 1.000 — VSA was the
vehicle, not the cause.

**Iteration dynamics (Exp 22):** gram fails at t=0 already (e₀ ~ 7×10³).
The loop is not the mechanism.

**BSC (Exp 18b):** 0.192 → 0.995, paired +0.80.

**FHRR (Exp 27):** 0.131 → 1.000, paired +0.868, closes the circle on the
originating algebra.

### 5. Universality across algebras
HRR (V3), BSC (11b, 18b), MAP (11c — single-shot, no loop, doesn't collapse),
FHRR (27). Transformers (12b) don't collapse (softmax, no Gram inverse).

### 6. Rust crate (fhrr-resilient)
A conservative baseline decoder router. Currently implements the *legacy*
κ-threshold routing (see `src/lib.rs` CAVEAT); a future version will route on
operator placement + geometry. 4 Rust tests green.

---

## How to run

```bash
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the numbers in the paper
python verify.py                       # 31 checks against data/*.json

# Tests
pytest tests/ -v                       # 13 Python tests
cd fhrr-resilient && cargo test        # 4 Rust tests

# Reproduce causal experiments
cd src
python exp_V18_frame_duality.py        # HRR
python exp_V18b_bsc_duality.py         # BSC
python exp_V27_fhrr_duality.py         # FHRR
python exp_V16_scaling_n.py            # spectral scaling
```

Compile the paper: see `paper/README.md`.

## Repository layout

```
fhrr-rho-collapse/
├── README.md
├── LICENSE  (MIT)
├── requirements.txt      ← numpy, matplotlib, pytest
├── pytest.ini
├── CITATION.cff          ← GitHub citation metadata
├── verify.py             ← reproduction checks
├── docs/
│   ├── ROADMAP.md        ← falsification program (closed)
│   └── archive/          ← older internal notes (historical)
├── paper/
│   ├── main.tex          ← the paper
│   ├── main.pdf          ← compiled (tectonic)
│   ├── references.bib
│   ├── README.md         ← how to compile
│   └── figures/
├── src/                  ← exp_V*.py (18 experiments)
├── tests/
│   ├── test_pipeline.py  ← 5 tests
│   ├── test_frame_duality.py ← 8 tests
│   └── README.md
├── fhrr-resilient/       ← Rust crate (legacy κ-router)
├── data/                 ← every number is a JSON
└── figures/              ← paper figures
```

## Claims → evidence

| Claim | Evidence |
|-------|----------|
| Point failure at ρ=1 | `data/out_V9_fine_rho.json` |
| κ ~ 4n² scaling | `data/exp16_scaling_n.json` |
| κ-band refutation | `data/exp17_kappa_vs_acc.json` |
| Frame-Dual Stability (HRR) | `data/exp18_summary.json` |
| Independent replication | `data/exp19_independent_replica.json` |
| BSC replication | `data/exp18_bsc_summary.json` |
| FHRR replication | `data/exp27_fhrr_duality.json` |
| Ensemble sweep | `data/exp20_ensembles_rho.json` |
| Linear-only | `data/exp21_linear_amp.json` |
| Dynamics | `data/exp22_iteration_dynamics.json` |
| Operator zoo | `data/exp23_operator_zoo.json` |
| Frame bounds | `data/exp24_frame_bounds.json` |
| Counterexamples | `data/exp25_contraejemplos.json` |
| Stability bound | `data/exp26_cota_stability.json` |
| BSC/MAP phase | `data/exp11_bsc_rho.json`, `exp11c_map_rho.json` |

## Citation

```bibtex
@article{nieto2026observer,
  title={Operator placement, not representation: a resonator-decoder failure
         mode at the square-Wishart hard edge},
  author={Nieto, Luciano Benjamín},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Rylow999/fhrr-rho-collapse}
}
```

## Acknowledgments

Anonymous external reviewers whose critiques sharpened the formulation.

---

*Per Aspera, Ad Astra.*
