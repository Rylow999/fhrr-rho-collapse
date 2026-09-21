# Observer-Induced Collapse in Resonator Decoding: A Frame-Duality Mechanism

**Author:** Luciano Benjamín Nieto
**Location:** General Alvear, Mendoza, Argentina
**License:** MIT

> The "binary collapse" of resonator decoding at high superposition is not a
> limit of the representation — it is a property of the decoder's wiring.
> At ρ = n/d = 1 the Gram matrix enters the square-Wishart *hard edge*:
> λ_min ~ n⁻², and applying M⁻¹ directly to the state blows up by 1/λ_min.
> The same resolvent, applied in coefficient space
> (CᵀM⁻¹C), is a projector of norm 1 and decodes perfectly.
> The collapse belongs to the observer. Exp. 18 demonstrates this with a
> controlled causal intervention: same codebook, same resolvent, only the
> placement changes.

---

## Summary

Code, data, and figures for the paper
*"Observer-induced collapse in resonator decoding: a frame-duality mechanism"*
(the causal core uses HRR-real and BSC; the FHRR phase-diagram experiments are
the original vantage point).

We identify a phase diagram controlled by a single scalar:

```
rho = (distinct codevectors per block) / (block dimensionality)
```

| Regime | rho | Gram matrix M | Decoding behaviour |
|--------|-----|---------------|--------------------|
| **I. Over-complete** | > 1 ($n > d$) | rank-deficient (singular) | Gram-inverse fails; pinv/pure stable |
| **II. Square** | = 1 ($n = d$) | critically conditioned Wishart (κ ~ 4n², verified n ≤ 512) | ambient Gram collapses; dual Gram does not |
| **III. Under-complete** | < 1 ($n < d$) | full-rank, well-conditioned frame | stable |

## Key results

### 1. The ρ=1 collapse is a point failure of ambient Gram decoding
Fine sweep (Exp 9, 14 ρ values, 10 seeds × 20 facts): gram and pinv drop to
0.148 at ρ=1.000 exactly while pure stays at 0.988. (Implementation note:
outside ρ=1 the ambient Gram correction is dimensionally inapplicable in our
implementation, so gram ≡ pure there; the meaningful cross-decoder comparison
is the square point.)

### 2. Random-matrix anchoring: M = C Cᵀ is a scaled square Wishart
At ρ=1 the lower edge of the Marchenko–Pastur support touches zero: the square Wishart is at its *hard edge*, λ_min ~ n⁻², λ_max → 4 (MP upper edge), and κ ~ 4n² (Edelman). Verified by direct measurement across n ∈ {32, 64, 128,
256, 512} (Exp 16, 12 seeds each, `data/exp16_scaling_n.json`). Scope: Exp 16 validates the *spectral* scaling of the square Gram; its decoder component is a simplified resonator and is not used as evidence about the observer-level collapse.

### 3. The trigger is NOT a critical κ band
Per-seed audit (Exp 17, 40 seeds at ρ=1): per-block κ spans [2.3e2, 1.9e5],
every seed collapses regardless of band membership (two seeds with
κ_min = 234 and 445 collapse identically), corr(log κ, acc) = −0.38.
Uniform failure at the square point — the trigger is the hard-edge conditioning of
the square Gram, not a scalar condition-number window.

### 4. Causal proof: the Frame-Dual Stability Principle (Exp 18, replicated Exp 19)
Same C, M, M⁻¹, initial states, facts, iteration count; only the operator
wiring changes:

| Observer | Operator | Accuracy at ρ=1 |
|----------|----------|-----------------|
| pure | f | 0.985 [0.982, 0.987] |
| gram | M⁻¹f | 0.160 [0.152, 0.169] |
| pinv | M⁺f | 0.161 [0.152, 0.169] |
| dual | CᵀM⁺Cf | 0.985 [0.982, 0.987] |
| **dual-same** | **CᵀM⁻¹Cf** (same resolvent) | **0.985 [0.982, 0.987]** |

Statistics per seed (N=200 codebooks, bootstrap 95% CI). Paired per-seed
difference dual-same − gram = +0.824, positive in 200/200 seeds
(p ≤ 2⁻²⁰⁰, exact sign test). ‖M⁻¹‖₂ exact = 1/λ_min (median 3.5×10³),
‖CᵀM⁻¹C‖₂ = 1.000, ‖CᵀM⁻¹C − I‖_F median 6×10⁻¹³.

**Independent replication (Exp 19, no shared code):** pure 0.982, gram
0.159, dual-same 0.982, paired +0.823, 100% of seeds.

**Ensembles (Exp 20A):** Gaussian 0.160→0.979, Rademacher 0.128→0.986,
Sphere 0.164→0.983, Toeplitz 0.175→0.947, DFT-tight does not collapse,
near-duplicate pathological (mis-specified frame breaks everything).

**ρ sweep (Exp 20B):** ambient collapse localized at hard edge ρ≈1;
benign below 0.95, rank-deficient above 1.

**De-VSA-ified (Exp 21):** in a purely linear pipeline (no resonator),
ambient gain = 3.6×10² median, dual gain = 1.000 — VSA was the vehicle,
not the cause.

**Iteration dynamics (Exp 22):** the ambient error saturates at the FIRST
application (e₀ ~ 7×10³); the loop doesn't cause it.

**BSC (Exp 18b):** 0.192 → 0.995, paired +0.80.

### 5. Universality across algebras
HRR real (V3), BSC binary (Exp 11b + causal replication Exp 18b), MAP (Exp 11c — the single-shot control: the same κ at ρ=1 does NOT collapse MAP because it is not closed-loop), and LiDAR voxelization (Exp 13). After Exp 18 the operative design rule is **operator placement**, not a κ-threshold.

### 6. Transformers do not collapse (Exp 12b)
Attention uses softmax, not Gram inversion. A non-monotonic entropy valley appears at ρ≈2 instead — bandwidth saturation, not singular decoding failure.

### 7. Rust crate: frame-aware decoder (fhrr-resilient)
The crate routes decoder selection by **operator wiring** (ambient M⁻¹ in a closed loop → fail over to dual form or pure resonator; rank-deficient Gram at ρ>1 → pseudo-inverse). The legacy κ-band thresholds remain only as conservative fallbacks when the loop structure is not inspectable. 4 Rust tests green. See `fhrr-resilient/src/lib.rs`.

## What's next (falsification roadmap)

The Frame-Dual Stability Principle survives the falsification agenda of `docs/ROADMAP.md`:
independent reimplementation (Exp 19), six codebook ensembles (Exp 20),
continuous ρ sweep (Exp 20B), perturbation amplification (Exp 21),
de-VSA-ified (Exp 21), iteration dynamics (Exp 22). Open: operator zoo
(ridge, truncated SVD, Krylov) and FHRR-complex causal test.

## Repository structure

```
fhrr-rho-collapse/
├── README.md
├── LICENSE  (MIT)
├── requirements.txt
├── paper/
│   ├── main.tex            ← full paper
│   ├── main.pdf            ← compiled (tectonic)
│   ├── references.bib
│   └── figures/
├── src/
│   ├── exp_observer_taxonomy_v3.py   # BundleV3: base HRR harness (used by V9, V14, V17, V18)
│   ├── exp_V3_hrr_real.py            # HRR real: phase diagram
│   ├── exp_V9_fine_rho.py            # fine sweep (point singularity)
│   ├── exp_V10_residual_trajectory.py
│   ├── exp_V7_mlp_decoder.py         # MLP observer
│   ├── exp_V11_bsc_binary.py         # BSC base
│   ├── exp_V11b_bsc_persist.py       # BSC grid, persisted
│   ├── exp_V11c_map_persist.py       # MAP grid, persisted
│   ├── exp_V14_kappa_law.py          # kappa law measurement
│   ├── exp_V16_scaling_n.py          # κ ~ 4n² verification (n up to 512)
│   ├── exp_V17_kappa_vs_acc.py       # per-seed kappa vs accuracy audit
│   ├── exp_V18_frame_duality.py      # causal test (HRR)
│   ├── exp_V18b_bsc_duality.py       # causal test (BSC replication)
│   ├── fig_exp16_scaling.py          # scaling figure
│   └── fig_exp18_frame_duality.py    # causal figure
├── fhrr-resilient/                   # Rust crate: κ-router with dual failover
├── tests/              # 5 tests, pytest
├── data/               # all raw outputs (.json, .txt)
└── figures/            # paper figures
```

## How to run

```bash
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python exp_V9_fine_rho.py          # the point singularity
python exp_V16_scaling_n.py        # κ ~ 4n² verification
python exp_V17_kappa_vs_acc.py     # per-seed audit (band refutation)
python exp_V18_frame_duality.py    # causal test (the Frame-Dual Stability Principle)
python exp_V18b_bsc_duality.py     # BSC replication

cd .. && pytest tests/ -v          # 5 unit tests
cd fhrr-resilient && cargo test    # 4 Rust tests (router)
```

## Claims → evidence map

| Claim | Evidence |
|-------|----------|
| Point failure at ρ=1 | `data/out_V9_fine_rho.json` |
| κ ~ 4n² scaling | `data/exp16_scaling_n.json` |
| Band refutation | `data/exp17_kappa_vs_acc.json` |
| Frame-Dual Stability Principle | `data/exp18_summary.json` |
| BSC replication | `data/exp18_bsc_summary.json` |
| BSC/MAP phase data | `data/exp11_bsc_rho.json`, `data/exp11c_map_rho.json` |
| MLP survival at ρ=1 | `data/mlp_decoder_results.json` |

## Paper

Full paper in `paper/main.tex` (compiled `paper/main.pdf`). Structure:
phase diagram → empirical validation → RMT anchoring → causal test
(Exp 18) → failed alternatives → discussion → limitations.

Two named, citable objects:

- **Definition 1** (closed-loop hard-edge condition) — a decoder satisfies
  it iff the Gram ensemble has a hard edge at 0 (ρ=1) AND the decoder
  applies M⁻¹ repeatedly to the running estimate. Checkable a priori.
- **Proposition 1** (Frame-Dual Stability Principle) — the spectral form: for spectral T = g(M), Cᵀ T C = V Σ g(Σ²) Σ Vᵀ with singular values σ_i²g(σ_i²). Pinning down exactly which operators neutralize in dual placement (Exp 23) is the version of the principle that survives contact with the frame-multiplier literature.

## Related repositories

- **Rylow999/paloma-pi-v2** — applied demo: pure resonator over real
  bioacoustic data.
- **Rylow999/sddf** — Navier–Stokes spectral curvature G[u].
- **Rylow999/rho-law** — the unifying framework.

## Citation

```bibtex
@article{nieto2026observer,
  title={Observer-induced collapse in resonator decoding: a frame-duality mechanism},
  author={Nieto, Luciano Benjamín},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Rylow999/fhrr-rho-collapse}
}
```

## Acknowledgments

- Nexus (agent assistance with audit, experimental design, and the
  random-matrix analysis)
- The anonymous external reviewer whose critique of the κ-band claim led
  to the stronger formulation

---

*Per Aspera, Ad Astra.*
