# FHRR rho-Collapse: Frame Duality Governs Compositional Decoding

**Author:** Luciano Benjamín Nieto
**Location:** General Alvear, Mendoza, Argentina
**License:** MIT

> The "binary collapse" of resonator decoding at high superposition is not a
> limit of the representation — it is a property of the decoder's wiring.
> At ρ = n/d = 1 the Gram matrix is a square Wishart whose soft edge gives
> λ_min ~ n⁻²; applying M⁻¹ directly to the state multiplies the error by
> 1/λ_min per iteration. The same resolvent, applied in coefficient space
> (CᵀM⁻¹C), is numerically the identity and decodes perfectly.
> The collapse belongs to the observer. Exp. 18 proves it causally.

---

## Summary

Code, data, and figures for the paper
*"Frame duality governs compositional decoding in FHRR: a phase diagram in ρ"*.

We identify a phase diagram controlled by a single scalar:

```
rho = (distinct codevectors per block) / (block dimensionality)
```

| Regime | rho | Gram matrix M | Decoding behaviour |
|--------|-----|---------------|--------------------|
| **I. Under-complete** | < 1 | rank-deficient | Gram-inverse fails (singular) |
| **II. Square** | = 1 | numerically singular (κ ~ 4n², verified n ≤ 512) | ambient Gram collapses; dual Gram does not |
| **III. Over-complete** | > 1 | well-conditioned frame | stable |

## Key results

### 1. The ρ=1 collapse is a point failure of ambient Gram decoding
Fine sweep (Exp 9, 14 ρ values, 10 seeds × 20 facts): gram and pinv drop to
0.148 at ρ=1.000 exactly while pure stays at 0.988. (Implementation note:
outside ρ=1 the ambient Gram correction is dimensionally inapplicable in our
implementation, so gram ≡ pure there; the meaningful cross-decoder comparison
is the square point.)

### 2. Random-matrix anchoring: M = C Cᵀ is a scaled square Wishart
At ρ=1, λ_min ~ n⁻² (soft edge at 0), λ_max → 4 (MP upper edge), and
κ ~ 4n² (Edelman). Verified by direct measurement across n ∈ {32, 64, 128,
256, 512} (Exp 16, 12 seeds each, `data/exp16_scaling_n.json`).

### 3. The trigger is NOT a critical κ band
Per-seed audit (Exp 17, 40 seeds at ρ=1): per-block κ spans [2.3e2, 1.9e5],
every seed collapses regardless of band membership (two seeds with
κ_min = 234 and 445 collapse identically), corr(log κ, acc) = −0.38.
Uniform failure at the square point — the trigger is the soft edge plus the
asymmetric loop application, not a scalar condition-number window.

### 4. Causal proof: the Frame-Duality Law (Exp 18)
Same C, M, M⁻¹, initial states, facts, iteration count; only the operator
wiring changes:

| Observer | Operator | Accuracy at ρ=1 |
|----------|----------|-----------------|
| pure | f | 0.986 |
| gram | M⁻¹f | 0.161 |
| pinv | M⁺f | 0.162 |
| dual | CᵀM⁺Cf | 0.986 |
| **dual-same** | **CᵀM⁻¹Cf** (same resolvent) | **0.986** |

Operator norms: ‖M⁻¹‖₂ ~ 1.7×10³ (median) vs ‖CᵀM⁻¹C‖₂ = 1.000.
The identity CᵀM⁻¹C = I is verified numerically per block
(‖·‖_F median 6×10⁻¹³). Paired difference dual−same − gram = +0.824,
positive in 100% of the 2000 fact-decodings. Replicated in BSC
(Exp 18b): gram 0.192 → dual-same 0.995, paired +0.80.

### 5. Universality across algebras
HRR real (V3), BSC binary (Exp 11b), MAP (Exp 11c — the single-shot control:
the same κ at ρ=1 does NOT collapse MAP because it is not closed-loop),
and LiDAR voxelization (Exp 13).

### 6. Transformers do not collapse (Exp 12b)
Attention uses softmax, not Gram inversion. A non-monotonic entropy valley
appears at ρ≈2 instead — bandwidth saturation, not singular decoding failure.

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
python exp_V18_frame_duality.py    # causal test (the Frame-Duality Law)
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
| Frame-Duality Law | `data/exp18_summary.json` |
| BSC replication | `data/exp18_bsc_summary.json` |
| BSC/MAP phase data | `data/exp11_bsc_rho.json`, `data/exp11c_map_rho.json` |
| MLP survival at ρ=1 | `data/mlp_decoder_results.json` |

## Paper

Full paper in `paper/main.tex` (compiled `paper/main.pdf`). Structure:
phase diagram → empirical validation → RMT anchoring → causal test
(Exp 18) → failed alternatives → discussion → limitations.

Two named, citable objects:

- **Definition 1** (closed-loop soft-edge condition) — a decoder satisfies
  it iff the Gram ensemble has a soft edge at 0 (ρ=1) AND the decoder
  re-applies M⁻¹ on every iteration. Checkable a priori.
- **Proposition 1** (Frame-Duality Law, provisional) — the collapse is a
  property of the observer's wiring, not of the frame; the dual operator
  using the same resolvent recovers decoding.

## Related repositories

- **Rylow999/paloma-pi-v2** — applied demo: pure resonator over real
  bioacoustic data.
- **Rylow999/sddf** — Navier–Stokes spectral curvature G[u].
- **Rylow999/rho-law** — the unifying framework.

## Citation

```bibtex
@article{nieto2026fhrr,
  title={Frame duality governs compositional decoding in FHRR: a phase diagram in rho},
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
