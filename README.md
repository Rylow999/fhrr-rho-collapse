# FHRR rho-Collapse: Frame Duality Governs Compositional Decoding

**Author:** Luciano Benjamín Nieto  
**Location:** General Alvear, Mendoza, Argentina  
**License:** MIT  

> **v3 (2026-09-16):** La evidencia ya no es solo técnica sino estructural.
> La "transición de fase" en ρ=1 es una **singularidad puntual** (det(M)=0
> exacto), no un valle de transición — el barrido fino lo muestra: 14 puntos
> entre ρ=0.73 y ρ=1.23, todos los decoders estables **excepto en ρ=1.000
> exacto**, donde `gram` y `pinv` caen a 0.148 mientras `pure` y MLP quedan
> en 0.99. El paper compilado está en `paper/main.pdf` (9 páginas, 8 refs).

---

## Summary

This repository contains the code, data, and figures for the paper *"Frame duality governs compositional decoding in FHRR: a phase diagram in rho"*, plus the observer-relativity extension across algebras.

We identify a phase diagram that governs resonator-based decoding in Fourier Holographic Reduced Representations, controlled by a single scalar:

```
rho = (distinct codevectors per block) / (block dimensionality)
```

Three regimes emerge:

| Regime | rho | State of Gram matrix M | Decoding behaviour |
|--------|-----|------------------------|--------------------|
| **I. Under-complete** | < 1 | rank-deficient (singular) | `mat_inv` returns numerical garbage; collapse |
| **II. Square** | = 1 | invertible, cond ~ 1e3 | dual frame degenerate; partial collapse |
| **III. Over-complete** | > 1 | well-conditioned frame | stable decoding |

## What's new in v3 (2026-09-16)

- **Exp 9 (fine sweep):** 14 values of rho between 0.727–1.231, 10 seeds × 20 facts each. The anti-resonance is a **mathematical singularity, not a region** — gram and pinv collapse only at rho=1.000 exactly (0.148), recovering to 0.99 at rho=0.970 and rho=1.032.
- **Exp 10 (residual trajectory):** At rho=1.00, `gram` and `pinv` plateau at residual ≈ 1.6–1.7 (stuck in a spurious attractor), while `pure` converges to 0.
- **Exp 7 (MLP):** The learned observer reaches 0.992 accuracy at rho=1 (where gram fails at 0.119). The information is in the vector; the collapse belongs to the observer.
- **Exp 8 (full taxonomy):** 600 decoder configurations (4 modes × 5 iterations × 3 cleanup × 2 schemes × 5 rho values).
- **Paloma-π application:** Pandora's transducer now uses the pure resonator, demonstrating the decoder-relativity principle in a working cognitive system.

## Repository structure

```
fhrr-rho-collapse/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── paper/
│   ├── main.tex                    ← paper completo (7 secciones + refs)
│   ├── main.pdf                    ← compilado con tectonic (9 páginas)
│   ├── references.bib
│   └── figures/                    ← figuras del paper
├── src/
│   ├── base_fhrr.py                # FHRR base
│   ├── base_fhrr_corregida.py      # fixes de auditoría
│   ├── exp_F2.py                   # FHRR: variar T
│   ├── exp_H2.py                   # FHRR: grid de rho
│   ├── exp_V3_hrr_real.py          # HRR real: grid de rho
│   ├── exp_V4_hrr_hierarchical.py  # nested binding (resultado negativo)
│   ├── exp_V5_scaling_sqrtD.py     # M_max vs D scaling
│   ├── exp_V7_mlp_decoder.py       # MLP observer
│   ├── exp_observer_taxonomy_v3.py # taxonomía (4 decoders × 120 configs)
│   ├── exp_V9_fine_rho.py          # barrido fino (singularidad puntual)
│   ├── exp_V10_residual_trajectory.py  # trayectoria residuo
│   ├── diag_A4.py
│   └── diag_H2_cond.py
├── tests/
│   └── test_pipeline.py            # 5 tests, todos PASS
├── data/                           ← outputs completos (.txt, .json)
└── figures/                        ← 9 figuras listas para paper
```

## How to run

```bash
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python exp_H2.py                      # grid FHRR original
python exp_V3_hrr_real.py            # HRR real (anti-resonancia)
python exp_V9_fine_rho.py            # barrido fino (singularidad puntual)
python exp_V10_residual_trajectory.py # residuo del resonator
python exp_V7_mlp_decoder.py         # MLP observer (requiere torch)
python exp_observer_taxonomy_v3.py   # taxonomía completa (600 configs)

cd .. && pytest tests/ -v            # 5 tests verdes
```

## Key findings

1. **The phase diagram is real but singular.** rho=1 is a point where the Gram inverse breaks exactly (det M = 0), not a capacity cliff.
2. **The physical trigger is kappa, not rho.** The decoder fails when kappa(M) enters the critical band [10³, 10⁴]; the exact threshold is **κ* = 5.81×10³** (log₁₀ = 3.76), derived from noise amplification vs codebook margin (Exp 14). Outside that band (singular κ>10¹⁵ or well-conditioned κ<10²), decoding succeeds.
3. **Pure decoder works everywhere.** No Gram correction needed — accuracy 0.98–1.00 across the whole grid.
4. **Learned observers close the gap.** MLP with 2 hidden layers achieves 0.992 accuracy at rho=1 (where gram fails at 0.146). The information is in the vector; the reported collapse is observer-relative.
5. **Universality across algebras.** The same ρ=1 transition holds in HRR real (anti-resonance), BSC binary (gram=0.28 vs pure=1.00), and LiDAR voxelization (gram 1.0→0.33). The law is a property of Gram-inverse decoders, not of any specific algebra.
6. **Transformers do not collapse** (they use softmax, not Gram inverse) — but attention entropy shows a non-monotonic valley at ρ≈2 (Exp 12b), a bandwidth saturation signature.
7. **Cross-adaptive routing** (Exp 15): a kappa-based router works outside the critical band but needs threshold calibration — the adaptive decoder inherits the same kappa law.
8. **Scaling:** M_max grows as D^1.1, not sqrt(D). Pure resonators scale linearly with dimension.
9. **Multi-observer invariance** (RHO_LAW): on SDDF turbulence spectra, window ratios are invariant across models (CV≈0.24) while G_total varies (CV≈0.7) — the algebra separates observer artifacts from structural invariants.
10. **Collatz analogue:** the divergence threshold f_P* = 0.7075 (derived, conditional on LEH) plays the same role as κ*: an admissible-ratio interval that closes at the critical parameter (a=4, drift=0) — the same structure as Proposition 4.2 of the NS blowup audit (Zenodo 22820521).

## Paper

Full paper in `paper/main.tex` (compiled: `paper/main.pdf`, 9 pages). Covers phase diagram, empirical validation (F2/H2/V3/V7/V8/V9/V10), diagnosis of numerical artifacts, kappa-conditioned refinement with exact threshold κ*, cross-algebra validation (HRR, BSC, transformers), failed alternatives, and discussion.

## Related repositories

- **Rylow999/paloma-pi-v2** — the applied demonstration: real bioacoustic data (Columba livia, 83 clips), pure resonator recovery 100%, structural significance p=0.002.
- **Rylow999/sddf** — Navier-Stokes spectral curvature G[u]: exact closed-form law, post-audit fixes, Migdal periodogram with calibrated null model.
- **Rylow999/rho-law** — the unifying framework: three-layer structure (substrate/observer/instrument), four domains, the irreducible point.

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

- Nexus (agent assistance with audit and experimental design)

---

*Per Aspera, Ad Astra.*

