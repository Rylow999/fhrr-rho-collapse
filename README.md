# FHRR rho-Collapse: Frame Duality Governs Compositional Decoding

**Author:** Luciano Benjamín Nieto  
**Location:** General Alvear, Mendoza, Argentina  
**License:** MIT  

> **v2 (2026-09-15):** Validación cruzada con HRR real — la ley de rho es universal
> pero con **topologías diferentes por álgebra**. En HRR real (gaussianos +
> convolución circular) aparece **anti-resonancia en rho=1**: colapso catastrófico
> exactamente donde FHRR es singular. Ver `docs/INFORME_TRANSFERENCIA.md`.

---

## Summary

This repository contains the code, data, and figures for the paper *"Frame duality governs compositional decoding in FHRR: a phase diagram in rho"*.

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

The previously reported "binary collapse" at high superposition is **not** a capacity limit of the resonator network. It is an artifact of Regime I: when `rho < 1` the Gram matrix is singular, and the custom `mat_inv` routine produces an invalid inverse that destroys decoding. Pure resonator decoding (without Gram correction) works perfectly across all tested configurations.

The `rho = 1` boundary is particularly subtle: the pseudo-inverse (`pinv`) repairs Regime I by truncating near-zero modes, but it **fails** at the square boundary because the smallest eigenvalue is above the truncation threshold yet still amplifies noise. Only the pure resonator (no Gram matrix) succeeds here, producing a clean double dissociation.

## Cross-algebra validation: HRR real

The rho-law is **universal**, but its topology is **algebra-specific**:

| Regime | FHRR (complex phases) | HRR real (Gaussian + circular convolution) |
|--------|-----------------------|--------------------------------------------|
| rho < 1 | Gram singular → collapse | Gram works perfectly (1.000) |
| rho = 1 | partial collapse (0.80) | **catastrophic anti-resonance (0.167)** |
| rho > 1 | stable | stable (0.95–1.00) |

In **both** algebras: the `pure` resonator decoder works across the whole grid
(0.95–1.00), and the `gradient` decoder collapses on every multi-role case
(continuous optimization vs discrete selection). The HRR anti-resonance at
rho = 1 is hypothesized to be destructive interference in circular convolution
— open problem, see Line B in `docs/INFORME_TRANSFERENCIA.md`.

**Reproducibility:** fixed seeds (BlockBundle=7, make_fact=42), all outputs in
`data/`. Full transfer report: `docs/INFORME_TRANSFERENCIA.md`.

---

## Repository structure

```
fhrr-rho-collapse/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── base_fhrr.py              # Original implementation (preserves reproducibility)
│   ├── base_fhrr_corregida.py    # Corrected implementation (audit fixes)
│   ├── exp_F2.py                 # Experiment F: varying T, three decode modes
│   ├── exp_H2.py                 # Experiment H: rho grid with four decoders
│   ├── diag_A4.py                # Diagnosis of run_decode_0059h.py
│   └── diag_H2_cond.py           # Conditioning and eigenvalue diagnosis
├── data/
│   ├── resultados_reales_completos.txt
│   ├── out_F2.txt
│   ├── out_H2.txt
│   ├── out_diag_A4.txt
│   └── out_diag_H2_cond.txt
├── figures/
│   ├── fig1_phase_diagram.png
│   ├── fig2_accuracy_vs_rho.png
│   ├── fig3_double_dissociation.png
│   └── fig4_comparison_0059h.png
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

## Installation

Requires Python 3.8+ and NumPy.

```bash
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduction

All experiments use fixed random seeds (7 for `BlockBundle`, 42 for `make_fact`), and symbol-to-phase assignment now iterates over a `sorted()` set so process-level Python hash randomization can no longer perturb it. Outputs are bitwise identical across runs (verified by diffing two independent runs of each script).

```bash
cd src

# Diagnostics
python diag_H2_cond.py > ../data/out_diag_H2_cond.txt

# Main experiments
python exp_F2.py > ../data/out_F2.txt
python exp_H2.py > ../data/out_H2.txt
```

`diag_A4.py` is **not** part of the reproducible pipeline above: it depends on `run_decode_0059h.py`, the original pre-repo script that first surfaced this bug, which is not included here. Running it prints an explanatory message and exits instead of crashing. Its previously captured output is preserved as-is in `data/out_diag_A4.txt` (forensic record, not reproducible from a clean clone).

`data/resultados_reales_completos.txt` is an archival combined log from the audit process. Its A3/A4 sections predate a fix to `ROLE_CONFIGS[6]` in `base_fhrr.py` (TERR used to alias LOC's codebook; TIME had 14 unique symbols instead of 16) and no longer match the shipped code for the n=6 case — this is noted at the top of that file. For an up-to-date, reproducible comparison, use `data/out_F2.txt` and `data/out_H2.txt` instead.

---

## Key results

### F2: Collapse is a `mat_inv` artifact, not a resonator limit

| n_roles | original (`mat_inv`) | pure (no Gram) | `pinv` |
|---------|----------------------|----------------|--------|
| 2 | 0.000 - 0.050 | **1.000** | **1.000** |
| 3 | 0.067 - 0.100 | **1.000** | **1.000** |
| 4 | 0.000 - 0.150 | **1.000** | **1.000** |
| 6 | 0.033 - 0.133 | **1.000** | **1.000** |

### H2: Double dissociation at `rho = 1`

(T = 100; see `data/out_H2.txt` for the full T=25/T=100 breakdown)

| rho | gram | gradient | pure | pinv |
|-----|------|----------|------|------|
| 0.50 | 0.050 | 0.500 | **1.000** | **1.000** |
| 0.75 | 0.083 | 0.450 | **1.000** | **1.000** |
| 0.80 | 0.200 | 0.217 | **1.000** | 0.950 |
| **1.00** | **0.550** | **0.217** | **1.000** | **0.550** |
| 1.33 | 0.983 | 0.233 | **1.000** | 0.983 |
| 1.50 | 1.000 | 0.200 | **1.000** | 0.967 |

The row `rho = 1.00` is the critical result: `pinv` does **not** rescue the square case (0.550, essentially tied with plain `gram` at 0.550), but `pure` does (1.000). This demonstrates that the problem at `rho = 1` is not singularity but dual-frame degeneracy. (Exact values will drift slightly run-to-run only if the fixed seeds above are changed; with the seeds as shipped, they are bit-identical.)

---

## Citation

```bibtex
@article{nieto2026fhrr,
  title={Frame duality governs compositional decoding in {FHRR}: a phase diagram in rho},
  author={Nieto, Luciano Benjamín},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Audit changelog (2026-09)

- `base_fhrr.py`: symbol-to-phase assignment now iterates a `sorted()` set instead of a raw `set()`, removing a `PYTHONHASHSEED`-driven source of non-determinism. Runs are now verified bitwise-identical across processes.
- `data/out_F2.txt`, `data/out_H2.txt`, `data/out_diag_H2_cond.txt`: regenerated with the fix above; README tables updated to match exactly.
- `diag_A4.py`: fails with a clear message instead of an unhandled traceback when `run_decode_0059h.py` (not included) is missing; removed from the main reproduction flow.
- `data/resultados_reales_completos.txt` and `data/out_diag_A4.txt`: annotated with provenance notes — both predate a since-fixed `ROLE_CONFIGS[6]` bug (TERR aliasing LOC's codebook; TIME with 14 instead of 16 unique symbols) and are kept as archival records, not as the reproducibility reference.
- Stale comments in `base_fhrr_corregida.py`, `exp_F2.py`, `exp_H2.py` describing bugs no longer present in the shipped code were corrected.
- `figures/fig1_phase_diagram.png`: regenerated (cropped title, overlapping x-axis labels) using `figures/_make_fig1.py`, built from the rho values actually present in `data/out_F2.txt`/`data/out_H2.txt`. No generation script existed for figures 2-4, so those are unchanged from the original submission.
- `figures/fig4_comparison_parte1.png` renamed to `fig4_comparison_0059h.png` (no "parte2" exists; the old name implied a missing file).
- Not fixed here — left for the author: `paper/main.tex` has no body text yet (title/abstract only); figures 2-4 reflect one valid pre-fix run and could optionally be regenerated to match the numbers above if an exact match to `data/` is wanted.

---

## Contact

Luciano Benjamín Nieto  
GitHub: [Rylow999](https://github.com/Rylow999)

*Per Aspera, Ad Astra.*
