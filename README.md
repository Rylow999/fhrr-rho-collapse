# Operator placement, not representation

**The collapse of vector-symbolic decoding at the square point is a property of the observer, not the signal** — and the same geometry governs the keys of trained transformers. This repository is the complete research program behind that claim: 38 experiments across VSA algebras, trained language models, and controlled interventions, with every result labeled under an honest-audit protocol (pre-registration, evidence classes, explicit limitations).

## Summary

1. **The frame-dual observation (VSA core).** When the number of codevectors equals the block dimension (the *square point*, rho = 1), the standard ambient decoder `M^-1 x` collapses (accuracy ~0.15) while the dual projector `C^T M^-1 C x` stays exact (~0.99). Controlled intervention on 200/200 seeds: the trigger is *operator placement*, not the condition number. The hard edge of Wishart matrices (lambda_min ~ n^-2) explains the geometry (Exp 16, 28; Chen-Liu-Zhou, Edelman).
2. **The square point is where attention keys live (transformers).** Sweeping context length T in three architecture families (GPT-2, Pythia/GPTNeoX, OPT), the smallest eigenvalue of every head's key-gram crosses zero exactly at T = d_head and stays rank-deficient beyond (Exp 29, 30). The bridge hypothesis `trained keys ~ quasi-random frames` holds cross-family.
3. **The law of training exposure.** Pythia-160m (12 layers, ~84B tokens) is fully collapsed where GPT-2 124M (12 layers, ~10B tokens) retains variance — same depth, different training scale (Exp 37). The driver is *training exposure*, not depth: keys converge to the corpus's intrinsic redundancy as tokens accumulate.
4. **The mechanism is content, not position.** Zeroing or randomizing positional embeddings does not restore any spectral variance (Exp 38). The collapse is the redundancy of language itself, learned faithfully. Positional surgery will not fix it; decoder-side defenses (dual/resonator) are the only route.
5. **A zero-training head-importance signal (corrected direction).** Per-head lambda_min carries functional signal — but in the direction OPPOSITE to our initial pre-registration: real pruning shows that heads with the LOWEST lambda_min are the vital, selective ones (pruning them is catastrophic: perplexity 341k vs 18k random, Exp 41), while the most diffuse heads (highest lambda_min) are the safe pruning targets (perplexity 8k, better than random). This inverts Exp 32's correlation reading and CONFIRMS the direction reported by State Rank Stratification in linear attention from an independent measurement.

## Key results

| Finding | Evidence | Status |
|---|---|---|
| Placement causes the collapse (same codebook, only placement changes) | Exp 18/19: 200/200 + 100/100 seeds, +0.82 paired | Proven (controlled intervention) |
| Hard edge lambda_min ~ n^-2 at the square point | Exp 16, 28: 200-1200 seeds/size, exponent -2.058 | Proven anchor (CLZ) + strong evidence |
| Universal crossing at T = d_head in trained models | Exp 29/30: GPT-2, Pythia, OPT | Confirmed (3 families) |
| Law of training exposure (not depth) | Exp 37: 12L-vs-12L natural experiment | Confirmed (corpus confound declared) |
| Collapse driven by content redundancy, not position | Exp 38: wpe zeroed/randomized controls | Confirmed (single model/text) |
| Spectral pruning criterion (square point) | Exp 32: 50x separation, Spearman 0.558 | Confirmed (lightly trained) |
| Sub-square criterion for heavily trained models | Exp 34: pearson 0.236, p=0.018; head-stability r=0.66 | Confirmed |
| Layer-level spectral criterion | Exp 35: no signal (depth-importance confound r=0.85) | Refuted (honest negative) |
| Dual single-shot does not beat pure | Exp 31: pure degrades gracefully, dual adds pinv noise | P2 re-labeled: iterative regime only |

## The Rust crate: fhrr-resilient

Zero-dependency (std only) decoder router for VSA/FHRR blocks. Routes by operator placement (`Dual`, `AmbientGram`, `Pure`, `DualPinv`) with kappa demoted to a diagnostic. Validated in a real FHRR pipeline; square-point ambient collapse and dual recovery reproduce in the crate's own benchmark.

```bash
cargo add fhrr-resilient   # once published; or path/git dependency today
cargo test                 # 4/4 tests
cargo run --example fhrr_real_decoding   # the square-point benchmark
```

## How to run the research artifacts

```bash
git clone https://github.com/Rylow999/fhrr-rho-collapse.git
cd fhrr-rho-collapse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python verify.py                        # 31 checks against data/*.json
pytest tests/ -v                        # Python tests
cd fhrr-resilient && cargo test         # Rust tests

# Cross-architecture key spectra (Exp 29/30)
python experiments/exp29_cross_arch.py gpt2 EleutherAI/pythia-70m facebook/opt-125m
python experiments/exp30_sweep_gpt2.py

# Spectral pruning criterion (Exp 32/34)
python experiments/exp32_pruning.py     # GPT-2 124M square point
python experiments/exp34_subsquare.py   # sub-square recipe

# Confound and mechanism (Exp 37/38)
python experiments/exp37_confound.py
python experiments/exp38_mechanism.py
```

## Repository layout

```
experiments/    # Exp B..38: every experiment script + its results JSON
data/           # VSA experiment data (Exp 1-28 of the paper)
docs/           # transformers_frame_dual.md: the pre-registered P1-P4 + results
fhrr-resilient/ # the Rust crate
src/            # VSA harness (base_fhrr, exp_V1-V28)
tests/          # Python test suite
paper/          # LaTeX paper (main.tex, references.bib)
figures/        # all figures
```

## Evidence and status

Every claim maps to a versioned artifact in `experiments/` or `data/`. The Method Integral protocol (pre-registration, evidence classes, explicit limitations) governs all of it; `honest_label` fields in each experiment JSON carry the per-experiment audit.

| Claim | Class | Evidence | Status |
|---|---|---|---|
| Identity C^T M^-1 C = I (invertible square codebook) | algebra | \|P-I\|_F ~ 4e-14 in f64 | Proven |
| Placement causes the VSA collapse (only variable changed) | controlled intervention | Exp 18/19: 200/200 + 100/100 seeds, +0.82 paired | Proven |
| Hard edge lambda_min ~ n^-2 at the square point | theorem + measurement | CLZ anchor; Exp 28: -2.058 (200-1200 seeds/size) | Proven anchor + strong evidence |
| Kappa is NOT the trigger | refutation | Exp 17: 40 seeds, all collapse regardless of kappa | Refutative evidence |
| Universal crossing at T = d_head (trained keys ~ frames) | cross-family measurement | Exp 29/30: GPT-2, Pythia, OPT | Confirmed (bridge hypothesis declared) |
| Law of training exposure (not depth) | natural experiment | Exp 37: Pythia-160m 12L collapsed vs GPT-2 124M 12L variance | Confirmed (corpus confound declared) |
| Collapse driven by content, not position | ablation controls | Exp 38: wpe zeroed 144/144, wpe random 142/144 | Confirmed (single model/text) |
| Spectral head signal at square | real pruning test | Exp 32 correlation (Spearman 0.558) INVERTED by Exp 41 real-pruning: low lmin = vital, high lmin = prunable | Corrected (direction reversed) |
| Sub-square signal for heavily trained | importance correlation | Exp 34: pearson 0.236 (p=0.018), head-stable r=0.66 | Confirmed (as signal, direction as above) |
| Layer-level spectral criterion | refutation | Exp 35: depth-importance confound r=0.85 | Refuted (honest negative) |
| Dual single-shot beats pure | refutation + re-label | Exp 31: pure degrades gracefully, dual adds pinv noise | P2 re-labeled: iterative regime only |

Open items: P1-P4 pre-registrations at deployment scale (needs GPU); KV-cache compression pilot; text-sensitivity sweep of the collapse regime; circuit-level interpretation of M^-1 heads.

## Citation

```bibtex
@software{nieto2026fhrr,
  author = {Nieto, Luciano Benjam{\'i}n},
  title = {fhrr-rho-collapse: operator placement, not representation},
  year = {2026},
  url = {https://github.com/Rylow999/fhrr-rho-collapse}
}
```

## Acknowledgments

Anonymous external reviewers whose critiques sharpened the formulation.

## Motto

*Per Aspera, Ad Astra.*
