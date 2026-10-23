# fhrr-resilient

Resilient decoding for VSA / FHRR (holographic reduced representation)
systems, routed by **operator placement** instead of condition-number
thresholds.

**The finding** (Frame-Dual Stability): when the number of codevectors equals
the block dimension (the *square point*, rho = 1), the ambient Gram decoder
`M^-1 x` collapses (accuracy ~0.15) while the dual projector
`C^T M^-1 C x` stays exact (accuracy ~0.99). This is not a conditioning
artefact: the collapse is caused by *where the operator is placed*, and the
dual projector satisfies `||C^T M^-1 C - I||_F ~ 1e-13` in float64 even at
kappa ~ n^2. The hard-edge law `lambda_min ~ n^-2` (Chen-Liu-Zhou) explains
why: lambda_min hits its extreme-value floor exactly at rho = 1.

## Usage

```rust
use fhrr_resilient::*;

// Describe your decoder block: n codevectors, d dimension
let spec = DecoderSpec {
    n_cv: 16,          // codevectors
    d_blk: 16,         // block dimension (n_cv == d_blk => square point)
    ambient_resolvent: true,
    kappa: None,       // optional diagnostic
    full_rank: true,
};

let decision = route_placement(&spec);
assert_eq!(decision.placement, Placement::Dual);
println!("route: {:?} because {}", decision.placement, decision.reason);
```

The router:

| Situation | Placement | Why |
|---|---|---|
| n_cv > d_blk or singular | `DualPinv` | projection, not inversion |
| square point (rho = 1), ambient | `Dual` | exact projector, norm-1 |
| sub-square (rho < 1) | `AmbientGram` / `Pure` | everything is stable |
| anything else | `Dual` | safe fallback |

Kappa never routes: it only sets `kappa_warning`. Measured across 200/200
seeds (Exp 18-19 of the research repo): every seed collapses at rho=1
regardless of kappa in [2.3e2, 1.9e5].

## Full decoding example

See `examples/fhrr_real_decoding.rs` — a complete FHRR pipeline
(circular-convolution binding, resonator cleanup, three decoders compared)
with deterministic RNG. Run it with:

```bash
cargo run --example fhrr_real_decoding
```


## Spectral profiling and pruning (v0.4.0)

The campaign findings (see the repository README) are exposed as a small API:

```rust
use fhrr_resilient::*;

// 1) measure per-head key-gram spectra at your operating point T
let keys: Vec<Vec<f64>> = /* unit-normalized key rows, T rows x d_head cols */;
let (lmin, lmax) = head_key_spectrum(&keys).unwrap();

// 2) classify the slice
let profile = classify_profile(&lmins_per_head);

// 3) get the recipe the science prescribes
match prune_recommendation(&profile) {
    PruneAdvice::RankByLmin => { /* rank heads by lmin ascending; prune lowest (Exp 32) */ }
    PruneAdvice::ReMeasureSubSquare => { /* collapsed at square: re-measure at T = d_head/2 (Exp 34) */ }
    PruneAdvice::NotApplicable => { /* spectral pruning N/A for this slice */ }
}
```

The recommendation encodes the **law of training exposure** (Exp 37): lightly
trained models keep spectral variance at the square point, so ranking works
there; heavily trained models are fully collapsed, and the signal moves to
the sub-square regime.

## Test

```bash
cargo test
```

4/4 tests: square collapse routing, dual projector exactness
(< 1e-6 in f64), overcomplete routing, and kappa-warns-but-does-not-route.

## Research context

This crate is the engineering output of the framework developed in
[Rylow999/fhrr-rho-collapse](https://github.com/Rylow999/fhrr-rho-collapse),
where the collapse law is characterized across algebras (HRR, BSC, MAP, FHRR)
and observers, with 28 documented experiments and independent replication.

## License

MIT

## Motto

*Per Aspera, Ad Astra.*
