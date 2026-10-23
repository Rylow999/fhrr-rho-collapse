
// === v0.4.0: spectral profile + pruning recommendation (campaign 29-38) ===

/// Per-head spectral measurement of a key-gram (keys already unit-normalized rows).
/// Returns (lambda_min, lambda_max) of the key-gram K K^T.
/// Following the campaign protocol: measure at T tokens, d_head dimension.
pub fn head_key_spectrum(keys: &[Vec<f64>]) -> Option<(f64, f64)> {
    let n = keys.len();
    if n == 0 { return None; }
    let g = gram(keys);
    // symmetric eigenvalues via Jacobi rotation (n x n, small)
    let (mut a, _) = jacobi_eig(&g, n)?;
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Some((a[0], a[n - 1]))
}

/// Jacobi eigenvalue iteration for symmetric matrices (values only).
fn jacobi_eig(a_in: &Matrix, n: usize) -> Option<(Vec<f64>, ())> {
    let mut a = a_in.clone();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n { for q in (p + 1)..n { off += a[p * n + q] * a[p * n + q]; } }
        if off < 1e-24 { break; }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p * n + q].abs() < 1e-18 { continue; }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    Some(((0..n).map(|i| a[i * n + i]).collect(), ()))
}

/// The collapse profile of a model slice: fraction of heads rank-deficient
/// at the measured operating point T. This is Exp 29/30/33/36 as an API.
pub struct SpectralProfile {
    pub heads: usize,
    pub collapsed: usize,          // |lmin| < COLLAPSE_EPS at the measured T
    pub lmin_median: f64,
    pub regime: &'static str,      // "variance" | "collapsed"
}

pub const COLLAPSE_EPS: f64 = 1e-4;

/// Given per-head lambda_min measurements (from `head_key_spectrum` at
/// operating point T), classify the regime.
pub fn classify_profile(lmins: &[f64]) -> SpectralProfile {
    let collapsed = lmins.iter().filter(|l| l.abs() < COLLAPSE_EPS).count();
    let mut sorted = lmins.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if lmins.is_empty() { f64::NAN } else { sorted[lmins.len() / 2] };
    let regime = if collapsed * 2 >= lmins.len() { "collapsed" } else { "variance" };
    SpectralProfile { heads: lmins.len(), collapsed, lmin_median: med, regime }
}

/// The pruning recommendation (Exp 32 + 34 recipe as code).
///
/// Law of training exposure (Exp 37): heavily trained models are fully
/// collapsed at the square point, so the criterion loses signal there.
/// The measured profile tells you which operating point to trust:
///   - regime "variance"  -> rank by lmin; prune the lowest (Exp 32 recipe)
///   - regime "collapsed"  -> re-measure at sub-square T = d_head/2 (Exp 34 recipe);
///                            if still collapsed, the criterion is N/A for this slice.
pub enum PruneAdvice {
    /// Rank heads by lambda_min ascending; the lowest are pruning candidates.
    RankByLmin,
    /// Fully collapsed at the square point: measure at T = d_head/2 and call again.
    ReMeasureSubSquare,
    /// Collapsed even sub-square: spectral pruning not applicable here.
    NotApplicable,
}

pub fn prune_recommendation(profile: &SpectralProfile) -> PruneAdvice {
    if profile.heads == 0 { return PruneAdvice::NotApplicable; }
    let frac = profile.collapsed as f64 / profile.heads as f64;
    if frac < 0.5 {
        PruneAdvice::RankByLmin
    } else if profile.regime == "collapsed" {
        // Caller measured at square; the sub-square check is theirs to run —
        // we signal the recipe from Exp 34.
        PruneAdvice::ReMeasureSubSquare
    } else {
        PruneAdvice::NotApplicable
    }
}

#[cfg(test)]
mod spectral_tests {
    use super::*;

    #[test]
    fn well_conditioned_profile_has_variance() {
        // 8 random unit vectors in d=64: square point is 64, this is far sub-square
        let lmins: Vec<f64> = (0..8)
            .map(|i| {
                let keys = mk_keys(8 + i, 64);
                head_key_spectrum(&keys).map(|(l, _)| l).unwrap()
            })
            .collect();
        let p = classify_profile(&lmins);
        assert_eq!(p.regime, "variance");
        assert!(matches!(prune_recommendation(&p), PruneAdvice::RankByLmin));
    }

    #[test]
    fn square_random_profile_collapses() {
        // 64 random unit vectors in d=64: the square point (Exp 16/28: lmin ~ n^-2)
        let keys = mk_keys(64, 64);
        let (lmin, _) = head_key_spectrum(&keys).unwrap();
        // hard edge: lmin is tiny positive or numerically zero
        assert!(lmin < 1.0, "square lmin should be small, got {}", lmin);
        let p = classify_profile(&[lmin; 16]); // 16 heads all like this
        assert_eq!(p.regime, "collapsed");
        assert!(matches!(prune_recommendation(&p), PruneAdvice::ReMeasureSubSquare));
    }

    fn mk_keys(n: usize, d: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| {
            let mut s = 0x9E3779B97F4A7C15u64
                .wrapping_add((i as u64).wrapping_mul(0xBF58476D1CE4E5B9));
            let mut v: Vec<f64> = (0..d).map(|_| {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                (s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            }).collect();
            let nm = norm(&v);
            for x in v.iter_mut() { *x /= nm; }
            v
        }).collect()
    }
}
