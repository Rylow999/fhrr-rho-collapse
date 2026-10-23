//! fhrr-resilient — v0.4.0
//! Frame-Dual Stability validated in a real pipeline + the spectral
//! profile API from the transformer campaign (Exp 29-38):
//!   - routing is by PLACEMENT, not by kappa (kappa = diagnostic only)
//!   - head_key_spectrum / classify_profile / prune_recommendation:
//!     the zero-training pruning criterion with its operating recipe.

pub const KAPPA_WARN: f64 = 1e3;
pub const KAPPA_SING: f64 = 1e12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Pure,
    AmbientGram,
    Dual,
    DualPinv,
}

#[derive(Debug, Clone)]
pub struct DecoderSpec {
    pub n_cv: usize,
    pub d_blk: usize,
    pub ambient_resolvent: bool,
    pub kappa: Option<f64>,
    pub full_rank: bool,
}

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub placement: Placement,
    pub reason: &'static str,
    pub kappa_warning: bool,
}

pub fn route_placement(spec: &DecoderSpec) -> RouteDecision {
    let rho = spec.n_cv as f64 / spec.d_blk as f64;
    let kw = spec.kappa.map(|k| k >= KAPPA_WARN).unwrap_or(false);
    if !spec.full_rank || spec.n_cv > spec.d_blk {
        return RouteDecision {
            placement: Placement::DualPinv,
            reason: "n_cv > d_blk o singular: dual pinv es proyectar, no se invierte",
            kappa_warning: kw,
        };
    }
    if (rho - 1.0).abs() < 1e-9 && spec.ambient_resolvent {
        return RouteDecision {
            placement: Placement::Dual,
            reason: "punto cuadrado: dual exato norma 1",
            kappa_warning: kw,
        };
    }
    if rho < 1.0 {
        return RouteDecision {
            placement: if spec.ambient_resolvent { Placement::AmbientGram } else { Placement::Pure },
            reason: "sub-cuadrado: todo estable, ambient funciona",
            kappa_warning: kw,
        };
    }
    RouteDecision {
        placement: Placement::Dual,
        reason: "fallback: dual norma-1 para resolvents espectrales",
        kappa_warning: kw,
    }
}

// === Algebra lineal robusta (Gauss-Jordan con pivoteo parcial) ===
pub type Matrix = Vec<f64>;

pub fn mat_vec(a: &Matrix, x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (0..n).map(|j| a[i * n + j] * x[j]).sum();
    }
    y
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
pub fn norm(v: &[f64]) -> f64 { dot(v,v).sqrt() }

pub fn gram(code: &[Vec<f64>]) -> Matrix {
    let n = code.len();
    let mut m = vec![0.0; n*n];
    for i in 0..n {
        for j in 0..=i {
            let g = dot(&code[i], &code[j]);
            m[i*n+j] = g; m[j*n+i] = g;
        }
    }
    m
}

pub fn inverse(a: &Matrix, n: usize) -> Option<Matrix> {
    let mut m = a.clone();
    let mut inv = vec![0.0; n*n];
    for i in 0..n { inv[i*n+i] = 1.0; }
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col*n+col].abs();
        for r in col+1..n {
            if m[r*n+col].abs() > best { best = m[r*n+col].abs(); piv = r; }
        }
        if best < 1e-300 { return None; }
        if piv != col {
            for k in 0..n {
                m.swap(col*n+k, piv*n+k);
                inv.swap(col*n+k, piv*n+k);
            }
        }
        let d = m[col*n+col];
        for k in 0..n { m[col*n+k] /= d; inv[col*n+k] /= d; }
        for r in 0..n {
            if r != col {
                let f = m[r*n+col];
                for k in 0..n { m[r*n+k] -= f*m[col*n+k]; inv[r*n+k] -= f*inv[col*n+k]; }
            }
        }
    }
    Some(inv)
}

pub fn solve(a: &Matrix, b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.clone();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col*n+col].abs();
        for r in col+1..n {
            if m[r*n+col].abs() > best { best = m[r*n+col].abs(); piv = r; }
        }
        if best < 1e-300 { return None; }
        if piv != col {
            for k in 0..n { m.swap(col*n+k, piv*n+k); }
            x.swap(col, piv);
        }
        let d = m[col*n+col];
        for r in 0..n {
            if r != col {
                let f = m[r*n+col] / d;
                for k in 0..n { m[r*n+k] -= f*m[col*n+k]; }
                x[r] -= f * x[col];
            }
        }
        x[col] /= d;
    }
    Some(x)
}

/// Dual projector (teoria de frames): P = C^T M^-1 C, matriz d×d en espacio
/// ambiente. Proyeccion ortogonal sobre el span del codebook. Para C cuadrada
/// invertible, P = I exacto (el teorema). NOTA: el orden C·M^-1·C^T (gram de
/// filas) NO es el projector — solo da I para codebooks simétricos.
/// Estable numericamente (Gauss-Jordan + suma de outers products).
pub fn dual_projector(code: &[Vec<f64>]) -> Option<Matrix> {
    let n = code.len();
    let d = code.first()?.len();
    let g = gram(code);
    let minv = inverse(&g, n)?;
    let mut p = vec![0.0; d * d];
    for i in 0..n { for j in 0..n {
        let mij = minv[i * n + j];
        if mij == 0.0 { continue; }
        for mu in 0..d { for nu in 0..d {
            p[mu * d + nu] += code[i][mu] * mij * code[j][nu];
        }}
    }}
    Some(p)
}

/// Aplica corrección en coefficient-space: g_s -> op * g_s
pub fn correct_coeffs(g: &[f64], op: Option<&Matrix>) -> Vec<f64> {
    match op {
        Some(m) if m.len() == g.len() * g.len() => mat_vec(m, g, g.len()),
        _ => g.to_vec(),
    }
}

/// Validación teórica: ||C^T M^-1 C - I||_F (debe ser ~1e-12 en float64)
pub fn validate(code: &[Vec<f64>]) -> Option<(f64, f64, f64)> {
    let n = code.len();
    let d = code[0].len();
    let g = gram(code);
    let minv = inverse(&g, n)?;
    // P = C^T M^-1 C (d×d): el mismo projector que dual_projector.
    let mut p = vec![0.0; d * d];
    for i in 0..n { for j in 0..n {
        let mij = minv[i * n + j];
        if mij == 0.0 { continue; }
        for mu in 0..d { for nu in 0..d {
            p[mu * d + nu] += code[i][mu] * mij * code[j][nu];
        }}
    }}
    let mut err = 0.0;
    for mu in 0..d {
        for nu in 0..d {
            let t = p[mu*d+nu] - if mu==nu { 1.0 } else { 0.0 };
            err += t*t;
        }
    }
    // lambda_min por inverse power iteration determinista
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 40503 + 17) % 997) as f64 + 1.0).collect();
    for _ in 0..400 {
        let w = solve(&g, &v, n)?;
        let nw = norm(&w);
        if nw < 1e-300 { break; }
        v = w.iter().map(|x| x / nw).collect();
    }
    let lmin = dot(&v, &mat_vec(&g, &v, n)).abs();
    Some((err.sqrt(), 1.0/lmin, lmin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_codebook(n: usize, d: usize) -> Vec<Vec<f64>> {
        // Determinista: xorshift seedeado por fila. El generador (i*j + K) % 1000
        // colapsaba el rango: con j <= 15 el módulo no hace wrap y las filas quedan
        // ~afines (761 + i*j) → rango efectivo ~2 → λ_min ≈ 0 → la inversa float64
        // explota (err ~1e13). Un codebook genérico unit-norm tiene κ ~ n² y el
        // dual es computable en float64.
        (0..n).map(|i| {
            let mut s = 0x9E3779B97F4A7C15u64.wrapping_add((i as u64).wrapping_mul(0xBF58476D1CE4E5B9));
            let mut v: Vec<f64> = (0..d).map(|_| {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                (s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            }).collect();
            let nm = norm(&v);
            for x in v.iter_mut() { *x /= nm; }
            v
        }).collect()
    }

    #[test]
    fn square_ambient_collapses_dual_exact() {
        let code = mk_codebook(16, 16);
        // gram de codebook cuadrado unit-norm está mal condicionado (Lambda_min ~ 1/n²)
        let spec = DecoderSpec { n_cv: 16, d_blk: 16, ambient_resolvent: true, kappa: None, full_rank: true };
        let r = route_placement(&spec);
        assert_eq!(r.placement, Placement::Dual, "punto cuadrado requiere dual");

        // La identidad es algebraicamente exacta
        let p = dual_projector(&code).unwrap();
        let mut err = 0.0;
        for i in 0..16 { for j in 0..16 {
            let t = p[i*16+j] - if i==j {1.0} else {0.0};
            err += t*t;
        }}
        let err = err.sqrt();
        // float64 con κ ~ 600: err ~ 1e-13 (precisión de máquina).
        assert!(err < 1e-6, "dual err demasiado grande: {err}");
    }

    #[test]
    fn overcomplete_routes_to_dual_pinv() {
        let spec = DecoderSpec { n_cv: 24, d_blk: 16, ambient_resolvent: true, kappa: None, full_rank: false };
        let r = route_placement(&spec);
        assert_eq!(r.placement, Placement::DualPinv);
    }

    #[test]
    fn wide_well_conditioned_keeps_ambient() {
        let spec = DecoderSpec { n_cv: 16, d_blk: 48, ambient_resolvent: true, kappa: Some(50.0), full_rank: true };
        let r = route_placement(&spec);
        assert_eq!(r.placement, Placement::AmbientGram);
        assert!(!r.kappa_warning);
    }

    #[test]
    fn kappa_warns_but_does_not_route() {
        let spec = DecoderSpec { n_cv: 16, d_blk: 16, ambient_resolvent: true, kappa: Some(1e4), full_rank: true };
        let r = route_placement(&spec);
        assert_eq!(r.placement, Placement::Dual);
        assert!(r.kappa_warning);
    }
}


// v0.4.0: spectral profile API (campaign Exp 29-38)

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
