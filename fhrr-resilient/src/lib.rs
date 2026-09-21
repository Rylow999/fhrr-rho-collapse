//! fhrr-resilient — zero-dependency resilient VSA decoding.
//!
//! POST-EXP-17 WARNING: the kappa threshold routing below is the LEGACY
//! behavior from before we understood the mechanism. Exp 17 showed kappa
//! alone does not predict collapse; Exp 18/27 showed the collapse is about
//! operator placement, not conditioning. This crate is kept as a
//! conservative fallback for cases where the decoder structure cannot be
//! inspected. If you can inspect the decoder, route on placement:
//!   closed-loop ambient M^-1  ->  replace with dual or pure
//!   rank-deficient Gram        ->  pseudo-inverse
//! Kappa remains useful as a coarse diagnostic signal, no more.
//!
//! No external crates: matrices are flat Vec<f64>, row-major.

pub const KAPPA_SAFE: f64 = 1e2;
pub const KAPPA_CRIT_LO: f64 = 1e3;
pub const KAPPA_CRIT_HI: f64 = 1e4;
pub const KAPPA_SING: f64 = 1e12;

/// Routing decision for a Gram matrix of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Well conditioned: direct Gram inverse.
    GramInverse,
    /// Critical band: the empirical failure window. Never invert here.
    PureResonator,
    /// Numerically singular: truncated pseudo-inverse.
    PseudoInverse,
    /// Intermediate zone [1e2, 1e3): still safe to invert, but flagged.
    GramInverseGuarded,
}

pub fn route_kappa(kappa: f64) -> Route {
    if !kappa.is_finite() || kappa >= KAPPA_SING {
        Route::PseudoInverse
    } else if kappa >= KAPPA_CRIT_LO {
        Route::PureResonator // [1e3, 1e15): cubre la banda critica y mas
    } else if kappa >= KAPPA_SAFE {
        Route::GramInverseGuarded
    } else {
        Route::GramInverse
    }
}

// ------------------------- basic dense linear algebra -------------------------

/// A in R^{n x n}, row-major.
pub type Matrix = Vec<f64>;

pub fn mat_vec(a: &Matrix, x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Gram matrix of codevectors (rows of `code`, each of dim d).
pub fn gram(code: &[Vec<f64>]) -> Matrix {
    let n = code.len();
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let g = dot(&code[i], &code[j]);
            m[i * n + j] = g;
            m[j * n + i] = g;
        }
    }
    m
}

/// Solve A x = b with Gaussian elimination + partial pivot. None if singular.
pub fn solve(a: &Matrix, b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = a.clone();
    let mut x = b.to_vec();
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = aug[col * n + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                aug.swap(col * n + k, piv * n + k);
            }
            x.swap(col, piv);
        }
        let d = aug[col * n + col];
        for r in (col + 1)..n {
            let f = aug[r * n + col] / d;
            for k in col..n {
                aug[r * n + k] -= f * aug[col * n + k];
            }
            x[r] -= f * x[col];
        }
    }
    // back substitution
    let mut sol = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= aug[i * n + k] * sol[k];
        }
        sol[i] = s / aug[i * n + i];
    }
    Some(sol)
}

/// Full inverse of A (n x n) via Gauss-Jordan on [A | I]. None if singular.
pub fn inverse(a: &Matrix, n: usize) -> Option<Matrix> {
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    let mut m = a.clone();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                m.swap(col * n + k, piv * n + k);
                inv.swap(col * n + k, piv * n + k);
            }
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            for k in 0..n {
                m[r * n + k] -= f * m[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    Some(inv)
}

// ------------------------- kappa estimation, O(n^2) -------------------------

/// lambda_max via power iteration (SPD matrix -> dominant = largest).
fn lambda_max(a: &Matrix, n: usize, iters: usize) -> f64 {
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 2654435761) % 1000) as f64 + 1.0).collect();
    let mut lam = 0.0;
    for _ in 0..iters {
        let w = mat_vec(a, &v, n);
        let nw = norm(&w);
        if nw < 1e-300 {
            break;
        }
        v = w.iter().map(|x| x / nw).collect();
        lam = dot(&v, &mat_vec(a, &v, n)); // Rayleigh quotient
        let _ = lam;
    }
    dot(&v, &mat_vec(a, &v, n))
}

/// lambda_min via inverse power iteration: solve A w = v each step.
fn lambda_min(a: &Matrix, n: usize, iters: usize) -> Option<f64> {
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 40503 + 17) % 997) as f64 + 1.0).collect();
    for _ in 0..iters {
        let w = solve(a, &v, n)?;
        let nw = norm(&w);
        if nw < 1e-300 {
            return None;
        }
        v = w.iter().map(|x| x / nw).collect();
    }
    // Rayleigh quotient of A on the (approx) smallest eigenvector
    Some(dot(&v, &mat_vec(a, &v, n)).abs())
}

/// Estimated spectral condition number of a symmetric Gram matrix.
/// Returns (kappa, lambda_max, lambda_min). None => numerically singular.
pub fn estimate_kappa(gram_mtx: &Matrix, n: usize, iters: usize) -> Option<(f64, f64, f64)> {
    let lmax = lambda_max(gram_mtx, n, iters);
    let lmin = lambda_min(gram_mtx, n, iters)?;
    if lmin <= 0.0 {
        return None;
    }
    Some((lmax / lmin, lmax, lmin))
}

// ------------------------- router -------------------------

/// What the router computed for one block.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    pub kappa: Option<f64>,
    pub lambda_max: f64,
    pub lambda_min: f64,
    pub route: Route,
}

pub struct Router {
    pub iters: usize,
}

impl Default for Router {
    fn default() -> Self {
        Self { iters: 200 }
    }
}

impl Router {
    /// Inspect a block's codebook and decide the decoding route.
    pub fn decide(&self, codevectors: &[Vec<f64>]) -> RouterDecision {
        let n = codevectors.len();
        let g = gram(codevectors);
        match estimate_kappa(&g, n, self.iters) {
            Some((k, lmax, lmin)) => RouterDecision {
                kappa: Some(k),
                lambda_max: lmax,
                lambda_min: lmin,
                route: route_kappa(k),
            },
            None => RouterDecision {
                kappa: None,
                lambda_max: 0.0,
                lambda_min: 0.0,
                route: Route::PseudoInverse,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }

    fn good_codebook(n: usize, d: usize) -> Vec<Vec<f64>> {
        // pseudo-random decorrelated unit vectors (splitmix64 hash)
        (0..n)
            .map(|i| {
                let mut v: Vec<f64> = (0..d)
                    .map(|j| {
                        let h = mix((i as u64) << 32 | j as u64);
                        ((h % 2000) as f64 - 1000.0) / 1000.0
                    })
                    .collect();
                let nm = norm(&v);
                for x in v.iter_mut() {
                    *x /= nm;
                }
                v
            })
            .collect()
    }

    fn badly_scaled_codebook(n: usize, d: usize) -> Vec<Vec<f64>> {
        // near-duplicate vectors -> ill conditioned Gram
        let mut cb = good_codebook(n, d);
        cb[1] = cb[0].clone();
        cb[1][0] += 1e-6;
        cb
    }

    #[test]
    fn kappa_good_codebook_is_low() {
        let cb = good_codebook(8, 64);
        let g = gram(&cb);
        let (k, _, _) = estimate_kappa(&g, 8, 300).unwrap();
        assert!(k < 1e3, "kappa={k}");
    }

    #[test]
    fn kappa_bad_codebook_is_high() {
        let cb = badly_scaled_codebook(8, 64);
        let g = gram(&cb);
        let (k, _, _) = estimate_kappa(&g, 8, 500).unwrap();
        assert!(k > 1e5, "kappa={k}");
    }

    #[test]
    fn router_fails_over_in_critical_band() {
        let cb = good_codebook(8, 64);
        let r = Router::default().decide(&cb);
        assert!(matches!(
            r.route,
            Route::GramInverse | Route::GramInverseGuarded
        ));
        let cb2 = badly_scaled_codebook(8, 64);
        let r2 = Router::default().decide(&cb2);
        assert!(matches!(
            r2.route,
            Route::PureResonator | Route::PseudoInverse
        ));
    }

    #[test]
    fn inverse_solves() {
        let cb = good_codebook(4, 16);
        let g = gram(&cb);
        let inv = inverse(&g, 4).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = mat_vec(&inv, &b, 4);
        let bb = mat_vec(&g, &x, 4);
        for (u, v) in b.iter().zip(bb.iter()) {
            assert!((u - v).abs() < 1e-8);
        }
    }
}
