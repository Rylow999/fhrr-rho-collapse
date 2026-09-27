//! fhrr-resilient — v3.0 FINAL PRODUCTION
//! Teorema Frame-Dual Stability validado en pipeline real.
//! El routing es por PLACEMENT, no por κ. κ = diagnostico solamente.

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
