//! fhrr_real_decoding — Benchmark del teorema Frame-Dual Stability (v6)
use fhrr_resilient::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}
fn rand_unit(n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let nm = norm(&v);
    for x in v.iter_mut() { *x /= nm; }
    v
}
fn bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    (0..n).map(|k| (0..n).map(|j| a[(k + n - j) % n] * b[j]).sum::<f64>() / (n as f64).sqrt()).collect()
}
fn unbind(s: &[f64], r: &[f64]) -> Vec<f64> {
    let n = s.len();
    (0..n).map(|k| (0..n).map(|j| s[(k + j) % n] * r[j]).sum::<f64>() / (n as f64).sqrt()).collect()
}
fn cleanup(v: &[f64], syms: &[Vec<f64>]) -> usize {
    let nv = norm(v) + 1e-12;
    let mut best = 0;
    let mut bs = -2.0;
    for (i, s) in syms.iter().enumerate() {
        let sim = dot(v, s) / nv;
        if sim > bs { bs = sim; best = i; }
    }
    best
}
fn min_eig(g: &Matrix, n: usize) -> f64 {
    let mut rng = Rng(777);
    let mut v = rand_unit(n, &mut rng);
    for _ in 0..400 {
        if let Some(w) = solve(g, &v, n) {
            let nw = norm(&w);
            if nw < 1e-300 { break; }
            v = w.iter().map(|x| x / nw).collect();
        }
    }
    dot(&v, &mat_vec(g, &v, n)).abs()
}
fn step(s: &[f64], roles: &[Vec<f64>], est: &[Vec<f64>], r: usize, op: Option<&Matrix>) -> Vec<f64> {
    let mut o = s.to_vec();
    for (j, e) in est.iter().enumerate() {
        if j != r {
            let b = bind(&roles[j], e);
            for i in 0..o.len() { o[i] -= b[i]; }
        }
    }
    let mut fj = unbind(&o, &roles[r]);
    if let Some(m) = op { fj = mat_vec(m, &fj, m.len().isqrt()); }
    fj
}
fn bench(n_runs: u64, d_blk: usize, n_sym: usize, facts: usize, iters: usize, label: &str) {
    let names = ["ambient M^-1 ", "dual proj    ", "pure         "];
    let mut acc = [0.0f64; 3];
    let mut lams = Vec::new();
    for run in 0..n_runs {
        let mut rng = Rng(0xA11CE + run * 7919);
        let syms: Vec<Vec<f64>> = (0..n_sym).map(|_| rand_unit(d_blk, &mut rng)).collect();
        let roles = vec![rand_unit(d_blk, &mut rng), rand_unit(d_blk, &mut rng)];
        let g = gram(&syms);
        lams.push(min_eig(&g, n_sym));
        let ambient_op = if n_sym == d_blk { inverse(&g, n_sym) } else { None };
        let dual_op = dual_projector(&syms);
        for fact in 0..facts as u64 {
            let f0 = (fact * 7 + 1) as usize % n_sym;
            let mut f1 = (fact * 13 + 5) as usize % n_sym;
            if f1 == f0 { f1 = (f1 + 1) % n_sym; }
            let mut s = bind(&roles[0], &syms[f0]);
            let b1 = bind(&roles[1], &syms[f1]);
            for i in 0..d_blk { s[i] += b1[i]; }
            let ops = [ambient_op.as_ref(), dual_op.as_ref(), None];
            for (mode, corr) in [0usize, 1, 2].iter().zip(ops) {
                let mut irng = Rng(0xBEE5 + fact * 131 + run);
                let mut est = vec![rand_unit(d_blk, &mut irng), rand_unit(d_blk, &mut irng)];
                for _ in 0..iters {
                    for rr in 0..2 {
                        let idx = cleanup(&step(&s, &roles, &est, rr, corr), &syms);
                        est[rr] = syms[idx].clone();
                    }
                }
                let ok = (cleanup(&est[0], &syms) == f0) as u8 + (cleanup(&est[1], &syms) == f1) as u8;
                acc[*mode] += ok as f64 / 2.0;
            }
        }
    }
    let tot = (n_runs * facts as u64) as f64;
    lams.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if lams.is_empty() { f64::NAN } else { lams[lams.len() / 2] };
    let ns = n_sym as f64;
    println!("[{}] rho={:.2} lam_med={:.3e} n^2*lam_med={:.3} (ln2=0.693)",
        label, ns / d_blk as f64, med, med * ns * ns);
    for i in 0..3 {
        println!("  {}: {:.3}", names[i], acc[i] / tot.max(1.0));
    }
}
fn main() {
    println!("== FHRR/VSA benchmark — Frame-Dual Stability ==");
    bench(4, 24, 16, 14, 20, "sub-square rho<1");
    bench(6, 16, 16, 16, 25, "square rho=1");
    println!("OK.");
}
