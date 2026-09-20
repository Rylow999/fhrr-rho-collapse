//! demo: router sobre una Gram Wishart cuadrada (n=32) como la del paper.
//! Reproduce la decision del Exp 14: kappa ~ 1e3-1e4 -> PureResonator.
use fhrr_resilient::{estimate_kappa, gram, route_kappa, Route};

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn gaussian_codebook(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    // Box-Muller sobre splitmix64
    (0..n)
        .map(|i| {
            let mut v = Vec::with_capacity(d);
            for j in 0..d {
                let u1 = ((mix(seed ^ ((i as u64) << 32 | j as u64)) % 1_000_000) as f64 + 1.0)
                    / 1_000_001.0;
                let u2 = ((mix(seed ^ ((j as u64) << 32 | i as u64) ^ 0x9e37) % 1_000_000) as f64
                    + 1.0)
                    / 1_000_001.0;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                v.push(z / (d as f64).sqrt());
            }
            v
        })
        .collect()
}

fn main() {
    let n = 32;
    println!("rho |  d  | median kappa (10 seeds) | route");
    for (d, rho) in [(40usize, 0.8), (36, 0.889), (33, 0.970), (32, 1.0)] {
        let mut ks = Vec::new();
        for s in 0..10u64 {
            let cb = gaussian_codebook(n, d, 42 + s);
            let g = gram(&cb);
            if let Some((k, _, _)) = estimate_kappa(&g, n, 300) {
                ks.push(k);
            }
        }
        ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ks[ks.len() / 2];
        let route = route_kappa(med);
        let tag = match route {
            Route::PureResonator => "PURE RESONATOR (banda critica)",
            Route::GramInverse => "gram inverse",
            Route::GramInverseGuarded => "gram (guarded)",
            Route::PseudoInverse => "pinv",
        };
        println!("{rho:.3} | {d:3} | {med:12.3e}          | {tag}");
    }
}
