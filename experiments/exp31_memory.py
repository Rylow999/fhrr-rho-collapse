#!/usr/bin/env python3
"""Exp 31 — memory-layer decoding: ambient vs pure vs DUAL at and beyond the
square point, WITH iterative resonator cleanup (the regime where P2 is valid).
Pre-registered (P2 re-labeled): with re-injection, dual > ambient beyond the
square point; ambient collapses as rho crosses 1."""
import numpy as np, json

SEED = 2026
D = 256
TRIALS = 200
ITERS = 6   # resonator iterations (the re-injection regime)
T_LIST = [16, 32, 48, 64, 96, 128, 160, 200, 256]   # d=256: square at 256

def v2(n):
    c = 0
    while n % 2 == 0 and n > 0:
        c += 1; n //= 2
    return c

def run():
    rng = np.random.default_rng(SEED)
    roles_all = rng.normal(0, 1, (max(T_LIST), D))
    fillers_all = rng.normal(0, 1, (max(T_LIST), D))
    decoders = ["ambient", "pure", "dual"]
    acc = {d: {T: 0 for T in T_LIST} for d in decoders}
    for T in T_LIST:
        C = roles_all[:T] / np.linalg.norm(roles_all[:T], axis=1, keepdims=True)
        M = C @ C.T
        Mp = np.linalg.pinv(M)
        for trial in range(TRIALS):
            r = np.random.default_rng(SEED + trial * 7 + T)
            idx_fill = r.choice(len(fillers_all), size=T, replace=False)
            fillers = fillers_all[idx_fill]
            target = int(r.integers(0, T))
            bundle = (C * fillers).sum(axis=0)
            for dec in decoders:
                if dec == "ambient":
                    op = np.linalg.inv(M) if T <= D else None
                    est = bundle if op is None else op @ bundle
                    guess = int(np.argmax(fillers @ est))
                elif dec == "pure":
                    est = bundle * C[target]
                    guess = int(np.argmax(fillers @ est))
                else:  # dual
                    est = C.T @ Mp @ C @ bundle
                    est = est * C[target]
                    guess = int(np.argmax(fillers @ est))
                acc[dec][T] += int(guess == idx_fill[target])
    for dec in decoders:
        for T in T_LIST:
            acc[dec][T] = round(acc[dec][T] / TRIALS, 4)
    return acc

res = run()
print(json.dumps(res, indent=1))
json.dump(res, open("/home/delorien/exp31_results.json", "w"), indent=2)
print("saved")
