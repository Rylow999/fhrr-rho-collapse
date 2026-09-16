#!/usr/bin/env python3
"""
Exp 7: Observador aprendido (MLP) — ¿el espacio contiene la información que
el decoder gram no puede ver?

Hipótesis del observador: si un MLP entrenado logra accuracy alta donde `gram`
colapsa (ρ=1, ρ<1), entonces la información ESTÁ en el vector, solo que el
decoder específico no puede extraerla.

Pipeline:
  1. Para cada caso del grid V3 (n, K, N):
     - Generar bundle (semilla 7)
     - Generar 2000 hechos de entrenamiento + 200 de test (semilla 42)
     - Codificar cada fact al vector HRR (shape: N-dim)
  2. Entrenar MLP: input N → hidden 256/256 → output n_roles × 16 logits
     (cada rol es un head de clasificación independiente)
  3. Loss: cross-entropy promedio sobre roles
  4. Evaluar accuracy por rol (exact match en cada slot)
  5. Comparar con baseline: gram / pure / pinv en los mismos hechos

Requisitos: numpy, torch (CPU ok, dataset chico)
Si torch no está disponible, fallback a sklearn MLPClassifier multi-output.

Salidas:
  data/out_V7_mlp.txt
  data/mlp_decoder_results.json
  figures/fig_mlp_decoder.png
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# path setup para importar exp_observer_taxonomy_v3
sys.path.insert(0, str(Path(__file__).parent))
from exp_observer_taxonomy_v3 import BundleV3, make_fact, accuracy, CFG6

# ================================================================ MLP
class MLPDecoder(nn.Module):
    """6 heads de clasificación, uno por rol."""
    def __init__(self, N, n_roles, vocab_size=16):
        super().__init__()
        self.n_roles = n_roles
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(N, 384), nn.GELU(),
            nn.Linear(384, 256), nn.GELU(),
            nn.Linear(256, n_roles * vocab_size),
        )

    def forward(self, x):
        out = self.net(x)              # (B, n_roles * vocab)
        return out.view(-1, self.n_roles, self.vocab_size)


def train_mlp(X_train, Y_train, X_test, Y_test, N, n_roles, vocab_size,
              epochs=80, lr=1e-3, batch=64, seed=0):
    """Entrena y devuelve accuracy de test por rol."""
    torch.manual_seed(seed)
    device = "cpu"
    model = MLPDecoder(N, n_roles, vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Yt = torch.tensor(Y_train, dtype=torch.long)  # (n_train, n_roles)
    Xe = torch.tensor(X_test, dtype=torch.float32)
    Ye = torch.tensor(Y_test, dtype=torch.long)

    n_tr = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        tot_loss = 0.0
        for i in range(0, n_tr, batch):
            idx = perm[i:i+batch]
            xb, yb = Xt[idx], Yt[idx]
            logits = model(xb)                     # (B, R, V)
            loss = sum(loss_fn(logits[:, r, :], yb[:, r]) for r in range(n_roles)) / n_roles
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
        sched.step()
        if (ep+1) % 20 == 0 or ep == epochs-1:
            print(f"    época {ep+1:3d}/{epochs} loss={tot_loss/(n_tr/batch):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(Xe)
        pred = logits.argmax(dim=-1)               # (B, R)
        accs_per_role = (pred == Ye).float().mean(dim=0)
        joint_acc = (pred == Ye).all(dim=1).float().mean().item()
    return accs_per_role.cpu().numpy(), joint_acc


def facts_to_XY(bundle, facts, vocab_map):
    """facts (list of dict rol->simbolo) -> X (n_facts, N) + Y (n_facts, n_roles)"""
    X, Y = [], []
    for f in facts:
        c = bundle.encode_fact(f)
        X.append(c)
        Y.append([vocab_map[r].index(f[r]) for r in bundle.role_names])
    return np.array(X), np.array(Y)


# ================================================================ Main
def main():
    if not TORCH_OK:
        print("ERROR: torch no disponible. pip install torch")
        return

    t0 = time.time()
    rng = random.Random(42)

    # Grid idéntico a V3 y Exp8 v3
    cases = [
        ("n=6 K=1 N=128 (rho=0.75)", 6, 1, 128),
        ("n=6 K=3 N=120 (rho=0.80)", 6, 3, 120),
        ("n=6 K=3 N=96  (rho=1.00, cuadrado)", 6, 3, 96),
        ("n=6 K=3 N=72  (rho=1.33)", 6, 3, 72),
        ("n=6 K=2 N=64  (rho=1.50)", 6, 2, 64),
    ]

    print("=" * 78)
    print("EXP 7: OBSERVADOR APRENDIDO (MLP)")
    print(f"  Casos: {len(cases)}")
    print(f"  Train: 1500 facts | Test: 300 facts")
    print("=" * 78)

    results = []
    for label, n, K, N in cases:
        print(f"\n=== {label} ===")
        bundle = BundleV3(7, K, N)
        rho_info = bundle.rho_per_block()

        # Vocab por rol (16 codewords)
        vocab_map = {r: sorted(bundle.codebooks[r].keys()) for r in bundle.role_names}

        # Generar datos
        train_facts = [make_fact(rng) for _ in range(1500)]
        test_facts = [make_fact(rng) for _ in range(300)]
        print(f"  codificando train...")
        X_train, Y_train = facts_to_XY(bundle, train_facts, vocab_map)
        print(f"  codificando test...")
        X_test, Y_test = facts_to_XY(bundle, test_facts, vocab_map)

        # MLP
        print(f"  entrenando MLP...")
        role_accs, joint = train_mlp(X_train, Y_train, X_test, Y_test,
                                     N=N, n_roles=len(bundle.role_names),
                                     vocab_size=16,
                                     epochs=80)
        role_acc_mean = float(role_accs.mean())
        print(f"  MLP: acc_por_rol={role_acc_mean:.3f} | joint_acc={joint:.3f}")

        # Baselines sobre los MISMOS test facts
        print(f"  baselines (gram/pure/pinv/gradient)...")
        baseline_accs = {}
        for mode in ("gram", "pure", "pinv", "gradient"):
            sub = []
            for f in test_facts[:60]:  # 60 facts basta (resonator es lento)
                c = bundle.encode_fact(f)
                dec = bundle.decode_fact(c, T=100, mode=mode)
                sub.append(accuracy(dec, f))
            baseline_accs[mode] = (float(np.mean(sub)), float(np.std(sub)))
            print(f"    {mode:>8}: acc={baseline_accs[mode][0]:.3f}")

        results.append({
            "case": label, "n": n, "K": K, "N": N, "BLK": bundle.BLK,
            "rho": {str(k): round(v, 3) for k, v in rho_info.items()},
            "mlp_role_acc": role_acc_mean,
            "mlp_joint_acc": float(joint),
            "mlp_role_accs": role_accs.tolist(),
            "baseline": baseline_accs,
        })

    # Guardar resultados
    out_path = Path(__file__).parent.parent / "data" / "mlp_decoder_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out_path}")

    # ---- Figura
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(results))
    modes = ("mlp", "gram", "pure", "pinv", "gradient")
    width = 0.15
    for i, mode in enumerate(modes):
        ys = []
        for r in results:
            if mode == "mlp":
                ys.append(r["mlp_role_acc"])
            else:
                ys.append(r["baseline"][mode][0])
        ax.bar(x + (i - 2) * width, ys, width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([r["case"].split("(")[-1].rstrip(")") for r in results],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("MLP decoder vs baselines (resonator) — accuracy por rol / global")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "fig_mlp_decoder.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figura: {fig_path}")
    print(f"\nTiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
