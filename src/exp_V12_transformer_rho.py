#!/usr/bin/env python3
"""
Exp 12: Transformers y la ley rho.

Pregunta: si una capa de atencion implementa soft-binding VSA, entonces
cuando la "dimension efectiva por cabeza" cae a rho=1 (n_keys = d_head),
la recuperacion del valor correcto deberia colapsar — no gradualmente,
sino en punto singular, como en VSA.

Setup sintetico controlado (no entrenamiento):
  - Semilla fija: vocabulario de V=16 "tokens" con embeddings random en
    R^d. Cada posicion tiene un bind(rol_pos, token) como en un VSA.
  - Implementamos una capa de atencion SINGLE-HEAD matematica pura:
    Q, K, V aprendibles pero fijos (random seed) → no entrenamiento.
  - Simulamos el "resonator analogo": iterar softmax(QK^T)V hasta
    convergencia, medir si recupera el token correcto por posicion.
  - rho := V_activos / d_head. Variamos d_head de modo que rho cruce 1.

Esto NO es un transformer entrenado: es la MECANICA de atencion bajo
saturacion dimensional, aislada de aprendizaje. Si colapsa en rho=1,
la ley se extiende a atencion.
"""
import numpy as np
import random
import json
from pathlib import Path

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention_layer(X, Wq, Wk, Wv, Wo, d_head):
    """Capa de atencion: X (n, d_model) -> salida (n, d_model)."""
    Q = X @ Wq          # (n, d_head)
    K = X @ Wk          # (n, d_head)
    V = X @ Wv          # (n, d_head)
    S = Q @ K.T / np.sqrt(d_head)
    A = softmax(S, axis=-1)
    return (A @ V) @ Wo  # (n, d_head) -> (n, d_model)

def make_embeddings(rng, V, d_model):
    return rng.randn(V, d_model) / np.sqrt(d_model)

def run_case(V_active, d_model, d_head, n_seq=4, T=10, seed=7):
    """Secuencia sintetica: cada posicion i tiene el token i (una permutacion).
    Atencion debe poder recuperar-la. Medimos accuracy de argmax."""
    np_rng = np.random.RandomState(seed)
    rng = random.Random(seed)

    d_model_full = d_model
    E = make_embeddings(np_rng, V_active, d_model_full)

    # Pesos fijos (no aprendidos): proyección a d_head y vuelta a d_model
    Wq = np_rng.randn(d_model, d_head) / np.sqrt(d_model)
    Wk = np_rng.randn(d_model, d_head) / np.sqrt(d_model)
    Wv = np_rng.randn(d_model, d_head) / np.sqrt(d_model)
    Wo = np_rng.randn(d_head, d_model) / np.sqrt(d_head)

    # Secuencias: permutaciones de [0..V_active-1]
    accs = []
    for _ in range(40):
        seq = list(range(V_active))
        rng.shuffle(seq)
        seq = seq[:n_seq]
        X = E[seq]  # (n_seq, d_model)
        # posicion como rol: sumamos un vector posicional fijo por slot
        # (hace la analogia VSA bind(rol_pos, token))
        P = np_rng.randn(n_seq, d_model) / np.sqrt(d_model)
        Xin = X + 0.3 * P

        Y = Xin.copy()
        for _ in range(T):
            Y = Y + attention_layer(Y, Wq, Wk, Wv, Wo, d_head)  # residual
            # normalizamos para estabilidad numerica
            Y = Y / (np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-9)

        # Decode: para cada posicion, que token esta mas cerca?
        # Usamos similitud con embeddings originales
        H = Y @ E.T  # (n_seq, V_active)
        pred = H.argmax(axis=-1)
        accs.append((pred == np.array(seq)).mean())

    return float(np.mean(accs))


def main():
    print("=" * 72)
    print("EXP 12: Atencion sintetica y ley rho (V activos / d_head)")
    print("=" * 72)
    d_model = 128
    results = []
    for V_active in (8, 12, 16):           # "vocabulario" activo
        for ratio in (0.5, 0.75, 1.0, 1.25, 1.5):
            d_head = max(4, round(V_active / ratio))
            rho = V_active / d_head
            acc = run_case(V_active, d_model, d_head)
            results.append({"V": V_active, "d_head": d_head, "rho": rho, "acc": acc})
            print(f"  V={V_active:2d} d_head={d_head:3d} rho={rho:.2f}: acc={acc:.3f}")

    out = Path(__file__).parent.parent / "data" / "exp12_transformer_rho.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nGuardado: {out}")

if __name__ == "__main__":
    main()
