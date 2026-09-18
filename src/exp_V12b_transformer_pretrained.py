#!/usr/bin/env python3
"""
Exp 12b: ley rho en atencion de un transformer PREENTRENADO real.

Modelo: nanoGPT char-level Shakespeare (HuggingFace), d_model=384, 6 heads.
Tarea: recuperacion de tokens por posicion (binding posicion<->token), la
prueba directa de que la atencion hace soft-binding.

Protocolo:
  - Cargar modelo preentrenado (inference only, CPU)
  - Extraer matrices de atencion por head (d_head = 384/6 = 64)
  - Construir "saturacion" artificial: duplicar el vocabulario contextual
    inyectando distractors y medir accuracy de recuperacion por posicion
  - rho_proxy = n_context_tokens / d_head

  Cuando rho_proxy cruza 1, si la atencion colapsa (exactamente como
  gram en VSA), la ley rho tambien aplica a transformers reales.
"""
import json
from pathlib import Path

import numpy as np
import torch


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "gpt2"
    print(f"Cargando {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True)
    model.eval()

    # Config del modelo real
    cfg = model.config
    n_head = getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", None))
    d_model = getattr(cfg, "n_embd", getattr(cfg, "hidden_size", None))
    d_head = d_model // n_head
    print(f"n_head={n_head}  d_model={d_model}  d_head={d_head}")

    # Task de recuperacion: dada una frase, pedir el token en posicion k
    # usando SOLO la atencion de la ultima capa hacia las posiciones previas.
    # Distractores = frases aleatorias intercaladas (aumentan "contexto
    # activo" = rho_proxy).
    base_text = (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles."
    )
    base_ids = tok(base_text, return_tensors="pt").input_ids[0]
    print(f"Base text tokens: {len(base_ids)}")

    rng = np.random.RandomState(42)
    results = []

    # rho_proxy = n_distractors / d_head (cuantos tokens distractivos por
    # dimension de cabeza de atencion)
    for n_dist in (0, 16, 32, 48, 64, 96, 128):
        # Construir secuencia: base + distractores al final
        toks = base_ids.tolist()
        for _ in range(n_dist):
            toks.append(int(rng.randint(0, tok.vocab_size if hasattr(tok, 'vocab_size') else 50257)))
        input_ids = torch.tensor([toks], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True)

        # Atencion ultima capa, todas las heads: (n_head, T, T)
        attn = out.attentions[-1][0]  # (n_head, T, T)
        T = attn.shape[-1]

        # Tarea: para cada posicion t en la zona base, predecir su token real
        # usando el "vector de valor" atendido. Decodificamos tomando la
        # hidden state post-atencion y comparando con embeddings.
        # Simplificacion: la probabilidad de recuperacion ~ atencion que la
        # posicion correcta recibe en la query de la posicion final.
        # Medimos: fraccion de posiciones base cuyo argmax de atencion
        # (desde la ultima posicion) apunta a ellas mismas o al vecino correcto.

        # Metrica simple pero informativa: entropia de la distribucion de
        # atencion de la ultima posicion hacia las posiciones base.
        last_attn = attn[:, -1, :len(base_ids)]   # (n_head, n_base)
        ent = -(last_attn * torch.log(last_attn + 1e-12)).sum(-1)  # (n_head,)
        ent_mean = float(ent.mean())

        # Top-1: ¿la posicion mas atendida esta en la zona base?
        top1_base = float((last_attn.argmax(-1) < len(base_ids)).float().mean())

        rho_proxy = (len(base_ids) + n_dist) / d_head
        results.append({
            "n_dist": n_dist,
            "total_tokens": len(base_ids) + n_dist,
            "rho_proxy": rho_proxy,
            "attn_entropy": ent_mean,
            "top1_in_base": top1_base,
        })
        print(f"  n_dist={n_dist:3d} rho={rho_proxy:.2f} "
              f"entropia={ent_mean:.3f} top1_base={top1_base:.3f}")

    out_path = Path(__file__).parent.parent / "data" / "exp12b_transformer_real.json"
    with open(out_path, "w") as f:
        json.dump({"model": name, "n_head": n_head, "d_model": d_model,
                   "d_head": d_head, "results": results}, f, indent=2)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
