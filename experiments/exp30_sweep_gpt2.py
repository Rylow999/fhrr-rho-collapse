#!/usr/bin/env python3
"""Exp 30 — GPT-2 lambda_min(T) fine sweep (transformers 5.x layout: W is (in, out))."""
import torch, numpy as np, json
from transformers import GPT2Model, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
DH = model.config.n_embd // NH
D = model.config.n_embd

BASE = ("The quick brown fox jumps over the lazy dog. "
        "The Collatz conjecture states that every positive integer reaches one. "
        "Spectral methods reveal the hidden geometry of attention. ")
Ts = [8, 16, 32, 64, 96, 128, 160, 200]
curve = []
for T_target in Ts:
    text = (BASE * 20)[:T_target * 4]
    ids = tok(text, return_tensors="pt").input_ids[:, :T_target]
    T = ids.shape[1]
    keys_by_layer = {}
    for i, blk in enumerate(model.h):
        cat = blk.attn.c_attn
        def make_hook(idx, proj):
            def hook(mod, inp, out):
                hidden = inp[0].float()
                B, Tt, _ = hidden.shape
                qkv = hidden @ proj.weight.float() + proj.bias.float()  # W is (in,out) in 5.x
                K = qkv[:, :, D:2*D].view(B, Tt, NH, DH).permute(0, 2, 1, 3)
                keys_by_layer[idx] = K[0]
            return hook
        blk.attn.register_forward_hook(make_hook(i, cat))
    with torch.no_grad():
        _ = model(ids)
    for i, blk in enumerate(model.h):
        blk.attn._forward_hooks.clear()
    lmins_all = []
    for li in sorted(keys_by_layer):
        K = keys_by_layer[li].numpy()
        H, Tt, dh = K.shape
        for h in range(H):
            Kh = K[h] / np.linalg.norm(K[h], axis=1, keepdims=True)
            lmins_all.append(float(np.linalg.eigvalsh(Kh @ Kh.T)[0]))
    curve.append({"T": T, "lmin_median": float(np.median(lmins_all)),
                  "lmin_max": float(max(lmins_all))})
    print(f"GPT-2 T={T}: lmin_med={curve[-1]['lmin_median']:.2e} max={curve[-1]['lmin_max']:.2e}", flush=True)

json.dump({"model": "gpt2", "d_head": DH, "curve": curve},
          open("/home/delorien/exp30_gpt2.json", "w"), indent=2)
print("saved")
