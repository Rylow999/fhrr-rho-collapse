#!/usr/bin/env python3
"""Exp 38 — WHY do keys collapse? Positional saturation test (GPT-2 124M).
Pre-registered (2026-10-24):
  M1 (positional hypothesis): if the positional embedding saturates the key space,
      zeroing wpe at inference changes lmin dramatically (variance appears).
  M2 (content hypothesis): if content redundancy is the driver, zeroing wpe
      changes nothing (keys stay collapsed).
  Measurement: lmin at square point (T=64) per head, with and without wpe.
  Control: randomize wpe (same norm, random direction) to distinguish
      'position removed' from 'perturbation per se'."""
import torch, numpy as np, json, time
from transformers import GPT2Model, GPT2Tokenizer

t0 = time.time()
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
D = model.config.n_embd
DH = D // NH
NL = model.config.n_layer
print(f"gpt2-124M: d={D} L={NL} H={NH} d_head={DH}", flush=True)

TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms.")
ids = tok(TEXT, return_tensors="pt").input_ids[:, :64]
print("T =", ids.shape[1], flush=True)

def lmin_all(wpe_mode):
    # wpe_mode: "normal" | "zero" | "random"
    backup = model.wpe.weight.data.clone()
    if wpe_mode == "zero":
        model.wpe.weight.data.zero_()
    elif wpe_mode == "random":
        model.wpe.weight.data.normal_(0, backup.std())
    keys_by_layer = {}
    for i, blk in enumerate(model.h):
        cat = blk.attn.c_attn
        def make_hook(idx, proj):
            def hook(mod, inp, out):
                hidden = inp[0].float()
                B, T, _ = hidden.shape
                qkv = hidden @ proj.weight.float() + proj.bias.float()
                K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
                keys_by_layer[idx] = K[0]
            return hook
        blk.attn.register_forward_hook(make_hook(i, cat))
    with torch.no_grad():
        _ = model(ids)
    for i in range(NL):
        model.h[i].attn._forward_hooks.clear()
    m = {}
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
            m[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
    model.wpe.weight.data = backup
    v = np.array(list(m.values()))
    return m, float(v.min()), float(v.max()), int((np.abs(v) < 1e-4).sum())

res = {}
for mode in ["normal", "zero", "random"]:
    mm, mn, mx, col = lmin_all(mode)
    res[mode] = {"lmin_min": round(mn, 8), "lmin_max": round(mx, 8),
                 "collapsed": f"{col}/{len(mm)}", "lmin_map": mm}
    print(f"  wpe={mode}: lmin [{mn:.2e}, {mx:.2e}] collapsed {col}/{len(mm)}", flush=True)

out = {
 "experiment": "Exp 38 - positional saturation test (GPT-2 124M, the mechanism question)",
 "preregistered": {"M1": "zeroing wpe restores lmin variance -> positional saturation is the driver",
                    "M2": "zeroing wpe changes nothing -> content redundancy is the driver",
                    "control": "random wpe distinguishes 'remove position' from 'perturb'"},
 "results": {k: {kk: vv for kk, vv in v.items() if kk != "lmin_map"} for k, v in res.items()},
 "lmin_maps": {k: v["lmin_map"] for k, v in res.items()},
 "runtime_s": round(time.time() - t0, 1),
}
json.dump(out, open("/home/delorien/exp38_results.json", "w"), indent=2)
print("saved /home/delorien/exp38_results.json")
