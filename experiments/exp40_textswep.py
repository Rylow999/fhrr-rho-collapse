#!/usr/bin/env python3
"""Exp 40 — Text sensitivity: does the collapse regime depend on the input?
Pre-registered (2026-10-24):
  T1: natural text -> the known collapse at square point
  T2: source code -> if code is less redundant than prose, lambda_min should be
      HIGHER (more variance survives) — testable
  T3: random tokens -> if the collapse is content-driven, random tokens
      (no linguistic structure) should show MORE variance (keys are random)
      — the cleanest control
  T4: repeated single token (the degenerate case) -> maximum redundancy,
      should collapse hardest
This closes the Exp32-vs-Exp38 input-dependence flag."""
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

TEXTS = {
  "natural_prose": ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms."),
  "source_code": ("def process(items):\n    results = []\n    for i, item in enumerate(items):\n"
        "        if item.is_valid() and i not in seen:\n            results.append(transform(item))\n"
        "    return results\n\nclass Handler:\n    def __init__(self, config):\n"
        "        self.config = config\n        self.cache = {}\n\n    def handle(self, req):\n"
        "        key = req.hash()\n        if key in self.cache:\n            return self.cache[key]\n"
        "        result = self.process(req)\n        self.cache[key] = result\n        return result\n"),
  "random_tokens": None,  # filled below
  "repeated_token": "the the the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the the the the ",
}

def lmin_all(ids):
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
    lmins = []
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
            lmins.append(float(np.linalg.eigvalsh(Kh @ Kh.T)[0]))
    return lmins

results = {}
rng = np.random.default_rng(42)
for name, text in TEXTS.items():
    if name == "random_tokens":
        ids = torch.tensor(rng.integers(0, 50257, size=64)).unsqueeze(0)
    else:
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] < 64:
            ids = tok(text * 3, return_tensors="pt").input_ids
        ids = ids[:, :64]
    T = ids.shape[1]
    lmins = lmin_all(ids)
    v = np.array(lmins)
    results[name] = {
        "T": T, "n_heads": len(lmins),
        "lmin_median": round(float(np.median(v)), 10),
        "lmin_max": round(float(v.max()), 10),
        "lmin_min": round(float(v.min()), 10),
        "collapsed_fraction": f"{int((np.abs(v) < 1e-4).sum())}/{len(v)}",
    }
    print(f"  {name:16s}: lmin_med={results[name]['lmin_median']:.2e} "
          f"max={results[name]['lmin_max']:.2e} collapsed={results[name]['collapsed_fraction']} ({time.time()-t0:.0f}s)", flush=True)

json.dump(results, open("/home/delorien/exp40_results.json", "w"), indent=2)
print(f"saved /home/delorien/exp40_results.json ({time.time()-t0:.0f}s)")
