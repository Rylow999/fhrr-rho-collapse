#!/usr/bin/env python3
"""Exp 39 — Training checkpoint sweep: lambda_min as a function of exposure.
Pre-registered (2026-10-24):
  E1: lambda_min at square point decays monotonically with training steps
      (the exposure law is quantitative, not just qualitative).
  E2: the decay is consistent across model scales (70m, 160m if checkpoints fit RAM).
  E3 (falsification): if lambda_min is flat across checkpoints, the exposure
      law is wrong and the corpus confound is alive.
Pythia checkpoints: EleutherAI/pythia-70m-deduped has step-numbered checkpoints
(revision step0, step512, ... step143000). We use pythia-70m (smallest, fits easily)."""
import torch, numpy as np, json, time
from transformers import AutoModel, AutoTokenizer

t0 = time.time()
TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline. "
        "Spectral methods reveal the hidden geometry of attention mechanisms.")

def measure_checkpoint(model_name, revision):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
    model.eval()
    cfg = model.config
    NH = cfg.num_attention_heads
    D = cfg.hidden_size
    DH = D // NH
    NL = cfg.num_hidden_layers
    ids = tok(TEXT * 3, return_tensors="pt").input_ids[:, :64]
    keys_by_layer = {}
    for i, blk in enumerate(model.layers):
        qkv = blk.attention.query_key_value
        def make_hook(idx, proj):
            def hook(mod, inp, out):
                hidden = inp[0].float()
                B, T, _ = hidden.shape
                qo = hidden @ proj.weight.float().T + proj.bias.float()
                K = qo[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
                keys_by_layer[idx] = K[0]
            return hook
        qkv.register_forward_hook(make_hook(i, qkv))
    with torch.no_grad():
        _ = model(ids)
    for i in range(NL):
        model.layers[i].attention.query_key_value._forward_hooks.clear()
    lmins = []
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
            lmins.append(float(np.linalg.eigvalsh(Kh @ Kh.T)[0]))
    v = np.array(lmins)
    del model
    return {
        "revision": revision,
        "lmin_median": round(float(np.median(v)), 10),
        "lmin_max": round(float(v.max()), 10),
        "lmin_min": round(float(v.min()), 10),
        "lmin_p90": round(float(np.percentile(v, 90)), 10),
        "collapsed_fraction": f"{int((np.abs(v) < 1e-4).sum())}/{len(v)}",
    }

CHECKPOINTS = [
    ("step0", "0"), ("step512", "512"), ("step1000", "1000"),
    ("step2000", "2000"), ("step4000", "4000"), ("step8000", "8000"),
    ("step16000", "16000"), ("step32000", "32000"), ("step64000", "64000"),
    ("step128000", "128000"),
]
results = []
for rev_name, steps in CHECKPOINTS:
    try:
        r = measure_checkpoint("EleutherAI/pythia-70m-deduped", rev_name)
        r["steps"] = int(steps)
        results.append(r)
        print(f"  step={steps:>6s}: lmin_med={r['lmin_median']:.2e} max={r['lmin_max']:.2e} collapsed={r['collapsed_fraction']} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  step={steps:>6s}: SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True)

json.dump(results, open("/home/delorien/exp39_results.json", "w"), indent=2)
print(f"saved /home/delorien/exp39_results.json ({time.time()-t0:.0f}s)")
