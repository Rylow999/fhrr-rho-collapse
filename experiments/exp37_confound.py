#!/usr/bin/env python3
"""Exp 37 — depth vs training-tokens confound (Pythia scale sweep + GPT-2 cross-check).
Pre-registered (2026-10-24):
  D1: if collapse is driven by DEPTH, pythia-70m (6L) and 160m (12L) keep lmin variance
      at square point like GPT-2 124M (12L) does.
  D2: if collapse is driven by TRAINING EXPOSURE, even pythia-70m is fully collapsed,
      AND pythia-160m (12L, same depth as GPT-2 124M) is collapsed where GPT-2 124M is not
      -> smoking gun: same depth, different collapse = training is the driver.
Falsifiable both ways. All CPU, fp16, T=d_head=64 (square point for all three)."""
import torch, numpy as np, json, time
from transformers import AutoModel, AutoTokenizer

def measure(model_name):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    cfg = model.config
    NH = cfg.num_attention_heads
    D = cfg.hidden_size
    DH = D // NH
    NL = cfg.num_hidden_layers
    TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
            "The city council approved the budget for the new park after a long "
            "debate about the environmental impact and the cost of maintenance. "
            "Local residents supported the initiative but questioned the timeline. "
            "Spectral methods reveal the hidden geometry of attention mechanisms.")
    ids = tok(TEXT, return_tensors="pt").input_ids
    if ids.shape[1] < 64:
        ids = tok(TEXT * 3, return_tensors="pt").input_ids
    ids = ids[:, :64]
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
    lmin_map = {}
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
            lmin_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
    v = np.array(list(lmin_map.values()))
    res = {
      "model": model_name, "layers": NL, "heads": NH, "d_head": DH, "T": 64,
      "lmin_min": round(float(v.min()), 8), "lmin_max": round(float(v.max()), 8),
      "lmin_median": round(float(np.median(v)), 8),
      "collapsed_fraction": f"{int((np.abs(v) < 1e-4).sum())}/{len(v)}",
      "lmin_map": lmin_map,
      "runtime_s": round(time.time() - t0, 1),
    }
    del model
    return res

out = []
for m in ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]:
    print(f"=== {m} ===", flush=True)
    try:
        out.append(measure(m))
        r = out[-1]
        print(f"  L={r['layers']} H={r['heads']}: lmin [{r['lmin_min']:.2e}, {r['lmin_max']:.2e}] "
              f"collapsed {r['collapsed_fraction']}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
json.dump(out, open("/home/delorien/exp37_results.json", "w"), indent=2)
print("saved /home/delorien/exp37_results.json")
