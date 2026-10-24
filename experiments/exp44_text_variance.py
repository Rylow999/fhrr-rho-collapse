#!/usr/bin/env python3
"""Exp 44 — Benchmark variance across 5 diverse texts.
Pre-registered: the spectral method's advantage is (or isn't) robust
across input distributions. If variance across texts is high, the
method is text-sensitive and we must report it honestly."""
import torch, numpy as np, json, time, random
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

t0 = time.time()
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
D = model.config.n_embd
DH = D // NH
NL = model.config.n_layer
lm_head = model.wte

TEXTS = {
    "prose_diverse": (
        "The autumn leaves drifted slowly across the quiet courtyard. "
        "Quantum computing represents a fundamental shift in information "
        "processing. The recipe called for patience and two cups of flour. "
        "Historical records suggest extensive trade along the Silk Road. "
        "When the satellite transmitted its first images, the team applauded. "
        "Economic indicators point toward recovery in certain sectors. "
        "The musician tuned her violin one final time before the stage. "
        "Volcanic eruptions release ash into the stratosphere."
    ),
    "technical": (
        "The transformer architecture processes tokens through multi-head "
        "attention where queries, keys, and values are projected from the "
        "input embeddings. The attention mechanism computes dot products "
        "between queries and keys, applies a softmax normalization, and "
        "produces weighted combinations of value vectors. Residual "
        "connections and layer normalization stabilize the deep network "
        "gradients during training on large-scale text corpora."
    ),
    "conversational": (
        "Hey, did you see that movie last night? I couldn't stop laughing. "
        "Yeah, the part with the dog was hilarious. My sister said she "
        "cried at the ending but I thought it was kind of predictable. "
        "Anyway, are we still on for dinner Friday? I heard that new place "
        "has amazing pasta. Let me know what time works for you."
    ),
    "mixed_domains": (
        "Photosynthesis converts carbon dioxide and water into glucose "
        "using light energy captured by chlorophyll. Meanwhile, the stock "
        "market reacted positively to the earnings report. The quantum "
        "state of the electron remains in superposition until measured. "
        "She walked along the beach at sunset, thinking about everything "
        "that had changed. The recipe requires three eggs and vanilla extract."
    ),
    "code": (
        "def process(items):\\n    results = []\\n    for i, item in enumerate(items):\\n"
        "        if item.is_valid():\\n            results.append(transform(item))\\n"
        "    return results\\nclass Handler:\\n    def __init__(self, config):\\n"
        "        self.config = config\\n        self.cache = {}\\n"
        "    def handle(self, req):\\n        key = req.hash()\\n"
        "        return self.cache.get(key, self.process(req))\\n"
    ),
}

def lmin_all(ids):
    keys_by_layer = {}
    for i, blk in enumerate(model.h):
        cat = blk.attn.c_attn
        def mk(idx, proj):
            def hook(mod, inp, out):
                hidden = inp[0].float()
                B, T, _ = hidden.shape
                qkv = hidden @ proj.weight.float() + proj.bias.float()
                K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
                keys_by_layer[idx] = K[0]
            return hook
        blk.attn.register_forward_hook(mk(i, cat))
    with torch.no_grad():
        _ = model(ids)
    for i in range(NL):
        model.h[i].attn._forward_hooks.clear()
    lmin_map = {}
    for li, K in keys_by_layer.items():
        Kn = K.numpy()
        for h in range(NH):
            Kh = Kn[h] / (np.linalg.norm(Kn[h], axis=1, keepdims=True) + 1e-12)
            lmin_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
    return lmin_map

def prune_and_ppl(prune_set, eval_ids):
    hooks = []
    for li in range(NL):
        heads = [h for (l2, h) in prune_set if l2 == li]
        if not heads: continue
        def ablate(mod, inp, out, hh=tuple(heads)):
            base = out[0] if isinstance(out, tuple) else out
            ao = base.clone()
            for h in hh:
                ao.view(ao.shape[0], ao.shape[1], NH, DH)[:, :, h, :] = 0
            return (ao,) if isinstance(out, tuple) else ao
        hooks.append(model.h[li].register_forward_hook(ablate))
    with torch.no_grad():
        W = min(128, eval_ids.shape[1] - 1)
        total_nll, total_tok = 0.0, 0
        for start in range(0, eval_ids.shape[1] - W, max(1, W // 2)):
            ids_w = eval_ids[:, start:start+W]
            o = model(ids_w)
            lg = o.last_hidden_state[0, :-1] @ lm_head.weight.T
            t = ids_w[0, 1:]
            total_nll += F.cross_entropy(lg, t, reduction="sum").item()
            total_tok += len(t)
    for h in hooks: h.remove()
    return float(np.exp(total_nll / total_tok))

all_results = {}
for text_name, text in TEXTS.items():
    eval_ids = tok(text, return_tensors="pt").input_ids
    if eval_ids.shape[1] < 100:
        eval_ids = tok(text * 3, return_tensors="pt").input_ids
    eval_ids = eval_ids[:, :min(eval_ids.shape[1], 400)]
    
    # Measure lmin on this text (context-sensitive!)
    ids_ctx = tok(text, return_tensors="pt").input_ids
    if ids_ctx.shape[1] < 64:
        ids_ctx = tok(text * 3, return_tensors="pt").input_ids
    ids_ctx = ids_ctx[:, :64]
    
    lmin_map = lmin_all(ids_ctx)
    
    # Base perplexity
    base_ppl = prune_and_ppl(set(), eval_ids)
    
    # Spectral: descending lmin (prune most diffuse)
    all_heads = list(lmin_map.keys())
    n_prune = int(len(all_heads) * 0.10)  # 10% (the sweet spot)
    sorted_desc = sorted(all_heads, key=lambda k: lmin_map[k], reverse=True)
    spec_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in sorted_desc[:n_prune]}
    
    # Random
    random.seed(42)
    rand_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in random.sample(all_heads, n_prune)}
    
    ppl_spec = prune_and_ppl(spec_prune, eval_ids)
    ppl_rand = prune_and_ppl(rand_prune, eval_ids)
    
    all_results[text_name] = {
        "baseline_ppl": round(base_ppl, 1),
        "spectral_ppl": round(ppl_spec, 1),
        "random_ppl": round(ppl_rand, 1),
        "n_eval_tokens": eval_ids.shape[1],
    }
    print(f"  {text_name:16s}: base={base_ppl:.0f} spec={ppl_spec:.0f} rand={ppl_rand:.0f} "
          f"ratio={ppl_rand/ppl_spec:.2f} ({time.time()-t0:.0f}s)", flush=True)

# Summarize
spec_ppls = [r["spectral_ppl"] for r in all_results.values()]
rand_ppls = [r["random_ppl"] for r in all_results.values()]
ratios = [r["random_ppl"]/r["spectral_ppl"] for r in all_results.values()]
print(f"\nSpectral vs Random across {len(TEXTS)} texts:")
print(f"  Ratio (random/spectral): mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}, "
      f"range=[{min(ratios):.2f}, {max(ratios):.2f}]")
print(f"  Spectral wins on {sum(1 for r in ratios if r > 1)}/{len(ratios)} texts")

result = {
 "experiment": "Exp 44 — benchmark variance across 5 diverse texts",
 "texts": list(TEXTS.keys()),
 "results": all_results,
 "summary": {
   "ratio_mean": round(float(np.mean(ratios)), 2),
   "ratio_std": round(float(np.std(ratios)), 2),
   "spectral_wins": f"{sum(1 for r in ratios if r > 1)}/{len(ratios)}",
 },
 "runtime_s": round(time.time() - t0, 1),
}
json.dump(result, open("/home/delorien/exp44_results.json", "w"), indent=2)
print("saved")
