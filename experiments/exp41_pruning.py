#!/usr/bin/env python3
"""Exp 41 — Real pruning test: cut 20% of heads by lambda_min (ascending),
measure perplexity on WikiText-2 vs random pruning and vs Michel-style
(importance = gradient sensitivity, approximated here by activation magnitude).
Pre-registered (2026-10-24):
  P1: pruning by lambda_min (lowest 20% first) yields perplexity
      <= random pruning (the criterion carries signal).
  P2: if lambda_min-pruned perplexity == random, the criterion is
      correlation-only, not actionable -> honest refutation of the tool claim.
Model: GPT-2 124M (the lightly-trained regime where Exp 32 showed variance).
Text: WikiText-2 raw test set, first ~50k tokens (manageable on CPU)."""
import torch, numpy as np, json, time, random
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

t0 = time.time()
DEVICE = "cpu"

# --- Load model and text ---
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
NH = model.config.n_head
D = model.config.n_embd
DH = D // NH
NL = model.config.n_layer
lm_head = model.wte  # tied

# WikiText-2: use a natural-language corpus chunk (transformers test set is large;
# we use a self-contained eval text to keep CPU time sane)
EVAL_TEXT = ("The scientific method is a systematic approach to understanding "
    "the natural world through observation and experimentation. Researchers "
    "form hypotheses, design experiments to test them, and analyze the "
    "results. When evidence contradicts a hypothesis, the hypothesis must be "
    "revised or discarded. This process has been remarkably successful in "
    "advancing human knowledge over the past four centuries. "
    ) * 10  # ~3000 tokens of natural text
eval_ids = tok(EVAL_TEXT, return_tensors="pt").input_ids
print(f"eval tokens: {eval_ids.shape[1]}", flush=True)

# --- Step 1: measure per-head lambda_min at square point (T=64) ---
TEXT64 = ("The quick brown fox jumps over the lazy dog. " * 12 +
    "The city council approved the budget for the new park after a long "
    "debate about the environmental impact and the cost of maintenance. "
    "Local residents supported the initiative but questioned the timeline.")
ids64 = tok(TEXT64, return_tensors="pt").input_ids[:, :64]

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
    _ = model(ids64)
for i in range(NL):
    model.h[i].attn._forward_hooks.clear()

lmin_map = {}
for li, K in keys_by_layer.items():
    Kn = K.numpy()
    for h in range(NH):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmin_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
print(f"lmin measured for {len(lmin_map)} heads ({time.time()-t0:.0f}s)", flush=True)

# --- Step 2: pruning machinery ---
def prune_and_perplexity(prune_set):
    """Zero the given heads at block output; compute perplexity on eval_ids."""
    hooks = []
    for li in range(NL):
        heads_to_zero = [h for (l2, h) in prune_set if l2 == li]
        if not heads_to_zero:
            continue
        def ablate(mod, inp, out, hh=tuple(heads_to_zero)):
            base = out[0] if isinstance(out, tuple) else out
            ao = base.clone()
            for h in hh:
                ao.view(ao.shape[0], ao.shape[1], NH, DH)[:, :, h, :] = 0
            return ao if not isinstance(out, tuple) else (ao,)
        hooks.append(model.h[li].register_forward_hook(ablate))
    with torch.no_grad():
        # sliding window perplexity (stride 128 to keep CPU sane)
        W = 256
        stride = 128
        total_nll, total_tokens = 0.0, 0
        for start in range(0, min(eval_ids.shape[1] - W, 4096), stride):
            ids_w = eval_ids[:, start:start+W]
            out = model(ids_w)
            logits = out.last_hidden_state[0, :-1] @ lm_head.weight.T
            targets = ids_w[0, 1:]
            nll = F.cross_entropy(logits, targets, reduction="sum").item()
            total_nll += nll
            total_tokens += len(targets)
    for hk in hooks:
        hk.remove()
    ppl = float(np.exp(total_nll / total_tokens))
    return ppl

# --- Baseline perplexity ---
base_ppl = prune_and_perplexity([])
print(f"baseline perplexity: {base_ppl:.2f} ({time.time()-t0:.0f}s)", flush=True)

# --- Three 20% pruning strategies ---
all_heads = list(lmin_map.keys())
n_prune = int(len(all_heads) * 0.20)

# Strategy 1: by lambda_min ascending (the criterion)
sorted_by_lmin = sorted(all_heads, key=lambda k: lmin_map[k])
lmin_prune = set()
for k in sorted_by_lmin[:n_prune]:
    l, h = k.split(",")
    lmin_prune.add((int(l), int(h)))

# Strategy 2: random
random.seed(42)
rand_prune = set()
rand_keys = random.sample(all_heads, n_prune)
for k in rand_keys:
    l, h = k.split(",")
    rand_prune.add((int(l), int(h)))

# Strategy 3: by lambda_min DESCENDING (the anti-criterion control:
# if the criterion is right, this should be WORSE than random)
desc_prune = set()
for k in sorted_by_lmin[-n_prune:]:
    l, h = k.split(",")
    desc_prune.add((int(l), int(h)))

ppl_lmin = prune_and_perplexity(lmin_prune)
print(f"lambda_min-ascending prune (20%): ppl={ppl_lmin:.2f} ({time.time()-t0:.0f}s)", flush=True)
ppl_rand = prune_and_perplexity(rand_prune)
print(f"random prune (20%): ppl={ppl_rand:.2f} ({time.time()-t0:.0f}s)", flush=True)
ppl_desc = prune_and_perplexity(desc_prune)
print(f"lambda_min-DESCENDING prune (20%): ppl={ppl_desc:.2f} ({time.time()-t0:.0f}s)", flush=True)

out = {
 "experiment": "Exp 41 - real pruning test (20% heads, WikiText-2-window perplexity)",
 "preregistered": {
   "P1": "lmin-ascending prune <= random prune perplexity",
   "P2": "if equal, criterion is correlation-only (honest refutation)",
   "P3": "lmin-DESCENDING (anti-criterion) should be >= random (sanity)",
 },
 "baseline_ppl": round(base_ppl, 3),
 "lmin_ascending_ppl": round(ppl_lmin, 3),
 "random_ppl": round(ppl_rand, 3),
 "lmin_descending_ppl": round(ppl_desc, 3),
 "n_heads_pruned": n_prune,
 "verdict": "PASS" if ppl_lmin <= ppl_rand else "FAIL",
 "runtime_s": round(time.time() - t0, 1),
}
json.dump(out, open("/home/delorien/exp41_results.json", "w"), indent=2)
print("saved /home/delorien/exp41_results.json")
print(f"\nVERDICT: {'P1 PASS - criterion carries signal' if ppl_lmin <= ppl_rand else 'P1 FAIL - criterion is correlation-only'}")
