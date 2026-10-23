#!/usr/bin/env python3
"""Exp 34b — Sub-square criterion RETEST with real pruning (Pythia-410M).
Pre-registered (2026-10-24, after the Exp 41 inversion discovery):
  S1: pruning by lmin@T/2 ASCENDING (lowest first) yields WORSE perplexity
      than random (the inversion pattern extends to sub-square measurement).
  S2: pruning by lmin@T/2 DESCENDING yields better than random.
  S3: if neither holds, sub-square measurement is N/A for pruning in this regime.
Model: Pythia-410M (the fully-collapsed-at-square regime where sub-square
was supposed to restore signal). Eval: diverse text perplexity."""
import torch, numpy as np, json, time, random
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

t0 = time.time()
MODEL = "EleutherAI/pythia-410m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()
cfg = model.config
NH = cfg.num_attention_heads          # 16
D = cfg.hidden_size                   # 1024
DH = D // NH                          # 64
NL = cfg.num_hidden_layers            # 24

# Diverse eval text (same as V41)
EVAL_TEXT = (
    "The autumn leaves drifted slowly across the quiet courtyard as Maria "
    "finished her last cigarette before the meeting. She had spent three "
    "weeks preparing the proposal, and today the board would finally vote. "
    "Quantum computing represents a fundamental shift in how we process "
    "information, leveraging superposition to solve certain classes of "
    "problems exponentially faster than classical machines. The recipe "
    "called for two cups of flour, a pinch of salt, and patience. "
    "Historical records suggest the Roman Empire traded extensively with "
    "distant civilizations along the Silk Road. When the satellite finally "
    "transmitted its first images of the distant galaxy, the research team "
    "erupted in spontaneous applause. Economic indicators point toward a "
    "gradual recovery. The musician tuned her violin one final time. "
    "Volcanic eruptions release massive quantities of ash. Machine learning "
    "algorithms detect patterns invisible to human observers."
)
eval_ids = tok(EVAL_TEXT, return_tensors="pt").input_ids
print(f"eval tokens: {eval_ids.shape[1]}", flush=True)

# Measure per-head lambda_min at SUB-SQUARE (T=32 = d_head/2)
TEXT32 = ("The quick brown fox jumps over the lazy dog. " * 12 +
    "The city council approved the budget for the new park after a long "
    "debate about the environmental impact and the cost of maintenance. "
    "Local residents supported the initiative but questioned the timeline. "
    "Spectral methods reveal the hidden geometry of attention mechanisms.")
ids32 = tok(TEXT32 * 3, return_tensors="pt").input_ids[:, :32]
print(f"T = {ids32.shape[1]} (sub-square: d_head/2 = 32)", flush=True)

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
    _ = model(ids32)
for i in range(NL):
    model.layers[i].attention.query_key_value._forward_hooks.clear()

lmin32_map = {}
for li, K in keys_by_layer.items():
    Kn = K.numpy()
    for h in range(NH):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmin32_map[f"{li},{h}"] = float(np.linalg.eigvalsh(Kh @ Kh.T)[0])
v = np.array(list(lmin32_map.values()))
print(f"lmin@T/2: range [{v.min():.2e}, {v.max():.2e}], "
      f"collapsed(<1e-4): {(np.abs(v)<1e-4).sum()}/{len(v)}", flush=True)

# Pruning machinery (fp16-safe, same as V41)
def prune_and_perplexity(prune_set):
    hooks = []
    for li in range(NL):
        heads_to_zero = [h for (l2, h) in prune_set if l2 == li]
        if not heads_to_zero:
            continue
        def ablate(mod, inp, out, hh=tuple(heads_to_zero)):
            base = out[0] if isinstance(out, tuple) else out
            ao = base.clone().float()
            for h in hh:
                ao.view(ao.shape[0], ao.shape[1], NH, DH)[:, :, h, :] = 0
            ao = ao.to(base.dtype)
            return ao if not isinstance(out, tuple) else (ao,)
        hooks.append(model.layers[li].register_forward_hook(ablate))
    with torch.no_grad():
        W = min(256, eval_ids.shape[1] - 1)
        total_nll, total_tokens = 0.0, 0
        for start in range(0, eval_ids.shape[1] - W, max(1, W // 2)):
            ids_w = eval_ids[:, start:start+W]
            out = model(ids_w)
            # pythia uses embed_in; the lm head is embed_out (not loaded in AutoModel)
            # use the hidden state directly as a proxy: measure NLL via
            # cosine distance to the next-token's hidden (self-supervised proxy)
            # Better: load the lm head from the full model
            pass
    for h in hooks:
        h.remove()
    return None

# Actually: AutoModel doesn't have the lm head. Use AutoModelForCausalLM instead.
from transformers import AutoModelForCausalLM
full_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
full_model.eval()

def prune_and_ppl(prune_set):
    hooks = []
    for li in range(NL):
        heads_to_zero = [h for (l2, h) in prune_set if l2 == li]
        if not heads_to_zero:
            continue
        def ablate(mod, inp, out, hh=tuple(heads_to_zero)):
            base = out[0] if isinstance(out, tuple) else out
            ao = base.clone().float()
            for h in hh:
                ao.view(ao.shape[0], ao.shape[1], NH, DH)[:, :, h, :] = 0
            ao = ao.to(base.dtype)
            return ao if not isinstance(out, tuple) else (ao,)
        hooks.append(full_model.layers[li].register_forward_hook(ablate))
    with torch.no_grad():
        W = min(256, eval_ids.shape[1] - 1)
        total_nll, total_tokens = 0.0, 0
        for start in range(0, eval_ids.shape[1] - W, max(1, W // 2)):
            ids_w = eval_ids[:, start:start+W]
            logits = full_model(ids_w).logits[0, :-1]
            targets = ids_w[0, 1:]
            nll = F.cross_entropy(logits.float(), targets, reduction="sum").item()
            total_nll += nll
            total_tokens += len(targets)
    for h in hooks:
        h.remove()
    return float(np.exp(total_nll / total_tokens))

all_heads = list(lmin32_map.keys())
n_prune = int(len(all_heads) * 0.20)
sorted_by_lmin32 = sorted(all_heads, key=lambda k: lmin32_map[k])

def to_set(keys):
    s = set()
    for k in keys:
        l, h = k.split(",")
        s.add((int(l), int(h)))
    return s

# Strategy 1: lmin@T/2 ascending (lowest first)
asc_prune = to_set(sorted_by_lmin32[:n_prune])
# Strategy 2: random
random.seed(99)
rand_prune = to_set(random.sample(all_heads, n_prune))
# Strategy 3: lmin@T/2 descending (highest first)
desc_prune = to_set(sorted_by_lmin32[-n_prune:])

base_ppl = prune_and_ppl(set())
print(f"\nbaseline ppl: {base_ppl:.2f} ({time.time()-t0:.0f}s)", flush=True)
ppl_asc = prune_and_ppl(asc_prune)
print(f"lmin@T/2 ascending prune (20%): ppl={ppl_asc:.2f} ({time.time()-t0:.0f}s)", flush=True)
ppl_rand = prune_and_ppl(rand_prune)
print(f"random prune (20%): ppl={ppl_rand:.2f} ({time.time()-t0:.0f}s)", flush=True)
ppl_desc = prune_and_ppl(desc_prune)
print(f"lmin@T/2 DESCENDING prune (20%): ppl={ppl_desc:.2f} ({time.time()-t0:.0f}s)", flush=True)

result = {
 "experiment": "Exp 34b — sub-square criterion RETEST with real pruning (Pythia-410M)",
 "preregistered": {
   "S1": "ascending (lowest lmin@T/2) worse than random (inversion extends)",
   "S2": "descending better than random",
   "S3": "neither: sub-square N/A for pruning",
 },
 "lmin_at_T/2_range": [round(float(v.min()), 8), round(float(v.max()), 8)],
 "baseline_ppl": round(base_ppl, 2),
 "lmin32_ascending_ppl": round(ppl_asc, 2),
 "random_ppl": round(ppl_rand, 2),
 "lmin32_descending_ppl": round(ppl_desc, 2),
 "verdict": "S1" if ppl_asc > ppl_rand else ("S2" if ppl_desc < ppl_rand else "S3"),
 "runtime_s": round(time.time() - t0, 1),
}
json.dump(result, open("/home/delorien/exp34b_results.json", "w"), indent=2)
print(f"\nVERDICT: {result['verdict']}")
