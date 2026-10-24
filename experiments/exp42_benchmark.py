#!/usr/bin/env python3
"""Exp 42b — FAIR benchmark with working Michel-style method.
Fix: instead of fragile gradient capture, use activation magnitude per head
(a validated first-order proxy in the pruning literature — heads with small
output activations contribute little to the loss gradient at first order)."""
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

EVAL_TEXT = (
    "The autumn leaves drifted slowly across the quiet courtyard as Maria "
    "finished her last cigarette before the meeting. Quantum computing "
    "represents a fundamental shift. The recipe called for two cups of flour "
    "and patience. Historical records suggest the Roman Empire traded "
    "extensively along the Silk Road. When the satellite transmitted its "
    "first images, the research team erupted in applause. Economic indicators "
    "point toward recovery. The musician tuned her violin. Volcanic eruptions "
    "release massive quantities of ash. Machine learning algorithms detect "
    "patterns invisible to human observers."
)
eval_ids = tok(EVAL_TEXT, return_tensors="pt").input_ids

# ===== Method 1: SPECTRAL =====
t_spec = time.time()
TEXT64 = ("The quick brown fox jumps over the lazy dog. " * 12 +
    "The city council approved the budget for the new park after a long "
    "debate about the environmental impact and the cost of maintenance. "
    "Local residents supported the initiative but questioned the timeline.")
ids64 = tok(TEXT64, return_tensors="pt").input_ids[:, :64]

keys_by_layer = {}
for i, blk in enumerate(model.h):
    cat = blk.attn.c_attn
    def mk(idx, proj):
        def hook(mod, inp, out):
            h = inp[0].float()
            qkv = h @ proj.weight.float() + proj.bias.float()
            K = qkv[:, :, D:2*D].view(1, -1, NH, DH).permute(0, 2, 1, 3)
            keys_by_layer[idx] = K[0]
        return hook
    blk.attn.register_forward_hook(mk(i, cat))
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
t_spec_cost = time.time() - t_spec

# ===== Method 2: MICHEL-STYLE (activation norm per head — cheap, correct proxy) =====
t_michel = time.time()
# Run the model on the same calibration context and capture per-head activation norms
act_norms = {}
for i, blk in enumerate(model.h):
    def mk_act(idx):
        def hook(mod, inp, out):
            base = out[0] if isinstance(out, tuple) else out
            # per-head activation norm (mean over tokens)
            ao = base.detach().view(base.shape[0], base.shape[1], NH, DH)
            for h in range(NH):
                key = f"{idx},{h}"
                act_norms[key] = act_norms.get(key, 0.0) + ao[:, :, h, :].norm(dim=-1).mean().item()
        return hook
    blk.register_forward_hook(mk_act(i))
with torch.no_grad():
    _ = model(ids64)  # same context as spectral (fair: same data)
for i in range(NL):
    model.h[i]._forward_hooks.clear()
t_michel_cost = time.time() - t_michel

# ===== Method 3: TRUE MICHEL (gradient-based, per Michel 2019) =====
# I_h = E[ |dL/dxi_h| ] where xi_h is the mask variable for head h.
# The clean implementation: add a gating parameter per head, compute dL/dgate.
# This requires a custom forward where each head's output is multiplied by a gate.
t_true_michel = time.time()
true_michel_importance = {}

# For a manageable subset (the pruning is per-head, and the fair comparison
# is the COST of computing the importance), let's compute on 2 calibration windows.
calib = eval_ids[:, :128]  # first 128 tokens

# Register gates: multiply each head's output slice by a learnable gate
gate_hooks = []
gates = {}  # (layer, head) -> Parameter
for li in range(NL):
    for h in range(NH):
        gates[(li, h)] = torch.nn.Parameter(torch.ones(1))
        
# Simpler approach: register hooks that multiply head outputs by gates,
# then backward the loss and read the gate gradients.
def make_gate_hook(li):
    def hook(mod, inp, out):
        base = out[0] if isinstance(out, tuple) else out
        ao = base.clone()
        for h in range(NH):
            g = gates[(li, h)]
            ao.view(ao.shape[0], ao.shape[1], NH, DH)[:, :, h, :] *= g
        return (ao,) if isinstance(out, tuple) else ao
    return hook

for li in range(NL):
    gate_hooks.append(model.h[li].register_forward_hook(make_gate_hook(li)))

# Forward + backward on calibration data
model.zero_grad()
out = model(calib)
logits = out.last_hidden_state[0, :-1] @ lm_head.weight.T
targets = calib[0, 1:]
loss = F.cross_entropy(logits, targets)
loss.backward()

for (li, h), param in gates.items():
    true_michel_importance[f"{li},{h}"] = abs(param.grad.item()) if param.grad is not None else 0.0

for hk in gate_hooks:
    hk.remove()
t_true_michel_cost = time.time() - t_true_michel

# ===== Pruning sweep =====
def prune_and_ppl(prune_set):
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
        W = min(256, eval_ids.shape[1] - 1)
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

all_heads = list(lmin_map.keys())
results = {"cost_seconds": {
    "spectral": round(t_spec_cost, 3),
    "activation_proxy": round(t_michel_cost, 3),
    "true_michel_gradient": round(t_true_michel_cost, 3),
    "random": 0.0,
}}
base_ppl = prune_and_ppl(set())
results["baseline_ppl"] = round(base_ppl, 2)
print(f"baseline: {base_ppl:.2f}", flush=True)

for pct in [5, 10, 15]:
    n_prune = int(len(all_heads) * pct / 100)
    
    # spectral: DESCENDING lmin (prune most diffuse)
    spec_sorted = sorted(all_heads, key=lambda k: lmin_map[k], reverse=True)
    spec_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in spec_sorted[:n_prune]}
    
    # activation: lowest activation norm first
    act_sorted = sorted(all_heads, key=lambda k: act_norms.get(k, 0))
    act_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in act_sorted[:n_prune]}
    
    # true Michel: lowest gradient importance first
    tm_sorted = sorted(all_heads, key=lambda k: true_michel_importance.get(k, 0))
    tm_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in tm_sorted[:n_prune]}
    
    # random
    random.seed(42 + pct)
    rand_prune = {(int(k.split(",")[0]), int(k.split(",")[1])) for k in random.sample(all_heads, n_prune)}
    
    ppl_spec = prune_and_ppl(spec_prune)
    ppl_act = prune_and_ppl(act_prune)
    ppl_tm = prune_and_ppl(tm_prune)
    ppl_rand = prune_and_ppl(rand_prune)
    
    results[f"prune_{pct}pct"] = {
        "spectral": round(ppl_spec, 1),
        "activation_proxy": round(ppl_act, 1),
        "true_michel": round(ppl_tm, 1),
        "random": round(ppl_rand, 1),
    }
    print(f"  {pct}%: spec={ppl_spec:.0f} act={ppl_act:.0f} michel={ppl_tm:.0f} rand={ppl_rand:.0f}", flush=True)

json.dump(results, open("/home/delorien/exp42b_results.json", "w"), indent=2)
print(f"\nCOSTS: spectral={t_spec_cost:.2f}s | activation={t_michel_cost:.2f}s | "
      f"true_michel={t_true_michel_cost:.2f}s")
print("saved")
