#!/usr/bin/env python3
"""VERIFY Exp 39 — is step0 actually random init? And is that GOOD for our theory?
The VSA theory PREDICTS that random keys collapse at the square point (hard edge).
If step0 = random init and it's collapsed, that's the bridge hypothesis CONFIRMED,
not the exposure law refuted. The question becomes: when does training PERSIST
the variance we see in GPT-2 124M? Is GPT-2's variance the anomaly, not the rule?
Test: compare pythia-70m step0 (if random init) with a truly random GPT-2 init."""
import torch, numpy as np, json
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer

TEXT = ("The quick brown fox jumps over the lazy dog. " * 12 +
        "The city council approved the budget for the new park after a long "
        "debate about the environmental impact and the cost of maintenance. "
        "Local residents supported the initiative but questioned the timeline.")

def lmin_all_gpt2(model, ids):
    NH = model.config.n_head; D = model.config.n_embd; DH = D // NH; NL = model.config.n_layer
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

tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok(TEXT, return_tensors="pt").input_ids[:, :64]

# 1) Truly random GPT-2 (fresh init, same architecture)
torch.manual_seed(42)
random_gpt2 = GPT2Model(GPT2Model.config_class(
    vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12))
random_gpt2.eval()
lmins_random = lmin_all_gpt2(random_gpt2, ids)
v = np.array(lmins_random)
print(f"RANDOM-INIT GPT-2 (12L): lmin_med={np.median(v):.2e} max={v.max():.2e} "
      f"collapsed={int((np.abs(v)<1e-4).sum())}/{len(v)}")
del random_gpt2

# 2) Trained GPT-2 124M (the variance case)
trained_gpt2 = GPT2Model.from_pretrained("gpt2")
trained_gpt2.eval()
lmins_trained = lmin_all_gpt2(trained_gpt2, ids)
v2 = np.array(lmins_trained)
print(f"TRAINED GPT-2 124M (12L): lmin_med={np.median(v2):.2e} max={v2.max():.2e} "
      f"collapsed={int((np.abs(v2)<1e-4).sum())}/{len(v2)}")

# 3) Pythia step0 (the "already collapsed" checkpoint)
tok_p = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision="step0")
pythia_step0 = AutoModel.from_pretrained("EleutherAI/pythia-70m-deduped", revision="step0", torch_dtype=torch.float16)
pythia_step0.eval()
cfg = pythia_step0.config; NHp = cfg.num_attention_heads; Dp = cfg.hidden_size; DHp = Dp // NHp
ids_p = tok_p(TEXT, return_tensors="pt").input_ids[:, :64]
keys_p = {}
for i, blk in enumerate(pythia_step0.layers):
    qkv = blk.attention.query_key_value
    def mk(idx, proj):
        def hook(mod, inp, out):
            h = inp[0].float()
            qo = h @ proj.weight.float().T + proj.bias.float()
            K = qo[:, :, Dp:2*Dp].view(1, -1, NHp, DHp).permute(0, 2, 1, 3)
            keys_p[idx] = K[0]
        return hook
    qkv.register_forward_hook(mk(i, qkv))
with torch.no_grad():
    _ = pythia_step0(ids_p)
for i in range(cfg.num_hidden_layers):
    pythia_step0.layers[i].attention.query_key_value._forward_hooks.clear()
lmins_p = []
for li, K in keys_p.items():
    Kn = K.numpy()
    for h in range(NHp):
        Kh = Kn[h] / np.linalg.norm(Kn[h], axis=1, keepdims=True)
        lmins_p.append(float(np.linalg.eigvalsh(Kh @ Kh.T)[0]))
v3 = np.array(lmins_p)
print(f"PYTHIA-70M STEP0 (6L, 'pre-trained'): lmin_med={np.median(v3):.2e} max={v3.max():.2e} "
      f"collapsed={int((np.abs(v3)<1e-4).sum())}/{len(v3)}")

out = {
 "verification": "Exp 39 interpretation control: is the step0 collapse the bridge hypothesis?",
 "random_init_gpt2": {"lmin_med": round(float(np.median(v)), 10), "collapsed": f"{int((np.abs(v)<1e-4).sum())}/{len(v)}"},
 "trained_gpt2_124m": {"lmin_med": round(float(np.median(v2)), 10), "collapsed": f"{int((np.abs(v2)<1e-4).sum())}/{len(v2)}"},
 "pythia_step0": {"lmin_med": round(float(np.median(v3)), 10), "collapsed": f"{int((np.abs(v3)<1e-4).sum())}/{len(v3)}"},
}
json.dump(out, open("/home/delorien/verify39_results.json", "w"), indent=2)
print("\nKEY QUESTION: does random-init GPT-2 also collapse at square?")
print("If YES -> bridge hypothesis (random keys = collapsed) CONFIRMED at init")
print("         and GPT-2's variance is something training ACHIEVES, not the default")
print("If NO  -> the Pythia step0 checkpoint is not truly random init")
