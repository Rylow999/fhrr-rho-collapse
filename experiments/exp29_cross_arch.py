#!/usr/bin/env python3
import torch, numpy as np, json, sys
from transformers import AutoModel, AutoTokenizer

def measure(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    cfg = model.config
    NH = cfg.num_attention_heads
    DH = cfg.hidden_size // NH
    D = cfg.hidden_size
    TEXT = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tok(TEXT, return_tensors="pt").input_ids
    keys_by_layer = {}
    if hasattr(model, "layers"):
        blocks = model.layers
        kind = "neox"
    elif hasattr(model, "decoder"):
        blocks = model.decoder.layers
        kind = "opt"
    else:
        blocks = model.h
        kind = "gpt2"
    for i, blk in enumerate(blocks):
        attn = (blk.attention if kind == "neox" else
                blk.self_attn if kind == "opt" else blk.attn)
        if kind == "opt":
            kp = attn.k_proj
            def make_hook(idx, proj, NH=NH, DH=DH):
                def hook(mod, args, kwargs, out):
                    hidden = kwargs.get("hidden_states")
                    if hidden is None and args:
                        hidden = args[0]
                    if hidden is None:
                        return
                    hidden = hidden.float()
                    B, T, _ = hidden.shape
                    K = hidden @ proj.weight.float().T + proj.bias.float()
                    K = K.view(B, T, NH, DH).permute(0, 2, 1, 3)
                    keys_by_layer[idx] = K[0]
                return hook
            attn.register_forward_hook(make_hook(i, kp), with_kwargs=True)
        elif kind == "neox":
            proj = attn.query_key_value
            def make_hook(idx, proj, NH=NH, DH=DH, D=D):
                def hook(mod, inp, out):
                    hidden = inp[0].float()
                    B, T, _ = hidden.shape
                    qkv = hidden @ proj.weight.float().T + proj.bias.float()
                    K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
                    keys_by_layer[idx] = K[0]
                return hook
            attn.register_forward_hook(make_hook(i, proj))
        else:
            kproj = attn.c_attn
            def make_hook(idx, proj, NH=NH, DH=DH, D=D):
                def hook(mod, inp, out):
                    hidden = inp[0].float()
                    B, T, _ = hidden.shape
                    qkv = hidden @ proj.weight.float().T + proj.bias.float()
                    K = qkv[:, :, D:2*D].view(B, T, NH, DH).permute(0, 2, 1, 3)
                    keys_by_layer[idx] = K[0]
                return hook
            attn.register_forward_hook(make_hook(i, kproj))
    with torch.no_grad():
        _ = model(ids)
    layers_out = []
    idxs = sorted(keys_by_layer)
    assert idxs, "no hooks fired"
    for li in [idxs[0], idxs[len(idxs)//2], idxs[-1]]:
        K = keys_by_layer[li].numpy()
        H, T, dh = K.shape
        lmins = []
        for h in range(H):
            Kh = K[h] / np.linalg.norm(K[h], axis=1, keepdims=True)
            lmins.append(float(np.linalg.eigvalsh(Kh @ Kh.T)[0]))
        med = float(np.median(lmins))
        layers_out.append({"layer": li, "T": T, "lmin_median": med,
                           "lmin_min": min(lmins), "lmin_max": max(lmins)})
        print(f"  {model_name} layer {li}: T={T} lmin_med={med:.2e}", flush=True)
    del model
    return {"model": model_name, "d_head": DH, "layers": layers_out}

out = []
for m in sys.argv[1:-1]:
    try:
        out.append(measure(m))
    except Exception:
        import traceback; traceback.print_exc()
json.dump(out, open(sys.argv[-1], "w"), indent=2)
print("saved", sys.argv[-1])
