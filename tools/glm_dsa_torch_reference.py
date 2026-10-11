#!/usr/bin/env python3
"""Full GLM-5.3 independent reference (docs/glm53_plan.md G5, 2026-09-12):
transformers' own GlmMoeDsa layer code (modeling_glm_moe_dsa.py) over the
REAL checkpoint's first layers, against the engine's per-layer residual
dump (glm_dsa_forward_check --dump-states --layers N). The engine's other
gates compare it to a python reference written from the same reading of
the architecture; this one is the reading transformers ships (the rule
that found GLM-4.7's RoPE bug).

The pack-quantized tensors (compressed-tensors: `weight_packed` I32 [N,
K*bits/32] of unsigned codes, low nibble/byte first; `weight_scale` bf16
[N, K/64]; `weight_shape`) are dequantized as code x scale and loaded into
a bf16 model — one bf16 rounding of every weight that the engine's exact
dequant does not make (an 0.2 %-class difference, inside the budget).
Runs on the 5090 box's ~/.venvs/glm53q312 (transformers 5.14.1): four
layers in bf16 take ~26 GB on the GPU.

Every layer's output is captured by a forward hook (HF's last hidden state
carries the final norm, so the last layer would otherwise be lost). With
an engine dump the layers are compared twice: CHAINED (transformers' own
inputs, errors accumulate down the stack) and ISOLATED (a pre-hook feeds
layer l the engine's layer l-1 output, so each number is one layer's own
error). `--dtype float32` on the dense layers (`--layers 3`: layer 3's
256 experts do not fit the 5090 in fp32) measures the engine's bf16
pipeline against a near-exact reference instead of against another bf16
pipeline that rounds at other points.

  glm_dsa_torch_reference.py --checkpoint-dir DIR --ids 1,2,3 --layers 4 \
      --engine-dump states.bin [--device cuda] [--dtype float32]
"""
import argparse
import math
import json
import os
import sys

import torch
from safetensors import safe_open


class Shards:
    def __init__(self, d):
        self.d = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, name):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.d, shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name)

    def has(self, name):
        return name in self.index

    # The NF4I8 contract (src/models/glm_dsa/config.cpp): `weight_indices`
    # nibbles select a level of the fixed codebook, one bf16 scale per 128;
    # the routed experts were quantized in the H32-rotated input basis.
    NF4I8_CODEBOOK = torch.tensor([-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
                                  dtype=torch.float32)

    @staticmethod
    def hadamard32():
        """The normalized 32-wide Sylvester Hadamard (symmetric, its own inverse)."""
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < 32:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        return h / math.sqrt(32.0)

    def linear(self, prefix, fold_h32=False):
        """A Linear's weight as float32: BF16 verbatim or the packed triple
        dequantized. `fold_h32` (a routed expert of the NF4I8 checkpoint):
        the stored matrix W' acts on H32-rotated inputs, so W' . H32 is the
        weight on the plain input (H32 symmetric and orthonormal; the engine
        rotates the activation in bf16 instead — a bf16 rounding apart)."""
        if self.has(prefix + ".weight_packed") or self.has(prefix + ".weight_indices"):
            codebook = self.has(prefix + ".weight_indices")
            words = self.get(prefix + (".weight_indices" if codebook else ".weight_packed"))  # int32 [N, K*bits/32]
            scales = self.get(prefix + ".weight_scale").float()   # [N, K/group]
            shape = self.get(prefix + ".weight_shape").tolist()   # [N, K]
            n, k = int(shape[0]), int(shape[1])
            bits = words.shape[1] * 32 // k
            per = 32 // bits
            group = 128 if codebook else 64
            u = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            codes = torch.stack([(u >> (bits * j)) & ((1 << bits) - 1) for j in range(per)], dim=-1).reshape(n, k)
            if codebook:
                levels = self.NF4I8_CODEBOOK[codes]
            else:
                levels = codes.to(torch.float32) - float(1 << (bits - 1))
            w = levels * scales.repeat_interleave(group, dim=1)
            if fold_h32:
                h = self.hadamard32()
                w = (w.reshape(n, k // 32, 32) @ h).reshape(n, k)
            return w
        return self.get(prefix + ".weight").to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--engine-dump", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as mg
    cfg = AutoConfig.from_pretrained(args.checkpoint_dir)
    # The NF4I8 contract's rotated routed range (the fold in Shards.linear).
    qc = getattr(cfg, "quantization_config", None) or {}
    h32_layers = None
    if isinstance(qc, dict) and qc.get("quant_method") == "dgpp_nf4i8":
        t = qc["routed_experts"].get("input_transform") or {}
        if t.get("type") != "normalized_hadamard" or t.get("block_size") != 32:
            raise SystemExit("dgpp_nf4i8: the reference implements the 32-wide normalized Hadamard rotation")
        h32_layers = (int(qc["routed_experts"]["layers"][0]), int(qc["routed_experts"]["layers"][1]))
    cfg.num_hidden_layers = args.layers
    for k in ("indexer_types", "layer_types", "mlp_layer_types"):
        if getattr(cfg, k, None) is not None:
            setattr(cfg, k, list(getattr(cfg, k))[: args.layers])
    cfg.num_nextn_predict_layers = 0
    for k in ("quantization_config",):
        if hasattr(cfg, k):
            try:
                delattr(cfg, k)
            except Exception:
                pass
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    # The model on the meta device: the loaded tensors are ASSIGNED as its
    # parameters (no second copy of 26 GB of bf16 on a 62 GB host), every
    # tensor cast to the model's dtype as it is read.
    with torch.device("meta"):
        model = mg.GlmMoeDsaModel(cfg).eval()
    sh = Shards(args.checkpoint_dir)
    sd = model.state_dict()
    print("model params:", len(sd), "e.g.", [k for k in list(sd.keys())[:8]], file=sys.stderr)
    new = {}
    E = cfg.n_routed_experts
    for key in sd:
        if ".mlp.experts." in key and not key.split(".mlp.experts.")[1][0].isdigit():
            continue  # the fused expert parameters, filled below
        name = "model." + key
        if sh.has(name):
            new[key] = sh.get(name).to(sd[key].dtype)
        elif key.endswith(".weight") and sh.has(name[: -len(".weight")] + ".weight_packed"):
            new[key] = sh.linear(name[: -len(".weight")]).to(sd[key].dtype)
    for l in range(args.layers):
        q = f"layers.{l}."
        p = f"model.layers.{l}."
        if l < cfg.first_k_dense_replace:
            continue
        if q + "mlp.experts.gate_up_proj" in sd:
            gu = torch.empty(sd[q + "mlp.experts.gate_up_proj"].shape, dtype=sd[q + "mlp.experts.gate_up_proj"].dtype)
            dn = torch.empty(sd[q + "mlp.experts.down_proj"].shape, dtype=sd[q + "mlp.experts.down_proj"].dtype)
            fold = h32_layers is not None and h32_layers[0] <= l <= h32_layers[1]
            for e in range(E):
                g = sh.linear(p + f"mlp.experts.{e}.gate_proj", fold).to(gu.dtype)
                u = sh.linear(p + f"mlp.experts.{e}.up_proj", fold).to(gu.dtype)
                d = sh.linear(p + f"mlp.experts.{e}.down_proj", fold).to(dn.dtype)
                if gu.shape[1] == g.shape[0] + u.shape[0]:      # [E, 2I, H]
                    gu[e, : g.shape[0]] = g
                    gu[e, g.shape[0]:] = u
                else:                                             # [E, H, 2I]
                    gu[e, :, : g.shape[0]] = g.T
                    gu[e, :, g.shape[0]:] = u.T
                dn[e] = d if dn.shape[1] == d.shape[0] else d.T
            new[q + "mlp.experts.gate_up_proj"] = gu
            new[q + "mlp.experts.down_proj"] = dn
        else:
            for e in range(E):
                for m in ("gate_proj", "up_proj", "down_proj"):
                    new[q + f"mlp.experts.{e}.{m}.weight"] = sh.linear(p + f"mlp.experts.{e}.{m}", fold).to(dtype)
        print(f"layer {l} experts loaded", file=sys.stderr)
    missing = [k for k in sd if k not in new]
    extra = [k for k in new if k not in sd]
    print("missing:", missing[:10], "extra:", extra[:10], file=sys.stderr)
    if missing:
        print("ERROR: unmapped parameters", file=sys.stderr)
    model.load_state_dict(new, strict=False, assign=True)
    del new
    # The non-persistent buffers the meta construction left (the rotary
    # inv_freq): rebuilt for real; nothing may stay on the meta device.
    model.rotary_emb = mg.GlmMoeDsaRotaryEmbedding(config=cfg)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        print("ERROR: tensors left on the meta device:", left[:10], file=sys.stderr)
        sys.exit(1)
    model = model.to(args.device)
    ids = torch.tensor([[int(v) for v in args.ids.split(",")]], device=args.device)
    T, H = ids.shape[1], cfg.hidden_size
    captured = {}

    def capture(l):
        def hook(mod, inputs, output):
            captured[l] = (output[0] if isinstance(output, (tuple, list)) else output)[0].float().cpu()
        return hook

    for l, layer in enumerate(model.layers):
        layer.register_forward_hook(capture(l))
    # The router's boundary margin per row (the 8th minus the 9th score,
    # sigmoid + bias, no groups at n_group 1): a row whose margin is near
    # zero routes differently under the engine's 0.3 % hidden noise, and
    # its MoE output differs by one expert — the certification of the
    # worst rows of a MoE layer.
    route_margin = {}

    def route_hook(l):
        def hook(mod, inputs, output):
            s = output[0].float().sigmoid() + mod.e_score_correction_bias.float()
            top = s.topk(mod.top_k + 1, dim=-1)[0]
            route_margin[l] = (top[:, mod.top_k - 1] - top[:, mod.top_k]).cpu()
        return hook

    for l, layer in enumerate(model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and hasattr(gate, "e_score_correction_bias"):
            gate.register_forward_hook(route_hook(l))
    dump = None
    if args.engine_dump:
        raw = open(args.engine_dump, "rb").read()
        per = T * H * 2
        n_layers = (len(raw) - per) // per
        print(f"engine dump: {n_layers} layers + final of [{T}, {H}] bf16", file=sys.stderr)
        dump = [torch.frombuffer(bytearray(raw[l * per:(l + 1) * per]), dtype=torch.bfloat16).reshape(T, H)
                for l in range(n_layers + 1)]  # [l] = after layer l; [n_layers] = final post-norm

    def compare(tag, l, eng, ref):
        eng = eng.float()
        d = eng - ref
        l2 = d.norm() / (ref.norm() + 1e-30)
        row = (d.norm(dim=1) / (ref.norm(dim=1) + 1e-30))
        worst = row.argsort(descending=True)[:4]
        print(f"{tag} {l}: rel l2 {l2:.4g}  max|d| {d.abs().max():.4g}  ref rms {ref.pow(2).mean().sqrt():.4g} max {ref.abs().max():.4g}"
              f"  worst rows " + " ".join(f"t{int(i)}:{row[i]:.3g}" for i in worst))
        m = route_margin.get(l) if tag.endswith("layer") else None
        if m is not None:
            order = m.argsort()
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(m))
            print(f"{tag} {l} routing margins: median {m.median():.4g}, {(m < 1e-3).sum().item()} rows under 1e-3;"
                  f" worst rows " + " ".join(f"t{int(i)}:{m[i]:.2e}(#{int(rank[i])})" for i in worst))

    def run(tag):
        captured.clear()
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        final = out.hidden_states[-1][0].float().cpu()
        print(f"{tag}: {len(captured)} layer outputs captured, final post-norm {tuple(final.shape)}", file=sys.stderr)
        if dump is None:
            return
        for l in range(min(len(dump) - 1, len(captured))):
            compare(f"{tag} layer", l, dump[l], captured[l])
        if len(dump) - 1 == len(captured):
            compare(f"{tag} final (post-norm)", len(captured) - 1, dump[-1], final)
        else:
            print(f"{tag} final: the dump has {len(dump) - 1} layers, the model {len(captured)}; the post-norm rows differ by construction")

    run("chained")
    if dump is not None and len(model.layers) > 1:
        # ISOLATED: layer l >= 1 reads the engine's layer l-1 output. The
        # first four layers each own an indexer (no prev_topk_indices
        # crosses the seam), so the substitution is complete.
        def feed(l):
            def pre(mod, a, kw):
                x = dump[l - 1].to(device=args.device, dtype=dtype).unsqueeze(0)
                if "hidden_states" in kw:
                    kw["hidden_states"] = x
                else:
                    a = (x,) + tuple(a[1:])
                return a, kw
            return pre
        for l in range(1, min(len(model.layers), len(dump) - 1)):
            model.layers[l].register_forward_pre_hook(feed(l), with_kwargs=True)
        run("isolated")

if __name__ == "__main__":
    main()
