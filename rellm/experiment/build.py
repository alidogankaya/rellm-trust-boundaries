"""Build a genuinely smaller model by slicing weights — no masks, no hooks.

Supported layouts: LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM
(gated MLP gate/up/down, separate q/k/v/o projections, optional q/k/v bias, optional q/k norm, GQA).

Constraints that keep the result a standard HF model:
- every kept layer keeps the same number of query heads and MLP neurons (heads/neurons may differ per layer);
- under GQA every kv group keeps the same number of query heads and kv heads are kept; under MHA a removed query head
  takes its kv head with it.
The new model is created on the meta device, materialised empty and filled by copying; every parameter must be
written or the build fails, and non-persistent buffers (rotary inv_freq) are copied from the source.
"""
from __future__ import annotations

import copy
from typing import Any

import torch

SUPPORTED = ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "MistralForCausalLM")


class BuildError(ValueError):
    pass


def geometry(model) -> dict[str, int]:
    c = model.config
    n_heads = c.num_attention_heads
    n_kv = getattr(c, "num_key_value_heads", None) or n_heads
    d_head = getattr(c, "head_dim", None) or c.hidden_size // n_heads
    return {"n_layers": c.num_hidden_layers, "n_heads": n_heads, "n_kv": n_kv, "group": n_heads // n_kv, "d_head": d_head,
            "d_mlp": c.intermediate_size, "d_model": c.hidden_size}


def check_heads(heads: dict[int, list[int]], g: dict[str, int]) -> int:
    counts = {len(v) for v in heads.values()}
    if len(counts) != 1:
        raise BuildError(f"every kept layer must keep the same number of heads, got {sorted(counts)}")
    if g["group"] == 1:  # multi-head attention: each query head owns its kv head, so any subset works
        return counts.pop()
    for L, hs in heads.items():
        per = [sum(1 for h in hs if h // g["group"] == k) for k in range(g["n_kv"])]
        if len(set(per)) != 1 or per[0] == 0:
            raise BuildError(f"layer {L}: each kv group must keep the same (>=1) number of query heads, got {per}")
    return counts.pop()


@torch.no_grad()
def build_pruned(src, keep_layers: list[int] | None = None, heads: dict[int, list[int]] | None = None,
                 neurons: dict[int, torch.Tensor] | None = None) -> Any:
    if type(src).__name__ not in SUPPORTED:
        raise BuildError(f"{type(src).__name__} is not supported by the builder; supported: {SUPPORTED}")
    g = geometry(src)
    keep_layers = sorted(keep_layers if keep_layers is not None else range(g["n_layers"]))
    heads = {L: sorted((heads or {}).get(L, range(g["n_heads"]))) for L in keep_layers}
    neurons = {L: torch.as_tensor((neurons or {}).get(L, torch.arange(g["d_mlp"]))).long().sort().values for L in keep_layers}
    n_keep_heads = check_heads(heads, g)
    n_neur = {int(v.numel()) for v in neurons.values()}
    if len(n_neur) != 1 or n_neur == {0}:
        raise BuildError(f"every kept layer must keep the same (>=1) number of MLP neurons, got {sorted(n_neur)}")

    cfg = copy.deepcopy(src.config)
    cfg.num_hidden_layers = len(keep_layers)
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = [cfg.layer_types[L] for L in keep_layers]
    mha = g["group"] == 1
    cfg.num_attention_heads = n_keep_heads
    cfg.num_key_value_heads = n_keep_heads if mha else g["n_kv"]
    cfg.head_dim = g["d_head"]
    cfg.intermediate_size = n_neur.pop()
    p0 = next(src.parameters())
    old_default = torch.get_default_dtype()
    torch.set_default_dtype(p0.dtype)
    try:
        with torch.device("meta"):
            new = type(src)(cfg)
    finally:
        torch.set_default_dtype(old_default)
    new = new.to_empty(device=p0.device).eval()

    written: set[str] = set()
    params = dict(new.named_parameters())

    def put(name: str, value: torch.Tensor) -> None:
        if name not in params:
            return
        if params[name].shape != value.shape:
            raise BuildError(f"{name}: shape {tuple(params[name].shape)} != {tuple(value.shape)}")
        params[name].copy_(value)
        written.add(name)

    put("model.embed_tokens.weight", src.model.embed_tokens.weight)
    put("model.norm.weight", src.model.norm.weight)
    if "lm_head.weight" in params:
        put("lm_head.weight", src.lm_head.weight)
    d = g["d_head"]
    for i, L in enumerate(keep_layers):
        o, pre = src.model.layers[L], f"model.layers.{i}."
        rows = torch.cat([torch.arange(h * d, (h + 1) * d) for h in heads[L]]).to(p0.device)
        put(pre + "self_attn.q_proj.weight", o.self_attn.q_proj.weight[rows])
        if o.self_attn.q_proj.bias is not None:
            put(pre + "self_attn.q_proj.bias", o.self_attn.q_proj.bias[rows])
        for nm in ("k_proj", "v_proj"):  # GQA keeps every kv head; MHA keeps the kv heads of the kept query heads
            src_m = getattr(o.self_attn, nm)
            put(pre + f"self_attn.{nm}.weight", src_m.weight[rows] if mha else src_m.weight)
            if src_m.bias is not None:
                put(pre + f"self_attn.{nm}.bias", src_m.bias[rows] if mha else src_m.bias)
        put(pre + "self_attn.o_proj.weight", o.self_attn.o_proj.weight[:, rows])
        if o.self_attn.o_proj.bias is not None:
            put(pre + "self_attn.o_proj.bias", o.self_attn.o_proj.bias)
        for nm in ("q_norm", "k_norm"):
            if hasattr(o.self_attn, nm):
                put(pre + f"self_attn.{nm}.weight", getattr(o.self_attn, nm).weight)
        put(pre + "input_layernorm.weight", o.input_layernorm.weight)
        put(pre + "post_attention_layernorm.weight", o.post_attention_layernorm.weight)
        nk = neurons[L].to(p0.device)
        put(pre + "mlp.gate_proj.weight", o.mlp.gate_proj.weight[nk])
        put(pre + "mlp.up_proj.weight", o.mlp.up_proj.weight[nk])
        put(pre + "mlp.down_proj.weight", o.mlp.down_proj.weight[:, nk])
        for nm, sl in (("gate_proj", nk), ("up_proj", nk)):
            if getattr(getattr(o.mlp, nm), "bias", None) is not None:
                put(pre + f"mlp.{nm}.bias", getattr(o.mlp, nm).bias[sl])
        if getattr(o.mlp.down_proj, "bias", None) is not None:
            put(pre + "mlp.down_proj.bias", o.mlp.down_proj.bias)
    missing = set(params) - written
    if missing:
        raise BuildError(f"parameters not written by the builder: {sorted(missing)[:8]} ...")
    src_buffers = dict(src.named_buffers())
    for name, buf in new.named_buffers():
        key = name
        if ".layers." in name:  # per-layer buffers: map new index -> source index
            parts = name.split(".")
            k = parts.index("layers") + 1
            parts[k] = str(keep_layers[int(parts[k])])
            key = ".".join(parts)
        if key in src_buffers and src_buffers[key].shape == buf.shape:
            buf.copy_(src_buffers[key])
    if getattr(cfg, "tie_word_embeddings", False):
        new.tie_weights()
    return new
