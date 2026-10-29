"""Declared facts: what the checkpoint says about itself, without loading or running it.

Mirrors REA's `inspect_model_artifact`: header-only reads of safetensors files
(local directory or Hugging Face repo via the hub metadata API), then an
architecture reconstruction from tensor names and shapes. Everything here is
`declared` authority — derived from declarations, not measured behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Any

from ..session import SessionError, safetensors_header_digest

DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1, "F4": 0.5, "I4": 0.5, "U4": 0.5}

FAMILIES = [  # (family, layer regex, attention q-proj name, notes)
    ("llama-like", r"^model\.layers\.(\d+)\.", "self_attn.q_proj.weight", "Llama/Qwen/Mistral/Gemma/Phi-3 layout"),
    ("gpt2", r"^transformer\.h\.(\d+)\.", "attn.c_attn.weight", "GPT-2 fused qkv (Conv1D)"),
    ("gpt-neox", r"^gpt_neox\.layers\.(\d+)\.", "attention.query_key_value.weight", "fused qkv"),
    ("falcon", r"^transformer\.h\.(\d+)\.", "self_attention.query_key_value.weight", "fused qkv"),
    ("bloom", r"^h\.(\d+)\.", "self_attention.query_key_value.weight", "fused qkv"),
    ("mamba", r"^backbone\.layers\.(\d+)\.", "mixer.in_proj.weight", "state-space, no attention heads"),
]


def _read_local(root: str) -> list[dict[str, Any]]:
    files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    if not files:
        raise SessionError("no_safetensors", f"no .safetensors files in {root}",
                           "Convert pickle/GGUF checkpoints to safetensors first; REA-style header inspection needs the safetensors header.")
    out = []
    for f in files:
        p = os.path.join(root, f)
        with open(p, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        meta = header.pop("__metadata__", None)
        rows = [{"name": k, "dtype": v["dtype"], "shape": v["shape"], "file": f} for k, v in header.items()]
        out.append({"file": f, "size": os.path.getsize(p), "header_sha256": safetensors_header_digest(p)["header_sha256"],
                    "metadata": meta, "tensors": rows})
    return out


def _read_hub(repo_id: str, revision: str | None, token: str | None) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    meta = api.get_safetensors_metadata(repo_id, revision=revision)
    out = []
    for fname, fmeta in sorted(meta.files_metadata.items()):
        rows = [{"name": k, "dtype": t.dtype, "shape": list(t.shape), "file": fname} for k, t in fmeta.tensors.items()]
        out.append({"file": fname, "size": None, "header_sha256": None, "metadata": fmeta.metadata, "tensors": rows})
    return out


def _load_config(source: str, revision: str | None, token: str | None) -> dict[str, Any] | None:
    try:
        if os.path.isdir(source):
            p = os.path.join(source, "config.json")
            return json.load(open(p)) if os.path.exists(p) else None
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(source, "config.json", revision=revision, token=token)))
    except Exception:  # noqa: BLE001
        return None


def reconstruct_architecture(tensors: list[dict[str, Any]], config: dict[str, Any] | None) -> dict[str, Any]:
    names = [t["name"] for t in tensors]
    by_name = {t["name"]: t for t in tensors}
    family, pat, qname, note = None, None, None, None
    for fam, rx, q, n in FAMILIES:
        if any(re.match(rx, nm) for nm in names):
            family, pat, qname, note = fam, rx, q, n
            break
    limitations: list[str] = []
    layers = sorted({int(m.group(1)) for nm in names if pat and (m := re.match(pat, nm))})
    n_layers = (max(layers) + 1) if layers else None
    if layers and n_layers != len(layers):
        limitations.append(f"layer indices are not contiguous ({len(layers)} seen, max {n_layers - 1}); shard may be incomplete")

    embed = next((t for t in tensors if re.search(r"(embed_tokens|wte|embed_in|word_embeddings)\.weight$", t["name"])), None)
    lm_head = next((t for t in tensors if re.search(r"^(lm_head|embed_out)\.weight$", t["name"])), None)
    d_model = embed["shape"][1] if embed else None
    vocab = embed["shape"][0] if embed else None
    tied = lm_head is None and embed is not None

    q = by_name.get(f"model.layers.0.{qname}") if family == "llama-like" else None
    k = by_name.get("model.layers.0.self_attn.k_proj.weight")
    n_heads = n_kv = d_head = None
    if config:
        n_heads = config.get("num_attention_heads") or config.get("n_head")
        n_kv = config.get("num_key_value_heads", n_heads)
        d_head = config.get("head_dim") or ((config.get("hidden_size") or d_model) // n_heads if n_heads and (config.get("hidden_size") or d_model) else None)
    if q is not None and k is not None and d_model:
        # q: (n_heads*d_head, d_model), k: (n_kv*d_head, d_model)
        if n_heads and q["shape"][0] % n_heads == 0:
            d_head_obs = q["shape"][0] // n_heads
            n_kv_obs = k["shape"][0] // d_head_obs
            if d_head and d_head_obs != d_head:
                limitations.append(f"config head_dim {d_head} disagrees with q_proj shape ({d_head_obs})")
            if n_kv and n_kv_obs != n_kv:
                limitations.append(f"config num_key_value_heads {n_kv} disagrees with k_proj shape ({n_kv_obs})")
            d_head, n_kv = d_head_obs, n_kv_obs
        elif not n_heads:
            limitations.append("no config: head count cannot be derived from a q_proj shape alone (needs head_dim)")
    mlp_up = by_name.get("model.layers.0.mlp.up_proj.weight") or by_name.get("model.layers.0.mlp.gate_proj.weight") \
        or by_name.get("transformer.h.0.mlp.c_fc.weight")
    d_mlp = (mlp_up["shape"][0] if family == "llama-like" else mlp_up["shape"][1]) if mlp_up else (config or {}).get("intermediate_size")

    experts = sorted({int(m.group(1)) for nm in names if (m := re.search(r"\.experts\.(\d+)\.", nm))})
    lora = [nm for nm in names if re.search(r"lora_[AB]", nm)]
    quant_markers = Counter()
    for t in tensors:
        for mk in ("qweight", "qzeros", "scales", "absmax", "quant_map", "SCB", "weight_scale", "weight_packed"):
            if t["name"].endswith(mk) or f".{mk}." in t["name"]:
                quant_markers[mk] += 1
        if t["dtype"] in ("F8_E4M3", "F8_E5M2", "I8", "U8", "F4", "I4", "U4"):
            quant_markers[t["dtype"]] += 1
    param_total = 0
    dtype_hist: Counter = Counter()
    unknown_dtype = False
    for t in tensors:
        n = 1
        for d in t["shape"]:
            n *= d
        param_total += n
        dtype_hist[t["dtype"]] += n
        if t["dtype"] not in DTYPE_BYTES:
            unknown_dtype = True
    if unknown_dtype:
        limitations.append("unknown dtype(s) present; byte totals unavailable for them")
    norm_kind = "rmsnorm" if any(re.search(r"input_layernorm\.weight$", nm) for nm in names) and \
        not any(re.search(r"input_layernorm\.bias$", nm) for nm in names) else ("layernorm" if any(nm.endswith("ln_1.bias") or nm.endswith("input_layernorm.bias") for nm in names) else None)
    limitations.append("derived from tensor declarations, not from running the model; shared/tied weights are not deduplicated")
    return {"family": family, "family_note": note, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab, "n_heads": n_heads,
            "n_kv_heads": n_kv, "d_head": d_head, "d_mlp": d_mlp, "tied_embeddings": tied, "final_norm_kind": norm_kind,
            "moe": {"experts_per_layer": len(experts)} if experts else None,
            "lora_adapter": {"tensor_count": len(lora), "sample": lora[:4]} if lora else None,
            "quantization_markers": dict(quant_markers) or None,
            "declared_parameters": param_total, "declared_parameters_human": f"{param_total/1e9:.3f}B",
            "dtype_histogram": {k: v for k, v in dtype_hist.most_common()},
            "config_consistency": "no config" if not config else ("consistent" if not any("disagrees" in l for l in limitations) else "disagrees"),
            "limitations": limitations}


def inspect_model_artifact(source: str, revision: str | None = None, hf_token: str | None = None,
                           tensor_name_filter: str | None = None, limit: int = 64) -> dict[str, Any]:
    """Header-only inspection of a local safetensors directory or a Hugging Face repo id."""
    if os.path.isdir(source):
        shards, mode = _read_local(source), "local-header"
    else:
        try:
            shards, mode = _read_hub(source, revision, hf_token), "hub-metadata"
        except Exception as e:  # noqa: BLE001
            raise SessionError("artifact_unavailable", f"cannot read safetensors metadata for {source!r}: {type(e).__name__}: {e}",
                               "Give a local directory with .safetensors files, or a public/authorised HF repo id (HF_TOKEN).") from e
    config = _load_config(source, revision, hf_token)
    tensors = [t for s in shards for t in s["tensors"]]
    arch = reconstruct_architecture(tensors, config)
    limitations = arch.pop("limitations")
    rx = re.compile(tensor_name_filter) if tensor_name_filter else None
    selected = [t for t in tensors if not rx or rx.search(t["name"])]
    headers = "".join(s["header_sha256"] or "" for s in shards)
    return {"source": source, "read_mode": mode, "revision": revision,
            "identity": {"headers_sha256": hashlib.sha256(headers.encode()).hexdigest() if headers else None,
                         "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() if config else None,
                         "note": "header digests commit to tensor declarations, not to the weights"},
            "shards": [{k: v for k, v in s.items() if k != "tensors"} | {"n_tensors": len(s["tensors"])} for s in shards],
            "n_tensors": len(tensors), "architecture": arch,
            "config_headline": {k: config[k] for k in ("model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                                        "num_key_value_heads", "intermediate_size", "vocab_size", "torch_dtype", "tie_word_embeddings",
                                                        "rope_theta", "max_position_embeddings") if config and k in config} if config else None,
            "tensors": selected[:limit], "tensors_total_matching": len(selected),
            "coverage": {"status": "complete" if mode == "local-header" else "partial",
                         "note": None if mode == "local-header" else "hub metadata gives names/dtypes/shapes but no header digests or file sizes"},
            "limitations": limitations + ["no weights were read, loaded or executed"]}
