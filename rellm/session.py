"""ModelSession: one opened HF causal LM plus hook plumbing and identity.

REA parallels:
- open_model binds one backend + dtype + attention implementation immutably for
  the session (`binding`), allocates a `run_id`, and computes the subject identity
  (config digest + safetensors header digests; optional full digest).
- `analysis_profile` names the semantics every result was produced under; it is
  part of each Evidence identity.
- `facet_availability` is probed once after loading: e.g. attention weights are
  only observable with eager attention.
- close_model releases what the session owns and reports what was freed.

Everything works on plain `transformers` modules via forward hooks, so any
decoder-only model (Llama, Qwen, Mistral, Gemma, Phi, GPT-2 ...) works without a
TransformerLens conversion.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn

from ._version import __version__

HOOK_SEMANTICS_VERSION = 2


# ----------------------------------------------------------------------------- helpers
def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    """Config attribute read that tolerates heterogeneous (per-layer) configs raising on global access."""
    try:
        return getattr(cfg, name, default)
    except Exception:  # noqa: BLE001
        return default


def _first_attr(obj: Any, names: Iterable[str]) -> tuple[str | None, Any]:
    for n in names:
        if obj is not None and hasattr(obj, n):
            return n, getattr(obj, n)
    return None, None


def _out_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_out(output: Any, new: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    if isinstance(output, list):
        return [new] + list(output[1:])
    return new


class SessionError(RuntimeError):
    def __init__(self, code: str, message: str, remediation: str = "", **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "remediation": self.remediation, "details": self.details}}


class Budget:
    """Bounded work: stop a loop when seconds or passes run out, keeping what was computed."""

    def __init__(self, max_seconds: float | None, max_passes: int | None) -> None:
        self.t0 = time.time()
        self.max_seconds = max_seconds
        self.max_passes = max_passes
        self.passes = 0
        self.exhausted: str | None = None

    def tick(self) -> bool:
        self.passes += 1
        if self.max_passes is not None and self.passes >= self.max_passes:
            self.exhausted = "max_passes"
        elif self.max_seconds is not None and time.time() - self.t0 >= self.max_seconds:
            self.exhausted = "max_seconds"
        return self.exhausted is None

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.t0, 3)


def safetensors_header_digest(path: str) -> dict[str, Any]:
    """REA-style header hash: read only the JSON header, never the weights."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = fh.read(n)
    return {"file": os.path.basename(path), "size": os.path.getsize(path), "header_bytes": n,
            "header_sha256": hashlib.sha256(header).hexdigest()}


def full_file_digest(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ForwardResult:
    logits: torch.Tensor                      # (1, seq, vocab)
    resid_pre0: torch.Tensor | None = None    # (1, seq, d_model) input to layer 0
    resid_post: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_out: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_out: dict[int, torch.Tensor] = field(default_factory=dict)
    head_in: dict[int, torch.Tensor] = field(default_factory=dict)   # o_proj input (1, seq, n_heads*d_head)
    attentions: dict[int, torch.Tensor] = field(default_factory=dict)  # (1, heads, q, k)


# ----------------------------------------------------------------------------- session
class ModelSession:
    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.tok = None
        self.model_id: str | None = None
        self.revision: str | None = None
        self.dtype: str | None = None
        self.device: str | None = None
        self.run_id: str | None = None
        self.opened_at: str | None = None
        self.lock = threading.Lock()
        self._struct: dict[str, Any] = {}
        self._weights: dict[str, Any] = {}
        self._facets: dict[str, Any] = {}
        self._identity_cache: dict[str, Any] | None = None
        self.tokenizer_note: str | None = None

    # ---------------------------------------------------------------- loading
    def open(self, model_id: str, dtype: str = "bfloat16", device: str | None = None, revision: str | None = None,
             trust_remote_code: bool = False, hf_token: str | None = None, hash_weights: str = "header") -> dict[str, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in ("bfloat16", "float16", "float32"):
            raise SessionError("bad_dtype", "dtype must be bfloat16|float16|float32")
        if hash_weights not in ("header", "full", "none"):
            raise SessionError("bad_hash", "hash_weights must be header|full|none")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw: dict[str, Any] = dict(attn_implementation="eager", trust_remote_code=trust_remote_code)
        if revision:
            kw["revision"] = revision
        if hf_token:
            kw["token"] = hf_token
        if self.model is not None:
            self.close()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kw)
        model.to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=hf_token, trust_remote_code=trust_remote_code)
        self.tokenizer_note = None
        if len(tok.encode("the capital of", add_special_tokens=False)) == 0:
            # AutoTokenizer routes by config.model_type; a checkpoint whose tokenizer.json does not match that family's
            # tokenizer class can come back unable to encode anything. Fall back to the generic fast tokenizer.
            from transformers import PreTrainedTokenizerFast
            generic = PreTrainedTokenizerFast.from_pretrained(model_id, revision=revision, token=hf_token)
            if len(generic.encode("the capital of", add_special_tokens=False)) > 0:
                self.tokenizer_note = f"{type(tok).__name__} could not encode plain text; using {type(generic).__name__} from tokenizer.json"
                tok = generic
        weights = self._weight_identity(model_id, revision, hf_token, hash_weights)
        self.attach(model, tok, model_id=model_id, revision=revision, dtype=dtype, device=device, weights=weights)
        return self.session_info()

    def attach(self, model: nn.Module, tokenizer: Any, model_id: str = "custom", revision: str | None = None,
               dtype: str | None = None, device: str | None = None, weights: dict[str, Any] | None = None) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.model_id = model_id
        self.revision = revision
        p = next(model.parameters())
        self.dtype = dtype or str(p.dtype).replace("torch.", "")
        self.device = device or str(p.device)
        self.run_id = "run_" + uuid.uuid4().hex[:12]
        self.opened_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._struct = self._resolve_structure(model)
        self._weights = weights or {"mode": "none", "files": [], "weights_sha256": None}
        self._identity_cache = None
        self._facets = self._probe_facets()

    def close(self) -> dict[str, Any]:
        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        info = {"model_id": self.model_id, "run_id": self.run_id}
        self.model = None
        self.tok = None
        self._struct = {}
        self._facets = {}
        self._identity_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()
            info.update(gpu_bytes_before=before, gpu_bytes_after=after, gpu_freed_bytes=before - after,
                        cleanup=("complete" if after < 64 << 20 else "incomplete"))
        else:
            info.update(cleanup="complete")
        self.model_id = self.run_id = None
        return info

    def require(self) -> None:
        if self.model is None:
            raise SessionError("no_model", "No model is open.", "Call rellm_open_model first.")

    # ------------------------------------------------------------- identity
    @staticmethod
    def _weight_identity(model_id: str, revision: str | None, hf_token: str | None, mode: str) -> dict[str, Any]:
        if mode == "none":
            return {"mode": "none", "files": [], "weights_sha256": None}
        try:
            if os.path.isdir(model_id):
                root = model_id
            else:
                from huggingface_hub import snapshot_download
                root = snapshot_download(model_id, revision=revision, token=hf_token, allow_patterns=["*.safetensors", "*.json"])
            files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
            rows = [safetensors_header_digest(os.path.join(root, f)) for f in files]
            if mode == "full":
                for r, f in zip(rows, files):
                    r["sha256"] = full_file_digest(os.path.join(root, f))
            combined = hashlib.sha256("".join(r["header_sha256"] for r in rows).encode()).hexdigest() if rows else None
            full = hashlib.sha256("".join(r["sha256"] for r in rows).encode()).hexdigest() if (mode == "full" and rows) else None
            return {"mode": mode, "root": root, "files": rows, "headers_sha256": combined, "weights_sha256": full,
                    "note": None if rows else "no safetensors files found (pickle/GGUF checkpoints have no header identity here)"}
        except Exception as e:  # noqa: BLE001 - identity is best effort, never blocks analysis
            return {"mode": mode, "files": [], "weights_sha256": None, "note": f"weight identity unavailable: {type(e).__name__}: {e}"}

    def identity(self) -> dict[str, Any]:
        self.require()
        if self._identity_cache is None:
            cfg_json = self.model.config.to_json_string() if hasattr(self.model.config, "to_json_string") else json.dumps(vars(self.model.config), default=str)
            self._identity_cache = {
                "model_id": self.model_id, "revision": self.revision, "architecture": self.s["arch"],
                "dtype": self.dtype, "device_type": "cuda" if str(self.device).startswith("cuda") else str(self.device),
                "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
                "weights_sha256": self._weights.get("weights_sha256"),
                "weights_headers_sha256": self._weights.get("headers_sha256"),
            }
        return dict(self._identity_cache)

    def analysis_profile(self) -> dict[str, Any]:
        import transformers
        return {"rellm": __version__, "hook_semantics": HOOK_SEMANTICS_VERSION, "attn_implementation": "eager",
                "dtype": self.dtype, "norm_freeze": "final-residual-scale", "torch": torch.__version__.split("+")[0],
                "transformers": transformers.__version__}

    def session_info(self) -> dict[str, Any]:
        self.require()
        return {"run_id": self.run_id, "opened_at": self.opened_at, "binding": {"backend": "transformers-hooks", "device": self.device,
                "dtype": self.dtype, "attn_implementation": "eager"}, "identity": self.identity(),
                "weights": {k: v for k, v in self._weights.items() if k != "files"} | {"n_files": len(self._weights.get("files", []))},
                "analysis_profile": self.analysis_profile(), "facet_availability": self._facets, "facts": self.facts()}

    # ------------------------------------------------------------- structure
    @staticmethod
    def _resolve_structure(model: nn.Module) -> dict[str, Any]:
        cfg = getattr(model.config, "text_config", None) or model.config
        _, base = _first_attr(model, ["model", "transformer", "gpt_neox", "base_model"])
        if base is None:
            raise SessionError("unsupported_architecture", f"Cannot find base transformer in {type(model).__name__}")
        name_layers, layers = _first_attr(base, ["layers", "h", "blocks"])
        for inner in ("language_model", "decoder", "text_model"):  # multimodal wrappers (Gemma 3/4, Llama 4 ...)
            if layers is None and hasattr(base, inner):
                base = getattr(base, inner)
                name_layers, layers = _first_attr(base, ["layers", "h", "blocks"])
        if layers is None:
            raise SessionError("unsupported_architecture", "Cannot find decoder layer list (layers/h/blocks).")
        _, final_norm = _first_attr(base, ["norm", "ln_f", "final_layernorm", "final_layer_norm", "final_norm"])
        _, embed = _first_attr(base, ["embed_tokens", "wte", "embed_in", "word_embeddings"])
        _, unembed = _first_attr(model, ["lm_head", "embed_out"])
        if final_norm is None or unembed is None:
            raise SessionError("unsupported_architecture", "Cannot find final norm / unembedding.")
        attn_mods, mlp_mods, o_projs = [], [], []
        for layer in layers:
            _, attn = _first_attr(layer, ["self_attn", "attn", "attention"])
            _, mlp = _first_attr(layer, ["mlp", "feed_forward", "ffn"])
            _, o_proj = _first_attr(attn, ["o_proj", "c_proj", "out_proj", "dense", "wo"])
            attn_mods.append(attn)
            mlp_mods.append(mlp)
            o_projs.append(o_proj)
        n_heads = _cfg(cfg, "num_attention_heads", None) or _cfg(cfg, "n_head", None)
        d_model = _cfg(cfg, "hidden_size", None) or _cfg(cfg, "n_embd", None)
        d_heads = []
        for o in o_projs:  # per-layer head size from the output projection (Gemma 4 global layers use a larger head_dim)
            if o is None:
                d_heads.append(None)
            else:
                in_f = getattr(o, "in_features", None) or (o.weight.shape[0] if type(o).__name__ == "Conv1D" else o.weight.shape[1])
                d_heads.append(in_f // n_heads if n_heads else None)
        d_head = next((d for d in d_heads if d), None) or (d_model // n_heads if (d_model and n_heads) else None)
        first = layers[0]
        gemma_post_norms = hasattr(first, "pre_feedforward_layernorm") and hasattr(first, "post_feedforward_layernorm")
        per_layer_inputs = hasattr(first, "per_layer_input_gate") or hasattr(first, "post_per_layer_input_norm")
        layer_types = list(_cfg(cfg, "layer_types", None) or [])
        return {
            "gemma_post_norms": gemma_post_norms, "per_layer_inputs": per_layer_inputs,
            "layer_types": layer_types, "sliding_window": _cfg(cfg, "sliding_window", None),
            "base": base, "layers": layers, "final_norm": final_norm, "embed": embed, "unembed": unembed,
            "attn": attn_mods, "mlp": mlp_mods, "o_proj": o_projs,
            "n_layers": len(layers), "n_heads": n_heads, "d_model": d_model, "d_head": d_head, "d_head_per_layer": d_heads,
            "n_kv_heads": _cfg(cfg, "num_key_value_heads", n_heads),
            "d_mlp": _cfg(cfg, "intermediate_size", None) or _cfg(cfg, "n_inner", None),
            "vocab": _cfg(cfg, "vocab_size", None),
            "max_pos": _cfg(cfg, "max_position_embeddings", None) or _cfg(cfg, "n_positions", None),
            "rope_theta": _cfg(cfg, "rope_theta", None),
            "softcap": _cfg(cfg, "final_logit_softcapping", None),
            "tied": bool(_cfg(cfg, "tie_word_embeddings", False)),
            "norm_class": type(final_norm).__name__,
            "arch": type(model).__name__,
        }

    @property
    def s(self) -> dict[str, Any]:
        return self._struct

    def facts(self) -> dict[str, Any]:
        self.require()
        s = self.s
        quirks = []
        if s["softcap"]:
            quirks.append(f"final_logit_softcapping={s['softcap']} (logits are tanh-capped; logit lens applies the cap, DLA is pre-cap)")
        if "Gemma" in s["arch"]:
            quirks.append("Gemma: embeddings scaled by sqrt(d_model); RMSNorm uses (1+weight)")
        if s["gemma_post_norms"]:
            quirks.append("attention and MLP outputs pass through a norm before being added to the residual (DLA uses frozen per-position scales)")
        if s["per_layer_inputs"]:
            quirks.append("per-layer input embeddings add a third term to every layer's residual update (reported as 'layer_extra' in DLA)")
        if len({d for d in s["d_head_per_layer"] if d}) > 1:
            quirks.append(f"head size differs by layer: {sorted({d for d in s['d_head_per_layer'] if d})}")
        if s["layer_types"]:
            n_local = sum(t == "sliding_attention" for t in s["layer_types"])
            quirks.append(f"{n_local} sliding-window layers (window {s['sliding_window']}) + {len(s['layer_types']) - n_local} global layers: "
                          "tokens beyond the window are only reachable through the global layers")
        if s["n_kv_heads"] and s["n_heads"] and s["n_kv_heads"] != s["n_heads"]:
            quirks.append(f"grouped-query attention: {s['n_heads']} query heads share {s['n_kv_heads']} KV heads")
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "n_layers": s["n_layers"], "d_model": s["d_model"], "n_heads": s["n_heads"], "n_kv_heads": s["n_kv_heads"],
            "d_head": s["d_head"], "d_mlp": s["d_mlp"], "vocab_size": s["vocab"], "max_position_embeddings": s["max_pos"],
            "rope_theta": s["rope_theta"], "tied_embeddings": s["tied"], "final_norm_class": s["norm_class"],
            "layer_types": s["layer_types"] or None, "sliding_window": s["sliding_window"],
            "parameters": n_params, "parameters_human": f"{n_params/1e9:.2f}B",
            "quirks": quirks,
            "tokenizer": {"class": type(self.tok).__name__, "vocab_size": len(self.tok), "bos": self.tok.bos_token,
                          "eos": self.tok.eos_token, "chat_template": bool(getattr(self.tok, "chat_template", None)),
                          "note": self.tokenizer_note},
        }

    def _probe_facets(self) -> dict[str, Any]:
        """One tiny forward pass to learn which facets this model/backend actually exposes."""
        facets: dict[str, Any] = {}
        try:
            ids = torch.tensor([[self.tok.bos_token_id if self.tok.bos_token_id is not None else 0, 1]], device=self.device)
            res = self.run(ids, components=("resid", "attn_out", "mlp_out", "head_in"), want_attn=True)
            n = self.s["n_layers"]
            facets["residual_stream"] = {"available": res.resid_pre0 is not None and len(res.resid_post) == n}
            facets["attention_weights"] = {"available": len(res.attentions) == n} if len(res.attentions) == n else \
                {"available": False, "reason": f"only {len(res.attentions)}/{n} layers returned attention weights (non-eager kernel?)"}
            facets["attention_out"] = {"available": len(res.attn_out) == n}
            facets["mlp_out"] = {"available": len(res.mlp_out) == n}
            facets["per_head_output"] = {"available": len(res.head_in) == n} if all(self.s["o_proj"]) else \
                {"available": False, "reason": "attention output projection not found for every layer"}
        except Exception as e:  # noqa: BLE001
            facets["probe_error"] = f"{type(e).__name__}: {e}"
        return facets

    @property
    def facets(self) -> dict[str, Any]:
        return dict(self._facets)

    def require_facet(self, name: str) -> None:
        f = self._facets.get(name)
        if f and not f.get("available", True):
            raise SessionError("facet_unavailable", f"facet {name!r} is not observable for this model: {f.get('reason')}",
                               "Open the model with eager attention on a supported architecture.")

    # ------------------------------------------------------------ tokenizing
    def tokenize(self, text: str, add_special_tokens: bool = True) -> tuple[torch.Tensor, list[str]]:
        self.require()
        ids = self.tok(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"].to(self.device)
        if ids.shape[1] == 0:
            raise SessionError("empty_prompt", "prompt tokenizes to zero tokens")
        return ids, [self.tok.decode([int(i)]) for i in ids[0]]

    def token_strings(self, ids: Iterable[int]) -> list[str]:
        return [self.tok.decode([int(i)]) for i in ids]

    def answer_token(self, text: str) -> tuple[int, list[str], str | None]:
        """Resolve an answer string to the first token id; report multi-token answers."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            raise SessionError("bad_answer", f"Answer {text!r} tokenizes to nothing.")
        toks = self.token_strings(ids)
        lim = f"answer {text!r} is {len(ids)} tokens {toks}; only the first token {toks[0]!r} is scored" if len(ids) > 1 else None
        return int(ids[0]), toks, lim

    def apply_chat_template(self, user: str, system: str | None = None) -> str:
        if not getattr(self.tok, "chat_template", None):
            raise SessionError("no_chat_template", "this tokenizer has no chat template; use the raw prompt")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def position(self, position: int, seq: int) -> int:
        pos = position if position >= 0 else seq + position
        if not 0 <= pos < seq:
            raise SessionError("bad_position", f"position {position} outside sequence of {seq} tokens")
        return pos

    # -------------------------------------------------------------- unembed
    def unembed(self, resid: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """Residual-stream vector(s) -> logits exactly as the model's head would (norm + lm_head + softcap)."""
        x = resid
        if apply_norm:
            x = self.s["final_norm"](x)
        logits = self.s["unembed"](x)
        if self.s["softcap"]:
            logits = self.s["softcap"] * torch.tanh(logits / self.s["softcap"])
        return logits.float()

    # ---------------------------------------------------------------- hooks
    @contextlib.contextmanager
    def capture(self, result: ForwardResult, components: Iterable[str] = ("resid",), layers: Iterable[int] | None = None):
        """Register forward hooks that fill `result` during the next forward pass."""
        comps = set(components)
        layer_ids = list(layers) if layers is not None else list(range(self.s["n_layers"]))
        handles = []

        def pre0(mod, args, kwargs):
            result.resid_pre0 = (args[0] if args else kwargs["hidden_states"]).detach()

        if "resid" in comps:
            handles.append(self.s["layers"][0].register_forward_pre_hook(pre0, with_kwargs=True))
            for i in layer_ids:
                handles.append(self.s["layers"][i].register_forward_hook(lambda m, a, o, i=i: result.resid_post.__setitem__(i, _out_tensor(o).detach())))
        if "attn_out" in comps:
            for i in layer_ids:
                handles.append(self.s["attn"][i].register_forward_hook(lambda m, a, o, i=i: result.attn_out.__setitem__(i, _out_tensor(o).detach())))
        if "mlp_out" in comps:
            for i in layer_ids:
                handles.append(self.s["mlp"][i].register_forward_hook(lambda m, a, o, i=i: result.mlp_out.__setitem__(i, _out_tensor(o).detach())))
        if "head_in" in comps:
            for i in layer_ids:
                if self.s["o_proj"][i] is None:
                    raise SessionError("unsupported_architecture", f"layer {i}: attention output projection not found")
                handles.append(self.s["o_proj"][i].register_forward_pre_hook(lambda m, a, i=i: result.head_in.__setitem__(i, a[0].detach())))
        if "attn_weights" in comps:
            def attn_hook(m, a, o, i):
                if isinstance(o, (tuple, list)) and len(o) > 1 and o[1] is not None:
                    result.attentions[i] = o[1].detach()
            for i in layer_ids:
                handles.append(self.s["attn"][i].register_forward_hook(lambda m, a, o, i=i: attn_hook(m, a, o, i)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def patch(self, component: str, layer: int, positions: list[int], source: torch.Tensor | None, head: int | None = None,
              mode: str = "replace"):
        """Patch one component at `positions` during the next forward pass.

        component: resid_post | attn_out | mlp_out | head
        mode: replace (use `source` values) | zero | mean (mean over this run's positions)
        """
        pos = torch.as_tensor(positions, device=self.device)
        d_head = self.s["d_head_per_layer"][layer] or self.s["d_head"]

        def edit(t: torch.Tensor) -> torch.Tensor:
            t = t.clone()
            sl = slice(None) if head is None else slice(head * d_head, (head + 1) * d_head)
            if mode == "replace":
                t[:, pos, sl] = source[:, pos, sl].to(t.dtype)
            elif mode == "zero":
                t[:, pos, sl] = 0
            elif mode == "mean":
                t[:, pos, sl] = t[:, :, sl].mean(dim=1, keepdim=True).to(t.dtype)
            return t

        if component == "head":
            mod = self.s["o_proj"][layer]
            handle = mod.register_forward_pre_hook(lambda m, a: (edit(a[0]),) + tuple(a[1:]))
        else:
            mod = {"resid_post": self.s["layers"], "attn_out": self.s["attn"], "mlp_out": self.s["mlp"]}[component][layer]
            handle = mod.register_forward_hook(lambda m, a, o: _replace_out(o, edit(_out_tensor(o))))
        try:
            yield
        finally:
            handle.remove()

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def run(self, ids: torch.Tensor, components: Iterable[str] = ("resid",), want_attn: bool = False,
            layers: Iterable[int] | None = None) -> ForwardResult:
        self.require()
        res = ForwardResult(logits=None)  # type: ignore[arg-type]
        comps = list(components) + (["attn_weights"] if want_attn else [])
        with self.capture(res, comps, layers):
            kw: dict[str, Any] = {"use_cache": False}
            if want_attn:
                kw["output_attentions"] = True
            try:
                out = self.model(input_ids=ids, **kw)
            except TypeError:  # transformers versions without the output_attentions kwarg; hooks still capture eager weights
                kw.pop("output_attentions", None)
                out = self.model(input_ids=ids, **kw)
        res.logits = out.logits.float()
        if want_attn and not res.attentions:
            atts = getattr(out, "attentions", None)
            if atts:
                res.attentions = {i: a.detach() for i, a in enumerate(atts) if a is not None}
        return res

    @torch.no_grad()
    def logits_only(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, use_cache=False).logits.float()
