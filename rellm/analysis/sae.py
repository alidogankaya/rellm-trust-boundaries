"""Human-readable features: sparse autoencoder readout at one residual position.

Requires `sae_lens` and a published SAE whose hook point is a residual stream
(e.g. Gemma Scope `gemma-scope-2b-pt-res` / `layer_12/width_16k/canonical`).
Feature explanations are not downloaded; the result gives Neuronpedia links so
the agent can look them up (that text is untrusted target content).
"""
from __future__ import annotations

import re
from typing import Any

import torch

from ..session import ModelSession, SessionError

_cache: dict[tuple[str, str], Any] = {}


def _load_sae(release: str, sae_id: str, device: str):
    key = (release, sae_id)
    if key in _cache:
        return _cache[key]
    try:
        from sae_lens import SAE
    except ImportError as e:
        raise SessionError("sae_lens_missing", "sae_lens is not installed", "pip install sae-lens") from e
    loaded = SAE.from_pretrained(release, sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    _cache[key] = sae
    return sae


@torch.no_grad()
def sae_features(session: ModelSession, prompt: str, release: str, sae_id: str, position: int = -1, top_k: int = 20,
                 add_special_tokens: bool = True) -> dict[str, Any]:
    sae = _load_sae(release, sae_id, session.device)
    cfg = sae.cfg
    hook = getattr(cfg, "hook_name", None) or getattr(getattr(cfg, "metadata", None), "hook_name", None) or ""
    m = re.search(r"blocks\.(\d+)\.hook_(resid_pre|resid_post|resid_mid)", hook)
    if not m:
        raise SessionError("unsupported_hook", f"SAE hook point {hook!r} is not a residual stream hook; only resid_pre/resid_post are supported")
    layer, kind = int(m.group(1)), m.group(2)
    if layer >= session.s["n_layers"]:
        raise SessionError("bad_layer", f"SAE is for layer {layer} but the model has {session.s['n_layers']} layers")
    ids, toks = session.tokenize(prompt, add_special_tokens)
    pos = session.position(position, ids.shape[1])
    res = session.run(ids, components=("resid",))
    if kind == "resid_pre":
        x = res.resid_pre0[0, pos] if layer == 0 else res.resid_post[layer - 1][0, pos]
    else:
        x = res.resid_post[layer][0, pos]
    x = x.float().to(sae.device)
    d_in = getattr(cfg, "d_in", None)
    if d_in and x.shape[-1] != d_in:
        raise SessionError("dim_mismatch", f"SAE expects d_in={d_in}, model residual is {x.shape[-1]}; wrong model for this SAE")
    with torch.no_grad():
        acts = sae.encode(x.unsqueeze(0))[0]
        recon = sae.decode(acts.unsqueeze(0))[0]
    frac_var = 1 - float(((recon - x) ** 2).sum() / (((x - x.mean()) ** 2).sum() + 1e-9))
    vals, idx = acts.topk(min(top_k, acts.numel()))
    model_slug = (session.model_id or "").split("/")[-1].lower()
    feats = [{"feature": int(i), "activation": round(float(v), 4),
              "neuronpedia": f"https://www.neuronpedia.org/{model_slug}/{layer}-{release}/{int(i)}"}
             for v, i in zip(vals, idx) if float(v) > 0]
    return {"layer": layer, "hook": hook, "position": pos, "token_at_position": toks[pos], "release": release, "sae_id": sae_id,
            "n_active": int((acts > 0).sum()), "fraction_variance_explained": round(frac_var, 4), "top_features": feats,
            "coverage": {"status": "complete" if frac_var > 0.5 else "partial", "fraction_variance_explained": round(frac_var, 4)},
            "limitations": ["SAE features are a learned, lossy basis; variance explained says how much of the residual the SAE accounts for",
                            "Neuronpedia URL is a best-effort guess at the dashboard path; explanations there are model-generated text (untrusted)"]}
