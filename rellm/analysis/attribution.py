"""Direct logit attribution (DLA).

The residual stream at the final position is a sum: embedding + Σ heads + Σ MLPs.
The unembedding is linear once the final norm's scale is frozen to the value it
takes on the full residual, so each component's contribution to a logit
difference can be read off exactly (up to that frozen-scale approximation).
"""
from __future__ import annotations

from typing import Any

import torch

from ..session import ModelSession, SessionError


@torch.no_grad()
def effective_gamma(norm, x: torch.Tensor) -> torch.Tensor:
    """The per-dimension scale a (RMS)norm module applies, found empirically: Gemma 2/3 use (1+w), Gemma 4 and Llama use w."""
    w = getattr(norm, "weight", None)
    if w is None:
        return torch.ones_like(x.float())
    w = w.float()
    eps = getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", None) or 1e-6
    v = torch.randn_like(x.float()) + 1.0
    base = v / torch.sqrt((v * v).mean() + eps)
    out = norm(v.to(x.dtype)).float()
    for cand in (w, 1.0 + w):
        if torch.allclose(out, base * cand, atol=1e-3, rtol=1e-3):
            return cand
    return w


def _norm_projection(session: ModelSession, final_resid: torch.Tensor):
    """Return (f, bias_term, kind) where f(vec) = frozen-scale normalised vec, matching the final norm."""
    norm = session.s["final_norm"]
    eps = getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", None) or 1e-6
    x = final_resid.float()
    w = effective_gamma(norm, final_resid)
    bias = getattr(norm, "bias", None)
    if bias is not None:  # LayerNorm: centre, then scale by 1/std
        scale = 1.0 / torch.sqrt(x.var(unbiased=False) + eps)
        return (lambda v: (v.float() - v.float().mean()) * scale * w), bias.float(), "layernorm"
    scale = 1.0 / torch.sqrt((x * x).mean() + eps)   # RMSNorm
    return (lambda v: v.float() * scale * w), None, "rmsnorm"


@torch.no_grad()
def direct_logit_attribution(session: ModelSession, prompt: str, answer: str, counter_answer: str | None = None,
                             position: int = -1, top_k: int = 20, add_special_tokens: bool = True) -> dict[str, Any]:
    session.require_facet("per_head_output")
    ids, toks = session.tokenize(prompt, add_special_tokens)
    pos = session.position(position, ids.shape[1])
    res = session.run(ids, components=("resid", "head_in", "mlp_out", "attn_out"))
    n_layers, n_heads = session.s["n_layers"], session.s["n_heads"]

    ans_id, ans_toks, lim_a = session.answer_token(answer)
    limitations = [l for l in [lim_a] if l]
    W_U = session.s["unembed"].weight.float()  # (vocab, d_model)
    if counter_answer is not None:
        cnt_id, cnt_toks, lim_c = session.answer_token(counter_answer)
        if lim_c:
            limitations.append(lim_c)
        if cnt_id == ans_id:
            raise SessionError("same_token", "answer and counter_answer tokenize to the same first token")
        direction = W_U[ans_id] - W_U[cnt_id]
        target = {"kind": "logit_diff", "answer": ans_toks[0], "counter_answer": cnt_toks[0]}
    else:
        direction = W_U[ans_id]
        target = {"kind": "raw_logit", "answer": ans_toks[0]}
        limitations.append("attribution to a single raw logit includes components that raise all logits; prefer a counter_answer")

    final_resid = res.resid_post[n_layers - 1][0, pos]
    proj, norm_bias, norm_kind = _norm_projection(session, final_resid)
    dot = lambda v: float(proj(v) @ direction)

    components = [{"component": "embed", "layer": None, "head": None, "contribution": dot(res.resid_pre0[0, pos])}]
    recon = res.resid_pre0[0, pos].float().clone()
    gemma = session.s.get("gemma_post_norms", False)
    for L in range(n_layers):
        layer = session.s["layers"][L]
        o_proj = session.s["o_proj"][L]
        W_O = o_proj.weight.float()  # (d_model, n_heads*d_head)
        if type(o_proj).__name__ == "Conv1D":  # GPT-2 style stores (in, out)
            W_O = W_O.T
        hin = res.head_in[L][0, pos].float()
        d_head = session.s["d_head_per_layer"][L] or session.s["d_head"]
        head_vecs = [W_O[:, h * d_head:(h + 1) * d_head] @ hin[h * d_head:(h + 1) * d_head] for h in range(n_heads)]
        bias = o_proj.bias.float() if getattr(o_proj, "bias", None) is not None else None
        if gemma:  # residual += post_attention_layernorm(attn_out): freeze the norm's scale on the full attention output
            attn_full = res.attn_out[L][0, pos].float()
            pn = layer.post_attention_layernorm
            eps = getattr(pn, "eps", None) or getattr(pn, "variance_epsilon", None) or 1e-6
            scale = effective_gamma(pn, attn_full) / torch.sqrt((attn_full * attn_full).mean() + eps)
            head_vecs = [v * scale for v in head_vecs]
            bias = bias * scale if bias is not None else None
        for h, vec in enumerate(head_vecs):
            recon += vec
            components.append({"component": "head", "layer": L, "head": h, "contribution": dot(vec)})
        if bias is not None:
            recon += bias
            components.append({"component": "attn_bias", "layer": L, "head": None, "contribution": dot(bias)})
        mlp = res.mlp_out[L][0, pos].float()
        if gemma:
            with torch.no_grad():
                mlp = layer.post_feedforward_layernorm(res.mlp_out[L][0, pos]).float()
        recon += mlp
        components.append({"component": "mlp", "layer": L, "head": None, "contribution": dot(mlp)})
        if gemma or session.s.get("per_layer_inputs"):  # whatever else the layer added (per-layer input embeddings, residual quirks)
            layer_in = res.resid_pre0[0, pos] if L == 0 else res.resid_post[L - 1][0, pos]
            extra = res.resid_post[L][0, pos].float() - layer_in.float() - sum(head_vecs) - (bias if bias is not None else 0) - mlp
            if float(extra.abs().max()) > 1e-4 * (float(res.resid_post[L][0, pos].float().abs().max()) + 1e-9):
                recon += extra
                components.append({"component": "layer_extra", "layer": L, "head": None, "contribution": dot(extra)})
    recon_err = float((recon - final_resid.float()).abs().max())
    rel_err = recon_err / (float(final_resid.float().abs().max()) + 1e-9)
    coverage = {"status": "complete", "reconstruction_rel_err": round(rel_err, 5)}
    if rel_err > 0.05:
        coverage["status"] = "partial"
        limitations.append(f"component sum does not reconstruct the final residual (rel err {rel_err:.3f}); the architecture has extra "
                           "residual terms (parallel blocks, scaling, biases) not covered here")
    actual = session.unembed(final_resid)
    actual_target = float(actual[ans_id] - actual[cnt_id]) if counter_answer is not None else float(actual[ans_id])
    total = sum(c["contribution"] for c in components)
    if norm_bias is not None:
        bias_contrib = float(norm_bias @ direction)
        components.append({"component": "final_norm_bias", "layer": None, "head": None, "contribution": bias_contrib})
        total += bias_contrib
    if session.s["softcap"]:
        limitations.append("final logit softcapping makes the head nonlinear; attributions are to the pre-cap logit")
    for c in components:
        c["contribution"] = round(c["contribution"], 4)
    by_layer = [{"layer": L,
                 "attention": round(sum(c["contribution"] for c in components if c["layer"] == L and c["component"] in ("head", "attn_bias")), 4),
                 "mlp": round(sum(c["contribution"] for c in components if c["layer"] == L and c["component"] == "mlp"), 4),
                 "extra": round(sum(c["contribution"] for c in components if c["layer"] == L and c["component"] == "layer_extra"), 4)}
                for L in range(n_layers)]
    ranked = sorted(components, key=lambda c: -abs(c["contribution"]))
    limitations.append("DLA measures the *direct* path to the logit; a head that only feeds later heads has ~0 direct attribution yet can "
                       "be essential (check with patching)")
    return {"position": pos, "token_at_position": toks[pos], "target": target, "norm_kind": norm_kind,
            "actual_target_value": round(actual_target, 4), "sum_of_attributions": round(total, 4),
            "reconstruction_max_abs_err": round(recon_err, 5),
            "top_positive": [c for c in ranked if c["contribution"] > 0][:top_k],
            "top_negative": [c for c in ranked if c["contribution"] < 0][:top_k],
            "by_layer": by_layer, "components": components, "coverage": coverage, "limitations": limitations}
