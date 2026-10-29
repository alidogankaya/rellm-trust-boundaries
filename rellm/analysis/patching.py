"""Causal tools: which layer/position/head *matters* for the decision."""
from __future__ import annotations

import contextlib
from typing import Any

import torch

from ..session import Budget, ModelSession, SessionError

MAX_PASSES = 4096
COMPONENTS = ("resid_post", "attn_out", "mlp_out", "head")


def metric_factory(session: ModelSession, answer: str, counter_answer: str | None, position: int):
    ans_id, ans_toks, lim_a = session.answer_token(answer)
    limitations = [l for l in [lim_a] if l]
    if counter_answer is not None:
        cnt_id, cnt_toks, lim_c = session.answer_token(counter_answer)
        if lim_c:
            limitations.append(lim_c)
        if cnt_id == ans_id:
            raise SessionError("same_token", f"answer and counter_answer tokenize to the same first token {ans_toks[0]!r}")

        def metric(logits: torch.Tensor) -> float:  # logit difference, the standard patching metric
            row = logits[0, position]
            return float(row[ans_id] - row[cnt_id])
        desc = {"kind": "logit_diff", "answer": ans_toks[0], "counter_answer": cnt_toks[0], "answer_id": ans_id, "counter_id": cnt_id}
    else:
        def metric(logits: torch.Tensor) -> float:
            return float(torch.log_softmax(logits[0, position], -1)[ans_id])
        desc = {"kind": "logprob", "answer": ans_toks[0], "answer_id": ans_id}
    return metric, desc, limitations


@torch.no_grad()
def activation_patching(session: ModelSession, clean_prompt: str, corrupted_prompt: str, answer: str,
                        counter_answer: str | None = None, component: str = "resid_post",
                        layers: list[int] | None = None, positions: list[int] | None = None,
                        heads: list[int] | None = None, add_special_tokens: bool = True,
                        max_seconds: float | None = None, max_passes: int | None = None) -> dict[str, Any]:
    """Run the corrupted prompt, restore one clean activation, measure recovery.

    normalized = (patched - corrupted) / (clean - corrupted):  1.0 = fully restores the clean answer.
    The grid is computed within a time/pass budget; an exhausted budget yields partial coverage, never a silent gap.
    """
    if component not in COMPONENTS:
        raise SessionError("bad_component", f"component must be one of {COMPONENTS}, got {component!r}")
    if component == "head":
        session.require_facet("per_head_output")
    ids_c, toks_c = session.tokenize(clean_prompt, add_special_tokens)
    ids_x, toks_x = session.tokenize(corrupted_prompt, add_special_tokens)
    if ids_c.shape[1] != ids_x.shape[1]:
        raise SessionError("length_mismatch", f"clean has {ids_c.shape[1]} tokens, corrupted has {ids_x.shape[1]}",
                           "Activation patching needs position-aligned prompts; change the corrupted prompt so both tokenize to the same length.",
                           clean_tokens=toks_c, corrupted_tokens=toks_x)
    seq = ids_c.shape[1]
    metric, mdesc, limitations = metric_factory(session, answer, counter_answer, seq - 1)

    cap = {"resid_post": "resid", "attn_out": "attn_out", "mlp_out": "mlp_out", "head": "head_in"}[component]
    clean = session.run(ids_c, components=(cap,))
    store = {"resid_post": clean.resid_post, "attn_out": clean.attn_out, "mlp_out": clean.mlp_out, "head": clean.head_in}[component]
    m_clean = metric(clean.logits)
    m_corr = metric(session.logits_only(ids_x))
    denom = m_clean - m_corr
    if abs(denom) < 1e-6:
        limitations.append("clean and corrupted prompts give the same metric; normalized scores are undefined (reported raw)")

    layer_ids = layers if layers is not None else list(range(session.s["n_layers"]))
    pos_ids = positions if positions is not None else ([seq - 1] if component == "head" else list(range(seq)))
    pos_ids = [session.position(p, seq) for p in pos_ids]
    head_ids: list[int | None] = heads if heads is not None else (list(range(session.s["n_heads"])) if component == "head" else [None])
    for L in layer_ids:
        if not 0 <= L < session.s["n_layers"]:
            raise SessionError("bad_layer", f"layer {L} outside 0..{session.s['n_layers'] - 1}")
    n_cells = len(layer_ids) * len(pos_ids) * len(head_ids)
    if n_cells > MAX_PASSES:
        raise SessionError("too_many_passes", f"{n_cells} forward passes requested (max {MAX_PASSES})",
                           "Narrow layers/positions/heads, e.g. patch heads only at the last position.")

    budget = Budget(max_seconds, max_passes)
    rows, stopped = [], False
    for L in layer_ids:
        for h in head_ids:
            for p in pos_ids:
                with session.patch(component, L, [p], store[L], head=h):
                    m = metric(session.logits_only(ids_x))
                row: dict[str, Any] = {"layer": L, "position": p, "token": toks_x[p], "metric": round(m, 4),
                                       "normalized": round((m - m_corr) / denom, 4) if abs(denom) >= 1e-6 else None}
                if h is not None:
                    row["head"] = h
                rows.append(row)
                if not budget.tick():
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break
    key = lambda r: -(r["normalized"] if r["normalized"] is not None else r["metric"])
    top = sorted(rows, key=key)[:15]
    computed = {(r["layer"], r.get("head"), r["position"]) for r in rows}
    skipped = [{"layer": L, "head": h, "position": p} for L in layer_ids for h in head_ids for p in pos_ids if (L, h, p) not in computed]
    limitations.append("patching one component restores that component's *clean* value in the corrupted run; a large effect means the "
                       "information flows through there, not that the component computes it alone")
    if skipped:
        limitations.append(f"{len(skipped)} cells not computed (budget {budget.exhausted}); absent cells are unknown, not zero")
    return {"component": component, "metric": mdesc,
            "clean": {"tokens": toks_c, "metric": round(m_clean, 4)}, "corrupted": {"tokens": toks_x, "metric": round(m_corr, 4)},
            "n_forward_passes": len(rows), "top_effects": top, "grid": rows,
            "coverage": {"status": "complete" if not skipped else "partial", "cells_requested": n_cells, "cells_computed": len(rows),
                         "cells_skipped": skipped[:64], "budget_exhausted": budget.exhausted},
            "timing": {"elapsed_s": budget.elapsed, "passes": len(rows) + 2}, "limitations": limitations}


@torch.no_grad()
def ablate_heads(session: ModelSession, prompt: str, heads: list[list[int]], answer: str, counter_answer: str | None = None,
                 mode: str = "zero", top_k: int = 5, add_special_tokens: bool = True) -> dict[str, Any]:
    """Zero- or mean-ablate a set of (layer, head) pairs and report the damage to the answer."""
    if mode not in {"zero", "mean"}:
        raise SessionError("bad_mode", "mode must be zero or mean")
    session.require_facet("per_head_output")
    for pair in heads:
        if len(pair) != 2 or not (0 <= pair[0] < session.s["n_layers"]) or not (0 <= pair[1] < session.s["n_heads"]):
            raise SessionError("bad_head", f"head {pair} must be [layer, head] within the model")
    ids, toks = session.tokenize(prompt, add_special_tokens)
    seq = ids.shape[1]
    metric, mdesc, limitations = metric_factory(session, answer, counter_answer, seq - 1)
    base_logits = session.logits_only(ids)
    m_base = metric(base_logits)

    with contextlib.ExitStack() as stack:
        for L, h in heads:
            stack.enter_context(session.patch("head", L, list(range(seq)), None, head=h, mode=mode))
        abl_logits = session.logits_only(ids)
    m_abl = metric(abl_logits)

    def top(logits):
        probs = torch.softmax(logits[0, -1], -1)
        v, i = probs.topk(top_k)
        return [{"token": session.tok.decode([int(t)]), "prob": round(float(p), 5)} for p, t in zip(v, i)]

    singles = []
    for L, h in heads:
        with session.patch("head", L, list(range(seq)), None, head=h, mode=mode):
            singles.append({"layer": L, "head": h, "metric": round(metric(session.logits_only(ids)), 4)})
    if mode == "mean":
        limitations.append("mean ablation uses the mean over this prompt's positions, not a corpus mean")
    limitations.append("ablation shows necessity under this intervention only; compensation by other heads and off-distribution "
                       "activations can hide or exaggerate roles")
    return {"metric": mdesc, "mode": mode, "tokens": toks, "ablated_heads": heads,
            "baseline": {"metric": round(m_base, 4), "top": top(base_logits)},
            "ablated": {"metric": round(m_abl, 4), "top": top(abl_logits), "delta": round(m_abl - m_base, 4)},
            "single_head_ablations": singles, "limitations": limitations}
