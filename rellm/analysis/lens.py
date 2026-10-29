"""Where does the decision form? Next-token behaviour and the logit lens."""
from __future__ import annotations

from typing import Any

import torch

from ..session import ModelSession

LN2 = 0.6931471805599453


def topk(session: ModelSession, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(min(k, logits.shape[-1]))
    return [{"token": session.tok.decode([int(i)]), "token_id": int(i), "prob": round(float(p), 6), "logit": round(float(logits[i]), 4)}
            for p, i in zip(vals, idx)]


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def entropy_bits(logits: torch.Tensor) -> float:
    lp = torch.log_softmax(logits, -1)
    return round(float(-(lp.exp() * lp).sum()) / LN2, 4)


@torch.no_grad()
def next_token(session: ModelSession, prompt: str, top_k: int = 10, add_special_tokens: bool = True,
               track_tokens: list[str] | None = None) -> dict[str, Any]:
    ids, toks = session.tokenize(prompt, add_special_tokens)
    logits = session.logits_only(ids)[0, -1]
    out: dict[str, Any] = {"tokens": toks, "n_tokens": len(toks), "top": topk(session, logits, top_k), "entropy_bits": entropy_bits(logits),
                           "limitations": []}
    if track_tokens:
        lp = torch.log_softmax(logits, -1)
        out["tracked"] = {}
        for t in track_tokens:
            tid, _, lim = session.answer_token(t)
            out["tracked"][t] = {"token_id": tid, "rank": rank_of(logits, tid), "prob": round(float(lp[tid].exp()), 6)}
            if lim:
                out["limitations"].append(lim)
    return out


@torch.no_grad()
def generate(session: ModelSession, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0,
             add_special_tokens: bool = True) -> dict[str, Any]:
    ids, toks = session.tokenize(prompt, add_special_tokens)
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                          "pad_token_id": session.tok.pad_token_id if session.tok.pad_token_id is not None else session.tok.eos_token_id}
    if temperature > 0:
        kw["temperature"] = temperature
    out = session.model.generate(ids, **kw)
    new = out[0, ids.shape[1]:]
    return {"prompt_tokens": len(toks), "generated_tokens": session.token_strings(new.tolist()),
            "text": session.tok.decode(new, skip_special_tokens=True), "n_generated": int(new.numel()),
            "sampling": {"temperature": temperature, "greedy": temperature == 0},
            "limitations": ["a sampled continuation is one draw; greedy is deterministic for this dtype/device only"]}


@torch.no_grad()
def logit_lens(session: ModelSession, prompt: str, position: int = -1, top_k: int = 5, add_special_tokens: bool = True,
               track_tokens: list[str] | None = None) -> dict[str, Any]:
    ids, toks = session.tokenize(prompt, add_special_tokens)
    pos = session.position(position, ids.shape[1])
    res = session.run(ids, components=("resid",))
    final_logits = res.logits[0, pos]
    final_top = int(final_logits.argmax())
    final_lp = torch.log_softmax(final_logits, -1)
    limitations: list[str] = []
    tracked: list[tuple[str, int]] = []
    for t in track_tokens or []:
        tid, _, lim = session.answer_token(t)
        tracked.append((t, tid))
        if lim:
            limitations.append(lim)

    states = [("embed", res.resid_pre0)] + [(f"L{i}", res.resid_post[i]) for i in range(session.s["n_layers"])]
    rows = []
    for name, state in states:
        logits = session.unembed(state[0, pos])
        lp = torch.log_softmax(logits, -1)
        kl = float((final_lp.exp() * (final_lp - lp)).sum())
        row: dict[str, Any] = {"layer": name, "top": topk(session, logits, top_k), "final_answer_rank": rank_of(logits, final_top),
                               "final_answer_prob": round(float(lp[final_top].exp()), 6), "kl_to_final_bits": round(kl / LN2, 4),
                               "entropy_bits": entropy_bits(logits)}
        if tracked:
            row["tracked"] = {t: {"rank": rank_of(logits, tid), "prob": round(float(lp[tid].exp()), 6)} for t, tid in tracked}
        rows.append(row)

    decision = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]["final_answer_rank"] == 1:
            decision = rows[i]["layer"]
        else:
            break
    last_err = float((session.unembed(res.resid_post[session.s["n_layers"] - 1][0, pos]) - final_logits).abs().max())
    if last_err > 1e-2:
        limitations.append(f"lens at the last layer deviates from model logits by max {last_err:.3g}; the unembedding path differs (custom head?)")
    limitations.append("logit lens reads the residual stream through the final norm+unembed; intermediate layers were never trained for that "
                       "readout, so early-layer predictions are suggestive, not what the model 'thinks'")
    return {"position": pos, "token_at_position": toks[pos], "tokens": toks,
            "final_top": {"token": session.tok.decode([final_top]), "token_id": final_top, "prob": round(float(final_lp[final_top].exp()), 6)},
            "decision_layer": decision, "lens_last_layer_max_abs_err": round(last_err, 5), "layers": rows, "limitations": limitations}
