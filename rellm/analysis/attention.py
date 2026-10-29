"""What do the heads look at? Attention patterns and a diagnostic head catalog."""
from __future__ import annotations

from typing import Any

import torch

from ..session import Budget, ModelSession, SessionError


def _need_attn(res, session: ModelSession) -> None:
    if not res.attentions:
        raise SessionError("no_attention_weights", "The forward pass returned no attention weights.",
                           "Open the model with eager attention (rellm_open_model does this) on an architecture that returns weights; "
                           "SDPA/flash kernels do not.")


def _entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum() / 0.6931471805599453)


@torch.no_grad()
def attention_patterns(session: ModelSession, prompt: str, layers: list[int] | None = None, heads: list[int] | None = None,
                       query_position: int = -1, top_k: int = 5, add_special_tokens: bool = True) -> dict[str, Any]:
    session.require_facet("attention_weights")
    ids, toks = session.tokenize(prompt, add_special_tokens)
    q = session.position(query_position, ids.shape[1])
    res = session.run(ids, components=(), want_attn=True)
    _need_attn(res, session)
    layer_ids = layers if layers is not None else sorted(res.attentions)
    head_ids = heads if heads is not None else list(range(session.s["n_heads"]))
    out_layers, missing = [], []
    for L in layer_ids:
        if L not in res.attentions:
            missing.append(L)
            continue
        att = res.attentions[L][0]  # (heads, q, k)
        rows = []
        for h in head_ids:
            a = att[h, q, : q + 1].float()
            vals, idx = a.topk(min(top_k, q + 1))
            rows.append({"head": h,
                         "attends_to": [{"position": int(i), "token": toks[int(i)], "weight": round(float(v), 4)} for v, i in zip(vals, idx)],
                         "entropy_bits": round(_entropy(a), 3), "to_self": round(float(a[q]), 4),
                         "to_previous": round(float(a[q - 1]), 4) if q > 0 else None, "to_first_token": round(float(a[0]), 4)})
        out_layers.append({"layer": L, "heads": rows})
    return {"query_position": q, "query_token": toks[q], "tokens": toks, "layers": out_layers,
            "coverage": {"status": "complete" if not missing else "partial", "layers_requested": len(layer_ids), "layers_missing": missing},
            "limitations": ["attention weights show where information is read from, not what is computed with it; a high weight on the "
                            "first token is usually an attention sink, not a signal"]}


@torch.no_grad()
def head_summary(session: ModelSession, prompt: str, add_special_tokens: bool = True, top_k: int = 12) -> dict[str, Any]:
    """Per-head aggregate stats over the whole prompt (cheap first look)."""
    session.require_facet("attention_weights")
    ids, toks = session.tokenize(prompt, add_special_tokens)
    res = session.run(ids, components=(), want_attn=True)
    _need_attn(res, session)
    seq = ids.shape[1]
    rows = []
    for L, att in sorted(res.attentions.items()):
        att = att[0].float()  # heads, q, k
        idx = torch.arange(seq, device=att.device)
        for h in range(att.shape[0]):
            a = att[h]
            prev = a[idx[1:], idx[:-1]].mean() if seq > 1 else torch.tensor(0.0)
            diag = a[idx, idx].mean()
            first = a[:, 0].mean()
            ent = sum(_entropy(a[i, : i + 1]) for i in range(seq)) / seq
            rows.append({"layer": L, "head": h, "prev_token": round(float(prev), 4), "self": round(float(diag), 4),
                         "first_token": round(float(first), 4), "mean_entropy_bits": round(ent, 3)})
    by = lambda key: sorted(rows, key=lambda r: -r[key])[:top_k]
    return {"n_tokens": seq, "heads_total": len(rows), "top_previous_token_heads": by("prev_token"), "top_sink_heads": by("first_token"),
            "lowest_entropy_heads": sorted(rows, key=lambda r: r["mean_entropy_bits"])[:top_k], "all_heads": rows,
            "limitations": ["statistics over one prompt; a head's role on other text can differ"]}


@torch.no_grad()
def head_catalog(session: ModelSession, seq_len: int = 40, n_samples: int = 4, seed: int = 0, threshold: float = 0.3,
                 max_seconds: float | None = None) -> dict[str, Any]:
    """Classic diagnostic: random token sequence repeated twice.

    induction score  = attention from token i (second half) to the token *after* the previous occurrence (i-L+1)
    duplicate score  = attention to the previous occurrence itself (i-L)
    prev-token score = attention to i-1 (all positions)
    """
    session.require_facet("attention_weights")
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = session.s["vocab"]
    lo, hi = min(1000, max(1, vocab // 4)), vocab - 1
    bos = session.tok.bos_token_id
    n_layers, n_heads = session.s["n_layers"], session.s["n_heads"]
    ind = torch.zeros(n_layers, n_heads)
    dup = torch.zeros(n_layers, n_heads)
    prev = torch.zeros(n_layers, n_heads)
    budget = Budget(max_seconds, None)
    done = 0
    for _ in range(n_samples):
        rand = torch.randint(lo, hi, (seq_len,), generator=g)
        seq = torch.cat([rand, rand])
        if bos is not None:
            seq = torch.cat([torch.tensor([bos]), seq])
        off = 1 if bos is not None else 0
        ids = seq.unsqueeze(0).to(session.device)
        res = session.run(ids, components=(), want_attn=True)
        _need_attn(res, session)
        T = ids.shape[1]
        for L, att in res.attentions.items():
            a = att[0].float()
            qs = torch.arange(off + seq_len, T, device=a.device)
            ind[L] += a[:, qs, qs - seq_len + 1].mean(dim=1).cpu()
            dup[L] += a[:, qs, qs - seq_len].mean(dim=1).cpu()
            qa = torch.arange(1, T, device=a.device)
            prev[L] += a[:, qa, qa - 1].mean(dim=1).cpu()
        done += 1
        if not budget.tick():
            break
    ind /= done; dup /= done; prev /= done
    rows = [{"layer": L, "head": h, "induction": round(float(ind[L, h]), 3), "duplicate_token": round(float(dup[L, h]), 3),
             "previous_token": round(float(prev[L, h]), 3)} for L in range(n_layers) for h in range(n_heads)]
    top = lambda key: sorted(rows, key=lambda r: -r[key])[:15]
    return {"diagnostic": {"seq_len": seq_len, "n_samples_requested": n_samples, "n_samples_done": done, "seed": seed,
                           "token_range": [lo, hi], "bos_prepended": bos is not None},
            "induction_heads": [r for r in top("induction") if r["induction"] >= threshold],
            "duplicate_token_heads": [r for r in top("duplicate_token") if r["duplicate_token"] >= threshold],
            "previous_token_heads": [r for r in top("previous_token") if r["previous_token"] >= threshold],
            "top_by_induction": top("induction"), "all_heads": rows,
            "coverage": {"status": "complete" if done == n_samples else "partial", "samples_done": done, "samples_requested": n_samples,
                         "budget_exhausted": budget.exhausted}, "timing": {"elapsed_s": budget.elapsed},
            "limitations": ["scores come from random-token sequences; a head can act as an induction head on natural text without passing "
                            "this test and vice versa", f"threshold {threshold} is a convention, not a proof of role"]}
