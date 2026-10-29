"""Scoring and interventions shared by the reverse phase and every pruning arm.

Batching groups prompts by token length (no padding), so last-position logits and position ids are exact.
Interventions are *removals* (zero), the same operation a physically pruned model performs.
"""
from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from typing import Any, Iterable

import torch

from .tasks import Example


def encode(tok, text: str, special: bool = True) -> list[int]:
    return tok(text, add_special_tokens=special)["input_ids"]


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def length_batches(seqs: list[list[int]], batch_size: int) -> Iterable[tuple[list[int], torch.Tensor]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[len(s)].append(i)
    for _, idx in sorted(groups.items()):
        for k in range(0, len(idx), batch_size):
            part = idx[k: k + batch_size]
            yield part, torch.tensor([seqs[i] for i in part])


class Scorer:
    """Pre-tokenised scoring of a list of examples. margin = answer score - best other candidate score."""

    def __init__(self, tok, examples: list[Example], batch_size: int = 32):
        self.tok, self.examples, self.bs = tok, examples, batch_size
        self.prompt_ids = [encode(tok, e.prompt) for e in examples]
        self.cand_ids = [[encode(tok, c, special=False) for c in e.candidates] for e in examples]
        self.answer_idx = [e.candidates.index(e.answer) for e in examples]
        firsts_unique = all(len({c[0] for c in cs}) == len(cs) and all(len(c) > 0 for c in cs) for cs in self.cand_ids)
        self.mode = "first_token" if firsts_unique else "sequence_logprob"
        self.first_ids = [torch.tensor([c[0] for c in cs]) for cs in self.cand_ids] if firsts_unique else None

    @torch.no_grad()
    def candidate_scores(self, model) -> list[torch.Tensor]:
        dev = device_of(model)
        out: list[torch.Tensor | None] = [None] * len(self.examples)
        if self.mode == "first_token":
            for part, ids in length_batches(self.prompt_ids, self.bs):
                logits = model(input_ids=ids.to(dev), use_cache=False).logits[:, -1].float()
                for row, i in enumerate(part):
                    out[i] = logits[row, self.first_ids[i].to(dev)].cpu()
            return out  # type: ignore[return-value]
        seqs, owner = [], []
        for i, (p, cs) in enumerate(zip(self.prompt_ids, self.cand_ids)):
            for j, c in enumerate(cs):
                seqs.append(p + c)
                owner.append((i, j, len(p), len(c)))
        scores = [torch.zeros(len(cs)) for cs in self.cand_ids]
        for part, ids in length_batches(seqs, self.bs):
            lp = torch.log_softmax(model(input_ids=ids.to(dev), use_cache=False).logits.float(), -1).cpu()
            for row, k in enumerate(part):
                i, j, lp_len, lc = owner[k]
                tgt = ids[row, lp_len: lp_len + lc]
                scores[i][j] = lp[row, lp_len - 1: lp_len - 1 + lc].gather(-1, tgt.unsqueeze(-1)).sum()
        return scores

    def score(self, model, keep_per_example: bool = False) -> dict[str, Any]:
        cs = self.candidate_scores(model)
        margins, correct = [], []
        for s, a in zip(cs, self.answer_idx):
            other = s.clone()
            other[a] = -math.inf
            margins.append(float(s[a] - other.max()))
            correct.append(int(s.argmax()) == a)
        m = torch.tensor(margins)
        r = {"acc": round(sum(correct) / len(correct), 4), "margin_mean": round(float(m.mean()), 4), "margin_median": round(float(m.median()), 4),
             "margin_min": round(float(m.min()), 4), "n": len(correct), "mode": self.mode}
        if keep_per_example:
            r["margins"] = margins
            r["correct"] = correct
        return r


@torch.no_grad()
def lm_nll(model, tok, texts: list[str], batch_size: int = 8) -> dict[str, float]:
    dev = device_of(model)
    tot, n = 0.0, 0
    for _, ids in length_batches([encode(tok, t) for t in texts], batch_size):
        ids = ids.to(dev)
        lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits.float()[:, :-1], -1)
        nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        tot += float(nll.sum())
        n += nll.numel()
    return {"nll": round(tot / max(n, 1), 4), "ppl": round(math.exp(min(tot / max(n, 1), 50)), 3), "tokens": n}


@torch.no_grad()
def latency_ms(model, tok, prompts: list[str], reps: int = 10, batch_size: int = 32) -> float:
    dev = device_of(model)
    batches = [ids.to(dev) for _, ids in length_batches([encode(tok, p) for p in prompts], batch_size)]
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for b in batches:
        model(input_ids=b, use_cache=False)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        for b in batches:
            model(input_ids=b, use_cache=False)
    sync()
    return round((time.perf_counter() - t0) / reps * 1000, 3)


def param_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    body = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n and "lm_head" not in n)
    return {"params": total, "params_body": body}


# ----------------------------------------------------------------------------------------- removals
def _out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


@contextlib.contextmanager
def removed(s, heads: dict[int, list[int]] | None = None, mlps: Iterable[int] = (), neurons: dict[int, torch.Tensor] | None = None,
            skip: Iterable[int] = (), mode: str = "zero"):
    """Zero whole heads (o_proj input slice), whole MLPs, individual MLP neurons (down_proj input), or skip layers."""
    handles = []
    for L, hs in (heads or {}).items():
        if not hs:
            continue
        d = s.s["d_head_per_layer"][L] or s.s["d_head"]
        cols = torch.cat([torch.arange(h * d, (h + 1) * d) for h in hs])

        def hh(m, a, cols=cols):
            t = a[0].clone()
            c = cols.to(t.device)
            t[..., c] = t[..., c].mean(dim=1, keepdim=True) if mode == "mean" else 0
            return (t,) + tuple(a[1:])
        handles.append(s.s["o_proj"][L].register_forward_pre_hook(hh))
    for L in mlps:
        handles.append(s.s["mlp"][L].register_forward_hook(
            lambda m, a, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else torch.zeros_like(o)))
    for L, idx in (neurons or {}).items():
        down = getattr(s.s["mlp"][L], "down_proj", None)
        if down is None:
            raise RuntimeError("neuron removal needs a gated MLP with down_proj (Llama/Qwen/Mistral layout)")

        def nh(m, a, idx=idx):
            t = a[0].clone()
            t[..., idx.to(t.device)] = 0
            return (t,) + tuple(a[1:])
        handles.append(down.register_forward_pre_hook(nh))
    for L in skip:
        def passthrough(mod, args, kwargs, out):
            hidden = args[0] if args else kwargs["hidden_states"]
            return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden
        handles.append(s.s["layers"][L].register_forward_hook(passthrough, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def make_pairs(tok, examples: list[Example], max_pairs: int) -> list[tuple[Example, Example]]:
    """Clean/corrupted pairs for patching: same token length, same template, different answer, and the fewest differing
    tokens (minimal pairs), so the difference is the task-relevant token rather than incidental filler."""
    enc = [encode(tok, e.prompt) for e in examples]
    by: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, (e, ids) in enumerate(zip(examples, enc)):
        by[(len(ids), e.template)].append(i)
    pairs = []
    for i, e in enumerate(examples):
        best, best_d = None, None
        for j in by[(len(enc[i]), e.template)]:
            o = examples[j]
            if o.answer == e.answer or o.answer not in e.candidates:
                continue
            d = sum(x != y for x, y in zip(enc[i], enc[j]))
            if best_d is None or d < best_d:
                best, best_d = o, d
        if best is not None:
            pairs.append((e, best))
        if len(pairs) >= max_pairs:
            break
    return pairs
