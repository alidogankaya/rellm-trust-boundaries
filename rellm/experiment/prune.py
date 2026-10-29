"""Phase 2 — does causal (circuit-based) importance prune better than standard importance at the same budget?

Width pruning: keep the same number of query heads per kv group and the same number of MLP neurons in every layer
for every method, so all arms at one level have identical parameter counts and identical tensor shapes.
Depth pruning: drop the same number of layers.

Arms
  causal         heads: E|margin change| when the head is removed (measured); neurons: E|attribution-patching effect|;
                 layers: skip necessity. Ranks by the *size* of the causal effect.
  causal_signed  same measurements, signed means (removal predicted to help => removed first). Kept as an ablation:
                 it fails because first-order effects of many removed neurons do not add up linearly.
  wanda     structured Wanda: sum |W| * ||X|| over the weights a head/neuron owns, X from the same train prompts
  magnitude L2 norm of the weights a head/neuron owns; layers: ShortGPT block influence (1 - cos(in, out))
  random    uniform random (mean over seeds)
Every pruned model is built physically (see build.py) and scored on heldout and OOD prompts, general-text NLL,
parameter count and latency. Optional healing: the same KL distillation budget for every arm.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from ..session import ModelSession
from .build import build_pruned, geometry
from .evaluate import Scorer, encode, latency_ms, length_batches, lm_nll, param_counts
from .tasks import Task


# ------------------------------------------------------------------------------------------- scores
@torch.no_grad()
def activation_norms(s: ModelSession, prompts: list[str]) -> dict[str, list[torch.Tensor]]:
    """Per-feature L2 norms over all tokens of: o_proj input, MLP input (post-attention norm output), down_proj input."""
    n = s.s["n_layers"]
    acc = {"o_in": [None] * n, "mlp_in": [None] * n, "down_in": [None] * n}
    hooks = []

    def mk(kind, L):
        def hook(m, a):
            x = a[0].float().reshape(-1, a[0].shape[-1])
            sq = (x * x).sum(0).cpu()
            acc[kind][L] = sq if acc[kind][L] is None else acc[kind][L] + sq
        return hook
    for L in range(n):
        layer = s.s["layers"][L]
        hooks.append(s.s["o_proj"][L].register_forward_pre_hook(mk("o_in", L)))
        hooks.append(layer.mlp.register_forward_pre_hook(mk("mlp_in", L)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk("down_in", L)))
    try:
        for _, ids in length_batches([encode(s.tok, p) for p in prompts], 16):
            s.model(input_ids=ids.to(s.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {k: [v.sqrt() for v in vs] for k, vs in acc.items()}


@torch.no_grad()
def width_scores(method: str, s: ModelSession, rev: dict[str, Any], prompts: list[str], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = geometry(s.model)
    nL, nH, d, dm = g["n_layers"], g["n_heads"], g["d_head"], g["d_mlp"]
    if method in ("causal", "causal_signed"):
        t = rev["_tensors"]
        if "neuron_attribution" not in t:
            raise RuntimeError("causal arms need neuron attribution from Phase 1 (first-token scoring)")
        if method == "causal":   # effect size: E|removal effect| per head (measured) and per neuron (attribution patching)
            return t["head_removal_effect"].clone(), t["neuron_effect"].clone()
        return t["head_removal_drop"].clone(), t["neuron_attribution"].clone()  # signed means (v1; kept as an ablation)
    if method == "random":
        gen = torch.Generator().manual_seed(seed)
        return torch.rand(nL, nH, generator=gen), torch.rand(nL, dm, generator=gen)
    heads, neurons = torch.zeros(nL, nH), torch.zeros(nL, dm)
    norms = activation_norms(s, prompts) if method == "wanda" else None
    for L in range(nL):
        layer = s.s["layers"][L]
        Wq, Wo = layer.self_attn.q_proj.weight.float().cpu(), layer.self_attn.o_proj.weight.float().cpu()
        Wg, Wu, Wd = (layer.mlp.gate_proj.weight.float().cpu(), layer.mlp.up_proj.weight.float().cpu(), layer.mlp.down_proj.weight.float().cpu())
        for h in range(nH):
            sl = slice(h * d, (h + 1) * d)
            if method == "magnitude":
                heads[L, h] = Wq[sl].norm() + Wo[:, sl].norm()
            elif method == "wanda":
                heads[L, h] = (Wo[:, sl].abs() * norms["o_in"][L][sl]).sum()
            else:
                raise ValueError(method)
        if method == "magnitude":
            neurons[L] = Wg.norm(dim=1) + Wu.norm(dim=1) + Wd.norm(dim=0)
        else:
            x = norms["mlp_in"][L]
            neurons[L] = (Wg.abs() * x).sum(1) + (Wu.abs() * x).sum(1) + (Wd.abs() * norms["down_in"][L]).sum(0)
    return heads, neurons


@torch.no_grad()
def depth_scores(method: str, s: ModelSession, rev: dict[str, Any], prompts: list[str], seed: int = 0) -> torch.Tensor:
    nL = s.s["n_layers"]
    if method == "causal":
        return rev["_tensors"]["layer_skip_drop"].clone()
    if method == "random":
        return torch.rand(nL, generator=torch.Generator().manual_seed(seed))
    if method in ("shortgpt", "magnitude", "wanda"):  # all non-causal arms use the standard depth criterion: block influence
        bi = torch.zeros(nL)
        n = 0
        for _, ids in length_batches([encode(s.tok, p) for p in prompts], 16):
            res = s.run(ids.to(s.device), components=("resid",))
            states = [res.resid_pre0] + [res.resid_post[L] for L in range(nL)]
            for L in range(nL):
                bi[L] += float((1 - torch.nn.functional.cosine_similarity(states[L].float(), states[L + 1].float(), dim=-1)).sum())
            n += ids.numel()
        return bi / n
    raise ValueError(method)


@torch.no_grad()
def iterative_depth_drop(s: ModelSession, examples, n_drop: int) -> list[int]:
    """Greedy: at each step skip the layer whose removal (given the layers already skipped) costs the least mean margin."""
    from .evaluate import removed
    sc = Scorer(s.tok, examples)
    dropped: list[int] = []
    for _ in range(n_drop):
        best, best_m = None, None
        for L in range(s.s["n_layers"]):
            if L in dropped:
                continue
            with removed(s, skip=dropped + [L]):
                m = sc.score(s.model)["margin_mean"]
            if best_m is None or m > best_m:
                best, best_m = L, m
        dropped.append(best)
    return sorted(dropped)


def select_width(heads: torch.Tensor, neurons: torch.Tensor, keep_frac: float, g: dict[str, int]) -> tuple[dict, dict, int, int]:
    k_neur = max(1, round(keep_frac * g["d_mlp"]))
    mha = g["group"] == 1
    # GQA: same number of query heads kept in every kv group; MHA: any heads (their kv heads go with them)
    k_group = max(1, round(keep_frac * (g["n_heads"] if mha else g["group"])))
    groups = [list(range(g["n_heads"]))] if mha else [list(range(kv * g["group"], (kv + 1) * g["group"])) for kv in range(g["n_kv"])]
    hsel, nsel = {}, {}
    for L in range(g["n_layers"]):
        chosen = []
        for ids in groups:
            chosen += sorted(ids, key=lambda h: -float(heads[L, h]))[:k_group]
        hsel[L] = sorted(chosen)
        nsel[L] = neurons[L].argsort(descending=True)[:k_neur].sort().values
    return hsel, nsel, k_group * len(groups), k_neur


# ------------------------------------------------------------------------------------------ healing
def heal(student, teacher, tok, texts: list[str], steps: int, lr: float = 5e-5) -> dict[str, float]:
    if steps <= 0:
        return {}
    dev = next(student.parameters()).device
    batches = [ids for _, ids in length_batches([encode(tok, t) for t in texts], 8)]
    opt = torch.optim.AdamW([p for p in student.parameters() if p.is_floating_point()], lr=lr)
    student.train()
    last = 0.0
    for step in range(steps):
        ids = batches[step % len(batches)].to(dev)
        with torch.no_grad():
            t_lp = torch.log_softmax(teacher(input_ids=ids, use_cache=False).logits.float(), -1)
        s_lp = torch.log_softmax(student(input_ids=ids, use_cache=False).logits.float(), -1)
        loss = torch.nn.functional.kl_div(s_lp, t_lp, log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return {"final_kl": round(last, 5), "steps": steps}


# -------------------------------------------------------------------------------------------- runner
def evaluate_model(model, tok, task: Task) -> dict[str, Any]:
    ho = Scorer(tok, task.heldout).score(model)
    ood = Scorer(tok, task.ood).score(model) if task.ood else {"acc": None, "margin_mean": None}
    gen = lm_nll(model, tok, task.general_texts) if task.general_texts else {"nll": None}
    return {"heldout_acc": ho["acc"], "heldout_margin": ho["margin_mean"], "ood_acc": ood["acc"], "ood_margin": ood["margin_mean"],
            "general_nll": gen["nll"], **param_counts(model), "latency_ms": latency_ms(model, tok, [e.prompt for e in task.heldout])}


def run_prune(s: ModelSession, task: Task, rev: dict[str, Any], width_levels=(0.75, 0.5, 0.25), depth_fracs=(0.125, 0.25, 0.375),
              methods=("causal", "causal_signed", "wanda", "magnitude", "random"), depth_methods=("causal", "causal_iter", "shortgpt", "random"), random_seeds=(0, 1, 2), heal_steps: int = 0, log=print) -> dict[str, Any]:
    g = geometry(s.model)
    prompts = [e.prompt + e.answer for e in task.train]
    heal_texts = prompts
    full = evaluate_model(s.model, s.tok, task)
    log(f"  full model: {full}")
    rows: list[dict[str, Any]] = [{"kind": "full", "level": 1.0, "method": "-", **full}]

    def run_one(kind, level, method, build_args, seed=None):
        m = build_pruned(s.model, **build_args)
        healed = heal(m, s.model, s.tok, heal_texts, heal_steps) if heal_steps else {}
        r = evaluate_model(m, s.tok, task) | {"kind": kind, "level": level, "method": method, "seed": seed, "heal": healed}
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return r

    seen = set()
    for frac in width_levels:
        hsel_shape = select_width(torch.zeros(g["n_layers"], g["n_heads"]), torch.zeros(g["n_layers"], g["d_mlp"]), frac, g)
        key = (hsel_shape[2], hsel_shape[3])
        if key in seen:
            continue
        seen.add(key)
        for method in methods:
            seeds = random_seeds if method == "random" else (0,)
            res = []
            for sd in seeds:
                hs, ns = width_scores(method, s, rev, [e.prompt for e in task.train], seed=sd)
                hsel, nsel, kh, kn = select_width(hs, ns, frac, g)
                res.append(run_one("width", frac, method, {"heads": hsel, "neurons": nsel}, sd))
            r = _mean_rows(res) if len(res) > 1 else res[0]
            r["kept"] = {"heads_per_layer": kh, "neurons_per_layer": kn}
            rows.append(r)
            log(f"  width {frac:.2f} {method:9s}: heldout {r['heldout_acc']:.3f} (m {r['heldout_margin']:.2f})  ood {r['ood_acc']}  "
                f"nll {r['general_nll']}  body {r['params_body']}  {r['latency_ms']} ms")
    seen_d = set()
    for frac in depth_fracs:
        n_drop = max(1, round(frac * g["n_layers"]))
        if n_drop >= g["n_layers"] or n_drop in seen_d:
            continue
        seen_d.add(n_drop)
        for method in depth_methods:
            seeds = random_seeds if method == "random" else (0,)
            res = []
            for sd in seeds:
                if method == "causal_iter":
                    drop = set(iterative_depth_drop(s, task.train, n_drop))
                else:
                    sc = depth_scores(method, s, rev, [e.prompt for e in task.train], seed=sd)
                    drop = set(sc.argsort()[:n_drop].tolist())
                keep = [L for L in range(g["n_layers"]) if L not in drop]
                res.append(run_one("depth", n_drop, method, {"keep_layers": keep}, sd) | {"dropped_layers": sorted(drop)})
            r = _mean_rows(res) if len(res) > 1 else res[0]
            rows.append(r)
            log(f"  depth -{n_drop} {method:9s}: heldout {r['heldout_acc']:.3f} (m {r['heldout_margin']:.2f})  ood {r['ood_acc']}  "
                f"nll {r['general_nll']}  dropped {r.get('dropped_layers')}  {r['latency_ms']} ms")
    _check_budgets(rows)
    chance = sum(1 / len(e.candidates) for e in task.heldout) / len(task.heldout)
    return {"rows": rows, "verdict": verdict(rows, chance=chance), "heal_steps": heal_steps}


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(rows[0])
    for k, v in rows[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("level", "seed"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = round(sum(vals) / len(vals), 4) if vals else None
    out["seed"] = f"mean of {len(rows)}"
    out["heldout_acc_seeds"] = [r["heldout_acc"] for r in rows]
    return out


def _check_budgets(rows: list[dict[str, Any]]) -> None:
    by = defaultdict(set)
    for r in rows:
        if r["kind"] != "full":
            by[(r["kind"], r["level"])].add(r["params"])
    bad = {k: v for k, v in by.items() if len(v) > 1}
    if bad:
        raise AssertionError(f"unequal parameter budgets within a level: {bad}")


def verdict(rows: list[dict[str, Any]], margin_pp: float = 0.05, margin_ratio: float = 0.10, min_full_acc: float = 0.8,
            chance: float = 0.0, collapse_band: float = 0.15) -> dict[str, Any]:
    """Causal arm vs the best non-causal arm at every level.

    Primary metric: heldout accuracy (win/loss when the difference is >= margin_pp).
    When accuracies are within 1pp (saturated), the heldout margin decides (win/loss at >= margin_ratio relative difference).
    A win also requires OOD accuracy not worse than that same best arm's OOD accuracy.
    No verdict when the full model itself does not solve the task (heldout acc < min_full_acc).
    """
    full = next((r for r in rows if r["kind"] == "full"), None)
    if full is not None and full["heldout_acc"] < min_full_acc:
        return {"comparisons": [], "wins": 0, "losses": 0, "threshold_pp": margin_pp, "threshold_margin_rel": margin_ratio,
                "text": f"NOT APPLICABLE: the full model solves only {full['heldout_acc']:.0%} of heldout prompts (< {min_full_acc:.0%}); "
                        "there is no capability to preserve", "note": "pick a larger model or an easier task"}
    levels = defaultdict(dict)
    for r in rows:
        if r["kind"] != "full":
            levels[(r["kind"], r["level"])][r["method"]] = r
    comps = []
    for (kind, level), arms in sorted(levels.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        primary = "causal_iter" if (kind == "depth" and "causal_iter" in arms) else "causal"
        if primary not in arms:
            continue
        others = {m: a for m, a in arms.items() if not m.startswith("causal")}
        best = max(others.items(), key=lambda kv: (kv[1]["heldout_acc"], kv[1]["heldout_margin"]))
        c = arms[primary]
        d_acc = round(c["heldout_acc"] - best[1]["heldout_acc"], 4)
        d_m = round((c["heldout_margin"] - best[1]["heldout_margin"]) / (abs(best[1]["heldout_margin"]) + 1e-9), 4)
        d_ood = None if c["ood_acc"] is None or best[1]["ood_acc"] is None else round(c["ood_acc"] - best[1]["ood_acc"], 4)
        if max(c["heldout_acc"], best[1]["heldout_acc"]) < chance + collapse_band:
            outcome = "collapsed"   # both at chance: the level says nothing about which criterion is better
        elif d_acc >= margin_pp or (abs(d_acc) < 0.01 and d_m >= margin_ratio):
            outcome = "win" if (d_ood is None or d_ood >= -0.01) else "tie"
        elif d_acc <= -margin_pp or (abs(d_acc) < 0.01 and d_m <= -margin_ratio):
            outcome = "loss"
        else:
            outcome = "tie"
        comps.append({"kind": kind, "level": level, "causal_arm": primary, "causal_heldout": c["heldout_acc"], "causal_margin": c["heldout_margin"],
                      "best_other": best[0], "best_other_heldout": best[1]["heldout_acc"], "best_other_margin": best[1]["heldout_margin"],
                      "delta_heldout": d_acc, "delta_margin_rel": d_m, "causal_ood": c["ood_acc"], "best_other_ood": best[1]["ood_acc"],
                      "delta_ood": d_ood, "outcome": outcome})
    collapsed = sum(x["outcome"] == "collapsed" for x in comps)
    wins = sum(x["outcome"] == "win" for x in comps)
    losses = sum(x["outcome"] == "loss" for x in comps)
    if wins >= 2 and losses == 0:
        text = f"SUPPORTED: causal importance beats every baseline at {wins} levels and loses at none"
    elif losses > wins:
        text = f"REJECTED: standard criteria beat causal importance ({losses} losses vs {wins} wins)"
    else:
        text = f"INCONCLUSIVE: {wins} wins, {losses} losses, {len(comps) - wins - losses - collapsed} ties"
    return {"comparisons": comps, "wins": wins, "losses": losses, "collapsed": collapsed, "chance": round(chance, 4), "threshold_pp": margin_pp, "threshold_margin_rel": margin_ratio, "text": text,
            "note": "one task, one model: a supported verdict here is a hypothesis for the next scale, not a general result"}
