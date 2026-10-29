"""Phase 1 — reverse engineer how the model performs a task, aggregated over many prompts.

Every section is recorded as Evidence (authority: observation / derived / inference) and summarised for the report.
Scores needed by the pruning phase (head removal necessity, layer skip necessity, neuron attribution) are returned
as tensors and saved next to the report.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch

from .. import analysis
from ..evidence import EvidenceStore
from ..session import ModelSession
from .evaluate import Scorer, encode, length_batches, make_pairs, removed
from .tasks import Task


class Recorder:
    def __init__(self, s: ModelSession, store: EvidenceStore, task: str):
        self.s, self.store, self.task = s, store, task
        self.ids: dict[str, str] = {}

    def __call__(self, op: str, params: dict[str, Any], result: dict[str, Any], authority: str, limitations=(), coverage=None) -> str:
        rec = self.store.record(f"{self.task}:{op}", params, self.s.identity(), result, coverage, list(limitations), authority,
                                self.s.analysis_profile(), self.s.facets, untrusted=True, cacheable=False)
        self.ids[op] = rec["evidence_id"]
        return rec["evidence_id"]


def _top(d: dict, k: int, key=lambda kv: -kv[1]) -> list:
    return sorted(d.items(), key=key)[:k]


@torch.no_grad()
def _lens(s: ModelSession, sc: Scorer, sel: list[int]) -> dict[str, Any]:
    n_layers = s.s["n_layers"]
    frac_top = torch.zeros(n_layers + 1)
    prob = torch.zeros(n_layers + 1)
    rank = torch.zeros(n_layers + 1)
    decision_hist: dict[str, int] = defaultdict(int)
    by_answer: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in sel:
        ids = torch.tensor([sc.prompt_ids[i]], device=s.device)
        res = s.run(ids, components=("resid",))
        states = [res.resid_pre0] + [res.resid_post[L] for L in range(n_layers)]
        cands = sc.first_ids[i].to(s.device)
        a = sc.answer_idx[i]
        tops = []
        for L, st in enumerate(states):
            logits = s.unembed(st[0, -1])
            cl = logits[cands]
            tops.append(int(cl.argmax()) == a)
            frac_top[L] += tops[-1]
            prob[L] += float(torch.softmax(logits, -1)[cands[a]])
            rank[L] += float((cl > cl[a]).sum()) + 1
        dec = None
        for L in range(n_layers, -1, -1):
            if tops[L]:
                dec = L
            else:
                break
        key = "never" if dec is None else ("embed" if dec == 0 else f"L{dec - 1}")
        decision_hist[key] += 1
        by_answer[sc.examples[i].answer][key] += 1
    n = max(len(sel), 1)
    names = ["embed"] + [f"L{L}" for L in range(n_layers)]
    rows = [{"layer": names[L], "answer_top_among_candidates": round(float(frac_top[L] / n), 3),
             "mean_candidate_rank": round(float(rank[L] / n), 2), "mean_answer_prob": round(float(prob[L] / n), 4)} for L in range(n_layers + 1)]
    split = {a: dict(h) for a, h in by_answer.items()} if len(by_answer) <= 4 else None  # only for few-class tasks
    return {"n_prompts": len(sel), "layers": rows, "decision_layer_histogram": dict(decision_hist), "decision_layer_by_answer": split}


def _neuron_attribution(s: ModelSession, sc: Scorer, sel: list[int], absolute: bool = False) -> torch.Tensor:
    """Attribution-patching estimate of removing each MLP neuron: per example e_j = sum_pos a_j * dmargin/da_j.
    Returns E[e_j] (signed) or E[|e_j|] (effect size, absolute=True)."""
    n_layers = s.s["n_layers"]
    downs = [getattr(m, "down_proj", None) for m in s.s["mlp"]]
    if any(d is None for d in downs):
        raise RuntimeError("neuron attribution needs gated MLPs with down_proj")
    d_mlp = downs[0].in_features
    total = torch.zeros(n_layers, d_mlp)
    req = [p.requires_grad for p in s.model.parameters()]
    for p in s.model.parameters():
        p.requires_grad_(False)
    try:
        seqs = [sc.prompt_ids[i] for i in sel]
        for part, ids in length_batches(seqs, 16):
            acts: dict[int, torch.Tensor] = {}

            def mk(L):
                def hook(m, a):
                    x = a[0].detach().requires_grad_(True)
                    acts[L] = x
                    return (x,) + tuple(a[1:])
                return hook
            hs = [downs[L].register_forward_pre_hook(mk(L)) for L in range(n_layers)]
            try:
                with torch.enable_grad():
                    logits = s.model(input_ids=ids.to(s.device), use_cache=False).logits[:, -1].float()
                    metric = 0.0
                    for row, k in enumerate(part):
                        i = sel[k]
                        cands = sc.first_ids[i].to(s.device)
                        cl = logits[row, cands]
                        a = sc.answer_idx[i]
                        others = torch.cat([cl[:a], cl[a + 1:]])
                        metric = metric + (cl[a] - others.max())
                    metric.backward()
                for L in range(n_layers):
                    per_ex = (acts[L].detach() * acts[L].grad).sum(dim=1).float()  # (batch, d_mlp)
                    total[L] += (per_ex.abs() if absolute else per_ex).sum(0).cpu()
            finally:
                for h in hs:
                    h.remove()
    finally:
        for p, r in zip(s.model.parameters(), req):
            p.requires_grad_(r)
    return total / max(len(sel), 1)


def reverse_task(s: ModelSession, task: Task, store: EvidenceStore, max_pairs: int = 8, last_k: int = 8, top_n: int = 12,
                 log=print) -> dict[str, Any]:
    t_start = time.time()
    rec = Recorder(s, store, task.name)
    n_layers, n_heads = s.s["n_layers"], s.s["n_heads"]
    out: dict[str, Any] = {"task": task.summary(), "warnings": []}
    tensors: dict[str, torch.Tensor] = {}

    # 1. behaviour ------------------------------------------------------------------------------------
    sc_tr, sc_ho, sc_ood = Scorer(s.tok, task.train), Scorer(s.tok, task.heldout), Scorer(s.tok, task.ood)
    base_tr = sc_tr.score(s.model, keep_per_example=True)
    beh = {"train": {k: v for k, v in base_tr.items() if k not in ("margins", "correct")}, "heldout": sc_ho.score(s.model), "ood": sc_ood.score(s.model)}
    rec("behaviour", {"splits": ["train", "heldout", "ood"]}, beh, "observation")
    out["behaviour"] = beh
    log(f"  behaviour: train {beh['train']['acc']:.2f}  heldout {beh['heldout']['acc']:.2f}  ood {beh['ood']['acc']:.2f}  (scoring: {sc_tr.mode})")
    sel = [i for i, c in enumerate(base_tr["correct"]) if c] or list(range(len(task.train)))
    if beh["train"]["acc"] < 0.6:
        out["warnings"].append(f"model solves only {beh['train']['acc']:.0%} of train prompts; the circuit of a task the model cannot do is weak evidence")
    sel_ex = [task.train[i] for i in sel]
    sc_sel = Scorer(s.tok, sel_ex)
    base_sel = sc_sel.score(s.model, keep_per_example=True)
    base_m = torch.tensor(base_sel["margins"])

    # 2. logit lens -------------------------------------------------------------------------------------
    if sc_sel.mode == "first_token":
        lens = _lens(s, sc_sel, list(range(min(32, len(sel_ex)))))
        rec("logit_lens", {"n": lens["n_prompts"]}, lens, "inference", ["logit lens is a heuristic readout of intermediate layers"])
        out["logit_lens"] = lens
        log(f"  decision layers: {lens['decision_layer_histogram']}" + (f" | by answer: {lens['decision_layer_by_answer']}" if lens["decision_layer_by_answer"] else ""))
    else:
        out["warnings"].append("candidates share first tokens: lens, patching, DLA and neuron attribution use first tokens and are skipped")

    # 3. depth: truncation ------------------------------------------------------------------------------
    trunc = []
    for L in range(-1, n_layers):
        with removed(s, skip=range(L + 1, n_layers)):
            r = sc_sel.score(s.model)
        trunc.append({"through": "embed" if L < 0 else f"L{L}", "acc": r["acc"], "margin_mean": r["margin_mean"], "margin_min": r["margin_min"]})
    full_acc = trunc[-1]["acc"]
    sufficient = next((t["through"] for t in trunc if t["acc"] >= full_acc - 0.02), trunc[-1]["through"])
    rec("truncation", {"layers": n_layers}, {"rows": trunc, "sufficient_depth": sufficient}, "observation",
        ["truncation removes later layers entirely; the final norm and unembedding are kept"])
    out["truncation"] = {"rows": trunc, "sufficient_depth": sufficient}
    log(f"  sufficient depth (acc within 2pp of full): through {sufficient}")

    # 4. layer skip necessity -----------------------------------------------------------------------------
    layer_drop = torch.zeros(n_layers)
    layer_rows = []
    for L in range(n_layers):
        with removed(s, skip=[L]):
            r = sc_sel.score(s.model, keep_per_example=True)
        layer_drop[L] = float((base_m - torch.tensor(r["margins"])).mean())
        layer_rows.append({"layer": L, "margin_drop": round(float(layer_drop[L]), 4), "acc": r["acc"]})
    tensors["layer_skip_drop"] = layer_drop
    rec("layer_skip_necessity", {}, {"rows": layer_rows}, "observation")
    out["layer_skip"] = layer_rows

    # 5. head + MLP removal necessity --------------------------------------------------------------------
    head_drop = torch.zeros(n_layers, n_heads)
    head_effect = torch.zeros(n_layers, n_heads)
    for L in range(n_layers):
        for h in range(n_heads):
            with removed(s, heads={L: [h]}):
                r = sc_sel.score(s.model, keep_per_example=True)
            diff = base_m - torch.tensor(r["margins"])
            head_drop[L, h] = float(diff.mean())
            head_effect[L, h] = float(diff.abs().mean())
    mlp_drop = torch.zeros(n_layers)
    for L in range(n_layers):
        with removed(s, mlps=[L]):
            r = sc_sel.score(s.model, keep_per_example=True)
        mlp_drop[L] = float((base_m - torch.tensor(r["margins"])).mean())
    tensors["head_removal_drop"], tensors["head_removal_effect"], tensors["mlp_removal_drop"] = head_drop, head_effect, mlp_drop
    hd = {f"L{L}H{h}": round(float(head_drop[L, h]), 4) for L in range(n_layers) for h in range(n_heads)}
    md = {f"L{L}.mlp": round(float(mlp_drop[L]), 4) for L in range(n_layers)}
    flat = head_drop.flatten().abs().sort(descending=True).values
    conc = {f"top{k}_share": round(float(flat[:k].sum() / (flat.sum() + 1e-9)), 3) for k in (1, 5, 20) if k <= flat.numel()}
    nec = {"base_margin_mean": round(float(base_m.mean()), 4), "top_heads": _top(hd, max(top_n, 20)), "most_harmful_heads": _top(hd, 5, lambda kv: kv[1]),
           "mlps": md, "head_drop_concentration": conc}
    rec("removal_necessity", {"n_prompts": len(sel_ex)}, nec | {"all_heads": hd}, "observation",
        ["removal = zero ablation over all positions, the operation a pruned model performs; compensation by other heads hides joint necessity"])
    out["necessity"] = nec
    log(f"  top heads by removal: {nec['top_heads'][:5]} | MLPs: {_top(md, 3)}")

    pairs = make_pairs(s.tok, sel_ex, max_pairs) if sc_sel.mode == "first_token" else []
    out["n_pairs"] = len(pairs)
    if sc_sel.mode == "first_token" and not pairs:
        out["warnings"].append("no clean/corrupted pairs with equal token length and template: patching and DLA skipped")

    # 6. activation patching aggregated over pairs ----------------------------------------------------
    if pairs:
        resid = defaultdict(list)
        headp = defaultdict(list)
        mlpp = defaultdict(list)
        tok_at: dict[int, str] = {}
        for clean, corr in pairs:
            seq = len(encode(s.tok, clean.prompt))
            k = min(last_k, seq)
            pos = list(range(seq - k, seq))
            r = analysis.activation_patching(s, clean.prompt, corr.prompt, clean.answer, corr.answer, "resid_post", positions=pos)
            if abs(r["clean"]["metric"] - r["corrupted"]["metric"]) < 1e-3:
                continue
            for row in r["grid"]:
                resid[(row["layer"], row["position"] - seq)].append(row["normalized"])
                tok_at.setdefault(row["position"] - seq, row["token"])
            r = analysis.activation_patching(s, clean.prompt, corr.prompt, clean.answer, corr.answer, "head", positions=[seq - 1])
            for row in r["grid"]:
                headp[(row["layer"], row["head"])].append(row["normalized"])
            r = analysis.activation_patching(s, clean.prompt, corr.prompt, clean.answer, corr.answer, "mlp_out", positions=pos)
            for row in r["grid"]:
                mlpp[(row["layer"], row["position"] - seq)].append(row["normalized"])
        mean = lambda v: round(sum(v) / len(v), 4)
        grid = [{"layer": L, "pos_from_end": p, "token_example": tok_at.get(p), "normalized_mean": mean(v), "n": len(v)}
                for (L, p), v in sorted(resid.items())]
        hp = {f"L{L}H{h}": mean(v) for (L, h), v in headp.items()}
        mp = {f"L{L}@{p}": mean(v) for (L, p), v in mlpp.items()}
        # information flow. Sources = positions where clean and corrupted differ, ranked by a token-swap test
        # (put the clean token back at one position of the corrupted prompt: how much of the clean answer returns?).
        swap = defaultdict(list)
        for clean, corr in pairs:
            a, b = encode(s.tok, clean.prompt), encode(s.tok, corr.prompt)
            ans_id, cnt_id = encode(s.tok, clean.answer, False)[0], encode(s.tok, corr.answer, False)[0]
            metric_of = lambda ids: float((lambda lg: lg[ans_id] - lg[cnt_id])(s.logits_only(torch.tensor([ids], device=s.device))[0, -1]))
            m_clean, m_corr = metric_of(a), metric_of(b)
            if abs(m_clean - m_corr) < 1e-3:
                continue
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    ids = list(b)
                    ids[i] = x
                    swap[i - len(a)].append((metric_of(ids) - m_corr) / (m_clean - m_corr))
                    tok_at.setdefault(i - len(a), s.tok.decode([x]))
        ranked_src = sorted(swap.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        for p_src, vals in ranked_src[:2]:
            if sum(vals) / len(vals) < 0.05 or any(k[1] == p_src for k in resid):
                continue
            for clean, corr in pairs:  # source outside the last_k window: patch it over all layers
                seq = len(encode(s.tok, clean.prompt))
                if seq + p_src < 0:
                    continue
                r = analysis.activation_patching(s, clean.prompt, corr.prompt, clean.answer, corr.answer, "resid_post", positions=[seq + p_src])
                if abs(r["clean"]["metric"] - r["corrupted"]["metric"]) < 1e-3:
                    continue
                for row in r["grid"]:
                    resid[(row["layer"], p_src)].append(row["normalized"])
        grid = [{"layer": L, "pos_from_end": p, "token_example": tok_at.get(p), "normalized_mean": mean(v), "n": len(v)}
                for (L, p), v in sorted(resid.items())]
        by = {(row["layer"], row["pos_from_end"]): row["normalized_mean"] for row in grid}
        flow = []
        for p_src, vals in sorted(swap.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            eff = [by.get((L, p_src)) for L in range(n_layers)]
            still = [L for L, e in enumerate(eff) if e is not None and e >= 0.1]
            measured = any(e is not None for e in eff)
            flow.append({"source_pos_from_end": p_src, "token_example": tok_at.get(p_src), "token_swap_recovery": round(sum(vals) / len(vals), 4),
                         "pairs": len(vals), "effect_by_layer": eff,
                         "source_matters_through": (f"L{max(still)}" if still else "only before L0 output (read inside L0)") if measured
                         else "not measured"})
        flow = [f for f in flow if f["token_swap_recovery"] >= 0.05] or flow[:1]
        last = [by.get((L, -1)) for L in range(n_layers)]
        carried = next((L for L, e in enumerate(last) if e is not None and e >= 0.5), None)
        flow_summary = {"sources": flow, "last_position_effect_by_layer": last,
                        "last_position_carries_decision_from": f"L{carried}" if carried is not None else "never >= 0.5"}
        patch = {"n_pairs": len(pairs), "resid_grid": grid, "top_heads_at_last": _top(hp, top_n), "top_mlp": _top(mp, top_n), "flow": flow_summary}
        rec("activation_patching", {"pairs": [(c.prompt, x.prompt) for c, x in pairs], "last_k": last_k}, patch, "observation",
            ["normalized effects averaged over pairs; pairs share template and token length"])
        out["patching"] = patch
        tensors["head_patch_last"] = torch.tensor([[hp.get(f"L{L}H{h}", 0.0) for h in range(n_heads)] for L in range(n_layers)])
        src = flow_summary["sources"][0] if flow_summary["sources"] else {}
        log(f"  flow: source {src.get('source_pos_from_end')} ('{src.get('token_example')}', swap recovery {src.get('token_swap_recovery')}) "
            f"matters through {src.get('source_matters_through')}; "
            f"last position carries it from {flow_summary['last_position_carries_decision_from']} | heads@last {patch['top_heads_at_last'][:4]}")

        # 7. DLA aggregated --------------------------------------------------------------------------
        contrib = defaultdict(list)
        totals, errs = [], []
        for clean, corr in pairs:
            d = analysis.direct_logit_attribution(s, clean.prompt, clean.answer, corr.answer, top_k=5)
            totals.append(d["actual_target_value"])
            errs.append(d["reconstruction_max_abs_err"])
            for c in d["components"]:
                key = c["component"] if c["layer"] is None else (f"L{c['layer']}H{c['head']}" if c["head"] is not None else f"L{c['layer']}.{c['component']}")
                contrib[key].append(c["contribution"])
        dla = {k: round(sum(v) / len(v), 4) for k, v in contrib.items()}
        tgt = sum(totals) / len(totals)
        attn_total = sum(v for k, v in dla.items() if "H" in k)
        mlp_total = sum(v for k, v in dla.items() if k.endswith(".mlp"))
        dla_out = {"mean_logit_diff": round(tgt, 4), "attention_share": round(attn_total / (tgt + 1e-9), 3), "mlp_share": round(mlp_total / (tgt + 1e-9), 3),
                   "top_positive": _top(dla, top_n), "top_negative": _top(dla, 6, lambda kv: kv[1]), "max_reconstruction_err": round(max(errs), 6)}
        rec("direct_logit_attribution", {"n_pairs": len(pairs)}, dla_out | {"all": dla}, "derived",
            ["direct path only; final-norm scale frozen per prompt"])
        out["dla"] = dla_out
        log(f"  DLA: logit diff {dla_out['mean_logit_diff']}  attention {dla_out['attention_share']:.0%}  MLP {dla_out['mlp_share']:.0%}")

        # 8. attention of the key heads ----------------------------------------------------------------
        key_heads = []
        for k, _ in nec["top_heads"][:5] + patch["top_heads_at_last"][:5]:
            L, h = map(int, k[1:].split("H"))
            if (L, h) not in key_heads:
                key_heads.append((L, h))
        att_rows = []
        acc_att = {kh: defaultdict(float) for kh in key_heads}
        ans_mass = {kh: 0.0 for kh in key_heads}
        cnt_mass = {kh: 0.0 for kh in key_heads}
        tok_from_end: dict[int, str] = {}
        for clean, corr in pairs:
            ids = torch.tensor([encode(s.tok, clean.prompt)], device=s.device)
            toks = s.token_strings(ids[0].tolist())
            seq = ids.shape[1]
            res = s.run(ids, components=(), want_attn=True)
            a_id = encode(s.tok, clean.answer, False)[0]
            c_id = encode(s.tok, corr.answer, False)[0]
            for (L, h) in key_heads:
                w = res.attentions[L][0, h, -1].float().cpu()
                for p in range(seq):
                    acc_att[(L, h)][p - seq] += float(w[p]) / len(pairs)
                    tok_from_end.setdefault(p - seq, toks[p])
                ans_mass[(L, h)] += float(w[(ids[0].cpu() == a_id)].sum()) / len(pairs)
                cnt_mass[(L, h)] += float(w[(ids[0].cpu() == c_id)].sum()) / len(pairs)
        for (L, h) in key_heads:
            top3 = _top(acc_att[(L, h)], 3)
            att_rows.append({"head": f"L{L}H{h}", "attends_to": [{"pos_from_end": p, "token_example": tok_from_end.get(p), "weight": round(w, 3)} for p, w in top3],
                             "mass_on_answer_token": round(ans_mass[(L, h)], 3), "mass_on_counter_token": round(cnt_mass[(L, h)], 3)})
        rec("key_head_attention", {"heads": [f"L{L}H{h}" for L, h in key_heads]}, {"rows": att_rows}, "inference",
            ["attention weights show where a head reads, not what it computes"])
        out["key_head_attention"] = att_rows

        # 9. roles ------------------------------------------------------------------------------------
        base = float(base_m.mean()) + 1e-9
        roles = defaultdict(list)
        for key in set(hd) | set(md):
            nec_f = (hd.get(key) if key in hd else md.get(key)) / base
            dla_f = dla.get(key, 0.0) / (tgt + 1e-9)
            if nec_f >= 0.1 and dla_f >= 0.1:
                roles["writer (necessary + writes the answer)"].append(key)
            elif nec_f >= 0.1 and abs(dla_f) < 0.03:
                roles["indirect (necessary, writes ~nothing directly)"].append(key)
            elif dla_f >= 0.1 and nec_f < 0.03:
                roles["backed-up writer (writes the answer, removal compensated)"].append(key)
            elif dla_f <= -0.05 or nec_f <= -0.05:
                roles["suppressor (pushes against the answer / removal helps)"].append(key)
        src_positions = {f["source_pos_from_end"] for f in patch["flow"]["sources"]}
        for row in att_rows:
            if row["attends_to"] and row["attends_to"][0]["pos_from_end"] in src_positions and row["attends_to"][0]["weight"] >= 0.3:
                nec_f = hd.get(row["head"], 0.0) / base
                if nec_f >= 0.05:
                    roles["mover (necessary, reads the source position)"].append(row["head"])
        roles = {k: sorted(set(v)) for k, v in roles.items()}
        rec("component_roles", {"thresholds": {"necessary": 0.1, "direct": 0.1, "negligible": 0.03}}, roles, "derived",
            ["roles combine removal necessity (observation) and DLA (derived); thresholds are conventions"])
        out["roles"] = roles
        unknowns = []
        for k in roles.get("indirect (necessary, writes ~nothing directly)", [])[:4]:
            unknowns.append(store.record_unknown(f"[{task.name}] {k} is necessary but writes almost nothing to the answer: which later component reads it?",
                                                 "circuit", rec.ids.get("removal_necessity"))["unknown_id"])
        for k in roles.get("backed-up writer (writes the answer, removal compensated)", [])[:4]:
            unknowns.append(store.record_unknown(f"[{task.name}] {k} writes the answer but its removal is compensated: which component backs it up?",
                                                 "circuit", rec.ids.get("direct_logit_attribution"))["unknown_id"])
        out["unknowns"] = unknowns

    # 10. neuron attribution --------------------------------------------------------------------------
    if sc_sel.mode == "first_token" and all(getattr(m, "down_proj", None) is not None for m in s.s["mlp"]):
        attr = _neuron_attribution(s, sc_sel, list(range(len(sel_ex))))
        tensors["neuron_attribution"] = attr
        tensors["neuron_effect"] = _neuron_attribution(s, sc_sel, list(range(len(sel_ex))), absolute=True)
        rows = []
        for L in range(n_layers):
            pos = attr[L].clamp_min(0).sort(descending=True).values
            cum = pos.cumsum(0) / (pos.sum() + 1e-9)
            rows.append({"layer": L, "positive_mass": round(float(pos.sum()), 4), "negative_mass": round(float(attr[L].clamp_max(0).sum()), 4),
                         "neurons_for_50pct": int((cum < 0.5).sum()) + 1, "neurons_for_90pct": int((cum < 0.9).sum()) + 1})
        flat = attr.flatten()
        top = flat.topk(min(10, flat.numel()))
        d_mlp = attr.shape[1]
        nrep = {"per_layer": rows, "top_neurons": [{"neuron": f"L{int(i) // d_mlp}.n{int(i) % d_mlp}", "attribution": round(float(v), 4)} for v, i in zip(*top)],
                "d_mlp": d_mlp}
        rec("neuron_attribution", {"n_prompts": len(sel_ex)}, nrep, "inference",
            ["first-order (attribution patching) estimate of removing each neuron at all positions; large effects should be confirmed by removal"])
        out["neurons"] = nrep
        log(f"  neurons for 90% of positive attribution per layer: {[r['neurons_for_90pct'] for r in rows]} / {d_mlp}")

    out["evidence_ids"] = rec.ids
    out["elapsed_s"] = round(time.time() - t_start, 1)
    out["_tensors"] = tensors
    return out


def model_level(s: ModelSession, store: EvidenceStore, seq_lens=(12, 16, 24, 32, 48), log=print) -> dict[str, Any]:
    """Task-independent facts: head catalog at several lengths, flagging length-dependent (positional) heads."""
    cats = {}
    for L in seq_lens:
        cats[L] = {(r["layer"], r["head"]): r for r in analysis.head_catalog(s, seq_len=L, n_samples=4, seed=0)["all_heads"]}
    keys = list(next(iter(cats.values())).keys())
    rows = []
    for k in keys:
        ind = [cats[L][k]["induction"] for L in seq_lens]
        prev = [cats[L][k]["previous_token"] for L in seq_lens]
        rows.append({"head": f"L{k[0]}H{k[1]}", "induction_by_len": ind, "prev_token_by_len": prev})
    induction = [r for r in rows if min(r["induction_by_len"]) >= 0.3]
    positional = [r for r in rows if max(r["induction_by_len"]) >= 0.3 and min(r["induction_by_len"]) < 0.3 * max(r["induction_by_len"])]
    prev_tok = [r for r in rows if min(r["prev_token_by_len"]) >= 0.4]
    res = {"seq_lens": list(seq_lens), "stable_induction_heads": induction, "length_dependent_induction_heads": positional,
           "previous_token_heads": prev_tok}
    store.record("model:head_catalog_sweep", {"seq_lens": list(seq_lens)}, s.identity(), res, None,
                 ["a head scoring as induction at one length only is a positional-shortcut suspect, not a content-based induction head",
                  "a shortcut tied to a length outside the sweep is invisible here"],
                 "inference", s.analysis_profile(), s.facets, untrusted=False, cacheable=False)
    log(f"  induction heads (stable): {[r['head'] for r in induction][:10]} | length-dependent: {[r['head'] for r in positional][:10]} | "
        f"prev-token: {[r['head'] for r in prev_tok][:10]}")
    return res


def cross_task(results: dict[str, dict[str, Any]], k: int = 20) -> dict[str, Any]:
    tops = {t: {h for h, _ in r["necessity"]["top_heads"][:k]} if "necessity" in r else set() for t, r in results.items()}
    names = sorted(tops)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = tops[a] & tops[b]
            rows.append({"tasks": [a, b], "jaccard_top_heads": round(len(inter) / max(1, len(tops[a] | tops[b])), 3), "shared": sorted(inter)})
    return {"k": k, "pairs": rows}
