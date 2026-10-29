"""Markdown report: what the model does (Phase 1) and which pruning criterion preserves it (Phase 2)."""
from __future__ import annotations

from typing import Any


def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def reverse_md(model_facts: dict[str, Any], model_level: dict[str, Any] | None, results: dict[str, dict[str, Any]], cross: dict[str, Any] | None) -> str:
    f = model_facts
    L = [f"# Reverse report — {f.get('model_id')}", "",
         f"{f['n_layers']} layers · {f['n_heads']} heads ({f['n_kv_heads']} kv) · d_model {f['d_model']} · d_mlp {f['d_mlp']} · "
         f"{f['parameters_human']} params · quirks: {', '.join(f['quirks']) or 'none'}", ""]
    if model_level:
        L += ["## Model-level head roles (task independent, inference)",
              f"- stable induction heads: {', '.join(r['head'] for r in model_level['stable_induction_heads']) or 'none'}",
              f"- length-dependent 'induction' heads (positional-shortcut suspects): {', '.join(r['head'] for r in model_level['length_dependent_induction_heads']) or 'none'}",
              f"- previous-token heads: {', '.join(r['head'] for r in model_level['previous_token_heads']) or 'none'}", ""]
    for name, r in results.items():
        b = r["behaviour"]
        L += [f"## Task: {name} — {r['task']['description']}", "",
              f"Behaviour (observation): train acc {b['train']['acc']:.2f} · heldout {b['heldout']['acc']:.2f} (margin {b['heldout']['margin_mean']}) · "
              f"OOD {b['ood']['acc']:.2f} · scoring {b['train']['mode']}", ""]
        for w in r.get("warnings", []):
            L.append(f"> ⚠ {w}")
        if "logit_lens" in r:
            L += ["", f"**Where the decision forms** (logit lens, inference): decision-layer histogram {r['logit_lens']['decision_layer_histogram']}", "",
                  "| layer | answer top among candidates | mean candidate rank | mean answer prob |", "|---|---|---|---|"]
            L += [f"| {x['layer']} | {x['answer_top_among_candidates']} | {x['mean_candidate_rank']} | {x['mean_answer_prob']} |" for x in r["logit_lens"]["layers"]]
        t = r["truncation"]
        L += ["", f"**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **{t['sufficient_depth']}**", "",
              "| through | acc | margin mean | margin min |", "|---|---|---|---|"]
        L += [f"| {x['through']} | {x['acc']} | {x['margin_mean']} | {x['margin_min']} |" for x in t["rows"]]
        L += ["", "**Layer skip necessity** (observation, margin drop when one layer is skipped): " +
              ", ".join(f"L{x['layer']} {x['margin_drop']}" for x in r["layer_skip"])]
        n = r["necessity"]
        L += ["", f"**Removal necessity** (observation, base margin {n['base_margin_mean']}): top heads " +
              ", ".join(f"{k} {v}" for k, v in n["top_heads"][:10]) + f"; concentration {n['head_drop_concentration']}",
              "MLPs: " + ", ".join(f"{k} {v}" for k, v in n["mlps"].items()),
              "Heads whose removal *helps*: " + ", ".join(f"{k} {v}" for k, v in n["most_harmful_heads"] if v < 0)]
        if "patching" in r:
            p = r["patching"]
            L += ["", f"**Information flow** (activation patching over {p['n_pairs']} pairs, observation)",
                  "- sources (positions where clean/corrupted differ): " + "; ".join(
                      f"{x['source_pos_from_end']} ('{x['token_example']}', token-swap recovery {x['token_swap_recovery']}) matters through "
                      f"{x['source_matters_through']}, resid effect by layer {x['effect_by_layer']}"
                      for x in p["flow"]["sources"][:3]),
                  f"- last position carries the decision from {p['flow']['last_position_carries_decision_from']}; effect by layer {p['flow']['last_position_effect_by_layer']}",
                  "- heads at the last position: " + ", ".join(f"{k} {v}" for k, v in p["top_heads_at_last"][:8]),
                  "- MLP outputs: " + ", ".join(f"{k} {v}" for k, v in p["top_mlp"][:6])]
        if "dla" in r:
            d = r["dla"]
            L += ["", f"**Who writes the answer** (DLA, derived): logit diff {d['mean_logit_diff']} · attention {d['attention_share']:.0%} · MLP {d['mlp_share']:.0%} · "
                  f"max reconstruction err {d['max_reconstruction_err']}",
                  "- positive: " + ", ".join(f"{k} {v}" for k, v in d["top_positive"][:8]),
                  "- negative: " + ", ".join(f"{k} {v}" for k, v in d["top_negative"][:5])]
        if "key_head_attention" in r:
            L += ["", "**What the key heads read** (inference)", "", "| head | top positions (from end, token, weight) | mass on answer token | on counter token |", "|---|---|---|---|"]
            for x in r["key_head_attention"]:
                L.append(f"| {x['head']} | " + "; ".join(f"{a['pos_from_end']} '{a['token_example']}' {a['weight']}" for a in x["attends_to"]) +
                         f" | {x['mass_on_answer_token']} | {x['mass_on_counter_token']} |")
        if "roles" in r:
            L += ["", "**Component roles** (derived)"] + [f"- {k}: {', '.join(v)}" for k, v in r["roles"].items()]
        if "neurons" in r:
            L += ["", "**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer " +
                  ", ".join(f"L{x['layer']} {x['neurons_for_90pct']}/{r['neurons']['d_mlp']}" for x in r["neurons"]["per_layer"]),
                  "top: " + ", ".join(f"{x['neuron']} {x['attribution']}" for x in r["neurons"]["top_neurons"][:8])]
        if r.get("unknowns"):
            L += ["", f"Open unknowns recorded: {len(r['unknowns'])}"]
        L += ["", f"_elapsed {r['elapsed_s']} s · evidence ids: {', '.join(f'{k}={v[:14]}' for k, v in r['evidence_ids'].items())}_", ""]
    if cross and cross["pairs"]:
        L += ["## Cross-task circuit overlap (top heads by removal necessity)"] + \
             [f"- {' vs '.join(x['tasks'])}: Jaccard {x['jaccard_top_heads']} shared {', '.join(x['shared'][:10])}" for x in cross["pairs"]] + [""]
    return "\n".join(L)


def prune_md(task: str, pr: dict[str, Any]) -> str:
    L = [f"## Pruning — {task}" + (f" (healing {pr['heal_steps']} steps, all arms)" if pr["heal_steps"] else " (no healing)"), "",
         "| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |", "|---|---|---|---|---|---|---|---|---|"]
    for r in pr["rows"]:
        lvl = r["level"] if r["kind"] != "depth" else f"-{r['level']} layers"
        L.append(f"| {r['kind']} | {lvl} | {r['method']} | {r['params_body']} | {r['latency_ms']} | {_fmt(r['heldout_acc'])} | {_fmt(r['heldout_margin'])} | "
                 f"{_fmt(r['ood_acc'])} | {_fmt(r['general_nll'])} |")
    v = pr["verdict"]
    L += ["", f"**Verdict: {v['text']}** (accuracy threshold {v['threshold_pp']:.0%}; margin decides ties at ±{v['threshold_margin_rel']:.0%})", "", v["note"], "",
          "| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |", "|---|---|---|---|---|---|---|---|---|"]
    for c in v["comparisons"]:
        L.append(f"| {c['kind']} | {c['level']} | {_fmt(c['causal_heldout'])} / {_fmt(c['causal_margin'])} | {c['best_other']} | "
                 f"{_fmt(c['best_other_heldout'])} / {_fmt(c['best_other_margin'])} | {_fmt(c['delta_heldout'])} | {c['delta_margin_rel']:+.1%} | "
                 f"{_fmt(c['delta_ood'])} | **{c['outcome']}** |")
    return "\n".join(L) + "\n"
