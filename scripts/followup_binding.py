"""Which ticket does "this ticket" mean? Follow-up entity binding in an agentic tool-calling setting, across models.

Turn 1: "What's the status of PROJ-4xxx?"  ->  the assistant calls get_ticket(id="PROJ-4xxx") and answers.
Turn 2: "Who is assigned to this ticket?"  ->  the assistant must call get_ticket(id="PROJ-4xxx") again.
We teacher-force the tool-call prefix  get_ticket(id="PROJ-  and score the next token (the ticket number's first token).

Conditions
  wording            five phrasings of the follow-up (this ticket / that one / it / the same ticket / "And the priority?")
  distance           1, 3, 6 unrelated Q&A turns between the ticket and the follow-up
  competing_before   another ticket discussed BEFORE the target (recency agrees with the target)
  competing_after    another ticket discussed AFTER the target (recency disagrees: "this ticket" is ambiguous)
  pipeline           in the competing_after case: (a) "Active ticket: X" in the system prompt, (b) the follow-up rewritten
                     with the explicit id
Mechanistic, per model: heads reading the target id vs the competitor id at the answer position, split by layer type;
logit lens (first layer where the target id is on top); direct logit attribution (target vs competitor).

Usage:  python scripts/followup_binding.py --models google/gemma-4-12B-it,Qwen/Qwen3-8B,microsoft/phi-4 --out /content/binding
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import ModelSession, analysis  # noqa: E402

TICKETS = [f"PROJ-{(i % 9) + 1}{(i * 37) % 10}{(i * 53) % 10}{(i * 71) % 10}" for i in range(20)]   # first digit cycles 1..9
PEOPLE = ["Dana", "Marcus", "Priya", "Tomas", "Aisha", "Leo", "Mei", "Jonas", "Sara", "Diego"]
STATUSES = ["In Progress", "Open", "Blocked", "In Review"]
TITLES = ["Checkout page crashes on Safari", "Password reset email not delivered", "Export to CSV times out", "Dark mode toggle ignored",
          "Search returns deleted items", "Webhook retries fire twice", "Invoice PDF shows wrong currency", "Mobile app logs out randomly"]
FILLER_QA = [("What's the difference between a bug and an incident?", "A bug is a defect in the product; an incident is a live disruption of service that needs a response."),
             ("How do I link two tickets?", "Open either ticket, choose \"Link issue\" and pick the relationship, for example \"blocks\" or \"relates to\"."),
             ("What does the P1 priority mean?", "P1 means a critical issue affecting many users with no workaround; it needs a response within one hour."),
             ("Can I set a reminder for a sprint review?", "Yes, open the sprint board, choose \"Add reminder\" and pick the date and the people to notify."),
             ("What is a release branch?", "A release branch holds the code for a specific release so fixes can land there without taking in new features."),
             ("How long are logs retained?", "Application logs are retained for 30 days; audit logs for one year.")]
WORDINGS = {"this_ticket": "Who is assigned to this ticket?", "that_one": "What's the priority of that one?", "it": "When was it created?",
            "same_ticket": "Can you add a comment to the same ticket?", "no_noun": "And the priority?"}
SYSTEM = ("You are a support assistant with access to a ticket database. To look up a ticket, write a tool call on its own line "
          "exactly in the form get_ticket(id=\"<ticket id>\") before answering.")
PREFIX = 'get_ticket(id="PROJ-'


def build(target: str, wording: str, n_filler: int, competing: str | None, where: str, pipeline: str, rng: random.Random):
    sysmsg = SYSTEM + (f"\nActive ticket: {target}" if pipeline == "entity_slot" else "")
    t_title, t_person, t_status = rng.choice(TITLES), rng.choice(PEOPLE), rng.choice(STATUSES)
    comp_turns = [{"role": "user", "content": f"Who opened {competing}?"},
                  {"role": "assistant", "content": f'get_ticket(id="{competing}")\n{competing} ({rng.choice(TITLES)}) was opened by {rng.choice(PEOPLE)}.'}] if competing else []
    msgs = [{"role": "system", "content": sysmsg}]
    if competing and where == "before":
        msgs += comp_turns
    msgs += [{"role": "user", "content": f"What's the status of {target}?"},
             {"role": "assistant", "content": f'get_ticket(id="{target}")\n{target} ({t_title}) is {t_status} and assigned to {t_person}.'}]
    for q, a in rng.sample(FILLER_QA, min(n_filler, len(FILLER_QA))):
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if competing and where == "after":
        msgs += comp_turns
    follow = f"Who is assigned to {target}?" if pipeline == "rewritten" else WORDINGS[wording]
    msgs.append({"role": "user", "content": follow})
    return msgs


def _template(tok, msgs) -> str:
    try:  # Qwen3-style templates: no thinking block before the tool call
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render(tok, msgs) -> str:
    if getattr(tok, "chat_template", None):
        try:
            return _template(tok, msgs) + PREFIX
        except Exception:  # templates without a system role: fold it into the first user turn
            m2 = [dict(m) for m in msgs if m["role"] != "system"]
            m2[0]["content"] = msgs[0]["content"] + "\n\n" + m2[0]["content"]
            return _template(tok, m2) + PREFIX
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant: " + PREFIX


def number(t: str) -> str:
    return t.split("-")[1]


@torch.no_grad()
def score(s: ModelSession, text: str, target: str, competing: str | None):
    ids = s.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
    logits = s.logits_only(ids)[0, -1]
    first = lambda t: s.tok(number(t), add_special_tokens=False)["input_ids"][0]
    c = first(target)
    others = sorted({first(t) for t in TICKETS if first(t) != c})
    top = int(logits.argmax())
    return {"correct": top == c, "top_token": s.tok.decode([top]), "margin": round(float(logits[c] - logits[others].max()), 4),
            "picked_competitor": None if not competing else top == first(competing), "n_tokens": ids.shape[1]}


def run_model(model_id: str, a, log) -> dict:
    s = ModelSession()
    s.open(model_id, dtype=a.dtype, device=a.device, hash_weights="none")
    f = s.facts()
    lt_counts = {t: (f["layer_types"] or []).count(t) for t in sorted(set(f["layer_types"] or []))} or None
    log(f"\n=== {model_id}: {f['n_layers']}L {f['n_heads']}H, layer types {lt_counts}, window {f['sliding_window']}")
    rng = random.Random(0)
    fillers = [int(x) for x in a.fillers.split(",")]
    conditions = [("wording", w, 0, None, "after", "history") for w in WORDINGS] + \
                 [("distance", "this_ticket", nf, None, "after", "history") for nf in fillers if nf > 0] + \
                 [("competing_before", "this_ticket", nf, True, "before", "history") for nf in (1, 3)] + \
                 [("competing_after", "this_ticket", nf, True, "after", "history") for nf in (1, 3)] + \
                 [("pipeline", "this_ticket", 3, True, "after", p) for p in ("entity_slot", "rewritten")]
    rows = []
    for kind, wording, nf, comp, where, pipe in conditions:
        res = []
        for i in range(a.n):
            target = TICKETS[i % len(TICKETS)]
            competing = TICKETS[(i + 7) % len(TICKETS)] if comp else None
            r = score(s, render(s.tok, build(target, wording, nf, competing, where, pipe, rng)), target, competing)
            res.append(r | {"target": target, "competing": competing})
        acc = sum(r["correct"] for r in res) / len(res)
        pc = (sum(bool(r["picked_competitor"]) for r in res) / len(res)) if comp else None
        rows.append({"condition": kind, "wording": wording, "fillers": nf, "competing": where if comp else None, "pipeline": pipe,
                     "accuracy": round(acc, 3), "mean_margin": round(sum(r["margin"] for r in res) / len(res), 3), "picked_competitor": pc,
                     "mean_tokens": round(sum(r["n_tokens"] for r in res) / len(res)),
                     "wrong_examples": [f"{r['target']} -> '{r['top_token']}'" for r in res if not r["correct"]][:3], "_cases": res})
        log(f"  {kind:16s} {wording:12s} fillers={nf} competing={str(where if comp else None):6s} {pipe:11s}: acc {acc:.2f} margin {rows[-1]['mean_margin']:+.2f}"
            + (f" picked_competitor {pc:.2f}" if comp else "") + (f" wrong {rows[-1]['wrong_examples'][:2]}" if acc < 1 else ""))

    # mechanistic: A = hardest succeeding condition, B = competing_after (ambiguous)
    n_layers, n_heads = s.s["n_layers"], s.s["n_heads"]
    lt = s.s["layer_types"] or ["full_attention"] * n_layers

    def analyse(target_row, n_cases):
        att_t, att_c = {"ok": torch.zeros(n_layers, n_heads), "no": torch.zeros(n_layers, n_heads)}, {"ok": torch.zeros(n_layers, n_heads), "no": torch.zeros(n_layers, n_heads)}
        cnt = {"ok": 0, "no": 0}
        lens = {"ok": [], "no": []}
        dla = []
        rng2 = random.Random(1)
        comp_where = target_row["competing"]
        for i in range(n_cases):
            target = TICKETS[i % len(TICKETS)]
            competing = TICKETS[(i + 7) % len(TICKETS)] if comp_where else None
            text = render(s.tok, build(target, target_row["wording"], target_row["fillers"], competing, comp_where or "after", "history", rng2))
            ids = s.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
            seq = ids[0].tolist()

            def positions(t):
                tid = s.tok(number(t), add_special_tokens=False)["input_ids"]
                occ = [j for j in range(len(seq) - len(tid) + 1) if seq[j: j + len(tid)] == tid]
                return sorted({p for o in occ for p in range(o, o + len(tid))})
            tp, cp = positions(target), (positions(competing) if competing else [])
            res = s.run(ids, components=("resid",), want_attn=True)
            ok = "ok" if int(res.logits[0, -1].argmax()) == s.tok(number(target), add_special_tokens=False)["input_ids"][0] else "no"
            A = torch.stack([res.attentions[l][0, :, -1].float() for l in range(n_layers)]).cpu()
            att_t[ok] += A[:, :, tp].sum(-1) if tp else 0
            att_c[ok] += A[:, :, cp].sum(-1) if cp else 0
            cnt[ok] += 1
            tid0 = s.tok(number(target), add_special_tokens=False)["input_ids"][0]
            lens[ok].append([int(s.unembed(res.resid_post[l][0, -1]).argmax()) == tid0 for l in range(n_layers)])
            if len(dla) < a.mech_cases:
                cnt_t = competing if competing else TICKETS[(i + 3) % len(TICKETS)]
                try:
                    d = analysis.direct_logit_attribution(s, text, number(target), number(cnt_t), add_special_tokens=False, top_k=8)
                    name = lambda c: c["component"] if c["layer"] is None else f"L{c['layer']}" + (f"H{c['head']}" if c["head"] is not None else f".{c['component']}")
                    heads_sum = sum(c["contribution"] for c in d["components"] if c["component"] == "head")
                    dla.append({"target": target, "competitor": cnt_t, "correct": ok == "ok", "logit_diff": d["actual_target_value"],
                                "attention_share": round(heads_sum / (d["actual_target_value"] + 1e-9), 3),
                                "top_positive": [(name(c), c["contribution"]) for c in d["top_positive"][:5]], "top_negative": [(name(c), c["contribution"]) for c in d["top_negative"][:4]]})
                except Exception as e:  # noqa: BLE001
                    dla.append({"target": target, "error": f"{type(e).__name__}: {e}"})

        def top(M, n):
            M = M / max(n, 1)
            fl = M.flatten().topk(min(6, M.numel()))
            return [{"head": f"L{int(i) // n_heads}H{int(i) % n_heads}", "type": lt[int(i) // n_heads], "attention": round(float(v), 3)} for v, i in zip(fl.values, fl.indices) if float(v) > 0]
        by_type = lambda M, n: {t: round(float(sum(M[l].max() for l in range(n_layers) if lt[l] == t) / max(n, 1)), 3) for t in sorted(set(lt))}
        first = lambda LL: sorted(next((l for l, v in enumerate(x) if v and all(x[l:])), n_layers) for x in LL)
        return {"condition": {k: target_row[k] for k in ("condition", "wording", "fillers", "competing")}, "n_success": cnt["ok"], "n_failure": cnt["no"],
                "target_heads_success": top(att_t["ok"], cnt["ok"]), "target_heads_failure": top(att_t["no"], cnt["no"]),
                "competitor_heads_success": top(att_c["ok"], cnt["ok"]), "competitor_heads_failure": top(att_c["no"], cnt["no"]),
                "max_attention_on_target_by_layer_type": {"success": by_type(att_t["ok"], cnt["ok"]), "failure": by_type(att_t["no"], cnt["no"])},
                "lens_first_layer_target_on_top": {"success": first(lens["ok"]), "failure": first(lens["no"])}, "dla": dla}

    hist = [r for r in rows if r["pipeline"] == "history"]
    ok_rows = [r for r in hist if r["accuracy"] > 0]
    A_row = sorted(ok_rows, key=lambda r: (r["accuracy"], -r["fillers"]))[0] if ok_rows else hist[0]
    B_row = next((r for r in hist if r["condition"] == "competing_after" and r["fillers"] == 3), None)
    mech = {"A_hardest_success": analyse(A_row, min(a.mech_cases * 2, a.n))}
    if B_row is not None:
        mech["B_competing_after"] = analyse(B_row, min(a.mech_cases * 2, a.n))
    for k, m in mech.items():
        log(f"  [{k}] {m['condition']} ok={m['n_success']} fail={m['n_failure']} | target heads(ok) {m['target_heads_success'][:3]} | "
            f"competitor heads(fail) {m['competitor_heads_failure'][:3]} | lens {m['lens_first_layer_target_on_top']}")
    out = {"model": model_id, "facts": {k: f[k] for k in ("n_layers", "n_heads", "layer_types", "sliding_window", "quirks", "parameters_human")},
           "grid": [{k: v for k, v in r.items() if k != "_cases"} for r in rows], "per_case": {f"{r['condition']}|{r['wording']}|{r['fillers']}|{r['competing']}|{r['pipeline']}": r["_cases"] for r in rows},
           "mechanistic": mech}
    s.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemma-4-12B-it,Qwen/Qwen3-8B,microsoft/phi-4")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--fillers", default="1,3,6")
    ap.add_argument("--mech-cases", type=int, default=6)
    ap.add_argument("--out", default="./followup-binding")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    results = {}
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        t0 = time.time()
        try:
            results[m] = run_model(m, a, log)
            results[m]["elapsed_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            log(f"  !! {m} failed: {type(e).__name__}: {e}")
            results[m] = {"model": m, "error": f"{type(e).__name__}: {e}"}
        json.dump(results, open(os.path.join(a.out, "followup_binding.json"), "w"), indent=1, ensure_ascii=False)
    # cross-model table
    conds = None
    md = ["# Follow-up binding across models", "", "accuracy = the next token after `get_ticket(id=\"PROJ-` is the target ticket's number", ""]
    ok_models = [m for m, r in results.items() if "grid" in r]
    if ok_models:
        conds = [(r["condition"], r["wording"], r["fillers"], r["competing"], r["pipeline"]) for r in results[ok_models[0]]["grid"]]
        md += ["| condition | wording | fillers | competing | pipeline | " + " | ".join(m.split("/")[-1] for m in ok_models) + " |",
               "|---|---|---|---|---|" + "---|" * len(ok_models)]
        for c in conds:
            cells = []
            for m in ok_models:
                r = next((x for x in results[m]["grid"] if (x["condition"], x["wording"], x["fillers"], x["competing"], x["pipeline"]) == c), None)
                cells.append("–" if r is None else f"{r['accuracy']:.2f} ({r['mean_margin']:+.1f})")
            md.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | " + " | ".join(cells) + " |")
        for m in ok_models:
            r = results[m]
            md += ["", f"## {m}: {r['facts']['parameters_human']}, {r['facts']['n_layers']} layers, window {r['facts']['sliding_window']}"]
            for k, mm in r["mechanistic"].items():
                md += [f"- **{k}** {mm['condition']} (ok {mm['n_success']} / fail {mm['n_failure']})",
                       f"  - target-reading heads (ok): {mm['target_heads_success'][:4]}",
                       f"  - competitor-reading heads (fail): {mm['competitor_heads_failure'][:4]}",
                       f"  - max attention on target by layer type: {mm['max_attention_on_target_by_layer_type']}",
                       f"  - logit lens first layer target on top: {mm['lens_first_layer_target_on_top']}",
                       f"  - DLA: {mm['dla'][:2]}"]
    failed = [m for m in results if m not in ok_models]
    if failed:
        md += ["", "Failed: " + ", ".join(f"{m} ({results[m]['error']})" for m in failed)]
    open(os.path.join(a.out, "followup_binding.md"), "w").write("\n".join(md))
    log("\n".join(md[:5 + (len(conds) if conds else 0)]))
    log(f"wrote {a.out}/followup_binding.md")


if __name__ == "__main__":
    main()
