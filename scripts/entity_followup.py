"""Follow-up entity binding: "X firmasının adresi nedir?" ... "Bu firmanın VKN'si ne?"

The model must carry the company name from turn 1 into the query it writes at turn 2. We teacher-force the
query prefix so the answer is a single token (the first token of the company name), then:

Behavioural grid (what fails):
  distance   filler turns pushed between the company mention and the follow-up (tests the sliding-window limit)
  wording    "Bu firmanın" / "Şirketin" / "Onun" / "Aynı firmanın" / "Peki VKN'si ne" (no noun at all)
  competing  a second company mentioned in between (does the model bind to the most recent entity?)
  pipeline   (a) full history, (b) rewritten standalone question, (c) explicit entity slot in the system prompt

Mechanistic (why it fails), on the hardest condition that still has successes:
  - copy heads: attention from the answer position to the company-name tokens, per head, split by layer type
    (sliding vs global); which heads copy the name in successes and what they look at in failures
  - logit lens: at which layer the name becomes the top candidate (success vs failure)
  - DLA: direct writers of the name vs the competitor name
  - token-swap and residual patching at the name position: how far the name representation survives

Usage:  python scripts/entity_followup.py --model google/gemma-4-12B-it --out /content/entity-12b
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

COMPANIES = ["Yıldız Metal Sanayi", "Akdeniz Lojistik", "Karadeniz Gıda", "Marmara Tekstil", "Anadolu Yazılım", "Ege Mobilya",
             "Doğu Enerji", "Kuzey İnşaat", "Güney Turizm", "Boğaziçi Kimya", "Toros Tarım", "Zafer Otomotiv", "Umut Sağlık",
             "Delta Bilişim", "Pınar Ambalaj", "Atlas Madencilik", "Meriç Elektrik", "Sahra Plastik", "Vega Ulaşım", "Nil Kozmetik"]
STREETS = ["Atatürk Caddesi No:12", "İstiklal Sokak No:5", "Cumhuriyet Bulvarı No:48", "Bağdat Caddesi No:101", "Sanayi Sitesi B Blok No:7"]
CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Kocaeli", "Antalya", "Konya", "Adana"]
FILLER_QA = [("Bugün hava nasıl?", "Hava bilgisine erişimim yok, ancak size başka konularda yardımcı olabilirim."),
             ("Ticaret sicili nedir, kısaca açıklar mısın?", "Ticaret sicili, tacirlerin ve ticari işletmelerin kayıtlarının tutulduğu resmî sicildir."),
             ("Limited şirket ile anonim şirket arasındaki fark ne?", "Anonim şirkette sermaye paylara bölünmüştür ve en az 50.000 TL'dir; limited şirkette ortak sayısı 50 ile sınırlıdır."),
             ("Sicil gazetesi ilanı ne kadar sürede yayımlanır?", "Tescil işleminden sonra genellikle birkaç iş günü içinde yayımlanır."),
             ("VKN nedir?", "VKN, vergi kimlik numarasıdır; vergi mükelleflerine Gelir İdaresi tarafından verilen 10 haneli numaradır."),
             ("MERSİS numarası kaç hanelidir?", "MERSİS numarası 16 hanelidir.")]
WORDINGS = {"bu_firmanin": "Bu firmanın VKN'si ne?", "sirketin": "Şirketin VKN'si nedir?", "onun": "Onun VKN'si ne?",
            "ayni_firmanin": "Aynı firmanın VKN'sini de verir misin?", "no_noun": "Peki VKN'si ne?"}
SYSTEM = ("Sen bir ticaret sicili asistanısın. Kullanıcı bir firma hakkında bilgi istediğinde, veritabanı sorgusunu şu formatta yaz: "
          "MATCH (c:Company {name: \"<firma adı>\"}) RETURN c.<alan>")
QUERY_PREFIX = 'MATCH (c:Company {name: "'


def build_messages(company: str, wording: str, n_filler: int, competing: str | None, pipeline: str, rng: random.Random,
                   competing_where: str = "after"):
    """competing_where: 'after' = the other company is the most recent entity (recency disagrees with the target);
    'before' = the other company was discussed first, the target last (recency agrees with the target)."""
    addr = f"{rng.choice(STREETS)}, {rng.choice(CITIES)}"
    system = SYSTEM + (f"\nAktif firma: {company}" if pipeline == "entity_slot" else "")
    comp_turns = [{"role": "user", "content": f"{competing} hangi şehirde?"},
                  {"role": "assistant", "content": f"{competing} {rng.choice(CITIES)} ilinde kayıtlıdır."}] if competing else []
    msgs = [{"role": "system", "content": system}]
    if competing and competing_where == "before":
        msgs += comp_turns
    msgs += [{"role": "user", "content": f"{company} firmasının adresi nedir?"},
             {"role": "assistant", "content": f'MATCH (c:Company {{name: "{company}"}}) RETURN c.address\n{company} adresi: {addr}'}]
    fillers = rng.sample(FILLER_QA, min(n_filler, len(FILLER_QA)))
    for q, a in fillers:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if competing and competing_where == "after":
        msgs += comp_turns
    follow = f"{company} firmasının VKN'si ne?" if pipeline == "rewritten" else WORDINGS[wording]
    msgs.append({"role": "user", "content": follow})
    return msgs


def render(tok, msgs) -> str:
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + QUERY_PREFIX
        except Exception:  # some templates refuse a system role: fold it into the first user turn
            m2 = [dict(m) for m in msgs if m["role"] != "system"]
            m2[0]["content"] = msgs[0]["content"] + "\n\n" + m2[0]["content"]
            return tok.apply_chat_template(m2, tokenize=False, add_generation_prompt=True) + QUERY_PREFIX
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant: " + QUERY_PREFIX


@torch.no_grad()
def score_case(s: ModelSession, text: str, company: str, competing: str | None):
    """margin = logit(first token of company) - max(logit of competing company first token, other company first tokens)."""
    ids = s.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
    logits = s.logits_only(ids)[0, -1]
    first = lambda name: s.tok(name, add_special_tokens=False)["input_ids"][0]
    c_id = first(company)
    others = [first(x) for x in ([competing] if competing else []) + [x for x in COMPANIES if x != company and x != competing]]
    others = [o for o in others if o != c_id]
    top = int(logits.argmax())
    margin = float(logits[c_id] - logits[others].max())
    return {"correct_top": top == c_id, "top_token": s.tok.decode([top]), "margin": round(margin, 4),
            "beats_competitor": None if not competing else bool(logits[c_id] > logits[first(competing)]), "n_tokens": ids.shape[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=16, help="cases per condition")
    ap.add_argument("--fillers", default="0,1,3,6")
    ap.add_argument("--mech-cases", type=int, default=6)
    ap.add_argument("--out", default="./entity-followup")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    t0 = time.time()
    s = ModelSession()
    s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="none")
    f = s.facts()
    lt_counts = {t: (f["layer_types"] or []).count(t) for t in sorted(set(f["layer_types"] or []))} or None
    log(f"opened {a.model}: {f['n_layers']}L {f['n_heads']}H, layer types {lt_counts}, sliding_window {f['sliding_window']}, quirks: {f['quirks']}")
    rng = random.Random(0)
    fillers = [int(x) for x in a.fillers.split(",")]

    # ---- behavioural grid
    rows = []
    conditions = [("wording", w, 0, None, "history") for w in WORDINGS] + \
                 [("distance", "bu_firmanin", nf, None, "history") for nf in fillers if nf > 0] + \
                 [("competing_before", "bu_firmanin", 1, "before", "history"), ("competing_before", "bu_firmanin", 3, "before", "history")] + \
                 [("competing_after", "bu_firmanin", 1, "after", "history"), ("competing_after", "bu_firmanin", 3, "after", "history")] + \
                 [("pipeline", "bu_firmanin", 3, "after", p) for p in ("rewritten", "entity_slot")]
    for kind, wording, nf, comp, pipe in conditions:
        res = []
        for i in range(a.n):
            company = COMPANIES[i % len(COMPANIES)]
            competing = COMPANIES[(i + 7) % len(COMPANIES)] if comp else None
            msgs = build_messages(company, wording, nf, competing, pipe, rng, competing_where=comp or "after")
            r = score_case(s, render(s.tok, msgs), company, competing)
            res.append(r | {"company": company, "competing": competing})
        acc = sum(r["correct_top"] for r in res) / len(res)
        comp_rate = (sum(1 for r in res if r["beats_competitor"] is False) / len(res)) if comp else None
        n_tok = round(sum(r["n_tokens"] for r in res) / len(res))
        rows.append({"condition": kind, "wording": wording, "fillers": nf, "competing": comp or False, "pipeline": pipe, "accuracy": round(acc, 3),
                     "mean_margin": round(sum(r["margin"] for r in res) / len(res), 3), "picked_competitor_rate": comp_rate, "mean_tokens": n_tok,
                     "examples_wrong": [f"{r['company']} -> '{r['top_token']}'" for r in res if not r["correct_top"]][:4], "_cases": res})
        log(f"  {kind:16s} {wording:14s} fillers={nf} competing={str(comp or False):6s} {pipe:11s}: acc {acc:.2f}  margin {rows[-1]['mean_margin']:+.2f}  "
            f"tokens~{n_tok}" + (f"  picked competitor {comp_rate:.2f}" if comp else "") + (f"  wrong: {rows[-1]['examples_wrong'][:2]}" if acc < 1 else ""))
    window = f["sliding_window"]
    beyond = [r for r in rows if r["condition"] == "distance" and window and r["mean_tokens"] > window]
    log(f"  sliding window {window}: {len(beyond)} distance conditions exceed it" if window else "  no sliding window")

    # ---- mechanistic: (A) hardest condition that still succeeds -> which heads copy the name; (B) competitor-after -> what wins
    n_layers, n_heads = s.s["n_layers"], s.s["n_heads"]
    lt = s.s["layer_types"] or ["full_attention"] * n_layers

    def analyse(target, n_cases):
        att_succ, att_fail = torch.zeros(n_layers, n_heads), torch.zeros(n_layers, n_heads)
        comp_succ, comp_fail = torch.zeros(n_layers, n_heads), torch.zeros(n_layers, n_heads)
        n_s = n_f = 0
        lens_s, lens_f, dla_rows = [], [], []
        rng2 = random.Random(1)
        for i in range(n_cases):
            company = COMPANIES[i % len(COMPANIES)]
            comp = target["competing"] or None
            competing = COMPANIES[(i + 7) % len(COMPANIES)] if comp else None
            text = render(s.tok, build_messages(company, target["wording"], target["fillers"], competing, "history", rng2, competing_where=comp or "after"))
            ids = s.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
            seq = ids[0].tolist()
            c_ids = s.tok(company, add_special_tokens=False)["input_ids"]
            occ = [j for j in range(len(seq) - len(c_ids) + 1) if seq[j: j + len(c_ids)] == c_ids]
            name_pos = sorted({p for o in occ for p in range(o, o + len(c_ids))})
            comp_pos = []
            if competing:
                k_ids = s.tok(competing, add_special_tokens=False)["input_ids"]
                occ2 = [j for j in range(len(seq) - len(k_ids) + 1) if seq[j: j + len(k_ids)] == k_ids]
                comp_pos = sorted({p for o in occ2 for p in range(o, o + len(k_ids))})
            res = s.run(ids, components=("resid",), want_attn=True)
            ok = int(res.logits[0, -1].argmax()) == c_ids[0]
            A = torch.stack([res.attentions[l][0, :, -1].float() for l in range(n_layers)]).cpu()
            m_name = A[:, :, name_pos].sum(-1)
            m_comp = A[:, :, comp_pos].sum(-1) if comp_pos else torch.zeros(n_layers, n_heads)
            lens = [int(s.unembed(res.resid_post[l][0, -1]).argmax()) == c_ids[0] for l in range(n_layers)]
            if ok:
                att_succ += m_name; comp_succ += m_comp; n_s += 1; lens_s.append(lens)
            else:
                att_fail += m_name; comp_fail += m_comp; n_f += 1; lens_f.append(lens)
            if len(dla_rows) < a.mech_cases:
                cnt = competing if competing else COMPANIES[(i + 3) % len(COMPANIES)]
                try:
                    d = analysis.direct_logit_attribution(s, text, company, cnt, add_special_tokens=False, top_k=8)
                    heads_sum = sum(c["contribution"] for c in d["components"] if c["component"] == "head")
                    dla_rows.append({"company": company, "competitor": cnt, "correct": ok, "logit_diff": d["actual_target_value"],
                                     "attention_share": round(heads_sum / (d["actual_target_value"] + 1e-9), 3),
                                     "top_positive": [(c["component"] if c["layer"] is None else f"L{c['layer']}" + (f"H{c['head']}" if c["head"] is not None else f".{c['component']}"), c["contribution"]) for c in d["top_positive"][:5]],
                                     "top_negative": [(c["component"] if c["layer"] is None else f"L{c['layer']}" + (f"H{c['head']}" if c["head"] is not None else f".{c['component']}"), c["contribution"]) for c in d["top_negative"][:4]]})
                except Exception as e:  # noqa: BLE001
                    dla_rows.append({"company": company, "error": f"{type(e).__name__}: {e}"})

        def top_heads(M, n):
            M = M / max(n, 1)
            flat = M.flatten().topk(min(8, M.numel()))
            return [{"head": f"L{int(i) // n_heads}H{int(i) % n_heads}", "layer_type": lt[int(i) // n_heads], "attention": round(float(v), 3)}
                    for v, i in zip(flat.values, flat.indices) if float(v) > 0]
        by_type = lambda M, n: {t: round(float(sum(M[l].max() for l in range(n_layers) if lt[l] == t) / max(n, 1)), 3) for t in sorted(set(lt))}
        first_layer = lambda LL: (sorted(next((l for l, v in enumerate(x) if v and all(x[l:])), n_layers) for x in LL) if LL else [])
        return {"condition": {k: target[k] for k in ("condition", "wording", "fillers", "competing")}, "n_success": n_s, "n_failure": n_f,
                "name_heads_success": top_heads(att_succ, n_s), "name_heads_failure": top_heads(att_fail, n_f),
                "competitor_heads_success": top_heads(comp_succ, n_s), "competitor_heads_failure": top_heads(comp_fail, n_f),
                "max_attention_on_name_by_layer_type": {"success": by_type(att_succ, n_s), "failure": by_type(att_fail, n_f)},
                "lens_first_layer_name_on_top": {"success": first_layer(lens_s), "failure": first_layer(lens_f)}, "dla": dla_rows}

    hist = [r for r in rows if r["pipeline"] == "history"]
    ok_conds = [r for r in hist if r["accuracy"] > 0]
    target_a = sorted(ok_conds, key=lambda r: (r["accuracy"], -r["fillers"]))[0] if ok_conds else hist[0]
    target_b = next((r for r in hist if r["condition"] == "competing_after" and r["fillers"] == 3), None)
    mech = {"A_hardest_success": analyse(target_a, min(a.mech_cases * 2, a.n))}
    if target_b:
        mech["B_competitor_after"] = analyse(target_b, min(a.mech_cases * 2, a.n))
    for k, m in mech.items():
        log(f"  [{k}] {m['condition']} successes {m['n_success']} failures {m['n_failure']}")
        log(f"     name heads (success): {m['name_heads_success'][:4]}")
        log(f"     competitor heads (failure): {m['competitor_heads_failure'][:4]}")
        log(f"     lens first layer name on top: {m['lens_first_layer_name_on_top']}")
    out = {"model": a.model, "facts": {k: f[k] for k in ("n_layers", "n_heads", "layer_types", "sliding_window", "quirks")},
           "reading_guide": "competing_before: recency and the target agree (the target was discussed last). competing_after: they disagree "
                            "(the other company is the most recent entity) - the model's choice there is a pragmatic reading, not necessarily an error.",
           "grid": [{k: v for k, v in r.items() if k != "_cases"} for r in rows], "mechanistic": mech, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(a.out, "entity_followup.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({f"{r['condition']}|{r['wording']}|{r['fillers']}|{r['competing']}|{r['pipeline']}": r["_cases"] for r in rows},
              open(os.path.join(a.out, "per_case.json"), "w"), indent=1, ensure_ascii=False)
    md = [f"# Entity follow-up — {a.model}", "", f"{f['n_layers']} layers, sliding window {f['sliding_window']}, "
          f"layer types {dict((t, (f['layer_types'] or []).count(t)) for t in set(f['layer_types'] or []))}", "",
          "| condition | wording | fillers | competing | pipeline | accuracy | mean margin | picked competitor | ~tokens | wrong examples |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['condition']} | {r['wording']} | {r['fillers']} | {r['competing']} | {r['pipeline']} | {r['accuracy']} | {r['mean_margin']:+.2f} | "
           f"{'' if r['picked_competitor_rate'] is None else r['picked_competitor_rate']} | {r['mean_tokens']} | {'; '.join(r['examples_wrong'][:2])} |" for r in rows]
    for k, m in mech.items():
        md += ["", f"## Mechanistic {k}: {m['condition']} ({m['n_success']} successes, {m['n_failure']} failures)",
               f"- heads reading the target name (successes): {m['name_heads_success'][:6]}",
               f"- heads reading the target name (failures): {m['name_heads_failure'][:6]}",
               f"- heads reading the competitor (successes / failures): {m['competitor_heads_success'][:3]} / {m['competitor_heads_failure'][:4]}",
               f"- max attention on the name by layer type: {m['max_attention_on_name_by_layer_type']}",
               f"- logit lens, first layer where the name is on top: {m['lens_first_layer_name_on_top']}",
               "- DLA (direct writers, target vs competitor):"] + [f"  - {d}" for d in m["dla"][:4]]
    open(os.path.join(a.out, "entity_followup.md"), "w").write("\n".join(md))
    log(f"wrote {a.out}/entity_followup.md ({time.time() - t0:.0f}s)")
    s.close()


if __name__ == "__main__":
    main()
