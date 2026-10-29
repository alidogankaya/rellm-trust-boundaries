"""Predict factual hallucinations before generation — mechanistic features vs published baselines.

Data: "capital of <country>" for ~100 real countries (well known to obscure) plus fictional countries (any answer is a
hallucination). Label = 1 if the model's greedy answer does not start with the true capital.

Predictors, all computed from ONE forward pass over the prompt, before any token is generated:
  baselines (published)
    max_prob         confidence of the first answer token
    entropy          entropy of the next-token distribution
    probe_last       logistic regression on the last-layer residual at the last position (SAPLMA-style)
    probe_best       same, at the best layer chosen inside the training folds
  mechanistic (ours, 8 features, from the circuit found by rellm)
    movers           the 3 heads that attend most from the last position to the subject in the upper layers (found
                     automatically on correctly answered prompts)
    extraction       attention mass of the movers on the subject tokens; norm of their output
    enrichment       what the movers WOULD copy from the subject: OV projection of the subject residual through each
                     mover, unembedded -> max prob and entropy ("does the subject representation contain an answer?")
    handoff          logit-lens confidence just before and after the handoff layer
Evaluation: 5-fold cross-validated AUROC; transfer (train on one template, test on the other); few-shot (32 labels);
error typing: enrichment failure (the subject carries no clear answer) vs extraction failure (it does, movers miss it).

Usage (Colab):  python scripts/hallucination_probe.py --model Qwen/Qwen2.5-1.5B --out /content/halluc-1.5b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import ModelSession  # noqa: E402

REAL = [("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"), ("Japan", "Tokyo"), ("China", "Beijing"),
        ("Egypt", "Cairo"), ("Russia", "Moscow"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Norway", "Oslo"),
        ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
        ("Hungary", "Budapest"), ("Czechia", "Prague"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Turkey", "Ankara"),
        ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Jordan", "Amman"), ("Israel", "Jerusalem"),
        ("Saudi Arabia", "Riyadh"), ("Qatar", "Doha"), ("Oman", "Muscat"), ("Yemen", "Sanaa"), ("Kuwait", "Kuwait"), ("India", "New Delhi"),
        ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Bhutan", "Thimphu"), ("Bangladesh", "Dhaka"),
        ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"), ("Myanmar", "Naypyidaw"),
        ("Malaysia", "Kuala Lumpur"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Mongolia", "Ulaanbaatar"), ("South Korea", "Seoul"),
        ("North Korea", "Pyongyang"), ("Kazakhstan", "Astana"), ("Uzbekistan", "Tashkent"), ("Kyrgyzstan", "Bishkek"), ("Tajikistan", "Dushanbe"),
        ("Turkmenistan", "Ashgabat"), ("Azerbaijan", "Baku"), ("Armenia", "Yerevan"), ("Georgia", "Tbilisi"), ("Ukraine", "Kyiv"),
        ("Belarus", "Minsk"), ("Lithuania", "Vilnius"), ("Latvia", "Riga"), ("Estonia", "Tallinn"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
        ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Slovenia", "Ljubljana"), ("Slovakia", "Bratislava"), ("Albania", "Tirana"),
        ("North Macedonia", "Skopje"), ("Moldova", "Chisinau"), ("Iceland", "Reykjavik"), ("Malta", "Valletta"), ("Cyprus", "Nicosia"),
        ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Haiti", "Port-au-Prince"),
        ("Guatemala", "Guatemala City"), ("Honduras", "Tegucigalpa"), ("Nicaragua", "Managua"), ("Costa Rica", "San Jose"), ("Panama", "Panama City"),
        ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Ecuador", "Quito"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Argentina", "Buenos Aires"),
        ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"), ("Brazil", "Brasilia"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
        ("Fiji", "Suva"), ("Samoa", "Apia"), ("Tonga", "Nuku'alofa"), ("Vanuatu", "Port Vila"), ("Kiribati", "Tarawa"), ("Tuvalu", "Funafuti"),
        ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"), ("Ethiopia", "Addis Ababa"),
        ("Kenya", "Nairobi"), ("Uganda", "Kampala"), ("Rwanda", "Kigali"), ("Burundi", "Gitega"), ("Tanzania", "Dodoma"), ("Somalia", "Mogadishu"),
        ("Eritrea", "Asmara"), ("Djibouti", "Djibouti"), ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Senegal", "Dakar"), ("Mali", "Bamako"),
        ("Niger", "Niamey"), ("Chad", "N'Djamena"), ("Burkina Faso", "Ouagadougou"), ("Guinea", "Conakry"), ("Liberia", "Monrovia"),
        ("Sierra Leone", "Freetown"), ("Togo", "Lome"), ("Benin", "Porto-Novo"), ("Cameroon", "Yaounde"), ("Gabon", "Libreville"),
        ("Angola", "Luanda"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Mozambique", "Maputo"), ("Malawi", "Lilongwe"),
        ("Botswana", "Gaborone"), ("Namibia", "Windhoek"), ("Madagascar", "Antananarivo"), ("Mauritania", "Nouakchott"), ("Eswatini", "Mbabane"),
        ("Lesotho", "Maseru"), ("Comoros", "Moroni"), ("Mauritius", "Port Louis"), ("Seychelles", "Victoria"), ("Cape Verde", "Praia")]
FICTIONAL = ["Velorania", "Brastovia", "Quenmark", "Zalvoria", "Ostrenia", "Felmonia", "Darvistan", "Krelland", "Movaria", "Tessalind",
             "Orvania", "Glimmora", "Pentravia", "Xandoria", "Lumeria", "Borsland", "Cavellia", "Rothmere", "Siltaria", "Varnholt"]
TEMPLATES = ["The capital of {c} is", "Q: What is the capital city of {c}?\nA:"]


def build_items():
    items = []
    for t_i, t in enumerate(TEMPLATES):
        for c, cap in REAL:
            items.append({"country": c, "gold": cap, "fictional": False, "template": t_i, "prompt": t.format(c=c)})
        for c in FICTIONAL:
            items.append({"country": c, "gold": None, "fictional": True, "template": t_i, "prompt": t.format(c=c)})
    return items


def norm(x: str) -> str:
    return "".join(ch for ch in x.lower() if ch.isalnum())


# ------------------------------------------------------------------------------------------- small ML
def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def logreg_fit_predict(Xtr, ytr, Xte, l2=1.0, iters=200):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    y = ytr.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, y) + l2 * (w * w).sum() / len(y)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(Xte @ w + b)


def cv_auroc(X, y, folds=5, seed=0, l2=1.0):
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    pred = torch.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = [i for i in idx if i not in set(te)]
        pred[te] = logreg_fit_predict(X[tr], y[tr], X[te], l2)
    return auroc(pred.tolist(), y.tolist()), pred


# ------------------------------------------------------------------------------------------ features
@torch.no_grad()
def collect(s: ModelSession, items, gen_tokens: int = 8):
    L, H = s.s["n_layers"], s.s["n_heads"]
    n_kv = s.s["n_kv_heads"]
    group = H // n_kv
    rows = []
    for it in items:
        ids = s.tok(it["prompt"], return_tensors="pt")["input_ids"].to(s.device)
        toks = s.token_strings(ids[0].tolist())
        c_ids = s.tok(" " + it["country"], add_special_tokens=False)["input_ids"]
        seq = ids[0].tolist()
        subj = [i for i in range(len(seq) - len(c_ids) + 1) if seq[i: i + len(c_ids)] == c_ids]
        subj_pos = list(range(subj[-1], subj[-1] + len(c_ids))) if subj else [max(0, len(seq) - 2)]
        res = s.run(ids, components=("resid", "head_in"), want_attn=True)
        g = s.model.generate(ids, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=s.tok.pad_token_id or s.tok.eos_token_id)
        answer = s.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        wrong = 1 if it["fictional"] else int(not norm(answer).startswith(norm(it["gold"])))
        logits = res.logits[0, -1]
        lp = torch.log_softmax(logits, -1)
        lens = [float(torch.softmax(s.unembed(res.resid_post[l][0, -1]), -1).max()) for l in range(L)]
        att = torch.stack([res.attentions[l][0, :, -1, :].float() for l in range(L)])          # (L, H, seq)
        subj_att = att[:, :, subj_pos].sum(-1)                                                    # (L, H)
        head_norm = torch.stack([res.head_in[l][0, -1].float().view(H, -1).norm(dim=-1) for l in range(L)])  # (L, H), per-layer head size
        rows.append({"item": it, "answer": answer, "wrong": wrong, "toks": toks, "subj_pos": subj_pos,
                     "max_prob": float(lp.exp().max()), "entropy": float(-(lp.exp() * lp).sum()),
                     "resid_last": torch.stack([res.resid_post[l][0, -1].float().cpu() for l in range(L)]),
                     "resid_subj": torch.stack([res.resid_pre0[0, subj_pos[-1]].float().cpu()] +
                                               [res.resid_post[l][0, subj_pos[-1]].float().cpu() for l in range(L)]),
                     "lens": lens, "subj_att": subj_att.cpu(), "head_norm": head_norm.cpu()})
    return rows, group


@torch.no_grad()
def mover_ov(s: ModelSession, resid_subj_pre: torch.Tensor, layer: int, head: int, group: int) -> tuple[float, float]:
    """What head (layer, head) would write if it attended fully to the subject: W_O_h V_h(norm(x)) -> unembed."""
    attn = s.s["attn"][layer]
    x = s.s["layers"][layer].input_layernorm(resid_subj_pre.to(s.device, dtype=next(s.model.parameters()).dtype))
    d = s.s["d_head"]
    kv = head // group
    v = attn.v_proj(x)[kv * d:(kv + 1) * d]
    out = attn.o_proj.weight[:, head * d:(head + 1) * d] @ v
    p = torch.softmax(s.unembed(out, apply_norm=True), -1)
    return float(p.max()), float(-(p * p.clamp_min(1e-12).log()).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="./hallucination")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    s = ModelSession()
    s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="none")
    L, H = s.s["n_layers"], s.s["n_heads"]
    items = build_items()
    rows, group = collect(s, items)
    y = torch.tensor([r["wrong"] for r in rows])
    real = [r for r in rows if not r["item"]["fictional"]]
    log(f"{len(rows)} prompts | hallucination rate: real {sum(r['wrong'] for r in real)}/{len(real)}, fictional 100% by construction")

    # movers: upper-half heads with the most attention from the last position to the subject on CORRECT prompts
    correct = [r for r in rows if r["wrong"] == 0]
    mean_att = torch.stack([r["subj_att"] for r in correct]).mean(0) if correct else torch.zeros(L, H)
    mean_att[: L // 2] = -1
    flat = mean_att.flatten().topk(3).indices.tolist()
    movers = [(i // H, i % H) for i in flat]
    lens_mean = torch.tensor([r["lens"] for r in correct]).mean(0) if correct else torch.zeros(L)
    handoff = int((lens_mean[1:] - lens_mean[:-1]).argmax()) + 1
    log(f"movers (auto): {[f'L{l}H{h}' for l, h in movers]} | handoff layer L{handoff}")

    feats, names = [], []
    for r in rows:
        f = []
        for (l, h) in movers:
            f.append(float(r["subj_att"][l, h]))
            f.append(float(r["head_norm"][l, h]))
        ov = [mover_ov(s, r["resid_subj"][l], l, h, group) for (l, h) in movers]   # resid_subj[l] = input to layer l
        r["ov_maxprob"] = max(o[0] for o in ov)
        f += [r["ov_maxprob"], min(o[1] for o in ov), r["lens"][handoff - 1], r["lens"][min(handoff, L - 1)]]
        feats.append(f)
    names = [f"{k}_L{l}H{h}" for (l, h) in movers for k in ("subj_att", "out_norm")] + ["ov_maxprob", "ov_entropy", "lens_pre_handoff", "lens_post_handoff"]
    Xm = torch.tensor(feats)

    res = {}
    res["max_prob (no training)"] = auroc([-r["max_prob"] for r in rows], y.tolist())
    res["entropy (no training)"] = auroc([r["entropy"] for r in rows], y.tolist())
    Xlast = torch.stack([r["resid_last"][-1] for r in rows])
    res["probe_last (d=%d)" % Xlast.shape[1]] = cv_auroc(Xlast, y, l2=10.0)[0]
    layer_scores = {l: cv_auroc(torch.stack([r["resid_last"][l] for r in rows]), y, l2=10.0)[0] for l in range(0, L, max(1, L // 8))}
    best_l = max(layer_scores, key=layer_scores.get)
    res[f"probe_best (layer {best_l}, selected on test folds = optimistic)"] = layer_scores[best_l]
    res["mechanistic (10 features)"] = cv_auroc(Xm, y)[0]
    Xmp = torch.cat([Xm, torch.tensor([[r["max_prob"], r["entropy"]] for r in rows])], 1)
    res["mechanistic + confidence"] = cv_auroc(Xmp, y)[0]
    real_idx = [i for i, r in enumerate(rows) if not r["item"]["fictional"]]
    yr = y[real_idx]
    res_real = {"max_prob": auroc([-rows[i]["max_prob"] for i in real_idx], yr.tolist()),
                "probe_last": cv_auroc(Xlast[real_idx], yr, l2=10.0)[0], "mechanistic": cv_auroc(Xm[real_idx], yr)[0],
                "mechanistic + confidence": cv_auroc(Xmp[real_idx], yr)[0]}

    # transfer: train on template 0, test on template 1 (and reverse)
    tr = [i for i, r in enumerate(rows) if r["item"]["template"] == 0]
    te = [i for i, r in enumerate(rows) if r["item"]["template"] == 1]
    transfer = {}
    for name, X, l2 in [("probe_last", Xlast, 10.0), ("mechanistic", Xm, 1.0), ("mechanistic + confidence", Xmp, 1.0)]:
        transfer[name] = round((auroc(logreg_fit_predict(X[tr], y[tr], X[te], l2).tolist(), y[te].tolist()) +
                                auroc(logreg_fit_predict(X[te], y[te], X[tr], l2).tolist(), y[tr].tolist())) / 2, 4)
    transfer["max_prob (no training)"] = round(auroc([-rows[i]["max_prob"] for i in te], y[te].tolist()), 4)
    transfer = {k: (None if v != v else v) for k, v in transfer.items()}  # NaN -> None: one split had a single class

    # few-shot: 32 labelled examples, mean over 10 draws
    few = {}
    for name, X, l2 in [("probe_last", Xlast, 10.0), ("mechanistic", Xm, 1.0)]:
        vals = []
        for sd in range(10):
            idx = list(range(len(rows)))
            random.Random(sd).shuffle(idx)
            trn, tst = idx[:32], idx[32:]
            if len(set(y[trn].tolist())) < 2:
                continue
            vals.append(auroc(logreg_fit_predict(X[trn], y[trn], X[tst], l2).tolist(), y[tst].tolist()))
        few[name] = round(sum(vals) / len(vals), 4) if vals else None

    # error typing on real-country hallucinations
    thr = sorted(r["ov_maxprob"] for r in correct)[len(correct) // 10] if correct else 0.0
    types = {"enrichment failure (subject carries no clear answer)": [], "extraction failure (subject knows, movers miss)": []}
    for r in real:
        if r["wrong"]:
            k = "enrichment failure (subject carries no clear answer)" if r["ov_maxprob"] < thr else "extraction failure (subject knows, movers miss)"
            types[k].append(f"{r['item']['country']} -> '{r['answer'].strip()[:24]}' (gold {r['item']['gold']})")

    out = {"model": a.model, "n": len(rows), "hallucinations": int(y.sum()), "movers": [f"L{l}H{h}" for l, h in movers], "handoff_layer": handoff,
           "features": names, "auroc_all": {k: round(v, 4) for k, v in res.items()}, "auroc_real_countries_only": {k: round(v, 4) for k, v in res_real.items()},
           "transfer_template": transfer, "few_shot_32": few, "error_types": {k: {"count": len(v), "examples": v[:12]} for k, v in types.items()}}
    json.dump(out, open(os.path.join(a.out, "hallucination.json"), "w"), indent=1)
    md = [f"# Hallucination prediction — {a.model}", "", f"{out['n']} prompts, {out['hallucinations']} hallucinations "
          f"(real countries {sum(r['wrong'] for r in real)}/{len(real)}, fictional {len(rows) - len(real)}). Movers {out['movers']}, handoff L{handoff}.", "",
          "| predictor | AUROC all | AUROC real only |", "|---|---|---|"]
    md += [f"| {k} | {v:.3f} | |" for k, v in out["auroc_all"].items()]
    md += [f"| {k} (real only) | | {v:.3f} |" for k, v in out["auroc_real_countries_only"].items()]
    md += ["", "Transfer (train on one template, test on the other): " + ", ".join(f"{k} {v}" for k, v in transfer.items()),
           "", "Few-shot (32 labels): " + ", ".join(f"{k} {v}" for k, v in few.items()), "", "## Error types (real countries)"]
    for k, v in out["error_types"].items():
        md += [f"- **{k}**: {v['count']}"] + [f"  - {e}" for e in v["examples"]]
    open(os.path.join(a.out, "hallucination.md"), "w").write("\n".join(md))
    log("\n".join(md))
    s.close()


if __name__ == "__main__":
    main()
