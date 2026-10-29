"""Confirmation experiment: is few-shot sentiment classification in Qwen2.5 a "default label + override" computation?

Hypothesis (from the reverse runs): with the demonstrations ordered (positive, negative), the model prefers the most
recent demonstration label ("negative") from layer 0 on, and one mid-layer head (L14H5 in 0.5B) overrides it for
positive reviews.

Pre-registered predictions (written before this script was run on any model):
  P1 order flip: with demonstrations (negative, positive) the default flips to "positive": on neutral and empty
     reviews the preferred label follows the LAST demonstration label in both orders.
  P2 decision layers by label: in each order, reviews whose answer equals the last demonstration label are decided
     early (embedding/L0-L2); the other label is decided at a single mid layer.
  P3 override head: the head with the largest patching effect at the last position is the same head in both orders.
  P4 ablation asymmetry: removing that head drops accuracy for the NON-default label far more than for the default label,
     and which label suffers flips with the order.
  P5 early default source: in the first layers, attention from the last position to the most recent demonstration label
     token exceeds attention to the earlier one.
Each prediction is reported as SUPPORTED / NOT SUPPORTED by fixed criteria below; per-example results are saved so
paired tests (McNemar) can be computed.

Usage:  python scripts/sentiment_mechanism.py --model Qwen/Qwen2.5-0.5B --out /content/sentiment-0.5b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import ModelSession, analysis  # noqa: E402
from rellm.experiment import removed  # noqa: E402
from rellm.experiment.tasks import NEG, POS  # noqa: E402

DEMO_POS = ("An absolute delight, I loved every minute.", " positive")
DEMO_NEG = ("Far too long and nothing happens.", " negative")
NOUNS = ["movie", "film", "story", "acting", "plot", "ending", "soundtrack", "script"]
NEUTRAL = ["The film was released in March.", "The movie is two hours long.", "The story takes place in a small town.",
           "The plot follows a family.", "The script was written by two people.", "The soundtrack has twelve songs."]
CANDS = (" positive", " negative")


def prefix(order: str) -> str:
    demos = [DEMO_POS, DEMO_NEG] if order == "PN" else [DEMO_NEG, DEMO_POS]
    return "".join(f"Review: {r}\nSentiment:{lab}\n" for r, lab in demos)


def items(order: str):
    out = []
    for adj in POS + NEG:
        lab = " positive" if adj in POS else " negative"
        for n in NOUNS:
            out.append({"prompt": prefix(order) + f"Review: The {n} was {adj}.\nSentiment:", "answer": lab, "adj": adj, "noun": n})
    return out


@torch.no_grad()
def margins(s: ModelSession, prompts: list[str], ids_c: tuple[int, int]) -> list[float]:
    out = []
    for p in prompts:
        ids = s.tok(p, return_tensors="pt")["input_ids"].to(s.device)
        lg = s.logits_only(ids)[0, -1]
        out.append(float(lg[ids_c[0]] - lg[ids_c[1]]))  # >0 means " positive" preferred
    return out


@torch.no_grad()
def decision_layer(s: ModelSession, prompt: str, ids_c: tuple[int, int], answer_is_pos: bool) -> str:
    ids = s.tok(prompt, return_tensors="pt")["input_ids"].to(s.device)
    res = s.run(ids, components=("resid",))
    states = [res.resid_pre0] + [res.resid_post[l] for l in range(s.s["n_layers"])]
    ok = []
    for st in states:
        lg = s.unembed(st[0, -1])
        d = float(lg[ids_c[0]] - lg[ids_c[1]])
        ok.append(d > 0 if answer_is_pos else d < 0)
    dec = None
    for i in range(len(ok) - 1, -1, -1):
        if ok[i]:
            dec = i
        else:
            break
    return "never" if dec is None else ("embed" if dec == 0 else f"L{dec - 1}")


def bucket(layer: str, n_layers: int) -> str:
    if layer in ("embed", "never"):
        return layer
    L = int(layer[1:])
    return "early (<=L2)" if L <= 2 else ("mid" if L < n_layers - 2 else "late")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--out", default="./sentiment-mechanism")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    s = ModelSession()
    s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="none")
    L = s.s["n_layers"]
    c_ids = tuple(s.tok(c, add_special_tokens=False)["input_ids"][0] for c in CANDS)
    R: dict = {"model": a.model, "orders": {}}
    per_example: dict = {}
    for order in ("PN", "NP"):
        last_label = " negative" if order == "PN" else " positive"
        its = items(order)
        m = margins(s, [x["prompt"] for x in its], c_ids)
        correct = [(mm > 0) == (x["answer"] == " positive") for mm, x in zip(m, its)]
        neutral = margins(s, [prefix(order) + f"Review: {r}\nSentiment:" for r in NEUTRAL], c_ids)
        empty = margins(s, [prefix(order) + "Review:\nSentiment:"], c_ids)[0]
        dec = defaultdict(lambda: defaultdict(int))
        for x in its:
            d = decision_layer(s, x["prompt"], c_ids, x["answer"] == " positive")
            dec[x["answer"].strip()][bucket(d, L)] += 1
        # override head: patching at the last position, clean = non-default-label review, corrupted = same noun, default-label adjective
        non_default = " positive" if last_label == " negative" else " negative"
        nd_adj = POS if non_default == " positive" else NEG
        df_adj = NEG if non_default == " positive" else POS
        head_eff = defaultdict(list)
        n_pairs = 0
        for k in range(min(a.pairs, len(nd_adj))):
            clean = prefix(order) + f"Review: The movie was {nd_adj[k]}.\nSentiment:"
            for adj2 in df_adj:
                corr = prefix(order) + f"Review: The movie was {adj2}.\nSentiment:"
                if s.tok(clean, return_tensors="pt")["input_ids"].shape[1] == s.tok(corr, return_tensors="pt")["input_ids"].shape[1]:
                    r = analysis.activation_patching(s, clean, corr, non_default, last_label, component="head", positions=[-1])
                    for row in r["grid"]:
                        if row["normalized"] is not None:
                            head_eff[(row["layer"], row["head"])].append(row["normalized"])
                    n_pairs += 1
                    break
        mean_eff = {k: sum(v) / len(v) for k, v in head_eff.items()}
        top = sorted(mean_eff, key=lambda k: -mean_eff[k])[:5]
        oh = top[0] if top else None
        # ablation of the override head, accuracy by label
        abl = {}
        if oh is not None:
            with removed(s, heads={oh[0]: [oh[1]]}):
                m2 = margins(s, [x["prompt"] for x in its], c_ids)
            for lab in CANDS:
                idx = [i for i, x in enumerate(its) if x["answer"] == lab]
                abl[lab.strip()] = {"before": round(sum(correct[i] for i in idx) / len(idx), 3),
                                    "after": round(sum((m2[i] > 0) == (lab == " positive") for i in idx) / len(idx), 3)}
        # early-layer attention from the last position to the two demonstration label tokens
        p0 = its[0]["prompt"]
        ids = s.tok(p0, return_tensors="pt")["input_ids"].to(s.device)
        toks = ids[0].tolist()
        pos_tok = [i for i, t in enumerate(toks) if t == c_ids[0]]
        neg_tok = [i for i, t in enumerate(toks) if t == c_ids[1]]
        res = s.run(ids, components=(), want_attn=True)
        early = {}
        for l in range(min(3, L)):
            w = res.attentions[l][0, :, -1].float()
            early[f"L{l}"] = {"to_positive_demo": round(float(w[:, pos_tok].sum(-1).max()), 4) if pos_tok else None,
                              "to_negative_demo": round(float(w[:, neg_tok].sum(-1).max()), 4) if neg_tok else None}
        R["orders"][order] = {
            "last_demo_label": last_label.strip(), "accuracy": round(sum(correct) / len(correct), 4), "n": len(its),
            "neutral_margin_pos_minus_neg": [round(x, 3) for x in neutral], "neutral_prefers": "positive" if sum(neutral) > 0 else "negative",
            "empty_margin_pos_minus_neg": round(empty, 3), "empty_prefers": "positive" if empty > 0 else "negative",
            "decision_layers_by_label": {k: dict(v) for k, v in dec.items()},
            "override_head": f"L{oh[0]}H{oh[1]}" if oh else None, "top_heads": [(f"L{k[0]}H{k[1]}", round(mean_eff[k], 3)) for k in top],
            "patch_pairs": n_pairs, "ablation_accuracy_by_label": abl, "early_attention_to_demo_labels": early}
        per_example[order] = [{"adj": x["adj"], "noun": x["noun"], "answer": x["answer"].strip(), "margin": round(mm, 4), "correct": bool(c)}
                              for x, mm, c in zip(its, m, correct)]
    o = R["orders"]
    verdict = {}
    verdict["P1 default follows last demo label"] = all(o[k]["neutral_prefers"] == o[k]["last_demo_label"] and o[k]["empty_prefers"] == o[k]["last_demo_label"] for k in o)
    def early_share(order, lab):
        d = o[order]["decision_layers_by_label"].get(lab, {})
        n = sum(d.values()) or 1
        return (d.get("embed", 0) + d.get("early (<=L2)", 0)) / n
    verdict["P2 default label decided early, other label not"] = all(
        early_share(k, o[k]["last_demo_label"]) >= 0.8 and early_share(k, "positive" if o[k]["last_demo_label"] == "negative" else "negative") <= 0.2 for k in o)
    verdict["P3 same override head in both orders"] = o["PN"]["override_head"] is not None and o["PN"]["override_head"] == o["NP"]["override_head"]
    def asym(order):
        ab = o[order]["ablation_accuracy_by_label"]
        dflt = o[order]["last_demo_label"]
        other = "positive" if dflt == "negative" else "negative"
        if not ab:
            return False
        return (ab[other]["before"] - ab[other]["after"]) >= 0.3 and (ab[dflt]["before"] - ab[dflt]["after"]) <= 0.1
    verdict["P4 ablation hurts only the non-default label"] = asym("PN") and asym("NP")
    def att(order):
        e = o[order]["early_attention_to_demo_labels"]
        key_last = "to_negative_demo" if o[order]["last_demo_label"] == "negative" else "to_positive_demo"
        key_first = "to_positive_demo" if key_last == "to_negative_demo" else "to_negative_demo"
        return all((v[key_last] or 0) > (v[key_first] or 0) for v in e.values())
    verdict["P5 early attention prefers the most recent demo label"] = att("PN") and att("NP")
    R["verdict"] = {k: ("SUPPORTED" if v else "NOT SUPPORTED") for k, v in verdict.items()}
    R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(a.out, "sentiment_mechanism.json"), "w"), indent=1)
    json.dump(per_example, open(os.path.join(a.out, "per_example.json"), "w"), indent=1)
    md = [f"# Sentiment default + override — {a.model}", ""]
    for k in ("PN", "NP"):
        x = o[k]
        md += [f"## Order {k} (last demonstration: {x['last_demo_label']})",
               f"- accuracy {x['accuracy']} on {x['n']} reviews",
               f"- default on neutral reviews: **{x['neutral_prefers']}** (margins pos−neg {x['neutral_margin_pos_minus_neg']}); empty review: **{x['empty_prefers']}** ({x['empty_margin_pos_minus_neg']})",
               f"- decision layers by label: {x['decision_layers_by_label']}",
               f"- override head: **{x['override_head']}** (top patching heads {x['top_heads']}, {x['patch_pairs']} pairs)",
               f"- ablating it, accuracy by label: {x['ablation_accuracy_by_label']}",
               f"- early-layer attention from the last token to demo labels: {x['early_attention_to_demo_labels']}", ""]
    md += ["## Pre-registered predictions"] + [f"- {k}: **{v}**" for k, v in R["verdict"].items()]
    open(os.path.join(a.out, "sentiment_mechanism.md"), "w").write("\n".join(md))
    print("\n".join(md))
    s.close()


if __name__ == "__main__":
    main()
