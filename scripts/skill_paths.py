"""Skill Paths — does a workload-specific, causally selected layer-skip draft give higher acceptance in lossless
self-speculative decoding than the standard draft choices?

Self-speculative decoding: the draft is the same model with some layers skipped; the full model verifies the drafted
tokens in one pass. With greedy verification the output is identical to the full model (lossless); speed depends on
how often draft and full model agree (acceptance) and on how cheap the draft is.

For each workload (translation, JSON extraction, short writing, code):
  1. reference = full-model greedy continuation of each prompt (gen_len tokens)
  2. agreement(S) = fraction of continuation positions where the model with layers S skipped predicts the same next
     token as the full model, teacher-forced on the reference (exact for greedy speculative decoding, because a draft
     is always conditioned on the verified prefix)
  3. skip sets of equal size chosen on CALIBRATION prompts by
       causal_greedy  iteratively skip the layer that keeps agreement highest (interventional, workload-specific)
       causal_oneshot rank layers by single-skip agreement
       early_exit     skip the last n layers (LayerSkip / Kangaroo style draft without training)
       shortgpt       lowest block influence 1 - cos(in, out)
       random_search  best of K random skip sets, same evaluation budget as causal_greedy (stand-in for SWIFT's search)
       generic        causal_greedy calibrated on the union of all workloads (is per-workload specialisation needed?)
  4. on HELD-OUT prompts: agreement, exact tokens per verification round (greedy simulation), estimated speedup
  5. wall clock: HF assisted generation with a physically built draft, output checked token-for-token against plain greedy

Usage (Colab):
  python scripts/skill_paths.py --model Qwen/Qwen2.5-0.5B-Instruct --skip 6,9 --out /content/skill-0.5b
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
from rellm import ModelSession  # noqa: E402
from rellm.experiment import build_pruned, removed  # noqa: E402

SENTENCES = ["The weather is cold today.", "My brother works in a small bakery.", "We are going to the museum tomorrow.",
             "She reads a book every evening.", "The train arrives at eight o'clock.", "I would like a cup of coffee.",
             "The children are playing in the garden.", "This restaurant serves excellent fish.", "He forgot his keys at home.",
             "Our teacher speaks three languages.", "The shop closes early on Sunday.", "Please open the window.",
             "They bought a new car last week.", "The river is very long.", "I need to finish my homework.",
             "The concert was wonderful.", "My grandmother lives near the sea.", "The doctor is very busy today.",
             "We watched a film after dinner.", "The library is next to the bank.", "It is raining in the city.",
             "Can you help me with this box?", "The cat sleeps on the sofa.", "I am learning to play the piano.",
             "The meeting starts at ten.", "He drinks tea without sugar.", "The flowers smell sweet.", "She lost her umbrella.",
             "The bus is late again.", "We need more bread and milk.", "The mountain is covered with snow.", "I like walking in the park."]
PEOPLE = [("Anna", "Lisbon", 34), ("Marco", "Turin", 41), ("Yuki", "Osaka", 27), ("Ahmet", "Izmir", 52), ("Sofia", "Kraków", 23),
          ("Liam", "Dublin", 38), ("Chen", "Chengdu", 45), ("Fatima", "Rabat", 31), ("Lucas", "Porto", 29), ("Elena", "Seville", 36),
          ("Noah", "Geneva", 47), ("Aisha", "Nairobi", 25), ("Pedro", "Recife", 58), ("Ingrid", "Bergen", 33), ("Omar", "Amman", 40),
          ("Mila", "Riga", 22), ("Tom", "Leeds", 61), ("Sara", "Malmö", 30), ("Ivan", "Kazan", 44), ("Leila", "Tabriz", 28),
          ("Hugo", "Lyon", 39), ("Nina", "Graz", 26), ("Ken", "Sapporo", 50), ("Zara", "Lahore", 35), ("Emil", "Aarhus", 42),
          ("Rosa", "Cádiz", 57), ("Arjun", "Pune", 32), ("Olga", "Odesa", 48), ("Felix", "Bremen", 24), ("Maya", "Haifa", 37),
          ("Diego", "Cusco", 46), ("Lena", "Basel", 29)]
TOPICS = ["autumn leaves", "a busy train station", "learning to swim", "an old lighthouse", "morning coffee", "a rainy weekend",
          "a city at night", "growing tomatoes", "a long bicycle ride", "the first day of school", "a quiet library", "a mountain lake",
          "baking bread", "a crowded market", "a thunderstorm", "a small fishing village", "a winter forest", "a football match",
          "visiting grandparents", "a desert road", "a jazz concert", "planting a tree", "a lost dog", "a summer festival",
          "an empty beach", "a broken clock", "a new neighbour", "the night sky", "a museum visit", "a ferry crossing",
          "a snowy morning", "a garden party"]
CODE = ["returns the square of a number", "reverses a string", "checks whether a number is even", "returns the largest item in a list",
        "counts the vowels in a string", "converts Celsius to Fahrenheit", "sums all numbers in a list", "removes duplicates from a list",
        "returns the factorial of n", "checks whether a string is a palindrome", "returns the first n Fibonacci numbers",
        "merges two dictionaries", "returns the average of a list", "capitalises every word in a sentence", "counts words in a text",
        "returns the minimum of three numbers", "checks whether a year is a leap year", "flattens a list of lists",
        "returns the length of the longest word", "converts a list of strings to integers", "returns every second element of a list",
        "checks whether a list is sorted", "returns the number of digits in an integer", "joins a list of words with commas",
        "returns the absolute difference of two numbers", "filters out negative numbers", "computes the area of a circle",
        "returns the last element of a list", "squares every number in a list", "returns True if a string is empty",
        "counts how often a value appears in a list", "returns the median of a sorted list"]

WORKLOADS = {
    "translate": [f"Translate to French: {x}" for x in SENTENCES],
    "json": [f"Extract the name, city and age as JSON. Text: {n} is {a} years old and lives in {c}." for n, c, a in PEOPLE],
    "write": [f"Write two sentences about {t}." for t in TOPICS],
    "code": [f"Write a short Python function that {t}." for t in CODE],
}


def render(tok, prompt: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt + "\n"


@torch.no_grad()
def references(s: ModelSession, prompts: list[str], gen_len: int) -> list[tuple[list[int], int]]:
    out = []
    for p in prompts:
        ids = s.tok(render(s.tok, p), return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
        g = s.model.generate(ids, max_new_tokens=gen_len, do_sample=False, pad_token_id=s.tok.pad_token_id or s.tok.eos_token_id)
        out.append((g[0].tolist(), ids.shape[1]))
    return out


@torch.no_grad()
def agreement_runs(s: ModelSession, refs, skip: list[int]) -> list[list[bool]]:
    """Per sequence: does the skipped model's argmax equal the full model's next token at every generated position?"""
    runs = []
    with removed(s, skip=skip):
        for seq, n_prompt in refs:
            ids = torch.tensor([seq], device=s.device)
            pred = s.model(input_ids=ids, use_cache=False).logits[0].argmax(-1)
            runs.append([bool(pred[p - 1] == ids[0, p]) for p in range(n_prompt, len(seq))])
    return runs


def agreement(s, refs, skip) -> float:
    r = agreement_runs(s, refs, skip)
    flat = [x for run in r for x in run]
    return sum(flat) / max(1, len(flat))


def tokens_per_round(runs: list[list[bool]], gamma: int) -> float:
    """Exact greedy speculative decoding: each round drafts up to gamma tokens, accepts the agreeing prefix and gains
    one verified token from the full model."""
    toks = rounds = 0
    for run in runs:
        p = 0
        while p < len(run):
            a = 0
            while a < gamma and p + a < len(run) and run[p + a]:
                a += 1
            adv = min(a + 1, len(run) - p)
            toks += adv
            rounds += 1
            p += adv
    return toks / max(1, rounds)


def select(method: str, s, calib, n: int, L: int, seed: int = 0) -> list[int]:
    if method == "early_exit":
        return list(range(L - n, L))
    if method == "causal_oneshot":
        single = {l: agreement(s, calib, [l]) for l in range(L)}
        return sorted(sorted(single, key=lambda l: -single[l])[:n])
    if method in ("causal_greedy", "generic"):
        S: list[int] = []
        for _ in range(n):
            best = max((l for l in range(L) if l not in S), key=lambda l: agreement(s, calib, S + [l]))
            S.append(best)
        return sorted(S)
    if method == "random_search":
        rng = random.Random(seed)
        budget = sum(L - i for i in range(n))  # same number of agreement evaluations as causal_greedy
        best, best_a = None, -1.0
        for _ in range(budget):
            S = sorted(rng.sample(range(1, L), n))  # never skip layer 0 (every method finds it essential)
            a = agreement(s, calib, S)
            if a > best_a:
                best, best_a = S, a
        return best
    if method == "shortgpt":
        bi = torch.zeros(L)
        with torch.no_grad():
            for seq, _ in calib:
                res = s.run(torch.tensor([seq], device=s.device), components=("resid",))
                st = [res.resid_pre0] + [res.resid_post[i] for i in range(L)]
                for i in range(L):
                    bi[i] += float((1 - torch.nn.functional.cosine_similarity(st[i].float(), st[i + 1].float(), dim=-1)).mean())
        return sorted(bi.argsort()[:n].tolist())
    raise ValueError(method)


@torch.no_grad()
def wall_clock(s: ModelSession, prompts: list[str], skip: list[int], gamma: int, gen_len: int) -> dict:
    draft = build_pruned(s.model, keep_layers=[l for l in range(s.s["n_layers"]) if l not in skip])
    draft.generation_config.num_assistant_tokens = gamma
    draft.generation_config.num_assistant_tokens_schedule = "constant"
    pad = s.tok.pad_token_id or s.tok.eos_token_id
    t_base = t_spec = 0.0
    same = 0
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for p in prompts:
        ids = s.tok(render(s.tok, p), return_tensors="pt", add_special_tokens=False)["input_ids"].to(s.device)
        sync(); t0 = time.perf_counter()
        a = s.model.generate(ids, max_new_tokens=gen_len, do_sample=False, pad_token_id=pad)
        sync(); t_base += time.perf_counter() - t0
        t0 = time.perf_counter()
        b = s.model.generate(ids, max_new_tokens=gen_len, do_sample=False, pad_token_id=pad, assistant_model=draft)
        sync(); t_spec += time.perf_counter() - t0
        same += int(a.shape == b.shape and bool((a == b).all()))
    del draft
    return {"baseline_s": round(t_base, 3), "speculative_s": round(t_spec, 3), "speedup": round(t_base / max(t_spec, 1e-9), 3),
            "identical_outputs": f"{same}/{len(prompts)}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workloads", default="translate,json,write,code")
    ap.add_argument("--skip", default="6,9", help="numbers of layers to skip in the draft")
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--calib", type=int, default=12)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--wall-clock", type=int, default=8, help="prompts per workload for the wall-clock check (0 = off)")
    ap.add_argument("--out", default="./skill-paths")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    s = ModelSession()
    s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="none")
    L = s.s["n_layers"]
    log(f"opened {a.model}: {L} layers on {s.device}")
    wls = [w.strip() for w in a.workloads.split(",")]
    data = {}
    for w in wls:
        ps = list(WORKLOADS[w])
        random.Random(0).shuffle(ps)
        cal, ev = ps[: a.calib], ps[a.calib:]
        t0 = time.time()
        data[w] = {"calib": references(s, cal, a.gen_len), "eval": references(s, ev, a.gen_len), "eval_prompts": ev}
        log(f"[{w}] references: {len(cal)} calib + {len(ev)} eval, {time.time() - t0:.0f}s")
    generic_calib = [r for w in wls for r in data[w]["calib"][: max(2, a.calib // len(wls))]]
    rows = []
    methods = ["causal_greedy", "causal_oneshot", "early_exit", "shortgpt", "random_search", "generic"]
    for n in [int(x) for x in a.skip.split(",")]:
        for w in wls:
            for m in methods:
                t0 = time.time()
                S = select(m, s, generic_calib if m == "generic" else data[w]["calib"], n, L)
                runs = agreement_runs(s, data[w]["eval"], S)
                flat = [x for r in runs for x in r]
                tpr = tokens_per_round(runs, a.gamma)
                c = (L - n) / L
                est = tpr / (a.gamma * c + 1.0)
                rows.append({"skip": n, "workload": w, "method": m, "layers_skipped": S, "eval_agreement": round(sum(flat) / len(flat), 4),
                             "tokens_per_round": round(tpr, 3), "draft_cost": round(c, 3), "est_speedup": round(est, 3),
                             "select_s": round(time.time() - t0, 1)})
                log(f"  skip {n} {w:9s} {m:15s} agree {rows[-1]['eval_agreement']:.3f}  tok/round {tpr:.2f}  est x{est:.2f}  skipped {S}")
    wc = []
    if a.wall_clock:
        for n in [int(x) for x in a.skip.split(",")]:
            for w in wls:
                cand = [r for r in rows if r["skip"] == n and r["workload"] == w]
                best = max(cand, key=lambda r: r["eval_agreement"])
                for r in {best["method"]: best, "early_exit": next(x for x in cand if x["method"] == "early_exit")}.values():
                    try:
                        res = wall_clock(s, data[w]["eval_prompts"][: a.wall_clock], r["layers_skipped"], a.gamma, a.gen_len)
                    except Exception as e:  # noqa: BLE001
                        res = {"error": f"{type(e).__name__}: {e}"}
                    wc.append({"skip": n, "workload": w, "method": r["method"], **res})
                    log(f"  wall-clock skip {n} {w:9s} {r['method']:15s} {res}")
    # verdict per skip level: causal_greedy vs best non-causal (early_exit, shortgpt, random_search) on agreement
    verdicts = []
    for n in [int(x) for x in a.skip.split(",")]:
        for w in wls:
            by = {r["method"]: r for r in rows if r["skip"] == n and r["workload"] == w}
            base = max(("early_exit", "shortgpt", "random_search"), key=lambda m: by[m]["eval_agreement"])
            d = by["causal_greedy"]["eval_agreement"] - by[base]["eval_agreement"]
            verdicts.append({"skip": n, "workload": w, "causal_greedy": by["causal_greedy"]["eval_agreement"], "best_baseline": base,
                             "baseline": by[base]["eval_agreement"], "delta": round(d, 4),
                             "specialisation_gain_vs_generic": round(by["causal_greedy"]["eval_agreement"] - by["generic"]["eval_agreement"], 4)})
    md = [f"# Skill Paths — {a.model}", "", f"gamma {a.gamma} · gen_len {a.gen_len} · calib {a.calib} prompts per workload", "",
          "| skip | workload | method | layers skipped | eval agreement | tokens/round | est. speedup |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['skip']} | {r['workload']} | {r['method']} | {r['layers_skipped']} | {r['eval_agreement']} | {r['tokens_per_round']} | {r['est_speedup']} |" for r in rows]
    md += ["", "## causal_greedy vs best standard draft", "", "| skip | workload | causal | best baseline | Δ agreement | gain vs generic path |", "|---|---|---|---|---|---|"]
    md += [f"| {v['skip']} | {v['workload']} | {v['causal_greedy']} | {v['best_baseline']} {v['baseline']} | {v['delta']:+.4f} | {v['specialisation_gain_vs_generic']:+.4f} |" for v in verdicts]
    if wc:
        md += ["", "## Wall clock (HF assisted generation, batch 1)", "", "| skip | workload | method | baseline s | speculative s | speedup | identical outputs |", "|---|---|---|---|---|---|---|"]
        md += [f"| {r['skip']} | {r['workload']} | {r['method']} | {r.get('baseline_s')} | {r.get('speculative_s')} | {r.get('speedup')} | {r.get('identical_outputs', r.get('error'))} |" for r in wc]
    md += ["", "Notes: agreement is measured teacher-forced on full-model greedy references, which is exact for greedy self-speculative",
           "decoding. Estimated speedup assumes draft cost proportional to kept layers and a verification pass costing one decode step.",
           "random_search is a stand-in for SWIFT-style search with the same evaluation budget as causal_greedy, not SWIFT itself."]
    open(os.path.join(a.out, "skill_paths.md"), "w").write("\n".join(md))
    json.dump({"rows": rows, "verdicts": verdicts, "wall_clock": wc, "args": vars(a)}, open(os.path.join(a.out, "skill_paths.json"), "w"), indent=1)
    log(f"wrote {a.out}/skill_paths.md")
    s.close()


if __name__ == "__main__":
    main()
