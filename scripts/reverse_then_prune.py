"""Reverse first, then test causal pruning — the full experiment, one command.

Colab (H100):
  python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-0.5B --tasks capital,ioi,sentiment --out /content/exp-0.5b
  python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-1.5B --tasks capital,ioi,sentiment --out /content/exp-1.5b --heal-steps 100
  python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-7B   --tasks capital,ioi --phase reverse --out /content/exp-7b
Offline demo model:
  python scripts/reverse_then_prune.py --model /home/claude/demo-qwen2/model --tasks syn_capital --dtype float32 --device cpu --out /tmp/exp

Outputs in --out: reverse_report.md, prune_report.md, results.json, scores_<task>.pt, evidence-bundle.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import EvidenceStore, ModelSession  # noqa: E402
from rellm.experiment import cross_task, get_task, model_level, prune_md, reverse_md, reverse_task, run_prune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="capital,ioi,sentiment", help="comma list of built-in tasks or .json task files")
    ap.add_argument("--phase", choices=["reverse", "prune", "all"], default="all")
    ap.add_argument("--out", default="./rellm-experiment")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--last-k", type=int, default=8)
    ap.add_argument("--width-levels", default="0.75,0.5,0.25")
    ap.add_argument("--depth-fracs", default="0.125,0.25,0.375")
    ap.add_argument("--heal-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-model-level", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda *x: print(*x, flush=True)

    s = ModelSession()
    info = s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="header")
    facts = info["facts"] | {"model_id": a.model}
    log(f"opened {a.model}: {facts['n_layers']}L {facts['n_heads']}H kv={facts['n_kv_heads']} d_mlp={facts['d_mlp']} {facts['parameters_human']} "
        f"on {s.device} {s.dtype}; tokenizer note: {facts['tokenizer'].get('note')}")
    store = EvidenceStore()
    tasks = [get_task(t.strip(), seed=a.seed) for t in a.tasks.split(",") if t.strip()]

    log("\n=== PHASE 1: reverse ===")
    ml = None if a.skip_model_level else model_level(s, store, log=log)
    results = {}
    for task in tasks:
        log(f"[{task.name}] {task.summary()['n_train']} train / {task.summary()['n_heldout']} heldout / {task.summary()['n_ood']} OOD")
        r = reverse_task(s, task, store, max_pairs=a.max_pairs, last_k=a.last_k, log=log)
        torch.save(r["_tensors"], os.path.join(a.out, f"scores_{task.name}.pt"))
        results[task.name] = r
    cross = cross_task(results) if len(results) > 1 else None
    open(os.path.join(a.out, "reverse_report.md"), "w").write(reverse_md(facts, ml, results, cross))
    log(f"wrote {a.out}/reverse_report.md")

    prune_results = {}
    if a.phase in ("prune", "all"):
        log("\n=== PHASE 2: equal-budget pruning (causal vs wanda vs magnitude/shortgpt vs random) ===")
        md = ["# Pruning report — " + a.model, ""]
        for task in tasks:
            log(f"[{task.name}]")
            pr = run_prune(s, task, results[task.name], width_levels=tuple(float(x) for x in a.width_levels.split(",")),
                           depth_fracs=tuple(float(x) for x in a.depth_fracs.split(",")), heal_steps=a.heal_steps, log=log)
            prune_results[task.name] = pr
            md.append(prune_md(task.name, pr))
            log(f"  verdict: {pr['verdict']['text']}")
        open(os.path.join(a.out, "prune_report.md"), "w").write("\n".join(md))
        log(f"wrote {a.out}/prune_report.md")

    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_tensors"} for k, v in results.items()}
    json.dump({"model": facts, "model_level": ml, "reverse": clean, "cross_task": cross, "prune": prune_results,
               "elapsed_s": round(time.time() - t0, 1), "args": vars(a)}, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    store.export_bundle(os.path.join(a.out, "evidence-bundle.json"))
    s.close()
    log(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
