"""Correctness of the experiment machinery on tiny random models (CPU):
- the physically built model equals the original with the same heads / neurons / layers removed by hooks (exact);
- scorer, pairs, reverse and prune run end to end; budgets are equal across arms."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import EvidenceStore, ModelSession  # noqa: E402
from rellm.experiment import Scorer, build_pruned, get_task, removed, reverse_task, run_prune  # noqa: E402


def tiny(arch: str):
    from scripts.demo_synthetic import build_tokenizer, make_model
    tok, vocab = build_tokenizer()
    torch.manual_seed(0)
    m = make_model(arch, len(vocab)).eval()
    s = ModelSession()
    s.attach(m, tok, model_id=f"tiny-{arch}")
    return s


def check_equivalence(arch: str) -> None:
    s = tiny(arch)
    ids = s.tok(["w1 the capital of france is w3 w4 w9"], return_tensors="pt")["input_ids"]
    # remove one head per kv group (different per layer), half the neurons, and layer 1
    heads_keep = {0: [0, 2], 2: [1, 3]} if s.s["n_kv_heads"] == 2 else {0: [0, 1], 2: [2, 3]}
    removed_heads = {L: [h for h in range(4) if h not in hs] for L, hs in heads_keep.items()}
    gen = torch.Generator().manual_seed(1)
    keep_n = {L: torch.randperm(256, generator=gen)[:100].sort().values for L in (0, 2)}
    rm_n = {L: torch.tensor([i for i in range(256) if i not in set(keep_n[L].tolist())]) for L in (0, 2)}
    with removed(s, heads=removed_heads, neurons=rm_n, skip=[1]), torch.no_grad():
        ref = s.model(input_ids=ids).logits
    small = build_pruned(s.model, keep_layers=[0, 2], heads=heads_keep, neurons=keep_n)
    with torch.no_grad():
        out = small(input_ids=ids).logits
    err = float((ref - out).abs().max())
    assert err < 1e-4, (arch, err)
    assert small.config.num_hidden_layers == 2 and small.config.num_attention_heads == 2 and small.config.intermediate_size == 100
    print(f"  {arch}: built model == hook removal (max abs err {err:.2e}); params {sum(p.numel() for p in small.parameters())}")


def check_pipeline() -> None:
    s = tiny("qwen2")
    task = get_task("syn_capital")
    sc = Scorer(s.tok, task.heldout)
    assert sc.mode == "first_token" and sc.score(s.model)["n"] == len(task.heldout)
    store = EvidenceStore()
    rev = reverse_task(s, task, store, max_pairs=3, last_k=4, log=lambda *a: None)
    for key in ("behaviour", "truncation", "layer_skip", "necessity", "neurons"):
        assert key in rev, key
    assert rev["_tensors"]["head_removal_drop"].shape == (3, 4) and rev["_tensors"]["neuron_attribution"].shape == (3, 256)
    pr = run_prune(s, task, rev, width_levels=(0.5,), depth_fracs=(0.34,), random_seeds=(0, 1), log=lambda *a: None)
    kinds = {(r["kind"], r["method"]) for r in pr["rows"]}
    assert {("width", "causal"), ("width", "wanda"), ("width", "magnitude"), ("width", "random"),
            ("depth", "causal"), ("depth", "shortgpt"), ("depth", "random")} <= kinds, kinds
    assert pr["verdict"]["text"]
    print(f"  pipeline OK: {len(pr['rows'])} rows, {len(store.summary()['records'])} evidence records, verdict: {pr['verdict']['text'][:40]}")


if __name__ == "__main__":
    for arch in ("llama", "qwen2", "qwen3"):
        check_equivalence(arch)
    check_pipeline()
    print("experiment OK")
