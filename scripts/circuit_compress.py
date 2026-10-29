"""Mechanistic Circuit Compression — prototype.

Question: can we find the minimal functional circuit a model uses for one task and build a
smaller, genuinely faster model that keeps that capability?

Steps (all measured on an evaluation set, never on one prompt):
  A. verify the decision really forms at the layer the logit lens claims, on every input,
     and that later layers never flip it (per-layer truncation accuracy + flip count);
  B. necessity map: ablate every head (mean) and every MLP (zero) -> accuracy / margin;
  C. minimal circuit: greedy backward elimination over {layers, heads, MLP} keeping 100% accuracy;
  D. MLP neuron pruning inside the kept layer, by task importance, with an accuracy sweep;
  E. build the compressed model as a real, smaller HF model (fewer layers, fewer heads, narrower MLP),
     measure parameters, CPU latency, task accuracy, and what was lost (the repeat task).

Run after scripts/demo_synthetic.py:  RELLM_DEMO_DIR=<dir> python scripts/circuit_compress.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import ModelSession  # noqa: E402
from scripts.demo_synthetic import CITIES, COUNTRIES, FILLER  # noqa: E402


# ----------------------------------------------------------------------------- eval set
def build_eval(tok, seed: int = 11, prefixes_per_fact: int = 5):
    rng = random.Random(seed)
    prompts, answers = [], []
    for c, city in zip(COUNTRIES, CITIES):
        for k in range(prefixes_per_fact):
            pre = rng.sample(FILLER, k % 4)  # 0..3 filler words before the fact, position of the country varies
            prompts.append(" ".join(pre + ["the", "capital", "of", c, "is"]))
            answers.append(tok.convert_tokens_to_ids(city))
    return prompts, torch.tensor(answers)


def city_ids(tok):
    return torch.tensor([tok.convert_tokens_to_ids(c) for c in CITIES])


@torch.no_grad()
def score(model, tok, prompts, answers, cities, logits_fn=None):
    """accuracy and margin (correct city logit minus best other city logit) at the last position."""
    correct, margins = 0, []
    for p, a in zip(prompts, answers):
        ids = tok(p, return_tensors="pt")["input_ids"]
        logits = (logits_fn(ids) if logits_fn else model(input_ids=ids, use_cache=False).logits)[0, -1].float()
        pred = int(logits.argmax())
        correct += pred == int(a)
        others = logits[cities].clone()
        others[(cities == a).nonzero()] = -1e9
        margins.append(float(logits[a] - others.max()))
    m = torch.tensor(margins)
    return {"acc": correct / len(prompts), "margin_mean": round(float(m.mean()), 3), "margin_min": round(float(m.min()), 3)}


@torch.no_grad()
def repeat_task_acc(model, tok, period: int = 19, n: int = 40, seed: int = 5):
    rng = random.Random(seed)
    c = t = 0
    for _ in range(n):
        half = [rng.choice(FILLER) for _ in range(period)]
        ids = tok(" ".join(half + half), return_tensors="pt")["input_ids"]
        pred = model(input_ids=ids, use_cache=False).logits[0].argmax(-1)
        for p in range(period + 1, ids.shape[1] - 1):
            c += int(pred[p] == ids[0, p + 1])
            t += 1
    return round(c / t, 3)


@torch.no_grad()
def latency_ms(model, tok, prompts, reps: int = 15):
    enc = tok(prompts, return_tensors="pt", padding=True)
    model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    return round((time.perf_counter() - t0) / reps * 1000, 2)


# ----------------------------------------------------------------------------- interventions
@contextlib.contextmanager
def truncate_after(s: ModelSession, last_layer: int):
    """Run only layers 0..last_layer: later decoder layers pass their input through unchanged."""
    handles = []
    for L in range(last_layer + 1, s.s["n_layers"]):
        def passthrough(mod, args, kwargs, out):
            hidden = args[0] if args else kwargs["hidden_states"]
            return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden
        handles.append(s.s["layers"][L].register_forward_hook(passthrough, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def circuit_context(s: ModelSession, keep_layers: set[int], drop_heads: set[tuple[int, int]], drop_mlps: set[int], seq_hint: int = 64):
    """Compose an intervention: layers not in keep_layers are skipped, listed heads mean-ablated, listed MLPs zeroed."""
    stack = contextlib.ExitStack()
    last_kept = max(keep_layers) if keep_layers else -1
    stack.enter_context(truncate_after(s, last_kept))
    all_pos = list(range(seq_hint))

    def mk_head(L, h):
        d = s.s["d_head"]

        def hook(m, a):
            t = a[0].clone()
            t[:, :, h * d:(h + 1) * d] = t[:, :, h * d:(h + 1) * d].mean(dim=1, keepdim=True)
            return (t,) + tuple(a[1:])
        return hook

    def mk_mlp(m, a, o):
        return torch.zeros_like(o) if not isinstance(o, tuple) else (torch.zeros_like(o[0]),) + tuple(o[1:])
    handles = []
    for (L, h) in drop_heads:
        if L in keep_layers:
            handles.append(s.s["o_proj"][L].register_forward_pre_hook(mk_head(L, h)))
    for L in drop_mlps:
        if L in keep_layers:
            handles.append(s.s["mlp"][L].register_forward_hook(mk_mlp))
    stack.callback(lambda: [h.remove() for h in handles])
    del all_pos
    return stack


# ----------------------------------------------------------------------------- compression
@torch.no_grad()
def mlp_neuron_importance(s: ModelSession, layer: int, prompts, answers, cities):
    """Task importance of each MLP neuron: mean |activation x contribution of its down-proj column to the answer direction|."""
    mlp = s.s["mlp"][layer]
    acts = []

    def hook(m, a):
        acts.append(a[0][0, -1].detach().clone())  # down_proj input = silu(gate)*up at last position
    h = mlp.down_proj.register_forward_pre_hook(hook)
    try:
        for p in prompts:
            s.model(input_ids=s.tok(p, return_tensors="pt")["input_ids"], use_cache=False)
    finally:
        h.remove()
    A = torch.stack(acts)  # (n_prompts, d_mlp)
    W_down = mlp.down_proj.weight.float()  # (d_model, d_mlp)
    W_U = s.s["unembed"].weight.float()
    imp = torch.zeros(A.shape[1])
    for i, a in enumerate(answers):
        direction = W_U[a] - W_U[cities].mean(0)
        imp += (A[i].float() * (W_down.T @ direction)).abs()
    return imp / len(prompts)


@torch.no_grad()
def build_compressed(src, keep_layers: list[int], keep_heads: dict[int, list[int]], keep_neurons: dict[int, torch.Tensor]):
    """Slice a real smaller model out of `src` (Llama / Qwen2 / Qwen3 layouts): fewer layers, query heads, kv heads, MLP neurons.

    Under grouped-query attention a kept query head needs its kv head; every kept kv group must keep the same number of
    query heads, otherwise the kept set is widened to whole groups.
    """
    cfg = copy.deepcopy(src.config)
    n_heads, n_kv = src.config.num_attention_heads, getattr(src.config, "num_key_value_heads", src.config.num_attention_heads)
    d_head = getattr(src.config, "head_dim", None) or src.config.hidden_size // n_heads
    group = n_heads // n_kv
    L0 = keep_layers[0]
    hs = sorted(keep_heads[L0])
    per_group = {}
    for h in hs:
        per_group.setdefault(h // group, []).append(h)
    if len({len(v) for v in per_group.values()}) > 1:
        hs = sorted(h for g in per_group for h in range(g * group, (g + 1) * group))  # widen to whole groups
        per_group = {g: list(range(g * group, (g + 1) * group)) for g in per_group}
    kv_keep = sorted(per_group)
    keep_heads = {L: hs for L in keep_layers}
    cfg.num_hidden_layers = len(keep_layers)
    if getattr(cfg, "layer_types", None):  # Qwen2/3, Gemma: per-layer attention types must follow the kept layers
        cfg.layer_types = [cfg.layer_types[L] for L in keep_layers]
    cfg.num_attention_heads = len(hs)
    cfg.num_key_value_heads = len(kv_keep)
    cfg.head_dim = d_head
    cfg.intermediate_size = int(keep_neurons[L0].numel())
    new = type(src)(cfg).eval()
    new.model.embed_tokens.weight.copy_(src.model.embed_tokens.weight)
    new.model.norm.load_state_dict(src.model.norm.state_dict())
    if not getattr(cfg, "tie_word_embeddings", False):
        new.lm_head.weight.copy_(src.lm_head.weight)
    q_rows = torch.cat([torch.arange(h * d_head, (h + 1) * d_head) for h in hs])
    kv_rows = torch.cat([torch.arange(g * d_head, (g + 1) * d_head) for g in kv_keep])

    def slice_linear(dst, srcm, rows=None, cols=None):
        w = srcm.weight
        w = w[rows] if rows is not None else w
        w = w[:, cols] if cols is not None else w
        dst.weight.copy_(w)
        if getattr(srcm, "bias", None) is not None and dst.bias is not None:
            dst.bias.copy_(srcm.bias[rows] if rows is not None else srcm.bias)

    for new_i, L in enumerate(keep_layers):
        old, cur = src.model.layers[L], new.model.layers[new_i]
        slice_linear(cur.self_attn.q_proj, old.self_attn.q_proj, rows=q_rows)
        slice_linear(cur.self_attn.k_proj, old.self_attn.k_proj, rows=kv_rows)
        slice_linear(cur.self_attn.v_proj, old.self_attn.v_proj, rows=kv_rows)
        slice_linear(cur.self_attn.o_proj, old.self_attn.o_proj, cols=q_rows)
        for name in ("q_norm", "k_norm"):  # Qwen3: per-head RMSNorm over head_dim, shared across heads
            if hasattr(old.self_attn, name):
                getattr(cur.self_attn, name).load_state_dict(getattr(old.self_attn, name).state_dict())
        cur.input_layernorm.load_state_dict(old.input_layernorm.state_dict())
        cur.post_attention_layernorm.load_state_dict(old.post_attention_layernorm.state_dict())
        nk = keep_neurons[L]
        cur.mlp.gate_proj.weight.copy_(old.mlp.gate_proj.weight[nk])
        cur.mlp.up_proj.weight.copy_(old.mlp.up_proj.weight[nk])
        cur.mlp.down_proj.weight.copy_(old.mlp.down_proj.weight[:, nk])
    return new


def main() -> None:
    out = os.environ.get("RELLM_DEMO_DIR", "/home/claude/demo")
    s = ModelSession()
    s.open(os.path.join(out, "model"), dtype="float32", device="cpu")
    tok, model = s.tok, s.model
    prompts, answers = build_eval(tok)
    cities = city_ids(tok)
    n_layers, n_heads = s.s["n_layers"], s.s["n_heads"]
    report = {}
    P = print

    P(f"eval set: {len(prompts)} prompts (16 facts x 5 prefixes, country at positions 4..7)")
    base = score(model, tok, prompts, answers, cities)
    report["baseline"] = base | {"params": sum(p.numel() for p in model.parameters()), "repeat_task_acc": repeat_task_acc(model, tok),
                                 "latency_ms": latency_ms(model, tok, prompts)}
    P(f"baseline: {report['baseline']}")

    # ---- A. decision stability: truncation after each layer, and flips
    P("\nA. where is the decision made? (truncation = later layers removed)")
    trunc = {}
    for L in range(-1, n_layers):
        with truncate_after(s, L):
            r = score(model, tok, prompts, answers, cities)
        trunc["embed" if L < 0 else f"L{L}"] = r
        P(f"   through {'embed' if L < 0 else 'L'+str(L):>5}: acc {r['acc']:.3f} margin mean {r['margin_mean']} min {r['margin_min']}")
    flips = 0
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt")["input_ids"]
            res = s.run(ids)
            l0_top = int(s.unembed(res.resid_post[0][0, -1]).argmax())
            flips += l0_top != int(res.logits[0, -1].argmax())
    P(f"   inputs where a later layer changes the L0 answer: {flips}/{len(prompts)}")
    report["truncation"] = trunc
    report["l0_flips"] = flips

    # ---- B. necessity map
    P("\nB. necessity map (ablate one component, full model)")
    nec = {}
    for L in range(n_layers):
        for h in range(n_heads):
            with circuit_context(s, set(range(n_layers)), {(L, h)}, set()):
                r = score(model, tok, prompts, answers, cities)
            nec[f"L{L}H{h}"] = r
        with circuit_context(s, set(range(n_layers)), set(), {L}):
            r = score(model, tok, prompts, answers, cities)
        nec[f"L{L}.mlp"] = r
    for k, r in sorted(nec.items(), key=lambda kv: kv[1]["margin_mean"]):
        P(f"   -{k:7s}: acc {r['acc']:.3f} margin mean {r['margin_mean']:7.3f} min {r['margin_min']:7.3f}")
    report["necessity"] = nec

    # ---- C. minimal circuit (greedy backward elimination, keep acc == 1.0 and margin_min >= threshold)
    P("\nC. minimal circuit (greedy: drop the component whose removal costs the least margin, stop before acc < 1.0)")
    threshold = max(1.0, 0.25 * base["margin_min"])
    keep_layers = set(range(n_layers))
    drop_heads: set[tuple[int, int]] = set()
    drop_mlps: set[int] = set()

    def evaluate():
        with circuit_context(s, keep_layers, drop_heads, drop_mlps):
            return score(model, tok, prompts, answers, cities)
    # first try truncation (cheapest structurally)
    for L in range(n_layers - 1, 0, -1):
        keep_layers.discard(L)
        r = evaluate()
        if r["acc"] < 1.0 or r["margin_min"] < threshold:
            keep_layers.add(L)
            break
        P(f"   dropped layer L{L}: acc {r['acc']:.3f} margin min {r['margin_min']}")
    while True:
        candidates = []
        for L in keep_layers:
            for h in range(n_heads):
                if (L, h) not in drop_heads:
                    drop_heads.add((L, h)); r = evaluate(); drop_heads.discard((L, h))
                    candidates.append((r["margin_min"], r["acc"], ("head", L, h)))
            if L not in drop_mlps:
                drop_mlps.add(L); r = evaluate(); drop_mlps.discard(L)
                candidates.append((r["margin_min"], r["acc"], ("mlp", L, None)))
        ok = [c for c in candidates if c[1] == 1.0 and c[0] >= threshold]
        if not ok:
            break
        best = max(ok)
        kind, L, h = best[2]
        (drop_heads.add((L, h)) if kind == "head" else drop_mlps.add(L))
        P(f"   dropped {kind} L{L}{'H'+str(h) if h is not None else ''}: acc 1.000 margin min {best[0]:.3f}")
    final = evaluate()
    keep_heads = {L: [h for h in range(n_heads) if (L, h) not in drop_heads] for L in sorted(keep_layers)}
    P(f"   minimal circuit: layers {sorted(keep_layers)}, heads {keep_heads}, mlps {[L for L in sorted(keep_layers) if L not in drop_mlps]} -> {final}")
    report["minimal_circuit"] = {"layers": sorted(keep_layers), "heads": keep_heads, "mlps": [L for L in sorted(keep_layers) if L not in drop_mlps],
                                 "score": final, "threshold_margin_min": threshold}

    # ---- D. MLP neuron pruning inside the kept layer(s)
    P("\nD. MLP neuron pruning (importance = |act x down-proj projected on the answer direction|)")
    keep_neurons = {}
    for L in sorted(keep_layers):
        if L in drop_mlps:
            keep_neurons[L] = torch.arange(0)
            continue
        imp = mlp_neuron_importance(s, L, prompts, answers, cities)
        order = imp.argsort(descending=True)
        chosen = None
        for k in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            k = min(k, imp.numel())
            keep = order[:k]
            mask = torch.zeros(imp.numel(), dtype=torch.bool); mask[keep] = True

            def masked(m, a, mask=mask):
                t = a[0].clone(); t[..., ~mask] = 0; return (t,) + tuple(a[1:])
            hh = s.s["mlp"][L].down_proj.register_forward_pre_hook(masked)
            try:
                with circuit_context(s, keep_layers, drop_heads, drop_mlps):
                    r = score(model, tok, prompts, answers, cities)
            finally:
                hh.remove()
            P(f"   L{L} keep {k:3d}/{imp.numel()} neurons: acc {r['acc']:.3f} margin min {r['margin_min']:.3f}")
            if chosen is None and r["acc"] == 1.0 and r["margin_min"] >= threshold:
                chosen = k
        keep_neurons[L] = order[: chosen or imp.numel()].sort().values
        P(f"   L{L}: keeping {keep_neurons[L].numel()} neurons")
    report["neurons_kept"] = {str(L): int(v.numel()) for L, v in keep_neurons.items()}

    # ---- E. build the real smaller model
    P("\nE. compressed model (real HF model, fewer layers/heads/neurons)")
    # validate on the *built* model, not on the hook proxy: mean-ablation, GQA widening and neuron pruning interact.
    sweep = [k for k in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)]
    imp_cache = {L: mlp_neuron_importance(s, L, prompts, answers, cities).argsort(descending=True) for L in sorted(keep_layers) if L not in drop_mlps}
    small, r_small = None, None
    for widen in range(len(sweep)):
        trial = {}
        for L in sorted(keep_layers):
            if L in drop_mlps:
                trial[L] = torch.arange(0)
            else:
                k0 = int(keep_neurons[L].numel())
                ks = [k for k in sweep if k >= k0] or [k0]
                k = min(ks[min(widen, len(ks) - 1)], imp_cache[L].numel())
                trial[L] = imp_cache[L][:k].sort().values
        small = build_compressed(model, sorted(keep_layers), keep_heads, trial)
        r_small = score(small, tok, prompts, answers, cities)
        P(f"   built with neurons {{{', '.join(f'L{L}:{int(v.numel())}' for L, v in trial.items())}}}: acc {r_small['acc']:.3f} margin min {r_small['margin_min']}")
        if r_small["acc"] == 1.0 and r_small["margin_min"] >= threshold:
            keep_neurons = trial
            break
    report["neurons_kept"] = {str(L): int(v.numel()) for L, v in keep_neurons.items()}
    stats = {"layers": small.config.num_hidden_layers, "heads": small.config.num_attention_heads, "d_mlp": small.config.intermediate_size,
             "params": sum(p.numel() for p in small.parameters()),
             "params_without_embeddings": sum(p.numel() for n, p in small.named_parameters() if "embed" not in n and "lm_head" not in n),
             "latency_ms": latency_ms(small, tok, prompts), "repeat_task_acc": repeat_task_acc(small, tok)} | r_small
    full_body = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n and "lm_head" not in n)
    P(f"   full : layers {n_layers} heads {n_heads} d_mlp {model.config.intermediate_size} params {report['baseline']['params']} (body {full_body}) "
      f"latency {report['baseline']['latency_ms']} ms  facts acc {base['acc']:.3f} margin min {base['margin_min']}  repeat {report['baseline']['repeat_task_acc']}")
    P(f"   small: layers {stats['layers']} heads {stats['heads']} d_mlp {stats['d_mlp']} params {stats['params']} (body {stats['params_without_embeddings']}) "
      f"latency {stats['latency_ms']} ms  facts acc {stats['acc']:.3f} margin min {stats['margin_min']}  repeat {stats['repeat_task_acc']}")
    P(f"   body params x{full_body / stats['params_without_embeddings']:.1f} smaller, {report['baseline']['latency_ms'] / stats['latency_ms']:.1f}x faster, "
      f"task kept: {stats['acc'] == 1.0}, repeat task lost: {stats['repeat_task_acc'] < 0.2}")
    # exact-equivalence check against the hook-based circuit
    with circuit_context(s, keep_layers, drop_heads, drop_mlps):
        ref = score(model, tok, prompts, answers, cities)
    P(f"   hook-circuit vs built model margin mean: {ref['margin_mean']} vs {r_small['margin_mean']} "
      f"(differ only by mean-ablation vs removal of dropped heads and pruned neurons)")
    report["compressed"] = stats
    small_dir = os.path.join(out, "model-compressed")
    small.save_pretrained(small_dir, safe_serialization=True)
    tok.save_pretrained(small_dir)
    json.dump(report, open(os.path.join(out, "circuit-compression-report.json"), "w"), indent=1, default=str)
    P(f"\nsaved {small_dir} and circuit-compression-report.json")
    s.close()


if __name__ == "__main__":
    main()
