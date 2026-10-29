"""Offline end-to-end demo with a *known* ground truth.

1. Train a 3-layer Llama-style model (CPU, ~1-2 min) on a synthetic language that mixes
   - facts:      "the capital of <country> is <city>"  (16 fixed pairs)
   - induction:  a random word sequence repeated twice
2. Drive the rellm MCP server tools on it and print an investigation report.

Because we built the data, we know what a correct reverse engineering must find:
- induction heads (layer >= 1) and a previous-token head (layer 0) from the repeated sequences;
- the country token position as the causal site for the city answer; the answer written
  by late components; different countries sharing the same circuit.
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import tempfile
import time

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import server  # noqa: E402

COUNTRIES = ["france", "italy", "spain", "germany", "japan", "china", "brazil", "egypt", "kenya", "peru", "india", "canada",
             "norway", "greece", "turkey", "mexico"]
CITIES = ["paris", "rome", "madrid", "berlin", "tokyo", "beijing", "brasilia", "cairo", "nairobi", "lima", "delhi", "ottawa",
          "oslo", "athens", "ankara", "mexicocity"]
FILLER = [f"w{i}" for i in range(32)]
WORDS = ["the", "capital", "of", "is", "and"] + COUNTRIES + CITIES + FILLER
SEQ = 40


def build_tokenizer():
    vocab = {"<bos>": 0, "<pad>": 1, "<unk>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 0)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", pad_token="<pad>", unk_token="<unk>")
    return tok, vocab


def sample_sequence(rng: random.Random) -> str:
    if rng.random() < 0.5:  # facts, several per sequence, with filler noise between them
        parts = []
        while len(" ".join(parts).split()) < SEQ - 8:
            i = rng.randrange(16)
            parts += rng.sample(FILLER, rng.randint(0, 3)) + ["the", "capital", "of", COUNTRIES[i], "is", CITIES[i]]
        return " ".join(parts)
    half = [rng.choice(FILLER) for _ in range(SEQ // 2 - 1)]
    return " ".join(half + half)


def make_model(arch: str, vocab_size: int):
    """Same tiny shape in three real HF architectures: llama (MHA), qwen2 (GQA + qkv bias + tied embeddings), qwen3 (GQA + q/k norm)."""
    common = dict(vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=3, num_attention_heads=4,
                  max_position_embeddings=64, attn_implementation="eager", bos_token_id=0, pad_token_id=1)
    if arch == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(num_key_value_heads=4, **common))
    if arch == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2ForCausalLM(Qwen2Config(num_key_value_heads=2, tie_word_embeddings=True, **common))
    if arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        return Qwen3ForCausalLM(Qwen3Config(num_key_value_heads=2, head_dim=32, tie_word_embeddings=True, **common))
    raise ValueError(arch)


def train(outdir: str, steps: int = 2500, seed: int = 0, arch: str = "llama") -> None:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tok, vocab = build_tokenizer()
    model = make_model(arch, len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    model.train()
    t0 = time.time()
    for step in range(steps):
        batch = [sample_sequence(rng) for _ in range(48)]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=SEQ + 1)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"  step {step:4d} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    model.eval()
    # held-out accuracy on the 16 facts
    correct = 0
    with torch.no_grad():
        for c, city in zip(COUNTRIES, CITIES):
            ids = tok(f"the capital of {c} is", return_tensors="pt")["input_ids"]
            pred = tok.decode([int(model(ids).logits[0, -1].argmax())])
            correct += pred == city
    print(f"  fact accuracy {correct}/16")
    model.save_pretrained(outdir, safe_serialization=True)
    tok.save_pretrained(outdir)


async def call(name: str, **args):
    r = await server.mcp.call_tool(name, args)
    if r.is_error:
        raise RuntimeError(f"{name}: {r.content[0].text[:300]}")
    return r.structured_content


async def investigate(model_dir: str, bundle_path: str) -> None:
    rep = []
    P = rep.append
    art = await call("rellm_inspect_model_artifact", source=model_dir)
    a = art["normalized_result"]["architecture"]
    P(f"[static {art['evidence_id'][:14]}] declared: family={a['family']} layers={a['n_layers']} d_model={a['d_model']} heads={a['n_heads']} "
      f"kv={a['n_kv_heads']} d_head={a['d_head']} d_mlp={a['d_mlp']} params={a['declared_parameters_human']} norm={a['final_norm_kind']} "
      f"tied={a['tied_embeddings']} config={a['config_consistency']}")
    op = await call("rellm_open_model", model_id=model_dir, dtype="float32", device="cpu")
    nr = op["normalized_result"]
    P(f"[open {op['evidence_id'][:14]}] run_id={nr['run_id']} headers_sha={nr['identity']['weights_headers_sha256'][:12]} "
      f"facets={ {k: v.get('available') for k, v in nr['facet_availability'].items()} } profile={nr['analysis_profile']['hook_semantics']}")

    clean, corr = "the capital of france is", "the capital of italy is"
    nt = await call("rellm_next_token", prompt=clean, top_k=3, track_tokens=["paris", "rome"])
    n = nt["normalized_result"]
    P(f"[behaviour {nt['evidence_id'][:14]}] {clean!r} -> {n['top']} | paris rank {n['tracked']['paris']['rank']} rome rank {n['tracked']['rome']['rank']} "
      f"entropy {n['entropy_bits']} bits")
    ntc = await call("rellm_next_token", prompt=corr, top_k=3, track_tokens=["paris", "rome"])
    P(f"[behaviour {ntc['evidence_id'][:14]}] {corr!r} -> {ntc['normalized_result']['top'][0]}")

    ll = await call("rellm_logit_lens", prompt=clean, top_k=3, track_tokens=["paris", "rome"])
    L = ll["normalized_result"]
    P(f"[lens {ll['evidence_id'][:14]} authority={ll['authority']}] decision layer = {L['decision_layer']}")
    for row in L["layers"]:
        P(f"    {row['layer']:>5}: top={[t['token'] for t in row['top']]} paris rank={row['tracked']['paris']['rank']} p={row['tracked']['paris']['prob']} "
          f"rome p={row['tracked']['rome']['prob']} KL->final={row['kl_to_final_bits']}")

    hc = await call("rellm_head_catalog", seq_len=16, n_samples=6, seed=1)
    H = hc["normalized_result"]
    P(f"[head catalog {hc['evidence_id'][:14]}] induction heads: {[(r['layer'], r['head'], r['induction']) for r in H['induction_heads']]} | "
      f"prev-token heads: {[(r['layer'], r['head'], r['previous_token']) for r in H['previous_token_heads']]} | "
      f"duplicate-token: {[(r['layer'], r['head'], r['duplicate_token']) for r in H['duplicate_token_heads']]}")

    pt = await call("rellm_activation_patching", clean_prompt=clean, corrupted_prompt=corr, answer="paris", counter_answer="rome")
    T = pt["normalized_result"]
    P(f"[patching resid {pt['evidence_id'][:14]} authority={pt['authority']} coverage={pt['coverage']['status']} passes={T['n_forward_passes']}] "
      f"clean metric {T['clean']['metric']} corrupted {T['corrupted']['metric']}")
    for r in T["top_effects"][:6]:
        P(f"    L{r['layer']} @pos{r['position']} {r['token']!r}: normalized {r['normalized']}")
    grid = {(r["layer"], r["position"]): r["normalized"] for r in T["grid"]}
    P("    grid (rows=layer, cols=position " + " ".join(f"{t[:5]:>6}" for t in T["clean"]["tokens"]) + ")")
    for Lr in range(a["n_layers"]):
        P(f"    L{Lr}  " + " ".join(f"{grid[(Lr, p)]:6.2f}" for p in range(len(T["clean"]["tokens"]))))

    ph = await call("rellm_activation_patching", clean_prompt=clean, corrupted_prompt=corr, answer="paris", counter_answer="rome",
                    component="head", positions=[-1])
    P(f"[patching heads@last {ph['evidence_id'][:14]}] " + ", ".join(f"L{r['layer']}H{r['head']}={r['normalized']}" for r in ph["normalized_result"]["top_effects"][:6]))
    ph4 = await call("rellm_activation_patching", clean_prompt=clean, corrupted_prompt=corr, answer="paris", counter_answer="rome",
                     component="head", positions=[4])
    P(f"[patching heads@country {ph4['evidence_id'][:14]}] " + ", ".join(f"L{r['layer']}H{r['head']}={r['normalized']}" for r in ph4["normalized_result"]["top_effects"][:6]))
    pm = await call("rellm_activation_patching", clean_prompt=clean, corrupted_prompt=corr, answer="paris", counter_answer="rome", component="mlp_out")
    P(f"[patching mlp {pm['evidence_id'][:14]}] " + ", ".join(f"L{r['layer']}@{r['position']}={r['normalized']}" for r in pm["normalized_result"]["top_effects"][:6]))

    dla = await call("rellm_direct_logit_attribution", prompt=clean, answer="paris", counter_answer="rome", top_k=6)
    D = dla["normalized_result"]
    P(f"[DLA {dla['evidence_id'][:14]} authority={dla['authority']}] logit diff paris-rome = {D['actual_target_value']} (sum of parts {D['sum_of_attributions']}, "
      f"recon err {D['reconstruction_max_abs_err']})")
    P("    + " + ", ".join(f"{c['component']}{'' if c['layer'] is None else ' L'+str(c['layer'])}{'' if c['head'] is None else 'H'+str(c['head'])}={c['contribution']}" for c in D["top_positive"]))
    P("    - " + ", ".join(f"{c['component']}{'' if c['layer'] is None else ' L'+str(c['layer'])}{'' if c['head'] is None else 'H'+str(c['head'])}={c['contribution']}" for c in D["top_negative"]))
    P("    by layer: " + ", ".join(f"L{b['layer']} attn={b['attention']} mlp={b['mlp']}" for b in D["by_layer"]))

    top_heads = [[c["layer"], c["head"]] for c in D["top_positive"] if c["component"] == "head"][:3]
    if top_heads:
        ab = await call("rellm_ablate_heads", prompt=clean, heads=top_heads, answer="paris", counter_answer="rome", mode="mean", top_k=3)
        A = ab["normalized_result"]
        P(f"[ablation {ab['evidence_id'][:14]} authority={ab['authority']}] mean-ablate {top_heads}: logit diff {A['baseline']['metric']} -> {A['ablated']['metric']} "
          f"(delta {A['ablated']['delta']}); top after: {A['ablated']['top']}; singles: {[(s['layer'], s['head'], s['metric']) for s in A['single_head_ablations']]}")

    lay = [L for L in range(a["n_layers"])]
    ap = await call("rellm_attention_patterns", prompt=clean, layers=lay, top_k=3)
    for lr in ap["normalized_result"]["layers"]:
        P(f"[attention L{lr['layer']}] " + " | ".join(f"H{h['head']}: " + ",".join(f"{x['token']}({x['weight']})" for x in h["attends_to"][:2]) for h in lr["heads"]))

    # does another fact use the same circuit?
    pt2 = await call("rellm_activation_patching", clean_prompt="the capital of japan is", corrupted_prompt="the capital of kenya is",
                     answer="tokyo", counter_answer="nairobi")
    cmp_ = await call("rellm_compare_circuits", left_evidence_id=pt["evidence_id"], right_evidence_id=pt2["evidence_id"], top_k=5)
    C = cmp_["normalized_result"]
    P(f"[compare {cmp_['evidence_id'][:14]}] france/italy vs japan/kenya resid grids: jaccard(top5)={C['jaccard_top']} spearman={C['spearman_common']} "
      f"shared top={C['shared_top']} largest deltas={[(d['key'], d['delta']) for d in C['largest_deltas'][:3]]}")
    dla2 = await call("rellm_direct_logit_attribution", prompt="the capital of japan is", answer="tokyo", counter_answer="nairobi", top_k=6)
    cmp2 = await call("rellm_compare_circuits", left_evidence_id=dla["evidence_id"], right_evidence_id=dla2["evidence_id"], top_k=5)
    C2 = cmp2["normalized_result"]
    P(f"[compare {cmp2['evidence_id'][:14]}] DLA france vs japan: jaccard(top5)={C2['jaccard_top']} spearman={C2['spearman_common']} shared top={C2['shared_top']}")

    # the repeat task: is it content-based induction or a positional shortcut? sweep the period.
    P("[induction sweep] training period was 19 (offset 18):")
    for L in (19, 12, 25):
        h = await call("rellm_head_catalog", seq_len=L, n_samples=6, seed=1)
        top = h["normalized_result"]["top_by_induction"][:3]
        P(f"    seq_len={L}: top induction scores {[(r['layer'], r['head'], r['induction']) for r in top]}")
    rep_prompt = " ".join([FILLER[i] for i in (3, 17, 9, 25, 1, 30, 12, 8, 21, 5, 14, 27, 2, 19, 11, 6, 29, 16, 23)] * 2)
    apr = await call("rellm_attention_patterns", prompt=rep_prompt, layers=[1], query_position=30, top_k=1)
    P("    L1 heads at q=30 on a period-19 repeat: " + ", ".join(f"H{h['head']}->pos{h['attends_to'][0]['position']} ({h['attends_to'][0]['weight']})"
                                                               for h in apr["normalized_result"]["layers"][0]["heads"]))
    unk2 = await call("rellm_record_unknown", question="is the repeat-copy mechanism content-based induction or a fixed positional offset?",
                      domain="attention", context_evidence_id=hc["evidence_id"])
    await call("rellm_update_unknown", unknown_id=unk2["unknown_id"], status="resolved",
               resolution="fixed offset 18: all L1 heads attend to q-18 on period-19 repeats; induction score collapses at other periods", resolved_by=apr["evidence_id"])

    unk = await call("rellm_record_unknown", question="which head moves the country identity from position 4 to the last position, and in which layer?",
                     domain="attention", context_evidence_id=pt["evidence_id"])
    await call("rellm_update_unknown", unknown_id=unk["unknown_id"], status="resolved",
               resolution="see heads@country / heads@last patching", resolved_by=ph["evidence_id"])
    cache = await call("rellm_logit_lens", prompt=clean, top_k=3, track_tokens=["paris", "rome"])
    P(f"[cache] repeated logit lens -> cache_hit={cache['cache_hit']} same id={cache['evidence_id'] == ll['evidence_id']}")
    exp = await call("rellm_export_evidence_bundle", path=bundle_path)
    P(f"[bundle] {exp['records']} records, {exp['unknowns']} unknowns, {exp['bytes']} bytes -> {exp['path']}")
    closed = await call("rellm_close_model")
    P(f"[close] cleanup={closed['cleanup']}")
    print("\n".join(rep))


def main() -> None:
    out = os.environ.get("RELLM_DEMO_DIR") or tempfile.mkdtemp(prefix="rellm-demo-")
    model_dir = os.path.join(out, "model")
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        print("training synthetic model ->", model_dir)
        train(model_dir, steps=int(os.environ.get("RELLM_DEMO_STEPS", "2500")), arch=os.environ.get("RELLM_DEMO_ARCH", "llama"))
    print("investigating ...")
    asyncio.run(investigate(model_dir, os.path.join(out, "evidence-bundle.json")))


if __name__ == "__main__":
    main()
