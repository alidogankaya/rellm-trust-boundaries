"""Smoke test: every analysis path on tiny random models (CPU, seconds). Run: python tests/test_smoke.py"""
from __future__ import annotations

import json
import os
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from rellm import EvidenceStore, ModelSession, analysis

WORDS = ["the", "capital", "of", "france", "is", "paris", "italy", "rome", "when", "mary", "and", "john", "went", "to", "store",
         "gave", "a", "drink"] + [f"w{i}" for i in range(46)]


def tiny_tokenizer():
    vocab = {"<bos>": 0, "<unk>": 1, "<eos>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 0)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>"), len(vocab)


def tiny_llama(tmpdir: str | None = None):
    tok, vocab = tiny_tokenizer()
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    s = ModelSession()
    s.attach(model, tok, model_id="tiny-llama-test")
    if tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True)
        tok.save_pretrained(tmpdir)
    return s


def tiny_gpt2():
    tok, vocab = tiny_tokenizer()
    cfg = GPT2Config(vocab_size=vocab, n_embd=64, n_layer=3, n_head=4, n_positions=128, n_inner=128, attn_implementation="eager")
    torch.manual_seed(1)
    s = ModelSession()
    s.attach(GPT2LMHeadModel(cfg).eval(), tok, model_id="tiny-gpt2-test")
    return s


CLEAN, CORR = "the capital of france is", "the capital of italy is"


def check_session(s: ModelSession, n_layers: int, n_heads: int, name: str):
    info = s.session_info()
    assert info["run_id"].startswith("run_") and info["analysis_profile"]["hook_semantics"] == 2
    assert info["facts"]["n_layers"] == n_layers and info["facts"]["n_heads"] == n_heads, info["facts"]
    for facet in ("residual_stream", "attention_weights", "attention_out", "mlp_out", "per_head_output"):
        assert info["facet_availability"][facet]["available"], (name, facet, info["facet_availability"])

    nt = analysis.next_token(s, CLEAN, track_tokens=["paris", "rome"])
    assert len(nt["top"]) == 10 and nt["tracked"]["paris"]["rank"] >= 1

    ll = analysis.logit_lens(s, CLEAN, track_tokens=["paris", "rome"])
    assert len(ll["layers"]) == n_layers + 1 and ll["layers"][-1]["final_answer_rank"] == 1 and ll["lens_last_layer_max_abs_err"] < 1e-3

    ap = analysis.attention_patterns(s, CLEAN, layers=[0, n_layers - 1])
    assert ap["coverage"]["status"] == "complete" and ap["layers"][0]["heads"][0]["attends_to"]

    hs = analysis.head_summary(s, CLEAN)
    assert hs["heads_total"] == n_layers * n_heads

    hc = analysis.head_catalog(s, seq_len=8, n_samples=2)
    assert len(hc["all_heads"]) == n_layers * n_heads and hc["coverage"]["status"] == "complete"

    pt = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome")
    assert pt["coverage"]["status"] == "complete" and pt["n_forward_passes"] == n_layers * 6
    # patching the last layer's residual at the last position restores the clean answer exactly -> normalized == 1
    full = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome", layers=[n_layers - 1], positions=[-1])
    assert abs(full["grid"][0]["normalized"] - 1.0) < 1e-3, full["grid"]

    pth = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome", component="head")
    assert pth["n_forward_passes"] == n_layers * n_heads and "head" in pth["grid"][0]
    for comp in ("attn_out", "mlp_out"):
        r = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome", component=comp, layers=[1])
        assert r["n_forward_passes"] == 6

    budget = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome", max_passes=5)
    assert budget["coverage"]["status"] == "partial" and budget["coverage"]["cells_computed"] == 5 and budget["coverage"]["cells_skipped"]

    dla = analysis.direct_logit_attribution(s, CLEAN, "paris", "rome")
    assert abs(dla["sum_of_attributions"] - dla["actual_target_value"]) < 2e-2, (name, dla["sum_of_attributions"], dla["actual_target_value"])
    assert dla["reconstruction_max_abs_err"] < 1e-3 and dla["coverage"]["status"] == "complete"

    ab = analysis.ablate_heads(s, CLEAN, [[0, 1], [n_layers - 1, 0]], "paris", "rome", mode="mean")
    assert len(ab["single_head_ablations"]) == 2
    analysis.ablate_heads(s, CLEAN, [[0, 1]], "paris", "rome", mode="zero")

    gen = analysis.generate(s, "the capital", max_new_tokens=3)
    assert gen["n_generated"] == 3
    return ll, pt, dla


def check_static(tmpdir: str) -> None:
    art = analysis.inspect_model_artifact(tmpdir)
    a = art["architecture"]
    assert art["read_mode"] == "local-header" and art["coverage"]["status"] == "complete"
    assert a["family"] == "llama-like" and a["n_layers"] == 4 and a["d_model"] == 64 and a["n_heads"] == 4 and a["n_kv_heads"] == 2, a
    assert a["d_head"] == 16 and a["d_mlp"] == 128 and a["final_norm_kind"] == "rmsnorm" and a["config_consistency"] == "consistent", a
    assert art["identity"]["headers_sha256"] and art["identity"]["config_sha256"]
    sel = analysis.inspect_model_artifact(tmpdir, tensor_name_filter=r"layers\.0\.self_attn")
    assert sel["tensors_total_matching"] == 4 and all("layers.0.self_attn" in t["name"] for t in sel["tensors"])


def check_store(s: ModelSession, ll, pt, dla) -> None:
    store = EvidenceStore()
    subj, prof = s.identity(), s.analysis_profile()
    rec = store.record("logit_lens", {"prompt": CLEAN}, subj, dict(ll), limitations=ll["limitations"], authority="inference", analysis_profile=prof)
    assert rec["evidence_id"].startswith("ev_") and store.lookup("logit_lens", {"prompt": CLEAN}, subj, prof)["evidence_id"] == rec["evidence_id"]
    assert store.lookup("logit_lens", {"prompt": CLEAN}, subj, dict(prof, hook_semantics=1)) is None  # profile is part of identity
    p1 = store.record("activation_patching", {"a": 1}, subj, dict(pt), coverage=pt["coverage"], authority="observation", analysis_profile=prof)
    pt2 = analysis.activation_patching(s, CLEAN, CORR, "paris", "rome", layers=[0, 1, 2, 3])
    p2 = store.record("activation_patching", {"a": 2}, subj, dict(pt2), coverage=pt2["coverage"], authority="observation", analysis_profile=prof)
    cmp_ = analysis.compare_circuits(p1, p2)
    assert cmp_["status"] == "same_subject" and cmp_["jaccard_top"] == 1.0 and cmp_["spearman_common"] == 1.0, cmp_
    view = store.view(p1["evidence_id"], "grid", offset=0, limit=5, fields=["layer", "position", "normalized"])
    assert view["total"] == len(pt["grid"]) and len(view["rows"]) == 5 and view["next_offset"] == 5
    u = store.record_unknown("why layer 2?", "attention", rec["evidence_id"])
    try:
        store.update_unknown(u["unknown_id"], "resolved", "x", "ev_bogus")
        raise AssertionError("bogus resolved_by accepted")
    except Exception as e:
        assert "not a retained" in str(e)
    store.update_unknown(u["unknown_id"], "resolved", "it was the induction head", rec["evidence_id"])
    with tempfile.TemporaryDirectory() as d:
        out = store.export_bundle(os.path.join(d, "bundle.json"))
        assert out["records"] == 3 and out["unknowns"] == 1
        bundle = json.load(open(out["path"]))
        bundle["records"][0]["normalized_result"]["tampered"] = True
        json.dump(bundle, open(os.path.join(d, "bad.json"), "w"))
        s2 = EvidenceStore()
        imp = s2.import_bundle(os.path.join(d, "bad.json"))
        assert imp["imported"] == 2 and len(imp["rejected"]) == 1, imp


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        s = tiny_llama(d)
        ll, pt, dla = check_session(s, 4, 4, "llama")
        check_static(d)
        check_store(s, ll, pt, dla)
        s3 = ModelSession()                      # real open() path: local dir, header identity, close
        info = s3.open(d, dtype="float32", device="cpu")
        assert info["identity"]["weights_headers_sha256"] and info["weights"]["n_files"] == 1, info["weights"]
        assert s3.close()["cleanup"] == "complete"
    g = tiny_gpt2()
    check_session(g, 3, 4, "gpt2")
    print("smoke OK: llama + gpt2 + static + store")


if __name__ == "__main__":
    main()
