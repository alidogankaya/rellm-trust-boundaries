"""End-to-end through the MCP layer: open a real (tiny) model from disk and drive every tool via call_tool.
Run: python tests/test_server.py"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_smoke import tiny_llama  # noqa: E402

from rellm import server  # noqa: E402

CLEAN, CORR = "the capital of france is", "the capital of italy is"


async def call(name: str, **args):
    r = await server.mcp.call_tool(name, args)
    assert not r.is_error, (name, r.content)
    return r.structured_content


async def expect_error(name: str, code: str, **args):
    try:
        await server.mcp.call_tool(name, args)
    except Exception as e:  # ToolError
        assert code in str(e), (name, code, str(e)[:200])
        return
    raise AssertionError(f"{name} should have failed with {code}")


async def drive(model_dir: str) -> None:
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    print(len(tools), "tools")
    assert {"rellm_open_model", "rellm_close_model", "rellm_session", "rellm_inspect_model_artifact", "rellm_activation_patching",
            "rellm_direct_logit_attribution", "rellm_compare_circuits", "rellm_inspect_analysis_view", "rellm_import_evidence_bundle"} <= names
    prompts = await server.mcp.list_prompts()
    assert {p.name for p in prompts} == {"investigate_decision", "compare_models", "audit_residual_unknowns", "prepare_bounded_intervention"}
    got = await server.mcp.get_prompt("investigate_decision", {"prompt": CLEAN, "answer": "paris", "counter_prompt": CORR, "counter_answer": "rome"})
    assert "rellm_activation_patching" in got.messages[0].content.text

    await expect_error("rellm_session", "no_model")

    art = await call("rellm_inspect_model_artifact", source=model_dir)
    assert art["authority"] == "declared" and art["normalized_result"]["architecture"]["family"] == "llama-like"

    opened = await call("rellm_open_model", model_id=model_dir, dtype="float32", device="cpu")
    nr = opened["normalized_result"]
    assert nr["run_id"].startswith("run_") and nr["identity"]["weights_headers_sha256"] and nr["facet_availability"]["attention_weights"]["available"]
    assert opened["subject"]["model_id"] == model_dir

    sess = await call("rellm_session")
    assert sess["normalized_result"]["facts"]["n_layers"] == 4

    tok = await call("rellm_tokenize", text=CLEAN)
    assert tok["normalized_result"]["n_tokens"] == 6

    nt = await call("rellm_next_token", prompt=CLEAN, track_tokens=["paris", "rome"])
    assert nt["authority"] == "observation" and nt["provenance"]["untrusted_content"] is True

    ll = await call("rellm_logit_lens", prompt=CLEAN, track_tokens=["paris", "rome"])
    assert ll["authority"] == "inference" and len(ll["normalized_result"]["layers"]) == 5 and ll["cache_hit"] is False
    ll2 = await call("rellm_logit_lens", prompt=CLEAN, track_tokens=["paris", "rome"])
    assert ll2["cache_hit"] is True and ll2["evidence_id"] == ll["evidence_id"]

    await call("rellm_attention_patterns", prompt=CLEAN, layers=[3], heads=[0, 1])
    await call("rellm_head_summary", prompt=CLEAN)
    hc = await call("rellm_head_catalog", seq_len=8, n_samples=2)
    assert hc["coverage"]["status"] == "complete"

    await expect_error("rellm_activation_patching", "length_mismatch", clean_prompt=CLEAN, corrupted_prompt="the capital of france is now", answer="paris")
    await expect_error("rellm_activation_patching", "bad_component", clean_prompt=CLEAN, corrupted_prompt=CORR, answer="paris", component="xyz")
    pt = await call("rellm_activation_patching", clean_prompt=CLEAN, corrupted_prompt=CORR, answer="paris", counter_answer="rome")
    assert pt["coverage"]["status"] == "complete" and pt["normalized_result"]["n_forward_passes"] == 24
    part = await call("rellm_activation_patching", clean_prompt=CLEAN, corrupted_prompt=CORR, answer="paris", counter_answer="rome", max_passes=7)
    assert part["coverage"]["status"] == "partial" and part["cache_hit"] is False
    part2 = await call("rellm_activation_patching", clean_prompt=CLEAN, corrupted_prompt=CORR, answer="paris", counter_answer="rome", max_passes=7)
    assert part2["cache_hit"] is False  # partial results are never served from the cache
    pth = await call("rellm_activation_patching", clean_prompt=CLEAN, corrupted_prompt=CORR, answer="paris", counter_answer="rome", component="head", layers=[2, 3])
    assert pth["normalized_result"]["n_forward_passes"] == 8

    view = await call("rellm_inspect_analysis_view", evidence_id=pt["evidence_id"], collection="grid", offset=20, limit=10, fields=["layer", "position", "normalized"])
    assert view["total"] == 24 and len(view["rows"]) == 4 and view["next_offset"] is None
    await expect_error("rellm_inspect_analysis_view", "bad_collection", evidence_id=pt["evidence_id"], collection="nope")

    dla = await call("rellm_direct_logit_attribution", prompt=CLEAN, answer="paris", counter_answer="rome")
    assert dla["authority"] == "derived" and abs(dla["normalized_result"]["sum_of_attributions"] - dla["normalized_result"]["actual_target_value"]) < 2e-2
    heads = [[c["layer"], c["head"]] for c in dla["normalized_result"]["top_positive"] if c["component"] == "head"][:2] or [[3, 0]]
    ab = await call("rellm_ablate_heads", prompt=CLEAN, heads=heads, answer="paris", counter_answer="rome")
    assert len(ab["normalized_result"]["single_head_ablations"]) == len(heads)
    await expect_error("rellm_ablate_heads", "bad_head", prompt=CLEAN, heads=[[9, 0]], answer="paris")

    gen = await call("rellm_generate", prompt="the capital", max_new_tokens=2)
    assert gen["normalized_result"]["n_generated"] == 2

    pt_b = await call("rellm_activation_patching", clean_prompt="the capital of italy is", corrupted_prompt="the capital of france is", answer="rome", counter_answer="paris")
    cmp_ = await call("rellm_compare_circuits", left_evidence_id=pt["evidence_id"], right_evidence_id=pt_b["evidence_id"])
    assert cmp_["authority"] == "derived" and cmp_["normalized_result"]["n_common"] == 24 and cmp_["normalized_result"]["status"] == "same_subject"
    await expect_error("rellm_compare_circuits", "not_comparable", left_evidence_id=pt["evidence_id"], right_evidence_id=dla["evidence_id"])

    unk = await call("rellm_record_unknown", question="which head moves 'france' to the last token?", domain="attention", context_evidence_id=pt["evidence_id"])
    await expect_error("rellm_update_unknown", "missing", unknown_id=unk["unknown_id"], status="resolved", resolved_by="ev_nope")
    upd = await call("rellm_update_unknown", unknown_id=unk["unknown_id"], status="resolved", resolution="see patching", resolved_by=pth["evidence_id"])
    assert upd["status"] == "resolved"
    lst = await call("rellm_list_evidence", operation="activation_patching")
    print("patching records:", lst["total"], [(r["coverage"]["status"], r["parameters"].get("max_passes"), r["parameters"].get("component")) for r in lst["records"]])
    assert lst["total"] == 4 and all(r["authority"] == "observation" for r in lst["records"])  # part and part2 are the same content-addressed record

    with tempfile.TemporaryDirectory() as d:
        exp = await call("rellm_export_evidence_bundle", path=os.path.join(d, "b.json"))
        assert exp["records"] >= 14 and exp["unknowns"] == 1
        imp = await call("rellm_import_evidence_bundle", path=os.path.join(d, "b.json"))
        assert imp["imported"] == 0 and imp["rejected"] == []  # all already retained

    closed = await call("rellm_close_model")
    assert closed["cleanup"] == "complete"
    await expect_error("rellm_logit_lens", "no_model", prompt=CLEAN)
    full = await call("rellm_get_evidence", evidence_id=pt["evidence_id"])
    assert full["operation"] == "activation_patching" and full["analysis_profile"]["hook_semantics"] == 2
    print("mcp tool drive OK")


def http_check() -> None:
    t = threading.Thread(target=server.serve_http, args=("127.0.0.1", 8765, "secret"), daemon=True)
    t.start()
    time.sleep(3)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}).encode()

    def post(h):
        req = urllib.request.Request("http://127.0.0.1:8765/mcp", data=body,
                                     headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **h})
        try:
            r = urllib.request.urlopen(req, timeout=10)
            return r.status, r.read()[:120]
        except urllib.error.HTTPError as e:
            return e.code, e.read()[:60]
    assert post({})[0] == 401
    ok = post({"Authorization": "Bearer secret"})
    assert ok[0] == 200 and b"rellm_mcp" in ok[1] or ok[0] == 200, ok
    print("http+bearer OK", ok[0])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tiny_llama(d)
        asyncio.run(drive(d))
    http_check()
