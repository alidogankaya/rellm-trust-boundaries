"""Derived comparisons between two retained Evidence records.

REA rules carried over: compare only like with like, require complete coverage
on both sides before concluding anything about absence, and report a textual /
numerical difference as exactly that — not as a behavioural difference.
"""
from __future__ import annotations

from typing import Any

from ..session import SessionError

COMPARABLE = {"activation_patching", "direct_logit_attribution", "head_catalog", "head_summary", "logit_lens"}


def _keyed(rec: dict[str, Any]) -> tuple[dict[str, float], str]:
    op = rec["operation"]
    nr = rec["normalized_result"]
    if op == "activation_patching":
        rows = nr["grid"]
        key = (lambda r: f"L{r['layer']}" + (f"H{r['head']}" if "head" in r else "") + f"@{r['position']}")
        val = (lambda r: r["normalized"] if r["normalized"] is not None else r["metric"])
        return {key(r): val(r) for r in rows}, "normalized patch effect"
    if op == "direct_logit_attribution":
        rows = nr["components"]
        key = (lambda c: c["component"] if c["layer"] is None else f"L{c['layer']}" + (f"H{c['head']}" if c["head"] is not None else f".{c['component']}"))
        return {key(c): c["contribution"] for c in rows}, "direct logit contribution"
    if op in ("head_catalog", "head_summary"):
        rows = nr["all_heads"]
        field = "induction" if op == "head_catalog" else "prev_token"
        return {f"L{r['layer']}H{r['head']}": r[field] for r in rows}, field
    if op == "logit_lens":
        return {r["layer"]: float(r["final_answer_prob"]) for r in nr["layers"]}, "final-answer probability"
    raise SessionError("not_comparable", f"operation {op!r} has no comparison projection")


def _spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else None


def compare_circuits(left: dict[str, Any], right: dict[str, Any], top_k: int = 15) -> dict[str, Any]:
    if left["operation"] != right["operation"]:
        raise SessionError("not_comparable", f"left is {left['operation']}, right is {right['operation']}; compare like with like")
    if left["operation"] not in COMPARABLE:
        raise SessionError("not_comparable", f"{left['operation']} results are not comparable; supported: {sorted(COMPARABLE)}")
    if left["operation"] == "activation_patching" and left["normalized_result"]["component"] != right["normalized_result"]["component"]:
        raise SessionError("not_comparable", "patching grids use different components")
    lk, label = _keyed(left)
    rk, _ = _keyed(right)
    common = sorted(set(lk) & set(rk))
    left_only, right_only = sorted(set(lk) - set(rk)), sorted(set(rk) - set(lk))
    both_complete = left["coverage"].get("status") == "complete" and right["coverage"].get("status") == "complete"
    top_l = {k for k, _ in sorted(lk.items(), key=lambda kv: -abs(kv[1]))[:top_k]}
    top_r = {k for k, _ in sorted(rk.items(), key=lambda kv: -abs(kv[1]))[:top_k]}
    shared_top = sorted(top_l & top_r, key=lambda k: -abs(lk.get(k, 0)) - abs(rk.get(k, 0)))
    deltas = sorted(({"key": k, "left": lk[k], "right": rk[k], "delta": round(rk[k] - lk[k], 4)} for k in common), key=lambda d: -abs(d["delta"]))
    same_subject = left["subject"].get("config_sha256") == right["subject"].get("config_sha256") and \
        left["subject"].get("weights_headers_sha256") == right["subject"].get("weights_headers_sha256")
    limitations = [f"values compared: {label}; equal numbers show the same measured effect, not the same mechanism",
                   "components are matched by (layer, head, position) label; across different models those labels are not the same objects"]
    if not both_complete:
        limitations.append("at least one side has partial/unknown coverage: one-sided absence is unknown, not a difference")
    status = "unknown" if not both_complete else ("same_subject" if same_subject else "different_subject")
    return {"operation": left["operation"], "left_evidence_id": left["evidence_id"], "right_evidence_id": right["evidence_id"],
            "subjects": {"left": left["subject"].get("model_id"), "right": right["subject"].get("model_id"), "same_weights": same_subject},
            "status": status, "n_common": len(common), "n_left_only": len(left_only), "n_right_only": len(right_only),
            "left_only_sample": left_only[:top_k], "right_only_sample": right_only[:top_k],
            "top_k": top_k, "shared_top": shared_top, "jaccard_top": round(len(top_l & top_r) / max(1, len(top_l | top_r)), 4),
            "left_top_only": sorted(top_l - top_r), "right_top_only": sorted(top_r - top_l),
            "spearman_common": _spearman([lk[k] for k in common], [rk[k] for k in common]),
            "largest_deltas": deltas[:top_k], "coverage": {"status": "complete" if both_complete else "unknown"},
            "limitations": limitations}
