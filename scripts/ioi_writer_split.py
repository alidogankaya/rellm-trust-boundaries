"""Does Qwen solve IOI by copying the indirect object (GPT-2 style name movers) or by suppressing the repeated subject?

For the heads with the largest direct effect on logit(IO) - logit(S), split that effect into
  +IO part: direct contribution to logit(IO)
  -S  part: minus the direct contribution to logit(S)
and report where each head looks (attention mass on the IO name vs the S name).
  copier     : effect mostly +IO, attends IO
  suppressor : effect mostly -S, attends S   (copy-suppression-like, McDougall et al. 2023)

Usage:  python scripts/ioi_writer_split.py --model Qwen/Qwen2.5-1.5B --out /content/ioi-split-1.5b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm import ModelSession, analysis  # noqa: E402
from rellm.experiment import Scorer, get_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="./ioi-split")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = ModelSession()
    s.open(a.model, dtype=a.dtype, device=a.device, hash_weights="none")
    task = get_task("ioi")
    sc = Scorer(s.tok, task.train)
    ok = [e for e, c in zip(task.train, sc.score(s.model, keep_per_example=True)["correct"]) if c][: a.n]
    io_part, s_part, diff = defaultdict(float), defaultdict(float), defaultdict(float)
    att_io, att_s = defaultdict(float), defaultdict(float)
    for e in ok:
        io, sub = e.answer, [c for c in e.candidates if c != e.answer][0]
        d_io = analysis.direct_logit_attribution(s, e.prompt, io, None, top_k=1)["components"]
        d_s = analysis.direct_logit_attribution(s, e.prompt, sub, None, top_k=1)["components"]
        for c_io, c_s in zip(d_io, d_s):
            if c_io["component"] != "head":
                continue
            k = (c_io["layer"], c_io["head"])
            io_part[k] += c_io["contribution"] / len(ok)
            s_part[k] += -c_s["contribution"] / len(ok)
            diff[k] += (c_io["contribution"] - c_s["contribution"]) / len(ok)
        ids = s.tok(e.prompt, return_tensors="pt")["input_ids"].to(s.device)
        res = s.run(ids, components=(), want_attn=True)
        io_id = s.tok(io, add_special_tokens=False)["input_ids"][0]
        s_id = s.tok(sub, add_special_tokens=False)["input_ids"][0]
        m_io, m_s = (ids[0] == io_id), (ids[0] == s_id)
        for L, att in res.attentions.items():
            w = att[0, :, -1].float()
            for h in range(w.shape[0]):
                att_io[(L, h)] += float(w[h][m_io].sum()) / len(ok)
                att_s[(L, h)] += float(w[h][m_s].sum()) / len(ok)
    top = sorted(diff, key=lambda k: -abs(diff[k]))[: a.top]
    rows = []
    for k in top:
        total = diff[k]
        share_io = io_part[k] / total if abs(total) > 1e-6 else float("nan")
        kind = "copier (+IO)" if share_io >= 0.6 else ("suppressor (-S)" if share_io <= 0.4 else "mixed")
        rows.append({"head": f"L{k[0]}H{k[1]}", "logit_diff_effect": round(total, 3), "+IO_part": round(io_part[k], 3), "-S_part": round(s_part[k], 3),
                     "share_from_IO": round(share_io, 3), "attn_on_IO": round(att_io[k], 3), "attn_on_S": round(att_s[k], 3), "kind": kind})
    md = [f"# IOI writer split — {a.model} ({len(ok)} correct prompts)", "",
          "| head | effect on logit(IO)-logit(S) | +IO part | -S part | share from IO | attention on IO | attention on S | kind |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['head']} | {r['logit_diff_effect']} | {r['+IO_part']} | {r['-S_part']} | {r['share_from_IO']} | {r['attn_on_IO']} | {r['attn_on_S']} | {r['kind']} |" for r in rows]
    pos = [r for r in rows if r["logit_diff_effect"] > 0]
    tot = sum(r["logit_diff_effect"] for r in pos) or 1.0
    sup = sum(r["logit_diff_effect"] for r in pos if r["kind"].startswith("suppressor")) / tot
    cop = sum(r["logit_diff_effect"] for r in pos if r["kind"].startswith("copier")) / tot
    md += ["", f"Of the positive effect carried by these heads: {cop:.0%} from copiers, {sup:.0%} from suppressors.",
           "Note: raw-logit DLA freezes the final-norm scale per prompt; the split is of the direct path only."]
    open(os.path.join(a.out, "ioi_writer_split.md"), "w").write("\n".join(md))
    json.dump({"rows": rows, "copier_share": cop, "suppressor_share": sup, "n": len(ok)}, open(os.path.join(a.out, "ioi_writer_split.json"), "w"), indent=1)
    print("\n".join(md))
    s.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
