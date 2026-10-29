"""Bar chart of followup_binding.json for the post.  python scripts/plot_binding.py <out dir>"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(os.path.join(sys.argv[1], "followup_binding.json")))
models = [m for m, r in d.items() if "grid" in r]
groups = [("5 wordings of\n'this ticket'", lambda r: r["condition"] == "wording"),
          ("6 unrelated turns\nin between", lambda r: r["condition"] == "distance" and r["fillers"] == 6),
          ("other ticket\nmentioned BEFORE", lambda r: r["condition"] == "competing_before" and r["fillers"] == 3),
          ("other ticket\nmentioned AFTER", lambda r: r["condition"] == "competing_after" and r["fillers"] == 3),
          ("+ 'Active ticket: X'\nin system prompt", lambda r: r["pipeline"] == "entity_slot"),
          ("+ follow-up rewritten\nwith the id", lambda r: r["pipeline"] == "rewritten")]
fig, ax = plt.subplots(figsize=(12, 5.4), dpi=160)
w = 0.8 / max(1, len(models))
colors = ["#1f4e79", "#c0504d", "#4f8a4f", "#7f6000"]
for k, m in enumerate(models):
    vals = []
    for _, sel in groups:
        rs = [r for r in d[m]["grid"] if sel(r)]
        vals.append(100 * sum(r["accuracy"] for r in rs) / max(1, len(rs)))
    xs = [i + (k - (len(models) - 1) / 2) * w for i in range(len(groups))]
    bars = ax.bar(xs, vals, w, label=m.split("/")[-1], color=colors[k % len(colors)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylim(0, 112); ax.set_ylabel("follow-up tool call uses the right ticket id (%)")
ax.set_title("\"Who is assigned to this ticket?\" - which ticket does the model mean?", fontsize=12, pad=10)
ax.legend(fontsize=9, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); out = os.path.join(sys.argv[1], "followup_binding.png"); plt.savefig(out); print("saved", out)
