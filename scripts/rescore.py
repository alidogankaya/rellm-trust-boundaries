"""Recompute verdicts from an existing results.json with the current rules (no model needed).
  python scripts/rescore.py /content/exp-0.5b/results.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rellm.experiment import get_task, prune_md  # noqa: E402
from rellm.experiment.prune import verdict  # noqa: E402

res = json.load(open(sys.argv[1]))
for name, pr in res["prune"].items():
    t = get_task(name)
    chance = sum(1 / len(e.candidates) for e in t.heldout) / len(t.heldout)
    pr["verdict"] = verdict(pr["rows"], chance=chance)
    print(f"{name}: {pr['verdict']['text']}")
    for c in pr["verdict"]["comparisons"]:
        print(f"   {c['kind']:5s} {str(c['level']):5s} causal {c['causal_heldout']:.3f} vs {c['best_other']:9s} {c['best_other_heldout']:.3f} -> {c['outcome']}")
out = os.path.join(os.path.dirname(sys.argv[1]), "prune_report_rescored.md")
open(out, "w").write("\n".join(prune_md(n, pr) for n, pr in res["prune"].items()))
print("wrote", out)
