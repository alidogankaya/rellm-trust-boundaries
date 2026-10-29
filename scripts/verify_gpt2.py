"""Verification lane (REA-style `verify:*`): run rellm on GPT-2 small and check against published circuits.

Ground truth from the literature:
- IOI (Wang et al. 2022): name-mover heads L9H9, L9H6, L10H0 carry most of the positive direct logit attribution
  on "When Mary and John went to the store, John gave a drink to" -> " Mary" vs " John".
- Induction heads in GPT-2 small (Olsson et al. 2022 / TransformerLens demos): L5H5, L5H1, L6H9, L7H2, L7H10 score high
  on the repeated-random-token diagnostic.

Needs network access to Hugging Face (downloads gpt2, ~500 MB) — run it on Colab, not in an offline sandbox.
Exit status 0 = all checks passed; 1 = a check failed (the printed report says which).
"""
from __future__ import annotations

import sys

from rellm import ModelSession, analysis

NAME_MOVERS = {(9, 9), (9, 6), (10, 0)}
INDUCTION = {(5, 5), (5, 1), (6, 9), (7, 2), (7, 10)}
CLEAN = "When Mary and John went to the store, John gave a drink to"
CORR = "When Mary and John went to the store, Mary gave a drink to"


def main() -> int:
    s = ModelSession()
    s.open("gpt2", dtype="float32", hash_weights="header")
    print("opened", s.identity()["model_id"], s.facts()["parameters_human"], "facets:", {k: v.get("available") for k, v in s.facets.items()})
    ok = True

    nt = analysis.next_token(s, CLEAN, track_tokens=[" Mary", " John"])
    print("next token:", nt["top"][0], "Mary rank", nt["tracked"][" Mary"]["rank"], "John rank", nt["tracked"][" John"]["rank"])
    ok &= nt["tracked"][" Mary"]["rank"] == 1

    dla = analysis.direct_logit_attribution(s, CLEAN, " Mary", " John")
    top_heads = [(c["layer"], c["head"]) for c in dla["top_positive"] if c["component"] == "head"][:5]
    print("DLA top heads:", top_heads, "| recon err", dla["reconstruction_max_abs_err"], "| sum", dla["sum_of_attributions"], "actual", dla["actual_target_value"])
    hit = len(set(top_heads) & NAME_MOVERS)
    print(f"  name movers in top-5: {hit}/3 (expect >= 2)")
    ok &= hit >= 2
    ok &= abs(dla["sum_of_attributions"] - dla["actual_target_value"]) < 0.1

    pt = analysis.activation_patching(s, CLEAN, CORR, " Mary", " John", component="head", layers=list(range(7, 12)), max_seconds=240)
    top_patch = [(r["layer"], r["head"]) for r in pt["top_effects"][:5]]
    print("head patching top:", top_patch, "| coverage", pt["coverage"]["status"], "| passes", pt["n_forward_passes"])
    ok &= len(set(top_patch) & NAME_MOVERS) >= 1

    ab = analysis.ablate_heads(s, CLEAN, [[9, 9], [9, 6], [10, 0]], " Mary", " John", mode="mean")
    print("ablate name movers: logit diff", ab["baseline"]["metric"], "->", ab["ablated"]["metric"], "(delta", ab["ablated"]["delta"], ")")
    ok &= ab["ablated"]["delta"] < -0.5

    hc = analysis.head_catalog(s, seq_len=50, n_samples=4, seed=0)
    found = {(r["layer"], r["head"]) for r in hc["induction_heads"]}
    print("induction heads found:", sorted(found), f"| known hits {len(found & INDUCTION)}/5 (expect >= 3)")
    ok &= len(found & INDUCTION) >= 3

    ll = analysis.logit_lens(s, "The capital of France is", track_tokens=[" Paris"])
    print("capital-of-France decision layer:", ll["decision_layer"], "final top:", ll["final_top"]["token"])

    print("\nVERIFY", "PASS" if ok else "FAIL")
    s.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
