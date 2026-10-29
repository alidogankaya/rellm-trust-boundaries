# How to read the result files

Every experiment writes three things: a `predictions.txt` (what was expected, time-stamped before the run),
a `.json` with per-case results, and a `.md` summary.

## predictions.txt

```
written_utc: 2026-10-29T07:27:13+00:00
script_sha256: 8150a2253c26d98e9...
Pre-registered thresholds (written before any real run): ...
```

The SHA-256 is of the script as it ran. Recompute it with `sha256sum scripts/<script>.py` to confirm the
thresholds belong to that exact code.

## trust_boundary.json

Per model:

- `behaviour` — per failure: `clean_accuracy`, `compliance`, `n`, and `direction_status` when a failure
  had too few complied/resisted cases to analyse.
- `injection_strength` — which attack level (weak/medium/strong) the calibration selected.
- `direction_cosine_own_best_layer` / `direction_cosine_shared_layer` + `shared_layer` — geometry, optimistic
  and fair.
- `causal_heads` — the top-8 set per failure, found on the discovery half.
- `head_validation_controls` — found vs random heads on the disjoint validation half, with `found_beats_random`.
- `head_jaccard`, `mean_jaccard` — overlap between sets.
- `cross_causal` — for each ablated set, every failure's `compliance_zero`, `compliance_mean`, `clean`
  (before, after) and a `capability_damaged` flag.
- `predictions` — T1–T5 against the pre-registered thresholds, and `verdict`.

## sycophancy.json

`cases` holds every prompt: condition, the generated reply, the label (correct / caved / other),
per-layer logit-lens probabilities for the correct and wrong answers, the `cave_layer`, and the
attention of the answer position to the asserted token.

## reverse_shield.json

`R1_dose_response`, `R2_steering`, `R3_placebo`, `R4_attention`, `R5_cost` and `predictions`.
Note: in the published run, injection obedience is 100% in every condition — the test was at ceiling
and is reported as uninformative, not as a refutation.
