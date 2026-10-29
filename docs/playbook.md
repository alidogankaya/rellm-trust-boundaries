# Investigation playbook

Each task gives a clean/corrupted prompt pair of equal token length and a single-token answer pair.
Check lengths with `rellm_tokenize` first; tokenizers differ (leading space matters: `" Paris"`).

## 1. Factual recall — "capital of"
- clean `The capital of France is` / corrupted `The capital of Italy is`
- answer `" Paris"` / counter `" Rome"`
- Expect: decision forms in the middle-late layers; patching resid at the *subject* position (France) in early-mid
  layers and at the last position in late layers; DLA dominated by a few late heads + mid MLPs (MLPs store the fact,
  heads move it to the last token).

## 2. Indirect object identification (IOI)
- clean `When Mary and John went to the store, John gave a drink to`
- corrupted `When Mary and John went to the store, Mary gave a drink to`
- answer `" Mary"` / counter `" John"`
- Expect: duplicate-token heads on the second name, S-inhibition heads, name-mover heads writing the answer (DLA top).

## 3. Greater-than
- clean `The war lasted from the year 1732 to the year 17` / corrupted `... from the year 1701 to the year 17`
- answer: track with logit lens; metric needs a probability mass over `>32` vs `<=32`, so use DLA/patching with
  `answer=" 5"`-style single digits as a proxy and say so in limitations.

## 4. Induction
- `rellm_head_catalog` first (random repeated tokens), then confirm on text:
  clean `Mr Dursley was ... Mr Durs` style repeated name; answer = the token that followed the first occurrence.

## 5. Instruction following / refusal (chat models)
- Render with `rellm_chat_prompt`, pass `add_special_tokens=false`.
- Compare a benign vs a refused request of the same length; answer `"Sure"` vs `"I"` (first token of refusal).
- Patching by layer at the last positions localises where the refusal decision is written; DLA shows the heads.

## Reporting template
```
Behaviour: <prompt> -> <answer> (p=..., ev_...)
Decision layer: L<k> (logit lens, ev_...)
Causal path: resid L<a>-<b> @ position "<token>" (norm. effect ...), heads [L.H ...] (patching ev_...)
Writers: heads [...] + MLP L[...] contribute ... of logit diff (DLA ev_...)
Necessity: ablating [...] drops logit diff by ... (ev_...)
Unknowns: unk_... (what moves the fact from subject to last position between L<a> and L<b>?)
```

## Budgets and caching
- `rellm_activation_patching` accepts `max_seconds` / `max_passes`; a 32-layer × 20-token resid grid is 640 passes
  (~30 s on an H100 for a short prompt); a head grid at one position is layers × heads passes — narrow `layers` first.
- Identical calls (same model, params, analysis profile) return the retained record with `cache_hit: true`; partial
  results are never cached. Page big results with `rellm_inspect_analysis_view` (`grid`, `components`, `all_heads`, `layers`).
- Compare two prompts or two models with `rellm_compare_circuits` on same-operation evidence ids.
