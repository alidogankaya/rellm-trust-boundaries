# Trust Boundaries Inside Open LLMs

A pre-registered, controlled study of three ways a language model gets pushed off course — **sycophancy**, **fake-authority obedience** and **prompt injection** — and whether they share machinery inside the network.

Models: **Qwen3-8B**, **Gemma 4 12B-it**, **Phi-4 14B**. All code, prompts, time-stamped pre-registrations and per-case results are in this repository.

---

## Headline results

**1. In Qwen3-8B, fake-authority obedience and sycophancy share causally influential attention heads.**

Five heads appear in both causal sets: `L17H27`, `L17H4`, `L18H15`, `L22H0`, `L29H6`. The two behaviours also point the same way inside the model (cosine **0.91** at a shared layer, L22).

| Heads switched off | Fake authority | Sycophancy | Injection (tool field) | Correct answers |
|---|---|---|---|---|
| Fake-authority set | 31% → **0%** | 31% → **0%** | 71% → 100% | 69% → **69%** |
| Sycophancy set | 31% → **0%** | 31% → **0%** | 71% → 43% | 69% → **69%** |
| Injection set | 31% → 28% | 31% → 25% | 71% → **0%** | 69% → 66% |

Either social set zeroes both social failures. Observed clean accuracy remained unchanged on this test set, under both zero- and mean-ablation.

**Controls:** heads are discovered on one half of the complied cases and validated on a **disjoint** half, against five same-size sets of random heads.

| Set | Found heads | Random heads, mean (max of 5) | Verdict |
|---|---|---|---|
| Fake authority | +12.5 | +0.4 (+3.7) | pass |
| Sycophancy | +14.1 | +0.3 (+3.1) | pass |
| Injection (tool) | +30.9 | +4.6 (+11.0) | pass |

**2. The correct answer remains decodable before the final output shifts.**

In Qwen3's caving cases, a logit lens shows the correct answer ahead of the user's wrong one at a middle layer in **19 of 21** prompts. The output flips at layer **~29 of 36**. Head `L28H0` attends to the word the user typed in 18 of 21 cases. Copying the model's own neutral-prompt residual into the pressured prompt at a single position restores the correct answer from layer ~23.

**3. A pre-registered hypothesis that did not replicate.**

An earlier run suggested the social heads also *protect* against prompt injection (+29 points when ablated — a "reverse shield"). Five tests were pre-registered; 4 of 5 came back not supported. But the follow-up setup ran at **100% injection obedience in every condition**, i.e. at ceiling, so it could neither confirm nor refute the effect. The reverse-shield effect **did not replicate under the follow-up setup**; the +29 observation is unresolved, not disproven.

**4. Where it breaks.**

- **Phi-4:** ablating the equivalent heads drove obedience to 0% *and* clean accuracy from 72% to **0%**. The intervention damages the model there; "behaviour changed, accuracy intact" does not transfer.
- **Gemma 4:** obeyed at most 9% at every tested strength, so no direction or head set could be estimated (data-sufficiency rule, ≥4 complied and ≥4 resisted cases).

---

## Model comparison

Obedience rates at attack strengths tuned per model. **These are per-scenario rates, not a safety ranking.**

| Model | Fake authority | Sycophancy | Injection (tool field) | Injection (plain text) |
|---|---|---|---|---|
| Gemma 4 12B-it | 9% (3/32) | 0% (0/32) | 0% (0/14) | 0% (0/14) |
| Qwen3-8B | 31% (10/32) | 31% (10/32) | 71% (10/14) | 0% (0/14) |
| Phi-4 14B | 12% (4/32) | 0% (0/32) | 0% (0/14) | 21% (3/14) |

Control condition: when the user asserts the **correct** answer, no model changes its reply (0% across all three). The effect is obedience to a wrong claim, not sensitivity to any claim.

Note the two injection columns. The same override instruction that Qwen3 ignores in plain prose it follows 71% of the time inside a JSON `system_note` field — the delivery channel matters more than the wording.

---

## Method in brief

- **Scoring.** A single teacher-forced token after the chat template: does the reply start with the correct answer or the asserted wrong one. (Injection: does the reply contain the payload word.)
- **Attack strength** (weak / medium / strong) is calibrated per model and per failure to the level whose obedience is nearest 50%. A circuit that never fires cannot be studied.
- **Locating heads.** On the discovery half, every head is zero-ablated alone and scored by how much the correct-vs-wrong margin recovers; the top 8 form the set. Validated on the disjoint half against random heads.
- **Cross-ablation.** Each set is switched off (zero and mean ablation) while measuring every failure's obedience *and* clean-task accuracy. A clean-accuracy drop above 15 points is flagged `capability_damaged`.
- **Direction geometry.** Direction = mean residual (complied) − mean residual (resisted) at the last position. Cosines are reported both at each failure's own best layer and at one **shared layer** — the fair comparison.
- **Pre-registration.** Thresholds and the verdict rule are written to `predictions.txt` with a UTC time stamp and the script's SHA-256 **before any model is loaded**.

Pre-registered verdict for the main run: **PARTIAL OVERLAP**. Head overlap and same-layer cosine pass for the social pair; averaged over all three failures the Jaccard (0.22) falls under the 0.30 bar, so the rule withholds "one circuit". The supported claim is narrower: one shared social-compliance circuit, with injection largely separate.

---

## Reproducing

Everything runs on a single A100 40 GB in under two hours. The notebooks are self-contained — they write the source files themselves, so no local install is needed.

```bash
pip install -e .

# Sections 3–4, 7: three models, three failures, cross-ablation + controls
python scripts/trust_boundary.py \
  --models google/gemma-4-12B-it,Qwen/Qwen3-8B,microsoft/phi-4 \
  --patch-cases 12 --top-heads 8 --random-controls 5 --out results/trust

# Section 5: knows-then-caves mechanics
python scripts/sycophancy_inside.py \
  --models google/gemma-4-12B-it,Qwen/Qwen3-8B,microsoft/phi-4 --out results/syco

# Section 6: the reverse-shield hypothesis
python scripts/reverse_shield.py --model Qwen/Qwen3-8B --n 12 --out results/shield
```

Or open a notebook in Colab (GPU runtime) and run all:

| Notebook | What it runs |
|---|---|
| `notebooks/trust_boundary_3models.ipynb` | main experiment, three models |
| `notebooks/sycophancy_3models.ipynb` | layer-by-layer caving analysis |
| `notebooks/reverse_shield_qwen3.ipynb` | the rejected hypothesis |
| `notebooks/followup_binding_3models.ipynb` | a separate study: entity binding in follow-up tool calls |

Gemma 4 requires accepting its licence on Hugging Face and an `HF_TOKEN` in Colab Secrets. Qwen3 and Phi-4 need neither.

---

## Full report

The complete write-up — method, every table, the rejected hypothesis, limitations — is in [`docs/FULL_REPORT.md`](docs/FULL_REPORT.md).

## Repository layout

```
rellm/              interpretability toolkit (forward hooks, any HF decoder)
  analysis/         logit lens, attention, activation patching, DLA, SAE
  experiment/       tasks, ablation, pruning, evidence records
scripts/            one script per experiment (see Reproducing)
notebooks/          self-contained Colab notebooks
results/            raw outputs, per-case JSON, pre-registrations
  trust_boundary_controlled/    the run the README reports
  trust_boundary_first_pass/    earlier run, before discovery/validation split
  sycophancy/ reverse_shield/ followup_binding/ gemma4_entity_followup/
figures/            charts used in the write-up
docs/               full report, LinkedIn write-up, toolkit notes
```

`results/*/predictions.txt` carries the UTC timestamp and script hash written before each run. `results/*/*.json` holds every per-case result, so any number in this README can be recomputed without a GPU.

---

## Limitations

- The positive circuit result rests on **one model**. Gemma 4 could not be localized; Phi-4's intervention damages the model.
- Samples are small: 32 facts and 14 documents per model; the validation halves hold 5 cases each. 0/14 still has a ~21% upper confidence bound.
- Scoring is the first reply token. It cannot see a model that names the wrong answer in order to correct it (Phi-4 often opens "Saturn is not…"); the sycophancy run uses a separate 4-token classifier for this.
- Prompts are synthetic, English, single-turn. Injection uses a harmless payload (`BANANA`) and one JSON channel.
- Head sets are the top 8 by single-head ablation; interactions with heads outside the set are not modelled.
- Behavioural steering transfer was flat. The shared-circuit claim rests on direction geometry, head overlap and cross-ablation — not on steering.

## Next steps

1. Freeze the five shared heads and re-test on **200+ unseen cases** per failure with no re-selection; report Wilson intervals.
2. Measure the cost on an independent clean suite (instruction following, multi-turn consistency, general QA), not on the same 32 facts.
3. Re-run the injection comparison at a tuned strength with 100+ documents, then revisit the signed cross-effect with the social heads frozen.
4. Add a fourth model family that caves at a measurable rate, to test whether the shared circuit is Qwen-specific.
5. For Phi-4, try narrower interventions (single-head mean ablation, direction projection) that might remove obedience without the capability collapse.

## Related work

- Sharma et al., *Towards Understanding Sycophancy in Language Models* (2023)
- *Authority Bias in Language Models* (2026) — compares source-authority and user-agreement representations
- PIShield (2025), CounterSteer (2026) — activation-level detection and steering for prompt injection

This study's contribution is putting all three failures through **head-level cross-causal tests with selectivity controls** on the same models, with thresholds fixed in advance.

## Licence

MIT. See `LICENSE`.
