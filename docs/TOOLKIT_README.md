# rellm — reverse engineer how an LLM decides

REA (morluto/rea) gives an agent MCP tools to reverse engineer binaries and apps with evidence.
`rellm` does the same for the *inside* of a language model you give it (open weights): where a
decision forms, which heads and MLPs produce it, which activations are causally necessary, and
how two prompts or two models differ. Every result is an Evidence record with `authority`,
`coverage`, `limitations` and an `analysis_profile`; open questions live in an unknown ledger.

Works on any HuggingFace decoder-only model through forward hooks (no TransformerLens conversion):
Llama, Qwen, Mistral, Gemma, Phi, GPT-2 ... Designed for a Colab H100 driven remotely by Claude Code,
Codex or any MCP client. Tested against transformers 5.x and mcp 2.x (also compatible with mcp 1.x).

## What it borrows from REA

| REA mechanism | rellm |
|---|---|
| open target → provider binding, `run_id`, `analysis_profile` in every identity | `rellm_open_model` binds backend/dtype/eager attention; profile = hook semantics + versions |
| subject identity = artifact digest | config sha256 + safetensors **header** digests (optional full hash) |
| `inspect_model_artifact` without loading | `rellm_inspect_model_artifact`: tensor inventory → architecture reconstruction (`declared`) |
| observation / inference / unknown kept distinct | `authority`: observation (patching, ablation) · derived (DLA, compare) · inference (lens, attention, SAE) · declared |
| producer-owned `coverage`, explanatory `limitations` | budgeted grids report `complete`/`partial` with the skipped cells; absent ≠ zero |
| `facet_availability` | probed on open: residual stream, attention weights, per-head output, … |
| retained Evidence references, `inspect_analysis_view` paging | `rellm_compare_circuits(left_evidence_id, right_evidence_id)`, `rellm_inspect_analysis_view` |
| snapshot cache keyed by target+op+params+profile | same call → `cache_hit: true`, no recomputation |
| bounded, cancellable work with partial observations | `max_seconds` / `max_passes` on patching and head catalog |
| untrusted target content marker | `provenance.untrusted_content` on every record carrying model text |
| guided MCP prompts | `investigate_decision`, `compare_models`, `audit_residual_unknowns`, `prepare_bounded_intervention` |
| verify lanes with real fixtures | `scripts/verify_gpt2.py`: IOI name-mover heads + induction heads on GPT-2 small |

## Tools (25)

| Tool | Question it answers |
|---|---|
| `rellm_inspect_model_artifact` | what does the checkpoint declare: family, layers, heads/GQA, MoE, LoRA, quantization, params — without loading |
| `rellm_open_model` / `rellm_close_model` / `rellm_session` | bind a model; identity, profile, facets, architecture facts; release memory |
| `rellm_tokenize` / `rellm_chat_prompt` | exact positions; chat-template rendering |
| `rellm_next_token` / `rellm_generate` | the behaviour to explain |
| `rellm_logit_lens` | at which layer the answer becomes top-1 ("decision layer"), per-layer top-k, tracked answers |
| `rellm_attention_patterns` / `rellm_head_summary` / `rellm_head_catalog` | what heads read; previous-token / induction / duplicate / sink heads |
| `rellm_activation_patching` | causal localisation by layer×position (resid/attn/mlp) or by head, within a budget |
| `rellm_direct_logit_attribution` | exact per-head / per-MLP contribution to the answer logit difference, with reconstruction check |
| `rellm_ablate_heads` | is a head necessary (zero / mean ablation, joint and single) |
| `rellm_sae_features` | readable feature labels via sae_lens (Gemma Scope etc.) |
| `rellm_compare_circuits` | shared/different components between two retained results (two prompts or two models) |
| `rellm_inspect_analysis_view` / `rellm_list_evidence` / `rellm_get_evidence` | page and discover retained evidence |
| `rellm_export_evidence_bundle` / `rellm_import_evidence_bundle` | portable, integrity-checked bundles |
| `rellm_record_unknown` / `rellm_update_unknown` / `rellm_list_unknowns` | open questions; resolving requires a retained evidence id |

## Quick start (Colab)

Open `colab/rellm_colab.ipynb` on an H100/A100 runtime and run it. It installs the package, runs the
self-tests and the GPT-2 verification lane, starts `python -m rellm.server --transport streamable-http`,
opens a Cloudflare quick tunnel and prints:

```
claude mcp add --transport http rellm https://<random>.trycloudflare.com/mcp --header "Authorization: Bearer <token>"
```

Run that locally, restart Claude Code, then use the `investigate_decision` prompt or ask:

```
Open Qwen/Qwen2.5-7B-Instruct with rellm and explain how it decides the capital of France.
Clean "The capital of France is", corrupted "The capital of Italy is", answer " Paris", counter " Rome".
```

## Local / library use

```bash
pip install -e ".[sae]"
python tests/test_smoke.py      # CPU: every analysis path on tiny random Llama + GPT-2, static inspection, evidence store
python tests/test_server.py     # CPU: every MCP tool through the server, cache, views, prompts, HTTP + bearer
python scripts/verify_gpt2.py   # GPU/network: ground-truth circuits on GPT-2 small
python -m rellm.server          # stdio, for a local MCP client
```

```python
from rellm import ModelSession, analysis
s = ModelSession(); s.open("google/gemma-2-2b")
analysis.logit_lens(s, "The capital of France is", track_tokens=[" Paris"])["decision_layer"]
analysis.activation_patching(s, "The capital of France is", "The capital of Italy is", " Paris", " Rome", max_seconds=120)["top_effects"]
analysis.direct_logit_attribution(s, "The capital of France is", " Paris", " Rome")["top_positive"]
analysis.sae_features(s, "The capital of France is", "gemma-scope-2b-pt-res", "layer_12/width_16k/canonical")
```

## Honest limits

- Mechanistic interpretability recovers *circuits* (which components carry which information), not
  source code. Clean, complete algorithms are only recoverable for narrow tasks.
- Logit lens is a heuristic readout (`inference`); early-layer tokens are suggestive only.
- Patching needs position-aligned clean/corrupted prompts and single-token answers.
- DLA freezes the final norm scale and reports the *direct* path; indirect heads show ~0.
- Eager attention is required for attention weights; keep prompts short on 7–9B models.
- Static inspection derives everything from tensor declarations; it never measures behaviour.
- Everything the model emits (tokens, generations, SAE explanations) is untrusted target content.
- Closed API models (GPT, Claude, Gemini) are out of scope: no weights, no internals.

## Environment variables

`HF_TOKEN` (gated models), `RELLM_EVIDENCE_LOG` (JSONL autosave), `RELLM_HOST`, `RELLM_PORT`,
`RELLM_TOKEN` (bearer token required on HTTP; the Colab notebook generates one).

## Offline demo with a known ground truth

`python scripts/demo_synthetic.py` trains a 3-layer model (CPU, ~4 min) on a synthetic language
(16 capital-city facts + repeated sequences) and drives every tool on it. `docs/demo-report.txt` and
`docs/demo-evidence-bundle.json` are one such run. Findings: the country identity is copied to the
last position by three layer-0 heads (99% attention on the country token), layers 0–2 MLPs at the last
position map country→city (≈78% of the logit difference), ablating the three heads drops the logit
difference from 10.1 to 3.7, and the "induction" the model learned is a positional shortcut (all
layer-1 heads attend at fixed offset 18; the copy task collapses to chance at any other period).

## Mechanistic Circuit Compression (prototype)

`python scripts/circuit_compress.py` (after the demo) answers: *can the minimal functional circuit for one
task be turned into a smaller, genuinely faster model?* It verifies the decision layer on an evaluation
set (not one prompt), builds a necessity map by ablation, greedily finds the minimal circuit, prunes MLP
neurons by task importance, then builds a real HF model with fewer layers/heads/neurons and measures it.
On the synthetic model: 3 layers/4 heads/256 neurons → 1 layer/1 head/24 neurons, body parameters 19×
smaller, 6× faster on CPU, 100% task accuracy kept, the unrelated repeat task lost (as expected).
See `docs/circuit-compression-report.txt`.

### Same pipeline on Qwen2 / Qwen3 layouts
`RELLM_DEMO_ARCH=qwen2 python scripts/demo_synthetic.py` trains the same synthetic task in a real
`Qwen2ForCausalLM` (GQA 4q/2kv, q/k/v biases, tied embeddings); `qwen3` adds q/k RMSNorm. All hooks,
DLA reconstruction (error 0.0) and the compression slicer (incl. `layer_types`, kv-group widening)
work unchanged. See `docs/demo-report-qwen2.txt` and `docs/circuit-compression-report-qwen2.txt`.
Finding there: the Qwen2 model implements "copy token q−18" as a two-hop positional relay
(layer 0 shifts by 3, layer 1 reads q−15), confirmed causally by zeroing layer-0 attention at q−15.

## Reverse-then-prune experiment (`rellm.experiment`)

Question: *does circuit-based (causal) importance preserve a task better than standard pruning criteria at the
same parameter budget?* One command runs both phases:

```bash
python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-0.5B --tasks capital,ioi,sentiment --out /content/exp-0.5b
python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-1.5B --tasks capital,ioi,sentiment --out /content/exp-1.5b --heal-steps 100
python scripts/reverse_then_prune.py --model Qwen/Qwen2.5-7B   --tasks capital,ioi --phase reverse --out /content/exp-7b
```

**Phase 1 — reverse** (per task, aggregated over many prompts, train / heldout / OOD splits, every section an Evidence record):
behaviour → logit lens decision-layer histogram → truncation depth → layer-skip necessity → removal necessity of
every head and MLP → information flow (source positions by token-swap test, when the source stops mattering, when
the last position carries the decision) → head patching at the last position → DLA (attention vs MLP share) →
what the key heads read → component roles (writer / mover / indirect / backed-up writer / suppressor) → MLP neuron
attribution (how many neurons hold 90% of the effect) → automatic unknowns. Model level: head catalog at several
lengths (length-dependent "induction" heads = positional-shortcut suspects); cross-task circuit overlap.

**Phase 2 — equal-budget pruning**: arms `causal` (effect size: E|removal effect| of heads, E|attribution| of
neurons, layer-skip necessity), `causal_signed` (signed means, kept as an ablation), `wanda` (structured),
`magnitude` / ShortGPT block influence for depth, `random` (3 seeds). Width levels keep the same number of query heads
per kv group and the same MLP width in every layer for every arm, so budgets and tensor shapes are identical. Every
pruned model is **physically built** (`build_pruned`, verified equal to hook removal to ~2e-7 on Llama/Qwen2/Qwen3)
and scored on heldout and OOD accuracy and margin, general-text NLL, parameters and latency. Optional healing gives
every arm the same KL-distillation budget. The verdict compares `causal` with the best non-causal arm at each level
(accuracy first, margin when accuracies saturate, OOD must not drop) and refuses to judge tasks the full model cannot do.

Findings so far (toy models on CPU, see `docs/experiment-examples/`):
- signed first-order neuron scores **lose** to Wanda (removing many large "negative" neurons does not add up linearly);
- ranking by effect size fixes it: at 25% width the task margin is +36% (Qwen2 layout, where the fix was found) and
  +19% (Llama layout, independent confirmation) over the best baseline, ties elsewhere, no losses;
- causal pruning keeps the task and spends general language ability (higher general NLL) — a specialist extractor;
- verdicts on toy models are INCONCLUSIVE because accuracy saturates; the real test is Qwen2.5 0.5B → 1.5B → 7B.

## Beyond the literature: two head-to-head experiments

**Skill Paths — lossless self-speculative decoding** (`scripts/skill_paths.py`). The draft is the same model with some
layers skipped; the full model verifies every drafted token, so greedy output is identical to the full model. Per workload
(translation, JSON extraction, short writing, code) the script picks equal-size skip sets by `causal_greedy` (iterative
intervention, workload specific), `causal_oneshot`, `early_exit` (LayerSkip/Kangaroo-style draft, no training),
`shortgpt`, `random_search` (stand-in for SWIFT-style search, same evaluation budget) and `generic` (causal path
calibrated on all workloads), then measures held-out token agreement, exact tokens per verification round, estimated
speedup, and wall-clock with HF assisted generation using a physically built draft (outputs checked token for token).

**Hallucination prediction before generation** (`scripts/hallucination_probe.py`). ~150 real countries plus fictional
ones, two templates. Baselines: first-token confidence, entropy, hidden-state probes (SAPLMA-style). Ours: 10
mechanistic features from the automatically found mover heads (attention to the subject, output norm), the OV
projection of the subject representation ("what would be copied": enrichment) and logit-lens confidence around the
handoff layer. Reports cross-validated AUROC, cross-template transfer, 32-label few-shot, and types each real-country
error as an enrichment failure or an extraction failure (Yu et al. 2024 taxonomy).

Pre-registered after the 0.5B seed-1 run: depth pruning uses `causal_iter` (greedy, re-measured after each skipped
layer) as the primary causal arm, because one-shot layer ranking failed on IOI by skipping a layer that is harmless
alone but essential in combination. The logit lens now splits decision layers by answer, to test a "default answer +
override head" mechanism suggested by the bimodal sentiment decisions (16 prompts at L0, 16 at L14 in both seeds).
