# Reverse report — /home/claude/demo-qwen2/model

3 layers · 4 heads (2 kv) · d_model 128 · d_mlp 256 · 0.00B params · quirks: grouped-query attention: 4 query heads share 2 KV heads

## Model-level head roles (task independent, inference)
- stable induction heads: none
- length-dependent 'induction' heads (positional-shortcut suspects): L1H0, L1H1, L1H2, L1H3
- previous-token heads: none

## Task: syn_capital — synthetic country -> city lookup

Behaviour (observation): train acc 1.00 · heldout 1.00 (margin 9.1509) · OOD 1.00 · scoring first_token


**Where the decision forms** (logit lens, inference): decision-layer histogram {'L0': 32}

| layer | answer top among candidates | mean candidate rank | mean answer prob |
|---|---|---|---|
| embed | 0.0 | 8.66 | 0.0 |
| L0 | 1.0 | 1.0 | 0.8392 |
| L1 | 1.0 | 1.0 | 0.9913 |
| L2 | 1.0 | 1.0 | 0.9987 |

**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **L0**

| through | acc | margin mean | margin min |
|---|---|---|---|
| embed | 0.0 | -0.6172 | -2.4139 |
| L0 | 1.0 | 4.4103 | 2.7933 |
| L1 | 1.0 | 6.7791 | 4.5776 |
| L2 | 1.0 | 9.1707 | 8.7221 |

**Layer skip necessity** (observation, margin drop when one layer is skipped): L0 13.0137, L1 3.0445, L2 2.3916

**Removal necessity** (observation, base margin 9.1707): top heads L0H3 1.9286, L0H0 1.3063, L0H2 1.2353, L0H1 0.2017, L1H2 0.161, L1H0 0.1538, L1H1 0.0912, L1H3 0.061, L2H3 0.0182, L2H0 0.0005; concentration {'top1_share': 0.373, 'top5_share': 0.935}
MLPs: L0.mlp 8.1106, L1.mlp 2.2463, L2.mlp 2.2964
Heads whose removal *helps*: L2H1 -0.0048, L2H2 -0.0044

**Information flow** (activation patching over 8 pairs, observation)
- sources (positions where clean/corrupted differ): -2 ('greece', token-swap recovery 1.0013) matters through only before L0 output (read inside L0), resid effect by layer [0.0003, 0.0001, 0.0]
- last position carries the decision from L0; effect by layer [0.9998, 0.9999, 1.0]
- heads at the last position: L0H3 0.4018, L0H2 0.3811, L0H0 0.1114, L1H2 0.0088, L1H3 0.0016, L2H1 0.0016, L2H2 0.0016, L2H0 0.0014
- MLP outputs: L0@-1 0.7375, L1@-1 0.3748, L2@-1 0.3401, L0@-8 0.0, L0@-7 -0.0, L0@-6 0.0

**Who writes the answer** (DLA, derived): logit diff 10.1319 · attention 12% · MLP 88% · max reconstruction err 0.0
- positive: L2.mlp 3.3451, L0.mlp 2.9827, L1.mlp 2.6078, L0H3 0.4737, L0H2 0.4453, L0H0 0.2008, L1H0 0.0415, L2H1 0.026
- negative: L1H2 -0.0377, L1H3 -0.0113, L0H1 -0.0019, embed 0.0001, L1H1 0.0048

**What the key heads read** (inference)

| head | top positions (from end, token, weight) | mass on answer token | on counter token |
|---|---|---|---|
| L0H3 | -2 'italy' 1.0; -3 'of' 0.0; -7 'w26' 0.0 | 0.0 | 0.0 |
| L0H0 | -2 'italy' 1.0; -4 'capital' 0.0; -7 'w26' 0.0 | 0.0 | 0.0 |
| L0H2 | -2 'italy' 1.0; -3 'of' 0.0; -7 'w26' 0.0 | 0.0 | 0.0 |
| L0H1 | -2 'italy' 0.461; -4 'capital' 0.359; -7 'w26' 0.111 | 0.0 | 0.0 |
| L1H2 | -5 'the' 0.459; -2 'italy' 0.223; -4 'capital' 0.194 | 0.0 | 0.0 |
| L1H3 | -5 'the' 0.901; -2 'italy' 0.04; -4 'capital' 0.029 | 0.0 | 0.0 |

**Component roles** (derived)
- writer (necessary + writes the answer): L0.mlp, L1.mlp, L2.mlp
- indirect (necessary, writes ~nothing directly): L0H0
- mover (necessary, reads the source position): L0H0, L0H2, L0H3

**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer L0 65/256, L1 53/256, L2 57/256
top: L2.n159 0.1959, L2.n198 0.1237, L2.n67 0.0756, L2.n130 0.0419, L2.n0 0.0407, L0.n88 0.0378, L0.n149 0.0378, L0.n225 0.0367

Open unknowns recorded: 1

_elapsed 4.4 s · evidence ids: behaviour=ev_a4964b3cd49, logit_lens=ev_14c57644ee1, truncation=ev_a42153b51db, layer_skip_necessity=ev_0ad95de4da7, removal_necessity=ev_207a69208a3, activation_patching=ev_e3c2976b058, direct_logit_attribution=ev_0f71f4cd210, key_head_attention=ev_72f7da9785c, component_roles=ev_5ed71d19b48, neuron_attribution=ev_f15b8286512_
