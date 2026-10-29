# Reverse report — /home/claude/demo/model

3 layers · 4 heads (4 kv) · d_model 128 · d_mlp 256 · 0.00B params · quirks: none

## Model-level head roles (task independent, inference)
- stable induction heads: none
- length-dependent 'induction' heads (positional-shortcut suspects): none
- previous-token heads: none

## Task: syn_capital — synthetic country -> city lookup

Behaviour (observation): train acc 1.00 · heldout 1.00 (margin 9.4113) · OOD 1.00 · scoring first_token


**Where the decision forms** (logit lens, inference): decision-layer histogram {'L0': 32}

| layer | answer top among candidates | mean candidate rank | mean answer prob |
|---|---|---|---|
| embed | 0.0 | 7.0 | 0.0467 |
| L0 | 1.0 | 1.0 | 0.9719 |
| L1 | 1.0 | 1.0 | 0.9944 |
| L2 | 1.0 | 1.0 | 0.9988 |

**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **L0**

| through | acc | margin mean | margin min |
|---|---|---|---|
| embed | 0.0 | -1.2376 | -2.2938 |
| L0 | 1.0 | 6.0023 | 3.2689 |
| L1 | 1.0 | 7.3099 | 5.6644 |
| L2 | 1.0 | 9.218 | 8.7991 |

**Layer skip necessity** (observation, margin drop when one layer is skipped): L0 10.5137, L1 1.2087, L2 1.9081

**Removal necessity** (observation, base margin 9.218): top heads L0H3 2.9919, L0H2 0.4262, L0H0 0.1764, L1H1 0.0382, L1H3 0.0377, L1H0 0.0076, L0H1 0.0034, L2H0 -0.0056, L2H2 -0.0162, L1H2 -0.0176; concentration {'top1_share': 0.791, 'top5_share': 0.97}
MLPs: L0.mlp 5.6308, L1.mlp 1.1675, L2.mlp 1.9587
Heads whose removal *helps*: L2H3 -0.0317, L2H1 -0.0299, L1H2 -0.0176, L2H2 -0.0162, L2H0 -0.0056

**Information flow** (activation patching over 8 pairs, observation)
- sources (positions where clean/corrupted differ): -2 ('greece', token-swap recovery 0.999) matters through only before L0 output (read inside L0), resid effect by layer [0.0002, 0.0001, 0.0]
- last position carries the decision from L0; effect by layer [0.9995, 1.0, 1.0]
- heads at the last position: L0H3 0.7234, L0H2 0.1923, L0H0 0.0639, L0H1 0.005, L2H0 0.002, L1H1 0.0017, L1H2 0.0011, L1H3 0.001
- MLP outputs: L0@-1 0.7338, L1@-1 0.2844, L2@-1 0.2433, L0@-3 0.0001, L0@-2 0.0001, L1@-7 0.0001

**Who writes the answer** (DLA, derived): logit diff 10.7609 · attention 16% · MLP 84% · max reconstruction err 0.0
- positive: L0.mlp 3.4401, L2.mlp 2.8825, L1.mlp 2.7075, L0H3 1.0354, L0H2 0.3407, L0H0 0.1655, L2H0 0.0617, L1H2 0.0393
- negative: L2H2 -0.0367, L2H3 -0.0226, L1H1 0.0027, L2H1 0.0134, embed 0.025

**What the key heads read** (inference)

| head | top positions (from end, token, weight) | mass on answer token | on counter token |
|---|---|---|---|
| L0H3 | -2 'italy' 0.987; -6 'w2' 0.005; -7 'w26' 0.004 | 0.0 | 0.0 |
| L0H2 | -2 'italy' 0.987; -5 'the' 0.004; -3 'of' 0.002 | 0.0 | 0.0 |
| L0H0 | -2 'italy' 0.941; -7 'w26' 0.013; -8 '<bos>' 0.012 | 0.0 | 0.0 |
| L1H1 | -4 'capital' 0.297; -5 'the' 0.25; -2 'italy' 0.148 | 0.0 | 0.0 |
| L1H3 | -4 'capital' 0.441; -7 'w26' 0.123; -6 'w2' 0.107 | 0.0 | 0.0 |
| L0H1 | -2 'italy' 0.41; -3 'of' 0.208; -5 'the' 0.178 | 0.0 | 0.0 |
| L2H0 | -3 'of' 0.483; -1 'is' 0.184; -5 'the' 0.177 | 0.0 | 0.0 |

**Component roles** (derived)
- writer (necessary + writes the answer): L0.mlp, L1.mlp, L2.mlp
- mover (necessary, reads the source position): L0H3

**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer L0 57/256, L1 52/256, L2 59/256
top: L1.n130 0.1562, L2.n185 0.0645, L2.n155 0.061, L2.n222 0.0587, L2.n238 0.0476, L1.n237 0.0429, L1.n165 0.0422, L2.n14 0.0415

_elapsed 4.2 s · evidence ids: behaviour=ev_17804989978, logit_lens=ev_d1087b06435, truncation=ev_9c98d6d90e6, layer_skip_necessity=ev_3383b50dfdc, removal_necessity=ev_eab1378fa2b, activation_patching=ev_3f3e136c8c6, direct_logit_attribution=ev_7d1924caf3a, key_head_attention=ev_22966216af9, component_roles=ev_f5e1a1e5300, neuron_attribution=ev_3fb36cbfe74_
