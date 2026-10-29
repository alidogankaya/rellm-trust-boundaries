# Reverse report — /home/claude/bpe/model

3 layers · 4 heads (2 kv) · d_model 128 · d_mlp 256 · 0.00B params · quirks: grouped-query attention: 4 query heads share 2 KV heads

## Model-level head roles (task independent, inference)
- stable induction heads: none
- length-dependent 'induction' heads (positional-shortcut suspects): none
- previous-token heads: none

## Task: capital — country -> capital city (factual recall)

Behaviour (observation): train acc 0.54 · heldout 0.71 (margin 0.6718) · OOD 0.03 · scoring first_token

> ⚠ model solves only 54% of train prompts; the circuit of a task the model cannot do is weak evidence

**Where the decision forms** (logit lens, inference): decision-layer histogram {'L0': 17, 'L1': 4, 'L2': 4, 'embed': 1}

| layer | answer top among candidates | mean candidate rank | mean answer prob |
|---|---|---|---|
| embed | 0.038 | 21.38 | 0.0 |
| L0 | 0.692 | 1.5 | 0.2072 |
| L1 | 0.846 | 1.19 | 0.3116 |
| L2 | 1.0 | 1.0 | 0.3418 |

**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **L2**

| through | acc | margin mean | margin min |
|---|---|---|---|
| embed | 0.0385 | -3.4283 | -5.1112 |
| L0 | 0.6923 | 0.3885 | -2.7247 |
| L1 | 0.8462 | 0.787 | -0.3587 |
| L2 | 1.0 | 0.7637 | 0.004 |

**Layer skip necessity** (observation, margin drop when one layer is skipped): L0 3.3078, L1 0.2517, L2 -0.0233

**Removal necessity** (observation, base margin 0.7637): top heads L0H0 0.8382, L0H1 0.4781, L0H2 0.0555, L1H1 0.0141, L2H0 0.0026, L1H0 -0.0011, L1H2 -0.0184, L2H3 -0.0329, L2H1 -0.036, L1H3 -0.0562; concentration {'top1_share': 0.478, 'top5_share': 0.908}
MLPs: L0.mlp 2.2892, L1.mlp 0.151, L2.mlp -0.0209
Heads whose removal *helps*: L0H3 -0.1394, L2H2 -0.0823, L1H3 -0.0562, L2H1 -0.036, L2H3 -0.0329

**Information flow** (activation patching over 8 pairs, observation)
- sources (positions where clean/corrupted differ): -2 (' Italy', token-swap recovery 1.0) matters through only before L0 output (read inside L0), resid effect by layer [0.0087, 0.006, 0.0]; -7 ('Cuba', token-swap recovery 1.0) matters through only before L0 output (read inside L0), resid effect by layer [0.0029, -0.0002, 0.0]
- last position carries the decision from L0; effect by layer [0.9853, 0.9918, 1.0]
- heads at the last position: L0H0 0.5609, L0H1 0.414, L1H2 0.0052, L2H3 0.0037, L0H2 0.0027, L1H0 0.0019, L2H1 0.0011, L0H3 0.0
- MLP outputs: L0@-1 0.8904, L1@-1 0.2511, L2@-1 0.0651, L0@-2 0.0073, L0@-4 0.0057, L1@-2 0.0022

**Who writes the answer** (DLA, derived): logit diff 4.4966 · attention 10% · MLP 90% · max reconstruction err 0.0
- positive: L0.mlp 2.6352, L1.mlp 1.1155, L2.mlp 0.304, L0H0 0.1644, L0H1 0.1606, L1H2 0.0585, L2H0 0.0584, L2H3 0.0187
- negative: L0H3 -0.0232, embed -0.0222, L1H1 -0.0057, L1H0 0.0009, L2H2 0.003

**What the key heads read** (inference)

| head | top positions (from end, token, weight) | mass on answer token | on counter token |
|---|---|---|---|
| L0H0 | -2 ' Greece' 0.738; -7 'Ireland' 0.247; -5 'The' 0.006 | 0.0 | 0.0 |
| L0H1 | -2 ' Greece' 0.737; -7 'Ireland' 0.249; -6 'The' 0.006 | 0.0 | 0.0 |
| L0H2 | -1 ' is' 0.578; -3 ' of' 0.099; -6 'The' 0.098 | 0.0 | 0.0 |
| L1H1 | -3 ' of' 0.372; -1 ' is' 0.169; -4 ' capital' 0.149 | 0.0 | 0.0 |
| L2H0 | -4 ' capital' 0.402; -2 ' Greece' 0.377; -5 'The' 0.152 | 0.0 | 0.0 |
| L1H2 | -3 ' of' 0.245; -6 'The' 0.225; -2 ' Greece' 0.196 | 0.0 | 0.0 |
| L2H3 | -2 ' Greece' 0.302; -3 ' of' 0.271; -5 'The' 0.13 | 0.0 | 0.0 |

**Component roles** (derived)
- writer (necessary + writes the answer): L0.mlp, L1.mlp
- suppressor (pushes against the answer / removal helps): L0H3, L1H3, L2H2
- mover (necessary, reads the source position): L0H0, L0H1

**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer L0 58/256, L1 60/256, L2 55/256
top: L0.n43 0.122, L0.n96 0.0602, L0.n66 0.0543, L0.n46 0.021, L0.n129 0.0194, L0.n255 0.0179, L0.n4 0.0173, L0.n20 0.0159

_elapsed 1.3 s · evidence ids: behaviour=ev_56d633fdc1a, logit_lens=ev_d764379afa8, truncation=ev_8db440d3741, layer_skip_necessity=ev_1de97b6e6b1, removal_necessity=ev_e93c8ba59ae, activation_patching=ev_d121545f2c8, direct_logit_attribution=ev_d97f018bc3f, key_head_attention=ev_d2865d73895, component_roles=ev_c1dcf528e01, neuron_attribution=ev_7841507f752_

## Task: ioi — indirect object identification (name moving)

Behaviour (observation): train acc 0.69 · heldout 0.56 (margin 0.0859) · OOD 0.38 · scoring first_token


**Where the decision forms** (logit lens, inference): decision-layer histogram {'embed': 20, 'L1': 4, 'L0': 7, 'L2': 1}

| layer | answer top among candidates | mean candidate rank | mean answer prob |
|---|---|---|---|
| embed | 0.75 | 1.25 | 0.0 |
| L0 | 0.875 | 1.12 | 0.1521 |
| L1 | 0.969 | 1.03 | 0.3109 |
| L2 | 1.0 | 1.0 | 0.3109 |

**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **L2**

| through | acc | margin mean | margin min |
|---|---|---|---|
| embed | 0.7576 | 0.5145 | -1.2922 |
| L0 | 0.8788 | 1.3755 | -0.7418 |
| L1 | 0.9697 | 1.6762 | -0.1139 |
| L2 | 1.0 | 1.6906 | 0.0028 |

**Layer skip necessity** (observation, margin drop when one layer is skipped): L0 1.6054, L1 0.3788, L2 0.0143

**Removal necessity** (observation, base margin 1.6906): top heads L1H2 0.195, L0H1 0.1802, L0H0 0.0983, L1H0 0.0514, L1H3 0.0453, L1H1 0.0154, L2H3 0.0133, L2H1 0.006, L2H2 -0.0001, L2H0 -0.0032; concentration {'top1_share': 0.251, 'top5_share': 0.84}
MLPs: L0.mlp 1.217, L1.mlp -0.1911, L2.mlp -0.005
Heads whose removal *helps*: L0H3 -0.1275, L0H2 -0.0407, L2H0 -0.0032, L2H2 -0.0001

**Information flow** (activation patching over 8 pairs, observation)
- sources (positions where clean/corrupted differ): -13 (' Daniel', token-swap recovery 0.9544) matters through only before L0 output (read inside L0), resid effect by layer [None, None, None]; -14 (' Alice', token-swap recovery 0.7399) matters through only before L0 output (read inside L0), resid effect by layer [None, None, None]; -12 (' James', token-swap recovery 0.3107) matters through only before L0 output (read inside L0), resid effect by layer [None, None, None]
- last position carries the decision from L1; effect by layer [0.483, 1.0007, 1.0]
- heads at the last position: L0H1 0.2945, L1H3 0.2022, L1H2 0.1936, L0H0 0.1453, L1H0 0.0896, L1H1 0.0453, L0H2 0.0355, L2H2 0.0042
- MLP outputs: L0@-1 0.3752, L1@-1 0.2319, L2@-1 0.0271, L1@-5 0.0004, L0@-7 0.0001, L1@-7 0.0001

**Who writes the answer** (DLA, derived): logit diff 1.4148 · attention 42% · MLP 57% · max reconstruction err 0.0
- positive: L1.mlp 0.4472, L0.mlp 0.2788, L1H3 0.1708, L1H2 0.1606, L1H0 0.1148, L2.mlp 0.0882, L1H1 0.059, L0H1 0.046
- negative: L0H2 -0.0049, L0H3 0.0042, L2H2 0.0043, L2H1 0.0045, L2H3 0.0053

**What the key heads read** (inference)

| head | top positions (from end, token, weight) | mass on answer token | on counter token |
|---|---|---|---|
| L1H2 | -13 ' Daniel' 0.868; -14 'When' 0.124; -11 ' James' 0.003 | 0.867 | 0.129 |
| L0H1 | -13 ' Daniel' 0.461; -11 ' James' 0.095; -4 ' gave' 0.072 | 0.45 | 0.212 |
| L0H0 | -13 ' Daniel' 0.335; -2 ' drink' 0.115; -7 ' store' 0.079 | 0.323 | 0.149 |
| L1H0 | -13 ' Daniel' 0.843; -14 'When' 0.115; -9 ' to' 0.009 | 0.833 | 0.127 |
| L1H3 | -13 ' Daniel' 0.874; -14 'When' 0.124; -3 ' a' 0.001 | 0.873 | 0.125 |

**Component roles** (derived)
- backed-up writer (writes the answer, removal compensated): L1.mlp, L1H3
- writer (necessary + writes the answer): L0.mlp, L1H2
- suppressor (pushes against the answer / removal helps): L0H3
- mover (necessary, reads the source position): L0H0, L0H1, L1H2

**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer L0 73/256, L1 63/256, L2 61/256
top: L0.n179 0.0353, L0.n229 0.0271, L0.n143 0.0262, L0.n190 0.0245, L0.n221 0.0212, L0.n156 0.015, L1.n8 0.0135, L1.n118 0.012

Open unknowns recorded: 2

_elapsed 1.6 s · evidence ids: behaviour=ev_e6a46ead78e, logit_lens=ev_2f52a5bbe3e, truncation=ev_37a431262ff, layer_skip_necessity=ev_2deaca5812c, removal_necessity=ev_a39bd184a3e, activation_patching=ev_7d2aa1b1431, direct_logit_attribution=ev_c9a175871e3, key_head_attention=ev_dd53dbd4025, component_roles=ev_287b85ad267, neuron_attribution=ev_d78d2debc03_

## Task: sentiment — few-shot sentiment classification

Behaviour (observation): train acc 1.00 · heldout 1.00 (margin 7.6303) · OOD 0.44 · scoring first_token


**Where the decision forms** (logit lens, inference): decision-layer histogram {'L0': 18, 'embed': 14}

| layer | answer top among candidates | mean candidate rank | mean answer prob |
|---|---|---|---|
| embed | 0.438 | 1.56 | 0.0 |
| L0 | 1.0 | 1.0 | 0.7286 |
| L1 | 1.0 | 1.0 | 0.8983 |
| L2 | 1.0 | 1.0 | 0.9969 |

**How deep the model must run** (truncation, observation): accuracy within 2pp of full after **L0**

| through | acc | margin mean | margin min |
|---|---|---|---|
| embed | 0.5 | 0.0 | -0.0436 |
| L0 | 1.0 | 4.427 | 3.5557 |
| L1 | 1.0 | 5.7581 | 4.2502 |
| L2 | 1.0 | 7.4944 | 6.924 |

**Layer skip necessity** (observation, margin drop when one layer is skipped): L0 7.49, L1 0.4332, L2 1.7363

**Removal necessity** (observation, base margin 7.4944): top heads L0H0 5.8405, L0H1 0.7141, L2H2 0.0783, L2H3 0.0494, L1H1 0.0144, L1H0 0.0142, L2H0 0.0032, L2H1 -0.0294, L0H2 -0.0586, L1H3 -0.0919; concentration {'top1_share': 0.817, 'top5_share': 0.965}
MLPs: L0.mlp 6.2107, L1.mlp 0.4574, L2.mlp 1.3632
Heads whose removal *helps*: L0H3 -0.1442, L1H2 -0.114, L1H3 -0.0919, L0H2 -0.0586, L2H1 -0.0294

**Information flow** (activation patching over 8 pairs, observation)
- sources (positions where clean/corrupted differ): -5 (' horrible', token-swap recovery 1.0) matters through only before L0 output (read inside L0), resid effect by layer [-0.0, 0.0, 0.0]
- last position carries the decision from L0; effect by layer [1.0013, 1.0001, 1.0]
- heads at the last position: L0H0 0.8666, L0H1 0.2159, L2H2 0.031, L2H3 0.0089, L2H0 0.0058, L2H1 0.0046, L1H3 0.0016, L1H1 0.0015
- MLP outputs: L0@-1 0.957, L2@-1 0.4899, L1@-1 0.3563, L0@-8 0.0, L0@-7 0.0, L0@-6 0.0

**Who writes the answer** (DLA, derived): logit diff 7.4667 · attention 7% · MLP 93% · max reconstruction err 0.0
- positive: L2.mlp 3.1697, L0.mlp 2.2567, L1.mlp 1.5364, L2H2 0.2607, L1H3 0.1182, L0H0 0.0722, L2H3 0.052, L2H0 0.0392
- negative: L0H3 -0.0529, L1H1 -0.0175, embed -0.0016, L1H0 -0.0006, L0H2 -0.0001

**What the key heads read** (inference)

| head | top positions (from end, token, weight) | mass on answer token | on counter token |
|---|---|---|---|
| L0H0 | -5 ' charming' 0.978; -21 ' too' 0.005; -7 ' acting' 0.003 | 0.001 | 0.0 |
| L0H1 | -5 ' charming' 0.542; -2 'Sentiment' 0.06; -21 ' too' 0.05 | 0.008 | 0.007 |
| L2H2 | -36 ' delight' 0.423; -37 ' absolute' 0.149; -35 ',' 0.05 | 0.009 | 0.021 |
| L2H3 | -36 ' delight' 0.117; -14 'Sentiment' 0.093; -20 ' long' 0.058 | 0.009 | 0.013 |
| L1H1 | -22 ' Far' 0.114; -29 '
' 0.086; -6 ' was' 0.076 | 0.028 | 0.013 |
| L2H0 | -20 ' long' 0.228; -17 ' happens' 0.161; -5 ' charming' 0.08 | 0.013 | 0.012 |

**Component roles** (derived)
- indirect (necessary, writes ~nothing directly): L0H0
- writer (necessary + writes the answer): L0.mlp, L2.mlp
- mover (necessary, reads the source position): L0H0, L0H1

**MLP neurons** (attribution patching, inference): neurons for 90% of positive attribution per layer L0 61/256, L1 48/256, L2 33/256
top: L2.n244 0.2112, L0.n138 0.108, L0.n31 0.0813, L2.n37 0.074, L0.n211 0.0649, L1.n156 0.0581, L0.n191 0.0556, L1.n40 0.055

Open unknowns recorded: 1

_elapsed 2.4 s · evidence ids: behaviour=ev_a98e142c194, logit_lens=ev_f17a9689426, truncation=ev_78cf6f94ee0, layer_skip_necessity=ev_72300c74a79, removal_necessity=ev_5bd1e51abe5, activation_patching=ev_4aa449e46f6, direct_logit_attribution=ev_45f3e116dae, key_head_attention=ev_f92c38f30c5, component_roles=ev_9397629cac1, neuron_attribution=ev_688628b96c9_

## Cross-task circuit overlap (top heads by removal necessity)
- capital vs ioi: Jaccard 1.0 shared L0H0, L0H1, L0H2, L0H3, L1H0, L1H1, L1H2, L1H3, L2H0, L2H1
- capital vs sentiment: Jaccard 1.0 shared L0H0, L0H1, L0H2, L0H3, L1H0, L1H1, L1H2, L1H3, L2H0, L2H1
- ioi vs sentiment: Jaccard 1.0 shared L0H0, L0H1, L0H2, L0H3, L1H0, L1H1, L1H2, L1H3, L2H0, L2H1
