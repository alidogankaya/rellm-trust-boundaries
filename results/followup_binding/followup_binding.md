# Follow-up binding across models

accuracy = the next token after `get_ticket(id="PROJ-` is the target ticket's number

| condition | wording | fillers | competing | pipeline | gemma-4-12B-it | Qwen3-8B | phi-4 |
|---|---|---|---|---|---|---|---|
| wording | this_ticket | 0 | None | history | 1.00 (+14.2) | 1.00 (+29.7) | 1.00 (+18.8) |
| wording | that_one | 0 | None | history | 1.00 (+14.8) | 1.00 (+28.4) | 1.00 (+20.2) |
| wording | it | 0 | None | history | 1.00 (+14.3) | 1.00 (+30.8) | 1.00 (+19.1) |
| wording | same_ticket | 0 | None | history | 1.00 (+14.1) | 1.00 (+26.9) | 1.00 (+20.1) |
| wording | no_noun | 0 | None | history | 1.00 (+14.4) | 1.00 (+26.8) | 1.00 (+21.4) |
| distance | this_ticket | 1 | None | history | 1.00 (+13.7) | 1.00 (+23.6) | 1.00 (+18.3) |
| distance | this_ticket | 3 | None | history | 1.00 (+11.9) | 1.00 (+20.1) | 1.00 (+10.6) |
| distance | this_ticket | 6 | None | history | 1.00 (+10.2) | 1.00 (+19.3) | 1.00 (+9.4) |
| competing_before | this_ticket | 1 | before | history | 0.94 (+2.2) | 1.00 (+6.6) | 1.00 (+1.7) |
| competing_before | this_ticket | 3 | before | history | 0.81 (+1.3) | 0.62 (+1.1) | 0.38 (-0.3) |
| competing_after | this_ticket | 1 | after | history | 0.00 (-7.6) | 0.06 (-11.4) | 0.00 (-7.5) |
| competing_after | this_ticket | 3 | after | history | 0.00 (-6.3) | 0.00 (-13.9) | 0.00 (-3.5) |
| pipeline | this_ticket | 3 | after | entity_slot | 0.62 (+0.3) | 0.31 (-3.3) | 0.56 (+0.3) |
| pipeline | this_ticket | 3 | after | rewritten | 1.00 (+15.4) | 1.00 (+33.9) | 1.00 (+17.9) |

## google/gemma-4-12B-it: 11.96B, 48 layers, window 1024
- **A_hardest_success** {'condition': 'competing_before', 'wording': 'this_ticket', 'fillers': 3, 'competing': 'before'} (ok 11 / fail 1)
  - target-reading heads (ok): [{'head': 'L33H8', 'type': 'sliding_attention', 'attention': 0.673}, {'head': 'L13H6', 'type': 'sliding_attention', 'attention': 0.586}, {'head': 'L35H14', 'type': 'full_attention', 'attention': 0.575}, {'head': 'L7H11', 'type': 'sliding_attention', 'attention': 0.476}]
  - competitor-reading heads (fail): [{'head': 'L35H14', 'type': 'full_attention', 'attention': 0.701}, {'head': 'L35H12', 'type': 'full_attention', 'attention': 0.681}, {'head': 'L35H7', 'type': 'full_attention', 'attention': 0.588}, {'head': 'L39H0', 'type': 'sliding_attention', 'attention': 0.583}]
  - max attention on target by layer type: {'success': {'full_attention': 2.436, 'sliding_attention': 6.523}, 'failure': {'full_attention': 2.032, 'sliding_attention': 6.759}}
  - logit lens first layer target on top: {'success': [38, 39, 40, 41, 43, 47, 47, 47, 47, 47, 47], 'failure': [48]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': True, 'logit_diff': 0.75, 'attention_share': 2451.31, 'top_positive': [('L11.layer_extra', 104801.2344), ('embed', 2345.416), ('L9.mlp', 1717.3389), ('L0.layer_extra', 1057.2336), ('L8.mlp', 939.9484)], 'top_negative': [('L11.mlp', -104620.3281), ('L9.layer_extra', -3372.0889), ('L0.mlp', -2162.4226), ('L6.mlp', -798.8044)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': True, 'logit_diff': 0.25, 'attention_share': 8449.7, 'top_positive': [('L11.layer_extra', 235016.1875), ('L0.layer_extra', 2552.3601), ('embed', 1897.2463), ('L1.mlp', 1609.0938), ('L9.mlp', 1493.0468)], 'top_negative': [('L11.mlp', -236152.5781), ('L0.mlp', -4726.3545), ('L9.layer_extra', -2694.3804), ('L1.layer_extra', -1629.8059)]}]
- **B_competing_after** {'condition': 'competing_after', 'wording': 'this_ticket', 'fillers': 3, 'competing': 'after'} (ok 0 / fail 12)
  - target-reading heads (ok): []
  - competitor-reading heads (fail): [{'head': 'L14H8', 'type': 'sliding_attention', 'attention': 0.836}, {'head': 'L35H14', 'type': 'full_attention', 'attention': 0.78}, {'head': 'L39H0', 'type': 'sliding_attention', 'attention': 0.721}, {'head': 'L13H6', 'type': 'sliding_attention', 'attention': 0.687}]
  - max attention on target by layer type: {'success': {'full_attention': 0.0, 'sliding_attention': 0.0}, 'failure': {'full_attention': 1.592, 'sliding_attention': 2.548}}
  - logit lens first layer target on top: {'success': [], 'failure': [48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': False, 'logit_diff': -6.375, 'attention_share': -97.065, 'top_positive': [('L11.layer_extra', 62355.9062), ('embed', 1615.3401), ('L9.mlp', 1081.0598), ('L0.layer_extra', 1057.5106), ('L47.layer_extra', 482.7887)], 'top_negative': [('L11.mlp', -62188.8945), ('L9.layer_extra', -1990.6389), ('L0.mlp', -1760.5911), ('L0H14', -450.9951)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': False, 'logit_diff': -7.625, 'attention_share': -49.691, 'top_positive': [('L11.layer_extra', 168113.7656), ('L0.layer_extra', 1931.4954), ('embed', 1358.655), ('L1.mlp', 1194.9595), ('L8.layer_extra', 907.2941)], 'top_negative': [('L11.mlp', -168917.75), ('L0.mlp', -3376.8577), ('L9.layer_extra', -1435.7051), ('L1.layer_extra', -1189.5378)]}]

## Qwen/Qwen3-8B: 8.19B, 36 layers, window None
- **A_hardest_success** {'condition': 'competing_after', 'wording': 'this_ticket', 'fillers': 1, 'competing': 'after'} (ok 0 / fail 12)
  - target-reading heads (ok): []
  - competitor-reading heads (fail): [{'head': 'L4H4', 'type': 'full_attention', 'attention': 0.92}, {'head': 'L13H13', 'type': 'full_attention', 'attention': 0.836}, {'head': 'L22H21', 'type': 'full_attention', 'attention': 0.807}, {'head': 'L29H11', 'type': 'full_attention', 'attention': 0.785}]
  - max attention on target by layer type: {'success': {'full_attention': 0.0}, 'failure': {'full_attention': 7.369}}
  - logit lens first layer target on top: {'success': [], 'failure': [36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': False, 'logit_diff': -8.0, 'attention_share': -0.212, 'top_positive': [('L35H27', 9.3806), ('L34H29', 1.0525), ('L30H6', 0.9683), ('L30H11', 0.745), ('L32.mlp', 0.6514)], 'top_negative': [('L34.mlp', -7.0301), ('L35H26', -4.5666), ('L32H10', -3.9101), ('L30H9', -2.3455)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': False, 'logit_diff': -13.75, 'attention_share': -0.051, 'top_positive': [('L35H27', 8.0904), ('L34H29', 1.2648), ('L30.mlp', 1.0631), ('L30H6', 1.0436), ('L34H31', 0.5765)], 'top_negative': [('L34.mlp', -5.4719), ('L35.mlp', -5.067), ('L35H26', -4.3852), ('L29.mlp', -2.8361)]}]
- **B_competing_after** {'condition': 'competing_after', 'wording': 'this_ticket', 'fillers': 3, 'competing': 'after'} (ok 0 / fail 12)
  - target-reading heads (ok): []
  - competitor-reading heads (fail): [{'head': 'L4H4', 'type': 'full_attention', 'attention': 0.985}, {'head': 'L13H13', 'type': 'full_attention', 'attention': 0.865}, {'head': 'L29H11', 'type': 'full_attention', 'attention': 0.842}, {'head': 'L28H0', 'type': 'full_attention', 'attention': 0.789}]
  - max attention on target by layer type: {'success': {'full_attention': 0.0}, 'failure': {'full_attention': 6.753}}
  - logit lens first layer target on top: {'success': [], 'failure': [36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': False, 'logit_diff': -10.5, 'attention_share': -0.202, 'top_positive': [('L35H27', 10.7543), ('L34H29', 1.3319), ('L35H24', 0.7504), ('L32.mlp', 0.7183), ('L30H11', 0.6806)], 'top_negative': [('L34.mlp', -8.2286), ('L35H26', -5.0717), ('L29H15', -2.8107), ('L32H10', -2.5483)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': False, 'logit_diff': -12.5, 'attention_share': -0.095, 'top_positive': [('L35H27', 7.2624), ('L34H29', 1.1745), ('L30.mlp', 1.152), ('L30H6', 0.9668), ('L34H31', 0.5949)], 'top_negative': [('L34.mlp', -5.0231), ('L35.mlp', -4.3017), ('L35H26', -4.0362), ('L29.mlp', -2.9029)]}]

## microsoft/phi-4: 14.66B, 40 layers, window None
- **A_hardest_success** {'condition': 'competing_before', 'wording': 'this_ticket', 'fillers': 3, 'competing': 'before'} (ok 5 / fail 7)
  - target-reading heads (ok): [{'head': 'L21H25', 'type': 'full_attention', 'attention': 0.753}, {'head': 'L17H22', 'type': 'full_attention', 'attention': 0.738}, {'head': 'L11H17', 'type': 'full_attention', 'attention': 0.726}, {'head': 'L18H18', 'type': 'full_attention', 'attention': 0.692}]
  - competitor-reading heads (fail): [{'head': 'L17H28', 'type': 'full_attention', 'attention': 0.873}, {'head': 'L26H29', 'type': 'full_attention', 'attention': 0.738}, {'head': 'L35H21', 'type': 'full_attention', 'attention': 0.718}, {'head': 'L35H22', 'type': 'full_attention', 'attention': 0.64}]
  - max attention on target by layer type: {'success': {'full_attention': 12.404}, 'failure': {'full_attention': 11.882}}
  - logit lens first layer target on top: {'success': [28, 35, 36, 38, 39], 'failure': [40, 40, 40, 40, 40, 40, 40]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': False, 'logit_diff': -1.125, 'attention_share': 1.29, 'top_positive': [('L35H16', 1.0932), ('L36.mlp', 0.8502), ('L27H36', 0.8081), ('L36H6', 0.7812), ('L31H2', 0.7232)], 'top_negative': [('L32H6', -1.4544), ('L39.mlp', -1.2419), ('L35H19', -1.0079), ('L28H18', -0.9229)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': True, 'logit_diff': 0.375, 'attention_share': -2.181, 'top_positive': [('L32H5', 1.7519), ('L36H6', 0.9752), ('L27H38', 0.7838), ('L35H6', 0.568), ('L35H16', 0.5561)], 'top_negative': [('L32H31', -1.5079), ('L34H38', -0.8125), ('L28H19', -0.7044), ('L27H39', -0.691)]}]
- **B_competing_after** {'condition': 'competing_after', 'wording': 'this_ticket', 'fillers': 3, 'competing': 'after'} (ok 0 / fail 12)
  - target-reading heads (ok): []
  - competitor-reading heads (fail): [{'head': 'L21H4', 'type': 'full_attention', 'attention': 0.927}, {'head': 'L24H24', 'type': 'full_attention', 'attention': 0.91}, {'head': 'L26H26', 'type': 'full_attention', 'attention': 0.897}, {'head': 'L11H17', 'type': 'full_attention', 'attention': 0.879}]
  - max attention on target by layer type: {'success': {'full_attention': 0.0}, 'failure': {'full_attention': 10.789}}
  - logit lens first layer target on top: {'success': [], 'failure': [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]}
  - DLA: [{'target': 'PROJ-1000', 'competitor': 'PROJ-8917', 'correct': False, 'logit_diff': -5.5, 'attention_share': 0.986, 'top_positive': [('L35H16', 2.5759), ('L27H36', 1.2043), ('L36.mlp', 1.0942), ('L36H6', 0.8162), ('L31H2', 0.7975)], 'top_negative': [('L35H19', -2.4736), ('L32H6', -1.5362), ('L32H20', -1.246), ('L28H18', -1.1233)]}, {'target': 'PROJ-2731', 'competitor': 'PROJ-9648', 'correct': False, 'logit_diff': -1.875, 'attention_share': 0.386, 'top_positive': [('L32H5', 1.9351), ('L35H16', 1.414), ('L36H6', 0.9857), ('L27H38', 0.9692), ('L32H35', 0.9039)], 'top_negative': [('L32H31', -1.744), ('L28H19', -0.9436), ('L34.mlp', -0.7444), ('L34H38', -0.7421)]}]