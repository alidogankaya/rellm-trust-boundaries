# Entity follow-up — google/gemma-4-E4B-it

42 layers, sliding window 512, layer types {'sliding_attention': 35, 'full_attention': 7}

| condition | wording | fillers | competing | pipeline | accuracy | mean margin | picked competitor | ~tokens | wrong examples |
|---|---|---|---|---|---|---|---|---|---|
| wording | bu_firmanin | 0 | False | history | 1.0 | +17.32 |  | 140 |  |
| wording | sirketin | 0 | False | history | 1.0 | +15.96 |  | 140 |  |
| wording | onun | 0 | False | history | 1.0 | +18.36 |  | 139 |  |
| wording | ayni_firmanin | 0 | False | history | 1.0 | +19.73 |  | 143 |  |
| wording | no_noun | 0 | False | history | 1.0 | +14.47 |  | 138 |  |
| distance | bu_firmanin | 1 | False | history | 1.0 | +14.86 |  | 178 |  |
| distance | bu_firmanin | 3 | False | history | 1.0 | +12.06 |  | 272 |  |
| distance | bu_firmanin | 6 | False | history | 1.0 | +8.98 |  | 402 |  |
| competing | bu_firmanin | 1 | True | history | 0.0 | -6.77 | 1.0 | 215 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |
| competing | bu_firmanin | 3 | True | history | 0.0 | -8.79 | 1.0 | 294 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |
| pipeline | bu_firmanin | 3 | True | rewritten | 1.0 | +21.50 | 0.0 | 307 |  |
| pipeline | bu_firmanin | 3 | True | entity_slot | 0.0 | -8.88 | 1.0 | 313 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |

## Mechanistic ({'condition': 'competing', 'wording': 'bu_firmanin', 'fillers': 3, 'competing': True}; 0 successes, 12 failures)
- copy heads in successes: []
- what the same positions attend in failures: [{'head': 'L5H7', 'layer_type': 'full_attention', 'attn_on_name': 0.936}, {'head': 'L11H5', 'layer_type': 'full_attention', 'attn_on_name': 0.324}, {'head': 'L23H3', 'layer_type': 'full_attention', 'attn_on_name': 0.32}, {'head': 'L17H1', 'layer_type': 'full_attention', 'attn_on_name': 0.248}, {'head': 'L35H7', 'layer_type': 'full_attention', 'attn_on_name': 0.211}, {'head': 'L12H1', 'layer_type': 'sliding_attention', 'attn_on_name': 0.181}]
- attention on the competitor in failures: [{'head': 'L37H6', 'layer_type': 'sliding_attention', 'attn_on_name': 0.818}, {'head': 'L30H4', 'layer_type': 'sliding_attention', 'attn_on_name': 0.762}, {'head': 'L39H7', 'layer_type': 'sliding_attention', 'attn_on_name': 0.717}, {'head': 'L32H7', 'layer_type': 'sliding_attention', 'attn_on_name': 0.661}]
- max attention on the name by layer type: {'success': {'full_attention': 0.0, 'sliding_attention': 0.0}, 'failure': {'full_attention': 2.286, 'sliding_attention': 1.593}}
- logit lens, first layer where the name is on top: success [] / failure [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
- DLA (direct writers of name vs competitor):
  - {'company': 'Yıldız Metal Sanayi', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Akdeniz Lojistik', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Karadeniz Gıda', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Marmara Tekstil', 'error': "KeyError: 'attention_share'"}