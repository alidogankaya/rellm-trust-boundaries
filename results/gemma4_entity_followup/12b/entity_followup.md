# Entity follow-up — google/gemma-4-12B-it

48 layers, sliding window 1024, layer types {'sliding_attention': 40, 'full_attention': 8}

| condition | wording | fillers | competing | pipeline | accuracy | mean margin | picked competitor | ~tokens | wrong examples |
|---|---|---|---|---|---|---|---|---|---|
| wording | bu_firmanin | 0 | False | history | 1.0 | +22.96 |  | 144 |  |
| wording | sirketin | 0 | False | history | 1.0 | +22.43 |  | 144 |  |
| wording | onun | 0 | False | history | 1.0 | +23.07 |  | 143 |  |
| wording | ayni_firmanin | 0 | False | history | 1.0 | +22.82 |  | 147 |  |
| wording | no_noun | 0 | False | history | 1.0 | +22.35 |  | 142 |  |
| distance | bu_firmanin | 1 | False | history | 1.0 | +20.31 |  | 182 |  |
| distance | bu_firmanin | 3 | False | history | 1.0 | +15.71 |  | 276 |  |
| distance | bu_firmanin | 6 | False | history | 1.0 | +13.94 |  | 406 |  |
| competing | bu_firmanin | 1 | True | history | 0.0 | -18.03 | 1.0 | 219 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |
| competing | bu_firmanin | 3 | True | history | 0.0 | -17.00 | 1.0 | 298 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |
| pipeline | bu_firmanin | 3 | True | rewritten | 1.0 | +23.46 | 0.0 | 311 |  |
| pipeline | bu_firmanin | 3 | True | entity_slot | 0.0 | -14.45 | 1.0 | 317 | Yıldız Metal Sanayi -> 'K'; Akdeniz Lojistik -> 'Gü' |

## Mechanistic ({'condition': 'competing', 'wording': 'bu_firmanin', 'fillers': 3, 'competing': True}; 0 successes, 12 failures)
- copy heads in successes: []
- what the same positions attend in failures: [{'head': 'L17H0', 'layer_type': 'full_attention', 'attn_on_name': 0.932}, {'head': 'L13H6', 'layer_type': 'sliding_attention', 'attn_on_name': 0.551}, {'head': 'L20H4', 'layer_type': 'sliding_attention', 'attn_on_name': 0.386}, {'head': 'L29H5', 'layer_type': 'full_attention', 'attn_on_name': 0.314}, {'head': 'L11H7', 'layer_type': 'full_attention', 'attn_on_name': 0.301}, {'head': 'L17H10', 'layer_type': 'full_attention', 'attn_on_name': 0.293}]
- attention on the competitor in failures: [{'head': 'L35H5', 'layer_type': 'full_attention', 'attn_on_name': 0.687}, {'head': 'L41H14', 'layer_type': 'full_attention', 'attn_on_name': 0.618}, {'head': 'L39H0', 'layer_type': 'sliding_attention', 'attn_on_name': 0.56}, {'head': 'L41H7', 'layer_type': 'full_attention', 'attn_on_name': 0.522}]
- max attention on the name by layer type: {'success': {'full_attention': 0.0, 'sliding_attention': 0.0}, 'failure': {'full_attention': 1.957, 'sliding_attention': 2.937}}
- logit lens, first layer where the name is on top: success [] / failure [48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
- DLA (direct writers of name vs competitor):
  - {'company': 'Yıldız Metal Sanayi', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Akdeniz Lojistik', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Karadeniz Gıda', 'error': "KeyError: 'attention_share'"}
  - {'company': 'Marmara Tekstil', 'error': "KeyError: 'attention_share'"}