# Pruning report — /home/claude/bpe/model

## Pruning — capital (no healing)

| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | - | 444032 | 11.354 | 0.708 | 0.672 | 0.031 | 12.437 |
| width | 0.5 | causal | 247232 | 8.602 | 0.521 | -0.071 | 0.031 | 11.298 |
| width | 0.5 | causal_signed | 247232 | 10.634 | 0.354 | -0.670 | 0.000 | 10.384 |
| width | 0.5 | wanda | 247232 | 8.492 | 0.396 | -0.674 | 0.000 | 11.455 |
| width | 0.5 | magnitude | 247232 | 8.204 | 0.479 | -0.009 | 0.031 | 10.841 |
| width | 0.5 | random | 247232.0 | 8.37 | 0.458 | -0.325 | 0.010 | 11.187 |
| width | 0.25 | causal | 173504 | 7.225 | 0.312 | -0.298 | 0.000 | 10.261 |
| width | 0.25 | causal_signed | 173504 | 7.748 | 0.354 | -0.862 | 0.031 | 9.848 |
| width | 0.25 | wanda | 173504 | 8.041 | 0.271 | -0.820 | 0.000 | 11.312 |
| width | 0.25 | magnitude | 173504 | 7.425 | 0.354 | -0.192 | 0.062 | 10.742 |
| width | 0.25 | random | 173504.0 | 7.97 | 0.250 | -0.758 | 0.021 | 10.883 |
| depth | -1 layers | causal | 296064 | 7.592 | 0.667 | 0.591 | 0.031 | 11.128 |
| depth | -1 layers | shortgpt | 296064 | 7.381 | 0.667 | 0.591 | 0.031 | 11.128 |
| depth | -1 layers | random | 296064.0 | 8.1237 | 0.639 | 0.407 | 0.010 | 10.276 |

**Verdict: NOT APPLICABLE: the full model solves only 71% of heldout prompts (< 80%); there is no capability to preserve** (accuracy threshold 5%; margin decides ties at ±10%)

pick a larger model or an easier task

| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |
|---|---|---|---|---|---|---|---|---|

## Pruning — ioi (no healing)

| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | - | 444032 | 15.556 | 0.562 | 0.086 | 0.375 | 12.437 |
| width | 0.5 | causal | 247232 | 10.836 | 0.604 | 0.102 | 0.531 | 11.497 |
| width | 0.5 | causal_signed | 247232 | 12.497 | 0.646 | 0.253 | 0.438 | 8.865 |
| width | 0.5 | wanda | 247232 | 11.533 | 0.583 | 0.139 | 0.438 | 10.954 |
| width | 0.5 | magnitude | 247232 | 11.199 | 0.583 | 0.098 | 0.375 | 10.841 |
| width | 0.5 | random | 247232.0 | 11.762 | 0.528 | 0.046 | 0.479 | 11.187 |
| width | 0.25 | causal | 173504 | 10.287 | 0.500 | 0.040 | 0.625 | 9.530 |
| width | 0.25 | causal_signed | 173504 | 10.399 | 0.646 | 0.246 | 0.469 | 8.618 |
| width | 0.25 | wanda | 173504 | 11.658 | 0.542 | 0.120 | 0.594 | 10.557 |
| width | 0.25 | magnitude | 173504 | 11.055 | 0.562 | 0.083 | 0.375 | 10.742 |
| width | 0.25 | random | 173504.0 | 9.6893 | 0.569 | 0.072 | 0.490 | 10.883 |
| depth | -1 layers | causal | 296064 | 10.907 | 0.521 | 0.104 | 0.375 | 11.128 |
| depth | -1 layers | shortgpt | 296064 | 11.352 | 0.521 | 0.104 | 0.375 | 11.128 |
| depth | -1 layers | random | 296064.0 | 11.1273 | 0.507 | 0.107 | 0.500 | 10.276 |

**Verdict: NOT APPLICABLE: the full model solves only 56% of heldout prompts (< 80%); there is no capability to preserve** (accuracy threshold 5%; margin decides ties at ±10%)

pick a larger model or an easier task

| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |
|---|---|---|---|---|---|---|---|---|

## Pruning — sentiment (no healing)

| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | - | 444032 | 27.438 | 1.000 | 7.630 | 0.438 | 12.437 |
| width | 0.5 | causal | 247232 | 19.905 | 1.000 | 7.062 | 0.438 | 9.524 |
| width | 0.5 | causal_signed | 247232 | 18.845 | 1.000 | 8.414 | 0.438 | 10.597 |
| width | 0.5 | wanda | 247232 | 21.65 | 1.000 | 6.889 | 0.438 | 10.852 |
| width | 0.5 | magnitude | 247232 | 19.206 | 1.000 | 6.560 | 0.438 | 10.841 |
| width | 0.5 | random | 247232.0 | 18.0893 | 0.833 | 3.852 | 0.438 | 11.187 |
| width | 0.25 | causal | 173504 | 15.724 | 1.000 | 7.180 | 0.438 | 9.159 |
| width | 0.25 | causal_signed | 173504 | 16.093 | 1.000 | 8.117 | 0.438 | 10.297 |
| width | 0.25 | wanda | 173504 | 20.998 | 1.000 | 6.170 | 0.438 | 10.175 |
| width | 0.25 | magnitude | 173504 | 15.7 | 1.000 | 3.829 | 0.438 | 10.742 |
| width | 0.25 | random | 173504.0 | 16.3857 | 1.000 | 2.536 | 0.479 | 10.883 |
| depth | -1 layers | causal | 296064 | 19.393 | 1.000 | 7.185 | 0.438 | 9.849 |
| depth | -1 layers | shortgpt | 296064 | 19.283 | 1.000 | 6.025 | 0.438 | 11.128 |
| depth | -1 layers | random | 296064.0 | 19.1223 | 1.000 | 6.798 | 0.438 | 10.276 |

**Verdict: INCONCLUSIVE: 1 wins, 0 losses, 2 ties** (accuracy threshold 5%; margin decides ties at ±10%)

one task, one model: a supported verdict here is a hypothesis for the next scale, not a general result

| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |
|---|---|---|---|---|---|---|---|---|
| depth | 1 | 1.000 / 7.185 | random | 1.000 / 6.798 | 0.000 | +5.7% | 0.000 | **tie** |
| width | 0.25 | 1.000 / 7.180 | wanda | 1.000 / 6.170 | 0.000 | +16.4% | 0.000 | **win** |
| width | 0.5 | 1.000 / 7.062 | wanda | 1.000 / 6.889 | 0.000 | +2.5% | 0.000 | **tie** |
