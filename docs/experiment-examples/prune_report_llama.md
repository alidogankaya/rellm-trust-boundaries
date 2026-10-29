# Pruning report — /home/claude/demo/model

## Pruning — syn_capital (no healing)

| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | - | 492416 | 30.415 | 1.000 | 9.411 | 1.000 | 1.752 |
| width | 0.75 | causal | 369536 | 23.869 | 1.000 | 9.257 | 1.000 | 1.848 |
| width | 0.75 | causal_signed | 369536 | 24.946 | 1.000 | 6.239 | 1.000 | 1.750 |
| width | 0.75 | wanda | 369536 | 30.107 | 1.000 | 9.228 | 1.000 | 1.752 |
| width | 0.75 | magnitude | 369536 | 25.367 | 1.000 | 9.244 | 1.000 | 1.752 |
| width | 0.75 | random | 369536.0 | 25.034 | 1.000 | 8.399 | 1.000 | 1.768 |
| width | 0.5 | causal | 246656 | 21.432 | 1.000 | 8.614 | 1.000 | 2.686 |
| width | 0.5 | causal_signed | 246656 | 22.89 | 1.000 | 5.305 | 1.000 | 1.839 |
| width | 0.5 | wanda | 246656 | 22.581 | 1.000 | 7.567 | 1.000 | 1.750 |
| width | 0.5 | magnitude | 246656 | 21.466 | 1.000 | 8.981 | 1.000 | 1.757 |
| width | 0.5 | random | 246656.0 | 22.92 | 0.896 | 3.902 | 0.875 | 1.855 |
| width | 0.25 | causal | 123776 | 16.414 | 1.000 | 5.140 | 1.000 | 3.243 |
| width | 0.25 | causal_signed | 123776 | 15.438 | 0.875 | 3.617 | 0.812 | 2.655 |
| width | 0.25 | wanda | 123776 | 15.464 | 1.000 | 3.783 | 1.000 | 3.345 |
| width | 0.25 | magnitude | 123776 | 16.026 | 1.000 | 4.318 | 1.000 | 2.260 |
| width | 0.25 | random | 123776.0 | 16.007 | 0.375 | 0.215 | 0.375 | 2.748 |
| depth | -1 layers | causal | 328320 | 19.657 | 1.000 | 8.711 | 1.000 | 3.489 |
| depth | -1 layers | shortgpt | 328320 | 19.467 | 1.000 | 7.568 | 1.000 | 2.002 |
| depth | -1 layers | random | 328320.0 | 20.513 | 1.000 | 8.330 | 1.000 | 2.993 |
| depth | -2 layers | causal | 164224 | 13.599 | 1.000 | 6.715 | 1.000 | 3.942 |
| depth | -2 layers | shortgpt | 164224 | 14.692 | 1.000 | 6.715 | 1.000 | 3.942 |
| depth | -2 layers | random | 164224.0 | 13.184 | 0.368 | 0.787 | 0.406 | 3.996 |

**Verdict: INCONCLUSIVE: 1 wins, 0 losses, 4 ties** (accuracy threshold 5%; margin decides ties at ±10%)

one task, one model: a supported verdict here is a hypothesis for the next scale, not a general result

| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |
|---|---|---|---|---|---|---|---|---|
| depth | 1 | 1.000 / 8.711 | random | 1.000 / 8.330 | 0.000 | +4.6% | 0.000 | **tie** |
| depth | 2 | 1.000 / 6.715 | shortgpt | 1.000 / 6.715 | 0.000 | +0.0% | 0.000 | **tie** |
| width | 0.25 | 1.000 / 5.140 | magnitude | 1.000 / 4.318 | 0.000 | +19.0% | 0.000 | **win** |
| width | 0.5 | 1.000 / 8.614 | magnitude | 1.000 / 8.981 | 0.000 | -4.1% | 0.000 | **tie** |
| width | 0.75 | 1.000 / 9.257 | magnitude | 1.000 / 9.244 | 0.000 | +0.1% | 0.000 | **tie** |
