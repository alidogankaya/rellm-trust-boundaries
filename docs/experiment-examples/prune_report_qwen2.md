# Pruning report — /home/claude/demo-qwen2/model

## Pruning — syn_capital (no healing)

| kind | level | method | params (body) | latency ms | heldout acc | heldout margin | OOD acc | general NLL |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | - | 444032 | 28.313 | 1.000 | 9.151 | 1.000 | 1.753 |
| width | 0.75 | causal | 370304 | 26.15 | 1.000 | 9.044 | 1.000 | 1.754 |
| width | 0.75 | causal_signed | 370304 | 29.579 | 1.000 | 6.181 | 1.000 | 1.758 |
| width | 0.75 | wanda | 370304 | 29.409 | 1.000 | 8.845 | 1.000 | 1.754 |
| width | 0.75 | magnitude | 370304 | 24.447 | 1.000 | 8.930 | 1.000 | 1.754 |
| width | 0.75 | random | 370304.0 | 26.4193 | 1.000 | 7.981 | 1.000 | 1.752 |
| width | 0.5 | causal | 247232 | 28.11 | 1.000 | 7.028 | 1.000 | 2.163 |
| width | 0.5 | causal_signed | 247232 | 22.691 | 1.000 | 2.591 | 1.000 | 2.412 |
| width | 0.5 | wanda | 247232 | 22.421 | 1.000 | 6.807 | 1.000 | 1.887 |
| width | 0.5 | magnitude | 247232 | 21.519 | 1.000 | 6.678 | 1.000 | 1.776 |
| width | 0.5 | random | 247232.0 | 22.0027 | 0.868 | 3.332 | 0.917 | 2.576 |
| width | 0.25 | causal | 173504 | 18.143 | 1.000 | 6.573 | 1.000 | 2.903 |
| width | 0.25 | causal_signed | 173504 | 18.245 | 0.896 | 2.289 | 1.000 | 2.967 |
| width | 0.25 | wanda | 173504 | 19.164 | 1.000 | 4.829 | 1.000 | 2.447 |
| width | 0.25 | magnitude | 173504 | 18.357 | 1.000 | 4.512 | 1.000 | 1.979 |
| width | 0.25 | random | 173504.0 | 18.9497 | 0.868 | 2.686 | 0.917 | 2.979 |
| depth | -1 layers | causal | 296064 | 18.434 | 1.000 | 6.737 | 1.000 | 1.776 |
| depth | -1 layers | shortgpt | 296064 | 20.011 | 1.000 | 6.737 | 1.000 | 1.776 |
| depth | -1 layers | random | 296064.0 | 23.492 | 1.000 | 6.096 | 1.000 | 2.916 |
| depth | -2 layers | causal | 148096 | 11.857 | 1.000 | 4.061 | 1.000 | 3.850 |
| depth | -2 layers | shortgpt | 148096 | 16.556 | 1.000 | 4.061 | 1.000 | 3.850 |
| depth | -2 layers | random | 148096.0 | 14.4993 | 0.389 | 0.430 | 0.406 | 4.720 |

**Verdict: INCONCLUSIVE: 1 wins, 0 losses, 4 ties** (accuracy threshold 5%; margin decides ties at ±10%)

one task, one model: a supported verdict here is a hypothesis for the next scale, not a general result

| kind | level | causal acc / margin | best other | its acc / margin | Δ acc | Δ margin (rel) | Δ OOD | outcome |
|---|---|---|---|---|---|---|---|---|
| depth | 1 | 1.000 / 6.737 | shortgpt | 1.000 / 6.737 | 0.000 | +0.0% | 0.000 | **tie** |
| depth | 2 | 1.000 / 4.061 | shortgpt | 1.000 / 4.061 | 0.000 | +0.0% | 0.000 | **tie** |
| width | 0.25 | 1.000 / 6.573 | wanda | 1.000 / 4.829 | 0.000 | +36.1% | 0.000 | **win** |
| width | 0.5 | 1.000 / 7.028 | wanda | 1.000 / 6.807 | 0.000 | +3.2% | 0.000 | **tie** |
| width | 0.75 | 1.000 / 9.044 | magnitude | 1.000 / 8.930 | 0.000 | +1.3% | 0.000 | **tie** |
