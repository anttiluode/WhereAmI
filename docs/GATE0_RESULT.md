# Gate 0 result — hidden-world calibration

Five seeds, 24-D GRU, no context label.

| quantity | mean ± SD |
| --- | ---: |
| memoryless accuracy | 0.5808 ± 0.0032 |
| GRU accuracy | **0.7596 ± 0.0023** |
| exact Bayes accuracy | 0.7608 ± 0.0024 |
| hidden -> context log-odds R² | **0.99557 ± 0.00051** |
| fitted stay | 0.95025 ± 0.00171 (true 0.960) |
| fitted emission peak | 0.57720 ± 0.00239 (true 0.580) |
| state-intervention L1 vs fitted update | 0.04738 ± 0.00630 |

The GRU nearly matches the exact Bayesian observer and its recurrent state exposes an almost linearly readable belief coordinate. For Gate 0 only, the action law lets us convert output probabilities into an implied context belief; fitting a tiny symmetric HMM to those neural beliefs recovers the hidden-world parameters closely.

## First kernel attacker

Held-out hidden dynamics:

| model | NMSE |
| --- | ---: |
| one shared affine operator | 0.1209 ± 0.0058 |
| shared operator + context bias | 0.1203 ± 0.0058 |
| context-specific affine operators | **0.0336 ± 0.0017** |

So this toy is not explained by merely adding a context-dependent offset. However the context-specific model has more parameters. The next attacker must be complexity-matched: shuffled context labels, random three-way partitions, or equal-size mixture-of-linear experts chosen without access to decoded context.

## What is earned

```text
ambiguous stream
  -> learned recurrent state
  -> near-Bayes hidden-world inference
  -> compact belief coordinate
  -> recoverable approximate update law
  -> provisional context-dependent local dynamics
```

This is a calibration organism, not evidence that brains or transformers literally run this HMM.

## Gate 1

Remove the convenient known action/context permutation. The decoder should have to discover a low-dimensional causal state from black-box trajectories and interventions before fitting an update family.
