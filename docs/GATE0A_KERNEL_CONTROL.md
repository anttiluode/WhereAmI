# Gate 0A — equal-complexity kernel controls

This is the promised attack on the strongest Gate-0 wording.

Five seeds, same trained 24-D GRU organism. The target is next-hidden-state prediction from current hidden state plus next symbol.

| partition / model | held-out NMSE |
| --- | ---: |
| one shared affine transition | `0.12086 ± 0.00583` |
| 3 experts, **decoded context** partition | **`0.03357 ± 0.00174`** |
| 3 experts, shuffled decoded-context labels | `0.12110 ± 0.00579` |
| 3 experts, random partitions | `0.12113 ± 0.00585` |
| 3 experts, PCA-first-axis terciles | `0.06496 ± 0.00363` |
| 3 experts, **k-means hidden-state partition** | **`0.03412 ± 0.00220`** |

The unlabeled k-means clusters align with the decoded context mode at best-permutation accuracy

```text
0.95484 ± 0.02024
```

across the five seeds.

## Verdict

The random/shuffled controls show that three times as many affine parameters do **not** by themselves buy the improvement.

But k-means kills the stronger interpretation.

An unsupervised geometric partition of hidden state recovers essentially the same three regions and the same local-linear advantage as the decoded-context partition. So Gate 0 has **not** established a special extra operator switch beyond ordinary nonlinear state-space regionalization.

The honest receipt is:

```text
context inference
    ->
three strongly separated hidden-state regions
    ->
local affine dynamics fit much better than one global affine map
```

That is compatible with a "situational kernel" picture, but it does not uniquely support it. In this organism the phrase is demoted from claim to hypothesis.

The actually less-obvious Gate-0 result remains the behavioral inversion: fitting a compact hidden-world update law back from the learned machine's outputs.
