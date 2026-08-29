# Gate 0B — sampled-choice persistence meter

Claude suggested reinterpreting Gate 0's recovered `stay` parameter as a black-box measure of how sticky a model's inferred world is.

The first version of that claim was too strong. Gate 0 used full output probabilities. Many closed model APIs do not expose comparable token probabilities.

So Gate 0B removes them.

## Observation interface

For each prompt/prefix, the observer gets only one or more sampled choices from the black box.

It does **not** read:

- weights;
- activations;
- logits;
- hidden state.

A naive plug-in estimator fails: empirical choice frequencies are noisy, and fitting belief dynamics to those noisy probabilities biases persistence downward.

The corrected estimator fits the candidate hidden-world law **directly to the sampled choices** under a multinomial likelihood.

## Calibration result

Three independently trained Gate-0 GRUs:

| interface | fitted stay | fitted reliability |
| --- | ---: | ---: |
| full soft outputs | `0.95618 ± 0.00243` | `0.57954 ± 0.00222` |
| **one sampled choice per prefix** | **`0.95397 ± 0.00165`** | `0.58461 ± 0.00187` |

The generating world uses `stay = 0.960`, `reliability = 0.580`.

This does **not** mean one API query is enough. "One sample per prefix" means one sample from each of many distinct queried prefixes. In a seed-0 budget sweep, 384 queried prefixes gave fitted stay about

```text
0.956 ± 0.006
```

over repeated sampled-choice runs.

So an output-only persistence estimator is technically viable in the calibration organism without logits.

## Collision with existing work

The broad claim "measure how LLM beliefs revise from black-box outputs" is already occupied.

Relevant examples:

- Wilie et al. 2024, **Belief-R** — evaluates whether LMs revise or maintain conclusions after new evidence:
  https://aclanthology.org/2024.emnlp-main.586/
- Farmer, Kochar & Lee 2026, **The α-Law of Observable Belief Revision in Large Language Model Inference** — fits an observable multiplicative revision law and model-family update fingerprints:
  https://arxiv.org/abs/2603.19262
- Myakala et al. 2026, **BeliefShift** — longitudinal belief consistency, drift and evidence sensitivity:
  https://arxiv.org/abs/2603.23848
- Dudley et al. 2026, **In-Context Learning Under Regime Change** — studies adaptation to unknown change points:
  https://arxiv.org/abs/2604.16988

So the possible niche is narrower:

> **estimate a model's implicit latent-regime persistence / hazard under controlled hidden-task switches.**

That is closer to change-point behavior than generic belief revision.

## Prompt injection is not yet implied

A benign latent-task switch and a prompt injection are not the same intervention. Prompt injection also involves instruction hierarchy, trust boundaries and adversarial content.

Therefore:

```text
low fitted benign-task persistence
    DOES NOT YET IMPLY
high prompt-injection susceptibility
```

That relationship is an empirical question for a later cross-model experiment.

## Next useful experiment

Build an API-facing forced-choice hidden-task benchmark where:

1. the surface wording is held fixed;
2. preceding demonstrations imply one latent task;
3. the latent task switches without an explicit "new task" label;
4. evidence strength and switch timing are controlled;
5. the fitted hazard/persistence is measured from sampled choices;
6. results are compared against ordinary switch-accuracy / belief-revision metrics.

Only if the fitted parameter is stable across prompt paraphrases and predicts adaptation curves better than simpler metrics does it become an instrument rather than a fitted story.
