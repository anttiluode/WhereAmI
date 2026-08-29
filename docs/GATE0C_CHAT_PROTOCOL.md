# Gate 0C — API-neutral latent-regime protocol

This is the first bridge from the synthetic GRU organism to real chat/sequence models.

It deliberately does **not** contain an OpenAI, Anthropic, Gemini, or local-model client. The benchmark and the measurement are separate from transport.

## Export

```bash
python experiments/gate0c_export_chat_benchmark.py --trials 16 --turns 24
```

This produces 384 independent prefix prompts.

The model is told:

- there are three possible modes;
- mode can change without announcement;
- evidence is noisy;
- the action rule depends on the hidden mode.

The model is **not** told:

- the true mode;
- the true persistence / switch probability;
- the true evidence reliability.

Each prompt contains the full evidence history so every API query is stateless and reproducible.

Ground-truth context/action fields are stored in the local JSONL record but are not part of the prompt text.

## Collect

Send each record's `prompt` to any model and save:

```json
{"id":"trial000_turn000","choice":2}
```

One row per sampled answer is enough for the basic protocol. Repeating an id creates multiple samples for that prefix.

The response must be forced into one of `0,1,2`. Transport-specific parsing belongs in an adapter, not in the scientific scorer.

## Score

```bash
python experiments/gate0c_score_choices.py \
  benchmarks/gate0c_hidden_mode.jsonl \
  responses/my_model.jsonl
```

The scorer reports:

- ordinary choice accuracy;
- accuracy near true latent switches;
- accuracy away from switches;
- fitted effective `stay`;
- fitted effective evidence reliability.

The fit uses sampled choices directly under a multinomial likelihood. It does not require logits.

## What the number means

For a real model, fitted `stay` is **behavioral**:

> under this particular controlled hidden-mode game, what persistence parameter makes the model's sequence of choices most resemble a simple latent-regime filter?

It is not proof that the model internally stores that scalar or runs an HMM.

## Required attackers before calling it a model fingerprint

1. paraphrase the instructions while keeping the generated histories fixed;
2. rename modes/symbols/actions;
3. permute the action mapping;
4. vary evidence reliability;
5. vary true switch rate;
6. change sequence length and query budget;
7. compare fitted stay with simple switch-lag and accuracy metrics;
8. repeat across temperatures / sampling settings.

A useful fingerprint should remain substantially more stable across these nuisance changes than a raw accuracy score.

## Security hypothesis kept separate

Only after a benign-regime persistence measure is reproducible should we ask whether it predicts prompt-injection susceptibility.

That experiment needs a separate adversarial benchmark because an injection is not merely "new evidence that the world changed."
