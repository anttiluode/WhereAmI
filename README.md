# WhereAmI

> **Can we decode how a learned machine infers which world it is in, and how that belief changes the computation it performs?**

This repo starts one level above [NeuralAlgorithmDecoding](https://github.com/anttiluode/NeuralAlgorithmDecoding).
That project asked whether fuzzy learned computation can be compressed back into causal equations / programs.
`WhereAmI` asks what happens when **the appropriate computation itself depends on an unobserved situation that must be inferred from history**.

The working decomposition is:

```text
slow learned machine G
        +
current history / evidence
        |
        v
latent belief q(world)
        |
        v
context-conditioned effective computation K(q)
        |
        v
action / prediction
```

The name was accidental. It stayed because it asks the right first question.

## Current receipt

Gate 0 is now runnable on five seeds. With no context label, a 24-D GRU reaches `0.7596 ± 0.0023` accuracy versus `0.5808 ± 0.0032` for a memoryless predictor and `0.7608 ± 0.0024` for the exact Bayesian observer. Its hidden state linearly exposes the true context log-odds with `R² = 0.9956 ± 0.0005`. Fitting a tiny symmetric HMM back from neural behavior recovers `stay = 0.9503` (true `0.960`) and emission peak `0.5772` (true `0.580`).

A first dynamics attacker also matters: shared affine hidden dynamics have held-out NMSE `0.1209`; adding only a context bias gives `0.1203`; context-specific affine operators give `0.0336`. That survives the first "it is only a bias" attack, but **does not yet earn a general situational-kernel claim** because the context-specific model has more parameters. See [`docs/GATE0_RESULT.md`](docs/GATE0_RESULT.md).

## Four sentences we do not silently change

**QUESTION:** Can a learned recurrent system's hidden situational inference be decoded into a compact causal belief-update law plus a family of context-conditioned effective operators?

**PAYOFF:** If yes, black-box decompilation can move from "what algorithm is this network running?" toward "which local algorithm did it load here, why, and how did evidence switch it?"

**FALSIFIER:** If ordinary output distillation / a context bias / a linear state probe explains the behavior as well as the proposed belief-and-operator story, the stronger "situational kernel" interpretation dies.

**INVARIANT:** Context is **not handed to the learner**. It must be inferred from preceding observations; the same current observation must support different behavior under different inferred worlds.

---

# Gate 0 — hidden-world calibration organism

Three latent worlds persist and occasionally switch.

```text
C_t in {A,B,C}
P(stay) = 0.96
```

Each world has a different but overlapping observation distribution. A single symbol is therefore evidence about the world, not an explicit world label.

The action law is deliberately simple:

```text
action = (symbol + hidden_world) mod 3
```

So for the exact same current symbol:

```text
world A -> action 0
world B -> action 1
world C -> action 2
```

(up to the symbol's cyclic offset).

A small GRU sees **only the symbol stream** and is trained to predict actions. It never receives the world identity.

For scoring we retain an exact Bayesian hidden-Markov filter:

```text
q_t(c)
  proportional to
P(x_t | c) * sum_c' P(c | c') q_{t-1}(c')
```

That exact filter is not used by the GRU.

## What the decoder tries to recover

Gate 0 asks four progressively stronger questions.

1. **Behavior:** does recurrence beat a memoryless `symbol -> action` predictor and approach the Bayes observer?
2. **Belief coordinate:** can the GRU hidden state linearly expose the true context log-odds?
3. **Update law:** if neural output probabilities are converted back into an implied context belief, can a two-parameter HMM update recover the hidden `stay` and observation-reliability parameters?
4. **Situational kernel:** holding the next symbol fixed, do high-belief states for A/B/C have measurably different recurrent Jacobians, and do explicit hidden-state swaps obey the fitted belief update?

The last item is intentionally attackable. Different hidden representations do **not** automatically imply different computations.

Run:

```bash
pip install -r requirements.txt
python experiments/gate0_hidden_world.py --seed 0
pytest -q
```

Results are written to `results/`.

---

## What would count as interesting later

Gate 0 is a calibration world where the ground-truth answer is known.

The project becomes more interesting only when we progressively remove gifts to the decoder:

```text
known 3-world family
    -> infer number of worlds
known action/context permutation
    -> infer latent belief coordinates causally
known HMM family
    -> choose update family
exact persistent contexts
    -> graded / compositional context
one recurrent organism
    -> same abstract inference in different realization geometries
small GRU
    -> in-context algorithm selection in a transformer
```

The long-term target is not a pretty manifold plot.

It is an executable receipt resembling:

```text
WHERE AM I?        q(world)
WHAT CHANGED?      q <- UPDATE(q, evidence)
WHAT DO I DO HERE? operator <- K(q)
```

and interventions that force the learned machine to follow that decoded abstraction.
