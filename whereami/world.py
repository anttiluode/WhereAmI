from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    """Symmetric 3-world hidden Markov environment."""

    stay: float = 0.96
    emission_peak: float = 0.58
    n_contexts: int = 3
    n_symbols: int = 3
    n_actions: int = 3

    def transition(self) -> np.ndarray:
        off = (1.0 - self.stay) / (self.n_contexts - 1)
        T = np.full((self.n_contexts, self.n_contexts), off, dtype=np.float64)
        np.fill_diagonal(T, self.stay)
        return T

    def emission(self) -> np.ndarray:
        off = (1.0 - self.emission_peak) / (self.n_symbols - 1)
        E = np.full((self.n_contexts, self.n_symbols), off, dtype=np.float64)
        for c in range(self.n_contexts):
            E[c, c] = self.emission_peak
        return E


def action_for(context: np.ndarray | int, symbol: np.ndarray | int) -> np.ndarray:
    """Each world gives the same symbol a different meaning."""
    return (np.asarray(context) + np.asarray(symbol)) % 3


def generate_batch(
    rng: np.random.Generator,
    batch_size: int,
    steps: int,
    cfg: WorldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = cfg.transition()
    E = cfg.emission()

    contexts = np.empty((batch_size, steps), dtype=np.int64)
    symbols = np.empty_like(contexts)
    actions = np.empty_like(contexts)

    contexts[:, 0] = rng.integers(0, cfg.n_contexts, size=batch_size)
    for t in range(steps):
        if t > 0:
            u = rng.random(batch_size)
            prev = contexts[:, t - 1]
            # Symmetric 3-state transition sampled without a Python loop over batch.
            stay_mask = u < cfg.stay
            contexts[:, t] = prev
            switch_idx = np.flatnonzero(~stay_mask)
            if switch_idx.size:
                jump = rng.integers(0, 2, size=switch_idx.size) + 1
                contexts[switch_idx, t] = (prev[switch_idx] + jump) % 3

        u = rng.random(batch_size)
        c = contexts[:, t]
        # Peak on symbol==context; the two alternatives share the remaining mass.
        peak_mask = u < cfg.emission_peak
        symbols[:, t] = c
        other = np.flatnonzero(~peak_mask)
        if other.size:
            jump = rng.integers(0, 2, size=other.size) + 1
            symbols[other, t] = (c[other] + jump) % 3
        actions[:, t] = action_for(c, symbols[:, t])

    return symbols, actions, contexts


def bayes_filter(symbols: np.ndarray, cfg: WorldConfig) -> np.ndarray:
    """Exact P(context_t | symbols_0:t) for scoring only."""
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.ndim == 1:
        symbols = symbols[None, :]
    B, Tn = symbols.shape
    trans = cfg.transition()
    emission = cfg.emission()
    q = np.empty((B, Tn, cfg.n_contexts), dtype=np.float64)
    prev = np.full((B, cfg.n_contexts), 1.0 / cfg.n_contexts, dtype=np.float64)

    for t in range(Tn):
        prior = prev if t == 0 else prev @ trans
        likelihood = emission[:, symbols[:, t]].T
        post = prior * likelihood
        post /= post.sum(axis=1, keepdims=True)
        q[:, t] = post
        prev = post
    return q


def action_distribution_from_belief(q: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Bayes action distribution induced by context belief."""
    q = np.asarray(q)
    symbols = np.asarray(symbols)
    out = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    for c in range(3):
        a = (symbols + c) % 3
        np.put_along_axis(out, a[..., None], q[..., c, None], axis=-1)
    return out
