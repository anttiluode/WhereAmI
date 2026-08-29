from __future__ import annotations

import numpy as np


def belief_from_symbols(symbols: np.ndarray, stay: float, emission_peak: float) -> np.ndarray:
    """Symmetric three-world filter used as a candidate behavioral law."""
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.ndim == 1:
        symbols = symbols[None, :]
    B, T = symbols.shape
    q = np.empty((B, T, 3), dtype=np.float64)
    prev = np.full((B, 3), 1.0 / 3.0, dtype=np.float64)

    for t in range(T):
        if t == 0:
            prior = prev
        else:
            off_t = (1.0 - stay) / 2.0
            prior = off_t + (stay - off_t) * prev

        off_e = (1.0 - emission_peak) / 2.0
        like = np.full((B, 3), off_e, dtype=np.float64)
        like[np.arange(B), symbols[:, t]] = emission_peak
        post = prior * like
        post /= post.sum(axis=1, keepdims=True)
        q[:, t] = post
        prev = post
    return q


def action_probs_from_belief(q: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=np.int64)
    out = np.zeros_like(q)
    rows = np.arange(q.shape[0])
    for t in range(q.shape[1]):
        for c in range(3):
            a = (symbols[:, t] + c) % 3
            out[rows, t, a] = q[:, t, c]
    return out


def sample_choice_counts(action_probs: np.ndarray, samples_per_prefix: int, rng: np.random.Generator) -> np.ndarray:
    p = np.asarray(action_probs, dtype=np.float64)
    flat = p.reshape(-1, 3)
    counts = np.array([rng.multinomial(samples_per_prefix, row) for row in flat], dtype=np.int64)
    return counts.reshape(p.shape)


def _choice_nll(counts: np.ndarray, symbols: np.ndarray, stay: float, peak: float) -> float:
    q = belief_from_symbols(symbols, stay, peak)
    probs = action_probs_from_belief(q, symbols)
    return float(-np.sum(counts * np.log(probs + 1e-12)) / max(float(np.sum(counts)), 1.0))


def fit_sampled_choice_hmm(counts: np.ndarray, symbols: np.ndarray) -> dict:
    """Fit stay/reliability directly to sampled black-box choices.

    Do not first convert noisy counts into q and then fit q-dynamics; that plug-in
    estimator is biased toward low persistence. This objective treats the observed
    choices themselves as multinomial measurements of the candidate hidden-world law.
    """
    counts = np.asarray(counts, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=np.int64)

    stay, peak = 0.94, 0.58
    ds, de = 0.08, 0.12
    best = (_choice_nll(counts, symbols, stay, peak), stay, peak)

    for _ in range(8):
        candidates = []
        for s in [stay - ds, stay, stay + ds]:
            for e in [peak - de, peak, peak + de]:
                s = float(np.clip(s, 0.50, 0.999))
                e = float(np.clip(e, 0.34, 0.95))
                candidates.append((_choice_nll(counts, symbols, s, e), s, e))
        best = min(candidates, key=lambda x: x[0])
        _, stay, peak = best
        ds *= 0.5
        de *= 0.5

    return {"nll": best[0], "stay": best[1], "emission_peak": best[2]}
