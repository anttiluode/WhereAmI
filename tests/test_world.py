import numpy as np

from whereami.world import WorldConfig, action_for, generate_batch, bayes_filter
from whereami.decode import behavior_to_context_belief, hmm_update


def test_same_symbol_has_three_contextual_meanings():
    assert [int(action_for(c, 1)) for c in range(3)] == [1, 2, 0]


def test_bayes_filter_normalizes():
    cfg = WorldConfig()
    rng = np.random.default_rng(0)
    symbols, _, _ = generate_batch(rng, 8, 20, cfg)
    q = bayes_filter(symbols, cfg)
    assert q.shape == (8, 20, 3)
    assert np.allclose(q.sum(axis=-1), 1.0)


def test_behavior_decode_inverts_action_permutation():
    q = np.array([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5]])
    symbols = np.array([0, 2])
    p = np.zeros_like(q)
    for i in range(len(q)):
        for c in range(3):
            p[i, (symbols[i] + c) % 3] = q[i, c]
    assert np.allclose(behavior_to_context_belief(p, symbols), q)


def test_hmm_update_normalizes():
    q = np.array([[0.8, 0.1, 0.1], [1/3, 1/3, 1/3]])
    out = hmm_update(q, np.array([2, 1]), 0.96, 0.58)
    assert np.allclose(out.sum(axis=-1), 1.0)
