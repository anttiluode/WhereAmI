import numpy as np

from whereami.blackbox import belief_from_symbols, action_probs_from_belief, fit_sampled_choice_hmm


def test_blackbox_probabilities_normalize():
    s = np.array([[0, 1, 2, 1, 0]])
    q = belief_from_symbols(s, 0.96, 0.58)
    p = action_probs_from_belief(q, s)
    assert np.allclose(q.sum(axis=-1), 1.0)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_direct_choice_fit_recovers_synthetic_parameters():
    rng = np.random.default_rng(7)
    symbols = rng.integers(0, 3, size=(64, 32))
    q = belief_from_symbols(symbols, 0.94, 0.62)
    p = action_probs_from_belief(q, symbols)
    counts = p * 1000.0
    fit = fit_sampled_choice_hmm(counts, symbols)
    assert abs(fit["stay"] - 0.94) < 0.02
    assert abs(fit["emission_peak"] - 0.62) < 0.02
