from __future__ import annotations

import numpy as np


def behavior_to_context_belief(action_probs: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Calibration decoder: fixed-symbol action probabilities are a permutation of context belief."""
    p = np.asarray(action_probs)
    s = np.asarray(symbols)
    q = np.empty_like(p)
    for c in range(3):
        a = (s + c) % 3
        q[..., c] = np.take_along_axis(p, a[..., None], axis=-1)[..., 0]
    q = np.clip(q, 1e-9, None)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def hmm_update(q_prev: np.ndarray, symbol: np.ndarray, stay: float, peak: float) -> np.ndarray:
    q_prev = np.asarray(q_prev, dtype=np.float64)
    symbol = np.asarray(symbol, dtype=np.int64)
    off_t = (1.0 - stay) / 2.0
    prior = off_t + (stay - off_t) * q_prev
    off_e = (1.0 - peak) / 2.0
    like = np.full_like(prior, off_e)
    rows = np.arange(q_prev.shape[0])
    like[rows, symbol] = peak
    q = prior * like
    q /= q.sum(axis=-1, keepdims=True)
    return q


def fit_symmetric_hmm(q: np.ndarray, symbols: np.ndarray, max_points: int = 5000) -> dict:
    """Fit the simplest two-parameter hidden-world law to decoded neural beliefs."""
    q = np.asarray(q, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=np.int64)
    prev = q[:, :-1].reshape(-1, 3)
    target = q[:, 1:].reshape(-1, 3)
    obs = symbols[:, 1:].reshape(-1)
    if len(prev) > max_points:
        idx = np.linspace(0, len(prev) - 1, max_points, dtype=int)
        prev, target, obs = prev[idx], target[idx], obs[idx]

    def loss(stay: float, peak: float) -> float:
        pred = hmm_update(prev, obs, stay, peak)
        return float(np.mean(np.sum(target * (np.log(target + 1e-9) - np.log(pred + 1e-9)), axis=1)))

    best = (float("inf"), None, None)
    # Coarse-to-fine grid: deterministic and dependency-free.
    for stay in np.linspace(0.75, 0.995, 21):
        for peak in np.linspace(0.36, 0.82, 21):
            v = loss(float(stay), float(peak))
            if v < best[0]:
                best = (v, float(stay), float(peak))
    _, s0, e0 = best
    for stay in np.linspace(max(0.5, s0 - 0.03), min(0.999, s0 + 0.03), 21):
        for peak in np.linspace(max(0.34, e0 - 0.06), min(0.95, e0 + 0.06), 21):
            v = loss(float(stay), float(peak))
            if v < best[0]:
                best = (v, float(stay), float(peak))
    return {"kl": best[0], "stay": best[1], "emission_peak": best[2]}


def fit_hidden_probe(hidden: np.ndarray, belief: np.ndarray, ridge: float = 1e-3) -> dict:
    """Affine probe from neural state to two context log-odds; reports held-out R^2."""
    H = np.asarray(hidden, dtype=np.float64).reshape(-1, hidden.shape[-1])
    q = np.asarray(belief, dtype=np.float64).reshape(-1, 3)
    z = np.stack([np.log(q[:, 0] + 1e-8) - np.log(q[:, 2] + 1e-8),
                  np.log(q[:, 1] + 1e-8) - np.log(q[:, 2] + 1e-8)], axis=1)
    n = len(H)
    split = n // 2
    Xtr = np.concatenate([H[:split], np.ones((split, 1))], axis=1)
    Xte = np.concatenate([H[split:], np.ones((n - split, 1))], axis=1)
    reg = ridge * np.eye(Xtr.shape[1]); reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ z[:split])
    pred = Xte @ W
    y = z[split:]
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"r2": float(r2), "weights": W}


def affine_dynamics_attack(hidden: np.ndarray, symbols: np.ndarray, belief: np.ndarray, ridge: float = 1e-3) -> dict:
    """Held-out attacker: shared dynamics vs context-bias vs context-specific operators."""
    H = np.asarray(hidden, dtype=np.float64)
    S = np.asarray(symbols, dtype=np.int64)
    Q = np.asarray(belief, dtype=np.float64)
    xh = H[:, :-1].reshape(-1, H.shape[-1])
    yh = H[:, 1:].reshape(-1, H.shape[-1])
    sym = S[:, 1:].reshape(-1)
    ctx = Q[:, :-1].argmax(axis=-1).reshape(-1)
    one_sym = np.eye(3)[sym]
    one_ctx = np.eye(3)[ctx]

    # Split by sequence to reduce leakage between adjacent timesteps.
    b, tm1 = H.shape[0], H.shape[1] - 1
    train_rows = np.arange((b // 2) * tm1)
    test_rows = np.arange((b // 2) * tm1, b * tm1)

    def fit_predict(X: np.ndarray, rows_train, rows_test):
        Xt, Yt = X[rows_train], yh[rows_train]
        reg = ridge * np.eye(X.shape[1]); reg[-1, -1] = 0.0
        W = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Yt)
        return X[rows_test] @ W

    shared_X = np.concatenate([xh, one_sym, np.ones((len(xh), 1))], axis=1)
    bias_X = np.concatenate([xh, one_sym, one_ctx, np.ones((len(xh), 1))], axis=1)
    pred_shared = fit_predict(shared_X, train_rows, test_rows)
    pred_bias = fit_predict(bias_X, train_rows, test_rows)

    pred_full = np.empty_like(yh[test_rows])
    test_ctx = ctx[test_rows]
    for c in range(3):
        tr = train_rows[ctx[train_rows] == c]
        te_local = np.flatnonzero(test_ctx == c)
        te = test_rows[te_local]
        if len(te) == 0:
            continue
        full_X = shared_X
        pred_full[te_local] = fit_predict(full_X, tr, te)

    denom = float(np.mean((yh[test_rows] - yh[train_rows].mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    def nmse(pred):
        return float(np.mean((yh[test_rows] - pred) ** 2) / denom)

    a = nmse(pred_shared); bnm = nmse(pred_bias); f = nmse(pred_full)
    return {
        "shared_affine_nmse": a,
        "shared_plus_context_bias_nmse": bnm,
        "context_specific_affine_nmse": f,
        "context_specific_vs_shared_ratio": f / max(a, 1e-12),
        "context_specific_vs_bias_ratio": f / max(bnm, 1e-12),
    }
