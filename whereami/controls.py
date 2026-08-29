from __future__ import annotations

import itertools
import numpy as np


def _fit_predict(X: np.ndarray, Y: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray, ridge: float) -> np.ndarray:
    Xt, Yt = X[train_rows], Y[train_rows]
    reg = ridge * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Yt)
    return X[test_rows] @ W


def _nmse(Y: np.ndarray, pred: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray) -> float:
    denom = float(np.mean((Y[test_rows] - Y[train_rows].mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    return float(np.mean((Y[test_rows] - pred) ** 2) / denom)


def _expert_nmse(X: np.ndarray, Y: np.ndarray, labels: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray, ridge: float) -> float:
    pred = np.empty_like(Y[test_rows])
    test_labels = labels[test_rows]
    for c in range(3):
        tr = train_rows[labels[train_rows] == c]
        local = np.flatnonzero(test_labels == c)
        te = test_rows[local]
        if len(te) == 0:
            continue
        if len(tr) < X.shape[1] + 2:
            pred[local] = _fit_predict(X, Y, train_rows, te, ridge)
        else:
            pred[local] = _fit_predict(X, Y, tr, te, ridge)
    return _nmse(Y, pred, train_rows, test_rows)


def _kmeans3(X_train: np.ndarray, X_test: np.ndarray, seed: int, iters: int = 30, max_fit: int = 15000):
    rng = np.random.default_rng(seed)
    X_fit = X_train
    if len(X_fit) > max_fit:
        X_fit = X_fit[rng.choice(len(X_fit), max_fit, replace=False)]

    centers = [X_fit[rng.integers(len(X_fit))]]
    for _ in range(2):
        d2 = np.min(np.stack([np.sum((X_fit - c) ** 2, axis=1) for c in centers]), axis=0)
        centers.append(X_fit[rng.choice(len(X_fit), p=d2 / (d2.sum() + 1e-12))])
    C = np.stack(centers)

    for _ in range(iters):
        labels = np.argmin(((X_fit[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1), axis=1)
        new_C = np.stack([X_fit[labels == k].mean(axis=0) if np.any(labels == k) else C[k] for k in range(3)])
        if np.max(np.abs(new_C - C)) < 1e-6:
            C = new_C
            break
        C = new_C

    train_labels = np.argmin(((X_train[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1), axis=1)
    test_labels = np.argmin(((X_test[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1), axis=1)
    return train_labels, test_labels


def _pca_terciles(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sample = X_train[: min(len(X_train), 15000)]
    _, _, vh = np.linalg.svd(sample - mu, full_matrices=False)
    v = vh[0]
    z_train = (X_train - mu) @ v
    z_test = (X_test - mu) @ v
    cuts = np.quantile(z_train, [1 / 3, 2 / 3])
    return np.digitize(z_train, cuts), np.digitize(z_test, cuts)


def _best_perm_accuracy(target: np.ndarray, labels: np.ndarray) -> float:
    best = 0.0
    for p in itertools.permutations(range(3)):
        mapped = np.array([p[x] for x in labels], dtype=np.int64)
        best = max(best, float(np.mean(mapped == target)))
    return best


def kernel_partition_controls(
    hidden: np.ndarray,
    symbols: np.ndarray,
    belief: np.ndarray,
    *,
    seed: int = 0,
    ridge: float = 1e-3,
    random_reps: int = 20,
) -> dict:
    """Equal-parameter attacks on the 'context-specific operator' interpretation."""
    H = np.asarray(hidden, dtype=np.float64)
    S = np.asarray(symbols, dtype=np.int64)
    Q = np.asarray(belief, dtype=np.float64)

    xh = H[:, :-1].reshape(-1, H.shape[-1])
    yh = H[:, 1:].reshape(-1, H.shape[-1])
    sym = S[:, 1:].reshape(-1)
    decoded_ctx = Q[:, :-1].argmax(axis=-1).reshape(-1)
    one_sym = np.eye(3)[sym]

    X = np.concatenate([xh, one_sym, np.ones((len(xh), 1))], axis=1)
    b, tm1 = H.shape[0], H.shape[1] - 1
    train_rows = np.arange((b // 2) * tm1)
    test_rows = np.arange((b // 2) * tm1, b * tm1)

    shared = _nmse(yh, _fit_predict(X, yh, train_rows, test_rows, ridge), train_rows, test_rows)
    decoded = _expert_nmse(X, yh, decoded_ctx, train_rows, test_rows, ridge)

    rng = np.random.default_rng(seed + 777)
    shuffled = decoded_ctx.copy()
    shuffled[train_rows] = rng.permutation(shuffled[train_rows])
    shuffled[test_rows] = rng.permutation(shuffled[test_rows])
    shuffled_nmse = _expert_nmse(X, yh, shuffled, train_rows, test_rows, ridge)

    random_scores = []
    for _ in range(random_reps):
        labels = np.empty(len(xh), dtype=np.int64)
        labels[train_rows] = rng.integers(0, 3, size=len(train_rows))
        labels[test_rows] = rng.integers(0, 3, size=len(test_rows))
        random_scores.append(_expert_nmse(X, yh, labels, train_rows, test_rows, ridge))

    km_train, km_test = _kmeans3(xh[train_rows], xh[test_rows], seed=seed)
    kmeans_labels = np.empty(len(xh), dtype=np.int64)
    kmeans_labels[train_rows] = km_train
    kmeans_labels[test_rows] = km_test
    kmeans_nmse = _expert_nmse(X, yh, kmeans_labels, train_rows, test_rows, ridge)
    kmeans_alignment = _best_perm_accuracy(decoded_ctx[test_rows], km_test)

    pc_train, pc_test = _pca_terciles(xh[train_rows], xh[test_rows])
    pc_labels = np.empty(len(xh), dtype=np.int64)
    pc_labels[train_rows] = pc_train
    pc_labels[test_rows] = pc_test
    pca_nmse = _expert_nmse(X, yh, pc_labels, train_rows, test_rows, ridge)

    return {
        "shared_affine_nmse": shared,
        "decoded_context_experts_nmse": decoded,
        "shuffled_context_experts_nmse": shuffled_nmse,
        "random_partition_experts_nmse_mean": float(np.mean(random_scores)),
        "random_partition_experts_nmse_min": float(np.min(random_scores)),
        "kmeans_hidden_experts_nmse": kmeans_nmse,
        "kmeans_vs_decoded_context_best_perm_accuracy": kmeans_alignment,
        "pca_tercile_experts_nmse": pca_nmse,
    }
