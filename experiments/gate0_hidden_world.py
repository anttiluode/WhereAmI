from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whereami.world import WorldConfig, generate_batch, bayes_filter, action_distribution_from_belief
from whereami.model import ContextGRU
from whereami.decode import behavior_to_context_belief, fit_symmetric_hmm, fit_hidden_probe, hmm_update, affine_dynamics_attack


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def memoryless_baseline(train_symbols, train_actions, test_symbols, test_actions):
    counts = np.ones((3, 3), dtype=np.float64)  # light Laplace smoothing
    for s, a in zip(train_symbols.reshape(-1), train_actions.reshape(-1)):
        counts[s, a] += 1
    probs = counts / counts.sum(axis=1, keepdims=True)
    pred = probs[test_symbols]
    acc = np.mean(pred.argmax(axis=-1) == test_actions)
    nll = -np.mean(np.log(np.take_along_axis(pred, test_actions[..., None], axis=-1)[..., 0] + 1e-9))
    return float(acc), float(nll)


def kernel_receipt(model, hidden, q_neural, device):
    H = hidden.reshape(-1, hidden.shape[-1])
    Q = q_neural.reshape(-1, 3)
    reps = []
    rep_q = []
    for c in range(3):
        idx = int(np.argmax(Q[:, c]))
        reps.append(H[idx])
        rep_q.append(Q[idx])

    jac = np.zeros((3, 3, model.hidden_size, model.hidden_size), dtype=np.float64)
    intervention_l1 = []
    with torch.no_grad():
        pass
    for c in range(3):
        h0 = torch.tensor(reps[c], dtype=torch.float32, device=device)
        for sym in range(3):
            def fn(hvec):
                _, hnew = model.step(sym, hvec)
                return hnew[0]
            J = torch.autograd.functional.jacobian(fn, h0).detach().cpu().numpy()
            jac[c, sym] = J
            with torch.no_grad():
                logits, _ = model.step(sym, h0)
                p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            q_after = behavior_to_context_belief(p[None, :], np.array([sym]))[0]
            q_expected = hmm_update(np.asarray(rep_q[c])[None, :], np.array([sym]),
                                    kernel_receipt.fit_stay, kernel_receipt.fit_peak)[0]
            intervention_l1.append(float(np.abs(q_after - q_expected).sum()))

    mean_j = jac.mean(axis=1)
    dists = []
    for a in range(3):
        for b in range(a + 1, 3):
            denom = 0.5 * (np.linalg.norm(mean_j[a]) + np.linalg.norm(mean_j[b])) + 1e-12
            dists.append(float(np.linalg.norm(mean_j[a] - mean_j[b]) / denom))
    return {
        "representative_beliefs": np.asarray(rep_q).tolist(),
        "mean_pairwise_kernel_distance": float(np.mean(dists)),
        "pairwise_kernel_distances": dists,
        "state_intervention_mean_l1_vs_fitted_update": float(np.mean(intervention_l1)),
    }


def run(seed: int, train_steps: int, batch_size: int, seq_len: int, hidden_size: int, device: str):
    seed_all(seed)
    cfg = WorldConfig()
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    model = ContextGRU(hidden_size=hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Keep a fixed baseline sample separate from optimization data.
    base_rng = np.random.default_rng(seed + 10000)
    tr_s, tr_a, _ = generate_batch(base_rng, 200, seq_len, cfg)
    te_s, te_a, _ = generate_batch(base_rng, 200, seq_len, cfg)
    mem_acc, mem_nll = memoryless_baseline(tr_s, tr_a, te_s, te_a)

    model.train()
    for step in range(train_steps):
        s, a, _ = generate_batch(rng, batch_size, seq_len, cfg)
        st = torch.from_numpy(s).to(dev)
        at = torch.from_numpy(a).to(dev)
        logits, _, _ = model(st)
        loss = F.cross_entropy(logits.reshape(-1, 3), at.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, train_steps // 4, train_steps // 2, (3 * train_steps) // 4, train_steps - 1}:
            print(f"seed={seed} step={step:4d} loss={loss.item():.5f}")

    model.eval()
    test_rng = np.random.default_rng(seed + 20000)
    s, a, c = generate_batch(test_rng, 512, max(seq_len, 96), cfg)
    q_true = bayes_filter(s, cfg)
    p_bayes = action_distribution_from_belief(q_true, s)
    with torch.no_grad():
        logits, states, _ = model(torch.from_numpy(s).to(dev))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        hidden = states.cpu().numpy()
    pred = probs.argmax(axis=-1)
    network_acc = float(np.mean(pred == a))
    network_nll = float(-np.mean(np.log(np.take_along_axis(probs, a[..., None], axis=-1)[..., 0] + 1e-9)))
    bayes_acc = float(np.mean(p_bayes.argmax(axis=-1) == a))
    bayes_nll = float(-np.mean(np.log(np.take_along_axis(p_bayes, a[..., None], axis=-1)[..., 0] + 1e-9)))

    q_neural = behavior_to_context_belief(probs, s)
    belief_kl = float(np.mean(np.sum(q_true * (np.log(q_true + 1e-9) - np.log(q_neural + 1e-9)), axis=-1)))
    belief_l1 = float(np.mean(np.abs(q_true - q_neural).sum(axis=-1)))
    probe = fit_hidden_probe(hidden, q_true)
    fit = fit_symmetric_hmm(q_neural, s)
    kernel_receipt.fit_stay = fit["stay"]
    kernel_receipt.fit_peak = fit["emission_peak"]
    kernel = kernel_receipt(model, hidden, q_neural, dev)
    dynamics_attack = affine_dynamics_attack(hidden, s, q_neural)

    result = {
        "seed": seed,
        "world": {"true_stay": cfg.stay, "true_emission_peak": cfg.emission_peak},
        "training": {"steps": train_steps, "batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size},
        "behavior": {
            "memoryless_accuracy": mem_acc,
            "memoryless_nll": mem_nll,
            "network_accuracy": network_acc,
            "network_nll": network_nll,
            "bayes_accuracy": bayes_acc,
            "bayes_nll": bayes_nll,
        },
        "belief_decode": {
            "true_vs_neural_kl": belief_kl,
            "true_vs_neural_mean_l1": belief_l1,
            "hidden_to_true_logodds_r2": probe["r2"],
        },
        "update_law_fit": fit,
        "situational_kernel": kernel,
        "dynamics_attacker": dynamics_attack,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = run(args.seed, args.train_steps, args.batch_size, args.seq_len, args.hidden_size, args.device)
    print(json.dumps(result, indent=2))
    out = args.out or f"results/gate0_seed{args.seed}.json"
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
