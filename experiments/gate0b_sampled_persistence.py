from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whereami.world import WorldConfig, generate_batch
from whereami.model import ContextGRU
from whereami.blackbox import fit_sampled_choice_hmm, sample_choice_counts


def train_blackbox(seed: int, steps: int = 500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = WorldConfig()
    rng = np.random.default_rng(seed)
    model = ContextGRU(hidden_size=24)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(steps):
        s, a, _ = generate_batch(rng, 128, 64, cfg)
        logits, _, _ = model(torch.from_numpy(s))
        loss = F.cross_entropy(logits.reshape(-1, 3), torch.from_numpy(a).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    s, _, _ = generate_batch(np.random.default_rng(seed + 30000), 64, 48, cfg)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(s))
        probs = torch.softmax(logits, dim=-1).numpy()
    return s, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--samples-per-prefix", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/gate0b_blackbox_persistence.json")
    args = ap.parse_args()

    rows = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        symbols, probs = train_blackbox(seed, args.steps)
        soft = fit_sampled_choice_hmm(probs * 1000.0, symbols)
        sampled = []
        for rep in range(args.reps):
            rng = np.random.default_rng(90000 + seed * 1000 + rep)
            counts = sample_choice_counts(probs, args.samples_per_prefix, rng)
            sampled.append(fit_sampled_choice_hmm(counts, symbols))
        rows.append({
            "seed": seed,
            "queried_prefixes": int(symbols.size),
            "samples_per_prefix": args.samples_per_prefix,
            "soft_output_fit": soft,
            "sampled_choice_fits": sampled,
        })

    result = {"rows": rows}
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
