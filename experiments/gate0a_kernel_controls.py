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
from whereami.decode import behavior_to_context_belief
from whereami.controls import kernel_partition_controls


def train_collect(seed: int, steps: int = 500, batch_size: int = 128, seq_len: int = 64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = WorldConfig()
    rng = np.random.default_rng(seed)

    model = ContextGRU(hidden_size=24)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        s, a, _ = generate_batch(rng, batch_size, seq_len, cfg)
        logits, _, _ = model(torch.from_numpy(s))
        loss = F.cross_entropy(logits.reshape(-1, 3), torch.from_numpy(a).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    s, a, _ = generate_batch(np.random.default_rng(seed + 20000), 512, 96, cfg)
    model.eval()
    with torch.no_grad():
        logits, states, _ = model(torch.from_numpy(s))
        probs = torch.softmax(logits, dim=-1).numpy()
    hidden = states.numpy()
    q = behavior_to_context_belief(probs, s)
    controls = kernel_partition_controls(hidden, s, q, seed=seed)
    controls["network_accuracy"] = float(np.mean(probs.argmax(axis=-1) == a))
    controls["seed"] = seed
    return controls


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "seed"]
    out = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows], "metrics": {}}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        out["metrics"][k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals.tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="results/gate0a_kernel_controls_summary.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = [train_collect(seed, steps=args.steps) for seed in seeds]
    result = summarize(rows)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
