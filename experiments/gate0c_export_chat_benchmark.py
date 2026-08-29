from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whereami.world import WorldConfig, generate_batch


INSTRUCTIONS = """You are playing a hidden-mode game.

There are three possible modes: M0, M1, M2.
The current mode can occasionally change without an explicit announcement.
At each turn you observe an evidence symbol: 0, 1, or 2.
The evidence tends to match the current mode, but it is noisy.

Your required action is:
  action = (current evidence symbol + current mode number) mod 3

Infer the current mode from the full evidence history.
Return ONLY the action digit 0, 1, or 2. Do not explain.
"""


def make_prompt(history: list[int]) -> str:
    hist = " ".join(str(x) for x in history)
    return (
        INSTRUCTIONS
        + "\nEvidence history (oldest -> newest):\n"
        + hist
        + "\n\nWhat action should you take for the newest evidence symbol?"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--out", default="benchmarks/gate0c_hidden_mode.jsonl")
    args = ap.parse_args()

    cfg = WorldConfig()
    rng = np.random.default_rng(args.seed)
    symbols, actions, contexts = generate_batch(rng, args.trials, args.turns, cfg)

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trial in range(args.trials):
            history = []
            for turn in range(args.turns):
                history.append(int(symbols[trial, turn]))
                rec = {
                    "id": f"trial{trial:03d}_turn{turn:03d}",
                    "trial": trial,
                    "turn": turn,
                    "prompt": make_prompt(history),
                    "symbol": int(symbols[trial, turn]),
                    "true_action": int(actions[trial, turn]),
                    "hidden_context": int(contexts[trial, turn]),
                }
                f.write(json.dumps(rec) + "\n")

    meta = {
        "seed": args.seed,
        "trials": args.trials,
        "turns": args.turns,
        "queried_prefixes": args.trials * args.turns,
        "generator_truth_not_shown_to_model": {
            "stay": cfg.stay,
            "emission_peak": cfg.emission_peak,
        },
    }
    (path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
