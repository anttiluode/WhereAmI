from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whereami.blackbox import fit_sampled_choice_hmm


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("responses", help="JSONL rows with at least {id, choice}; repeated ids count as repeated samples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = read_jsonl(Path(args.benchmark))
    replies = read_jsonl(Path(args.responses))
    by_id = {r["id"]: r for r in bench}
    trials = max(r["trial"] for r in bench) + 1
    turns = max(r["turn"] for r in bench) + 1

    symbols = np.zeros((trials, turns), dtype=np.int64)
    truth = np.zeros((trials, turns), dtype=np.int64)
    context = np.zeros((trials, turns), dtype=np.int64)
    counts = np.zeros((trials, turns, 3), dtype=np.int64)

    for r in bench:
        symbols[r["trial"], r["turn"]] = int(r["symbol"])
        truth[r["trial"], r["turn"]] = int(r["true_action"])
        context[r["trial"], r["turn"]] = int(r["hidden_context"])

    bad = []
    for r in replies:
        if r.get("id") not in by_id:
            bad.append({"id": r.get("id"), "reason": "unknown id"})
            continue
        try:
            choice = int(str(r["choice"]).strip())
        except Exception:
            bad.append({"id": r.get("id"), "reason": "choice not parseable"})
            continue
        if choice not in (0, 1, 2):
            bad.append({"id": r.get("id"), "reason": "choice outside 0/1/2"})
            continue
        b = by_id[r["id"]]
        counts[b["trial"], b["turn"], choice] += 1

    observed = counts.sum(axis=-1) > 0
    if not np.all(observed):
        missing = int((~observed).sum())
        raise SystemExit(f"missing responses for {missing} benchmark prefixes")

    fit = fit_sampled_choice_hmm(counts, symbols)
    majority = counts.argmax(axis=-1)
    accuracy = float(np.mean(majority == truth))

    switched = np.zeros((trials, turns), dtype=bool)
    switched[:, 1:] = context[:, 1:] != context[:, :-1]
    after1 = switched.copy()
    after2 = switched.copy()
    after4 = switched.copy()
    for dt in range(1, 2):
        after2[:, dt:] |= switched[:, :-dt]
    for dt in range(1, 4):
        after4[:, dt:] |= switched[:, :-dt]

    def acc(mask):
        if not np.any(mask):
            return None
        return float(np.mean(majority[mask] == truth[mask]))

    result = {
        "n_prefixes": int(trials * turns),
        "n_response_rows": len(replies),
        "samples_per_prefix_mean": float(counts.sum() / (trials * turns)),
        "choice_accuracy": accuracy,
        "accuracy_on_switch_turn": acc(after1),
        "accuracy_within_2_turns_of_switch": acc(after2),
        "accuracy_within_4_turns_of_switch": acc(after4),
        "accuracy_away_from_switch_4": acc(~after4),
        "fitted_latent_regime": fit,
        "invalid_response_rows": bad,
    }

    out = Path(args.out) if args.out else Path(args.responses).with_suffix(".score.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
