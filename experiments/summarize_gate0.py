from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "results").glob("gate0_seed*.json"))
if not files:
    raise SystemExit("no gate0_seed*.json files")
rows = [json.loads(p.read_text()) for p in files]

metrics = {
    "memoryless_accuracy": ("behavior", "memoryless_accuracy"),
    "network_accuracy": ("behavior", "network_accuracy"),
    "bayes_accuracy": ("behavior", "bayes_accuracy"),
    "belief_kl": ("belief_decode", "true_vs_neural_kl"),
    "hidden_logodds_r2": ("belief_decode", "hidden_to_true_logodds_r2"),
    "fit_stay": ("update_law_fit", "stay"),
    "fit_emission_peak": ("update_law_fit", "emission_peak"),
    "kernel_distance": ("situational_kernel", "mean_pairwise_kernel_distance"),
    "intervention_l1": ("situational_kernel", "state_intervention_mean_l1_vs_fitted_update"),
    "shared_affine_nmse": ("dynamics_attacker", "shared_affine_nmse"),
    "context_bias_nmse": ("dynamics_attacker", "shared_plus_context_bias_nmse"),
    "context_specific_nmse": ("dynamics_attacker", "context_specific_affine_nmse"),
    "context_specific_vs_shared": ("dynamics_attacker", "context_specific_vs_shared_ratio"),
    "context_specific_vs_bias": ("dynamics_attacker", "context_specific_vs_bias_ratio"),
}

summary = {
    "n_seeds": len(rows),
    "seeds": [r["seed"] for r in rows],
    "true_world": rows[0]["world"],
    "training": rows[0]["training"],
    "metrics": {},
}
for name, path in metrics.items():
    vals = np.array([r[path[0]][path[1]] for r in rows], dtype=float)
    summary["metrics"][name] = {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals.tolist(),
    }

out = ROOT / "results" / "gate0_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(out)
