"""
run_experiment.py

Train and evaluate one configuration across multiple seeds, then aggregate.
This is the core of the harness: single runs are anecdotes, and RL variance
across seeds is large enough that one run tells you very little.

Usage:
    python run_experiment.py --config baseline --seeds 5 --steps 200000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluate import evaluate


def run(config_name: str, n_seeds: int, steps: int, lr: float, episodes: int):
    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"CONFIG {config_name} | SEED {seed+1}/{n_seeds}")
        print('='*60)

        subprocess.run([
            sys.executable, "train.py",
            "--seed", str(seed),
            "--steps", str(steps),
            "--config", config_name,
            "--lr", str(lr),
        ], check=True)

        model_path = f"models/{config_name}_seed{seed}"
        res = evaluate(model_path, episodes)
        res["seed"] = seed
        all_results.append(res)

        print(f"  seed {seed}: return={res['mean_return']:.1f}  "
              f"dist={res['mean_distance']:.2f}m  "
              f"term_rate={res['termination_rate']:.0%}")

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/{config_name}_all_seeds.json"
    with open(out_path, "w") as f:
        json.dump({"config": config_name, "runs": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {config_name} across {n_seeds} seeds")
    print('='*60)
    for metric in ["mean_return", "mean_distance", "mean_length",
                   "termination_rate", "mean_energy"]:
        vals = [r[metric] for r in all_results]
        print(f"  {metric:18s}: {np.mean(vals):8.3f} +/- {np.std(vals):7.3f}  "
              f"(min {np.min(vals):.2f}, max {np.max(vals):.2f})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    run(args.config, args.seeds, args.steps, args.lr, args.episodes)