"""
check_training_health.py

Reads SB3's CSV training log and flags runs where key stability metrics
crossed known danger thresholds -- specifically approx_kl and clip_fraction,
based on the instability observed in tonight's Ant training run, where
approx_kl climbed from ~0.48 to ~1.68 and the final checkpoint performed
worse than an earlier one.

Usage:
    python check_training_health.py --log path/to/progress.csv
"""

import argparse
import csv

# Thresholds based on standard PPO guidance: approx_kl should typically
# stay under ~0.02-0.05; values above ~0.1 indicate the policy is
# changing too aggressively per update. clip_fraction above ~0.3-0.4
# suggests PPO's safety clipping is being hit constantly, not
# occasionally.
APPROX_KL_WARNING = 0.1
APPROX_KL_CRITICAL = 0.5
CLIP_FRACTION_WARNING = 0.4


def check_log(log_path):
    warnings = []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows):
        try:
            kl = float(row.get("train/approx_kl", 0))
            clip_frac = float(row.get("train/clip_fraction", 0))
            timesteps = row.get("time/total_timesteps", "?")
        except (ValueError, TypeError):
            continue

        if kl >= APPROX_KL_CRITICAL:
            warnings.append(
                f"CRITICAL at step {timesteps}: approx_kl={kl:.3f} "
                f"(>= {APPROX_KL_CRITICAL}) -- policy update badly unstable"
            )
        elif kl >= APPROX_KL_WARNING:
            warnings.append(
                f"WARNING at step {timesteps}: approx_kl={kl:.3f} "
                f"(>= {APPROX_KL_WARNING}) -- policy changing aggressively"
            )

        if clip_frac >= CLIP_FRACTION_WARNING:
            warnings.append(
                f"WARNING at step {timesteps}: clip_fraction={clip_frac:.3f} "
                f"(>= {CLIP_FRACTION_WARNING}) -- PPO clipping constantly, not occasionally"
            )

    if not warnings:
        print("No instability detected -- all metrics within healthy range.")
    else:
        print(f"Found {len(warnings)} instability warning(s):\n")
        for w in warnings:
            print(f"  {w}")
        print(
            "\nRecommendation: check whether an earlier checkpoint "
            "outperforms the final one before assuming later = better."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    args = parser.parse_args()

    check_log(args.log)