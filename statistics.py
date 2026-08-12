"""
statistics.py

Compare two configurations' results using Welch's t-test (does the
difference look real, given the noise?) and Cohen's d (is the difference
big enough to matter, independent of sample size?).

Usage:
    python statistics.py --config_a baseline --config_b high_lr
"""

import argparse
import json

import numpy as np
from scipy import stats


def load_metric_values(config_name: str, metric: str):
    with open(f"results/{config_name}_all_seeds.json") as f:
        data = json.load(f)
    return [run[metric] for run in data["runs"]]


def cohens_d(group_a, group_b):
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    # pooled standard deviation, weighted by each group's sample size
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def interpret_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compare(config_a: str, config_b: str, metrics: list[str]):
    print(f"Comparing '{config_a}' vs '{config_b}'\n")
    print(f"{'Metric':<20}{'A mean':>10}{'B mean':>10}{'t-stat':>10}{'p-value':>10}{'Cohens d':>10}  Verdict")
    print("-" * 90)

    for metric in metrics:
        a = load_metric_values(config_a, metric)
        b = load_metric_values(config_b, metric)

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
        d = cohens_d(a, b)

        significant = "significant" if p_value < 0.05 else "not significant"
        magnitude = interpret_d(d)
        verdict = f"{significant}, {magnitude} effect"

        print(f"{metric:<20}{np.mean(a):>10.2f}{np.mean(b):>10.2f}"
              f"{t_stat:>10.3f}{p_value:>10.4f}{d:>10.3f}  {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_a", type=str, default="baseline")
    parser.add_argument("--config_b", type=str, default="high_lr")
    args = parser.parse_args()

    metrics = ["mean_return", "mean_distance", "mean_length",
               "termination_rate", "mean_energy"]

    compare(args.config_a, args.config_b, metrics)