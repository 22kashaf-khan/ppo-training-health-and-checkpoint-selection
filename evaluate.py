"""
evaluate.py

Load a trained policy and run evaluation episodes, computing metrics that are
deliberately independent of the training reward function. Evaluating a policy
with the same signal it was trained on tells you little; these metrics can
disagree with reward, which is the point.

Usage:
    python evaluate.py --model models/baseline_seed0 --episodes 20
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate(model_path: str, n_episodes: int, eval_seed: int = 1000):
    model = PPO.load(model_path)
    env = gym.make("Ant-v5")

    returns = []
    lengths = []
    distances = []
    velocities = []
    terminated_early = []
    energies = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=eval_seed + ep)
        start_x = env.unwrapped.data.qpos[0]

        ep_return = 0.0
        ep_energy = 0.0
        steps = 0
        was_terminated = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            ep_energy += float(np.sum(np.square(action)))
            steps += 1

            if terminated or truncated:
                was_terminated = terminated
                break

        end_x = env.unwrapped.data.qpos[0]
        distance = end_x - start_x

        returns.append(ep_return)
        lengths.append(steps)
        distances.append(distance)
        velocities.append(distance / steps if steps > 0 else 0.0)
        terminated_early.append(1.0 if was_terminated else 0.0)
        energies.append(ep_energy / steps if steps > 0 else 0.0)

    env.close()

    results = {
        "model": model_path,
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_distance": float(np.mean(distances)),
        "mean_velocity": float(np.mean(velocities)),
        "termination_rate": float(np.mean(terminated_early)),
        "mean_energy": float(np.mean(energies)),
        "raw_returns": returns,
        "raw_distances": distances,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    results = evaluate(args.model, args.episodes)

    print(f"\nEvaluation of {args.model} over {args.episodes} episodes")
    print("-" * 55)
    print(f"  Mean return        : {results['mean_return']:.1f}")
    print(f"  Mean episode length: {results['mean_length']:.1f} steps")
    print(f"  Mean distance      : {results['mean_distance']:.3f} m")
    print(f"  Mean velocity      : {results['mean_velocity']:.4f} m/step")
    print(f"  Termination rate   : {results['termination_rate']:.1%}")
    print(f"  Mean energy cost   : {results['mean_energy']:.3f}")

    Path("results").mkdir(exist_ok=True)
    name = Path(args.model).name
    out_path = f"results/{name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")