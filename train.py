"""
train.py

Train a single PPO policy on Ant-v5 for one (config, seed) pair.
Kept deliberately minimal: this script trains and saves, nothing else.
Evaluation and metrics live in separate scripts so policies can be
re-evaluated later without retraining.

Usage:
    python train.py --seed 0 --steps 200000 --config baseline
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def train(seed: int, steps: int, config_name: str, learning_rate: float):
    env = gym.make("Ant-v5")
    env.reset(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=learning_rate,
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=steps)
    elapsed = time.time() - start

    Path("models").mkdir(exist_ok=True)
    save_path = f"models/{config_name}_seed{seed}"
    model.save(save_path)

    print(f"\nSaved to {save_path}.zip")
    print(f"Training took {elapsed/60:.1f} minutes for {steps} steps")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(args.seed, args.steps, args.config, args.lr)