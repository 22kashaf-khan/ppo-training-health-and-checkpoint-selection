"""
robustness_test.py

Test the trained Ant policy under varied physics conditions (friction,
mass) that differ from training, to measure how much performance
degrades — not just whether it walks on the exact conditions it saw
during training.

Usage:
    python robustness_test.py --model models_full/ant_full_500000_steps --vecnorm models_full/ant_full_vecnormalize_500000_steps.pkl
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def run_episodes(model, vec_env, n_episodes=10):
    successes = 0
    returns = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_return += reward[0]
            steps += 1
            done = dones[0]
        returns.append(ep_return)
        if steps >= 1000:
            successes += 1

    return successes / n_episodes, np.mean(returns)


def make_env_with_physics(friction_scale=1.0, mass_scale=1.0):
    def _make():
        env = gym.make("Ant-v5")
        model = env.unwrapped.model
        model.geom_friction[:, 0] *= friction_scale
        model.body_mass[:] *= mass_scale
        return env
    return _make


def main(model_path, vecnorm_path, checkpoint_name):
    model = PPO.load(model_path)

    scenarios = {
        "nominal (training conditions)": (1.0, 1.0),
        "low friction (-40%)": (0.6, 1.0),
        "very low friction (-60%)": (0.4, 1.0),
        "heavier robot (+20% mass)": (1.0, 1.2),
        "much heavier (+40% mass)": (1.0, 1.4),
    }

    print(f"Robustness test for checkpoint: {checkpoint_name}\n")
    print(f"{'Scenario':<32}{'Success rate':>14}{'Mean return':>14}")
    print("-" * 60)

    for label, (friction_scale, mass_scale) in scenarios.items():
        raw_env = DummyVecEnv([make_env_with_physics(friction_scale, mass_scale)])
        vec_env = VecNormalize.load(vecnorm_path, raw_env)
        vec_env.training = False
        vec_env.norm_reward = False

        success_rate, mean_return = run_episodes(model, vec_env, n_episodes=10)
        print(f"{label:<32}{success_rate:>13.0%}{mean_return:>14.1f}")
        vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, required=True)
    args = parser.parse_args()

    main(args.model, args.vecnorm, args.model)