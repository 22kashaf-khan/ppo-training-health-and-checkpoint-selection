"""
evaluate_normalized.py

Load a trained policy together with the VecNormalize statistics saved at the
same checkpoint, and run evaluation episodes.

Why this exists: models_full/ was trained with VecNormalize(norm_obs=True,
norm_reward=True). A policy trained that way expects normalized observations.
Evaluating it on a bare gym.make("Ant-v5") feeds it raw observations it never
saw during training. Because the normalization statistics keep updating
throughout training, later checkpoints are adapted to more converged statistics
than earlier ones, so stripping normalization penalizes later checkpoints more.
That alone can produce an apparent "later checkpoint is worse" result.

This script pairs each checkpoint with its own saved statistics, so a remaining
difference is attributable to the policy rather than to a normalization
mismatch.

At evaluation:
    training=False     stops the running statistics from updating
    norm_reward=False  reports raw environment reward, comparable across runs

Usage:
    python evaluate_normalized.py --model models_full/ant_full_7000000_steps
    python evaluate_normalized.py --model models_full/ant_full_final \
        --vecnormalize models_full/vecnormalize_final.pkl
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


def derive_stats_path(model_path: str) -> Path:
    """
    CheckpointCallback(name_prefix="ant_full", save_vecnormalize=True) writes:
        ant_full_7000000_steps.zip
        ant_full_vecnormalize_7000000_steps.pkl
    and the end-of-training save writes vecnormalize_final.pkl.
    """
    p = Path(model_path)
    stem = p.stem

    if stem.endswith("_final"):
        return p.parent / "vecnormalize_final.pkl"

    m = re.match(r"^(?P<prefix>.+?)_(?P<steps>\d+)_steps$", stem)
    if m:
        return p.parent / f"{m.group('prefix')}_vecnormalize_{m.group('steps')}_steps.pkl"

    raise ValueError(
        f"Could not derive a VecNormalize path from '{model_path}'. "
        "Pass --vecnormalize explicitly."
    )


def evaluate(model_path: str, n_episodes: int, stats_path: str = None,
             eval_seed: int = 1000):
    stats_path = Path(stats_path) if stats_path else derive_stats_path(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"VecNormalize statistics not found at {stats_path}. Evaluating "
            "without them would not be a valid comparison."
        )

    model = PPO.load(model_path)

    venv = make_vec_env("Ant-v5", n_envs=1, seed=eval_seed)
    env = VecNormalize.load(str(stats_path), venv)
    env.training = False
    env.norm_reward = False

    raw_env = env.venv.envs[0].unwrapped

    returns, lengths, distances = [], [], []
    velocities, terminated_early, energies = [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        start_x = float(raw_env.data.qpos[0])
        last_x = start_x

        ep_return = 0.0
        ep_energy = 0.0
        steps = 0
        was_terminated = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            ep_return += float(reward[0])
            ep_energy += float(np.sum(np.square(action[0])))
            steps += 1

            if done[0]:
                # DummyVecEnv auto-resets, so qpos already refers to the new
                # episode. Use the value captured before the final step, and
                # read the termination flag from the info dict.
                was_terminated = not infos[0].get("TimeLimit.truncated", False)
                break

            last_x = float(raw_env.data.qpos[0])

        env.close()
        venv = make_vec_env("Ant-v5", n_envs=1, seed=eval_seed + ep + 1)
        env = VecNormalize.load(str(stats_path), venv)
        env.training = False
        env.norm_reward = False
        raw_env = env.venv.envs[0].unwrapped

        distance = last_x - start_x
        returns.append(ep_return)
        lengths.append(steps)
        distances.append(distance)
        velocities.append(distance / steps if steps > 0 else 0.0)
        terminated_early.append(1.0 if was_terminated else 0.0)
        energies.append(ep_energy / steps if steps > 0 else 0.0)

    env.close()

    return {
        "model": model_path,
        "vecnormalize": str(stats_path),
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_distance": float(np.mean(distances)),
        "mean_velocity": float(np.mean(velocities)),
        "termination_rate": float(np.mean(terminated_early)),
        "mean_energy": float(np.mean(energies)),
        "raw_returns": returns,
        "raw_distances": distances,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    results = evaluate(args.model, args.episodes, args.vecnormalize)

    print(f"\nEvaluation of {args.model} over {args.episodes} episodes")
    print(f"Normalization: {results['vecnormalize']}")
    print("-" * 60)
    print(f"  Mean return        : {results['mean_return']:.1f} "
          f"(sd {results['std_return']:.1f})")
    print(f"  Mean episode length: {results['mean_length']:.1f} steps")
    print(f"  Mean distance      : {results['mean_distance']:.3f} m")
    print(f"  Mean velocity      : {results['mean_velocity']:.4f} m/step")
    print(f"  Termination rate   : {results['termination_rate']:.1%}")
    print(f"  Mean energy cost   : {results['mean_energy']:.3f}")

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/{Path(args.model).stem}_eval_normalized.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
