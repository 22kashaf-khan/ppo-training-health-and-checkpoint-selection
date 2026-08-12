"""
record_video.py (corrected)
"""

import argparse
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def record(model_path: str, vecnorm_path: str, out_dir: str, video_name: str):
    os.makedirs(out_dir, exist_ok=True)

    def make_env():
        env = gym.make("Ant-v5", render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=out_dir,
            name_prefix=video_name,
            episode_trigger=lambda ep: True,
        )
        return env

    raw_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnorm_path, raw_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)

    obs = vec_env.reset()

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        total_reward += reward[0]
        steps += 1
        done = dones[0]

    vec_env.close()
    print(f"Episode finished: {steps} steps, total reward {total_reward:.1f}")
    print(f"Video saved in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, required=True)
    parser.add_argument("--out", type=str, default="videos")
    parser.add_argument("--name", type=str, default="episode")
    args = parser.parse_args()

    record(args.model, args.vecnorm, args.out, args.name)