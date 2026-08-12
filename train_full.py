"""
train_full.py

Full-length PPO training on Ant-v5 using tuned hyperparameters from the
official Stable-Baselines3 Zoo (searched specifically for this task),
plus observation/reward normalization. Checkpoints periodically so you
can record a video from an intermediate point without waiting for the
full run, and so nothing is lost if it's interrupted.

Usage:
    python train_full.py
"""

import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

TOTAL_STEPS = 10_000_000
CHECKPOINT_EVERY = 500_000
SAVE_DIR = "models_full"

os.makedirs(SAVE_DIR, exist_ok=True)

env = make_vec_env("Ant-v5", n_envs=1, seed=0)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.ReLU,
    log_std_init=-2,
    ortho_init=False,
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1.90609e-05,
    n_steps=512,
    batch_size=32,
    n_epochs=10,
    gamma=0.98,
    gae_lambda=0.8,
    clip_range=0.1,
    ent_coef=4.9646e-07,
    vf_coef=0.677239,
    max_grad_norm=0.6,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=0,
)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path=SAVE_DIR,
    name_prefix="ant_full",
    save_vecnormalize=True,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback)

model.save(f"{SAVE_DIR}/ant_full_final")
env.save(f"{SAVE_DIR}/vecnormalize_final.pkl")
print("Training complete.")