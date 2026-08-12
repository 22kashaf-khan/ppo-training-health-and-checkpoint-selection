"""
check_leg_symmetry.py

Log each leg's hip/ankle joint angles over one episode and report the
range of motion per joint, to check whether one leg is moving noticeably
less than the others.
"""

import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    env = gym.make("Ant-v5")
    return env

raw_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("models_full/ant_full_vecnormalize_7000000_steps.pkl", raw_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("models_full/ant_full_7000000_steps")

inner_env = raw_env.envs[0].unwrapped
mj_model = inner_env.model
mj_data = inner_env.data

joint_names = ["hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4"]
addrs = {name: mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names}
logs = {name: [] for name in joint_names}

obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    done = dones[0]
    for name, addr in addrs.items():
        logs[name].append(mj_data.qpos[addr])

print(f"{'Joint':<10}{'Min (deg)':>12}{'Max (deg)':>12}{'Range (deg)':>14}")
print("-" * 48)
for name in joint_names:
    vals = np.rad2deg(logs[name])
    print(f"{name:<10}{vals.min():>12.1f}{vals.max():>12.1f}{(vals.max()-vals.min()):>14.1f}")