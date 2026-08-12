"""
record_friction_video.py
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    model = env.unwrapped.model
    model.geom_friction[:, 0] *= 0.6
    env = RecordVideo(env, video_folder="videos/low_friction", name_prefix="low_friction", episode_trigger=lambda ep: True)
    return env

raw_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("models_full/ant_full_vecnormalize_7000000_steps.pkl", raw_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("models_full/ant_full_7000000_steps")
obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    done = dones[0]
vec_env.close()
print("Saved to videos/low_friction/")