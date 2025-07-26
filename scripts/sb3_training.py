import myosuite
import gym
import skvideo.io
import numpy as np
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from IPython.display import HTML
from base64 import b64encode
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import torch
from multiprocessing import freeze_support
 
def show_video(video_path, video_width = 400):
   
  video_file = open(video_path, "r+b").read()
 
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")


tmp_path = "/tmp"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

import gym

def make_env():
    return gym.make('CenterReachOut-v0')

def train():
    env = SubprocVecEnv([make_env for _ in range(30)])
    env.render_mode = None
    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,
        tensorboard_log="logs",
        batch_size=960,
        n_steps=32,
        device="cuda",
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512], vf=[512, 512])],
            activation_fn=torch.nn.ReLU,  # or Tanh/ReLU
        ),
    )
    model.learn(total_timesteps=300000)
    model.save("test")

if __name__ == "__main__":
    freeze_support()          
    train()