import gym
import imageio
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay

# Video stuff 
from pathlib import Path
from IPython import display as ipythondisplay

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2


env_id = 'BipedalWalker-v2'
video_folder = '/videos'
video_length = 100

# set our inital enviorment
env = DummyVecEnv([lambda: gym.make(env_id)])
obs = env.reset()


# Evaluation Function
def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward


import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make('BipedalWalker-v2')])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


def show_videos(video_path='', prefix=''):
  html = []
  for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


model = PPO2(MlpPolicy, env, verbose=1)


# Random Agent, before training
mean_reward_before_train = evaluate(model, num_steps=10000)

# Train model
model.learn(total_timesteps=50000)

# Save model
model.save("ppo2-walker-50000")

# Random Agent, after training
mean_reward_after_train = evaluate(model, num_steps=1000)

# Record & show video
record_video('BipedalWalker-v2', model, video_length=1500, prefix='ppo2-walker-50000')
show_videos('videos', prefix='ppo2-walker-50000')


mean_reward_before_train = evaluate(model, num_steps=10000)

# Train model
model.learn(total_timesteps=500000)

# Save model
model.save("ppo2-walker-500000")

# Random Agent, after training
mean_reward_after_train = evaluate(model, num_steps=10000)

# Record & show video
record_video('BipedalWalker-v2', model, video_length=1500, prefix='ppo2-walker-500000')
show_videos('videos', prefix='ppo2-walker-500000')
