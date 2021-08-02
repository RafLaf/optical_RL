import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
#os.environ['DISPLAY'] = 'localhost:0.0'
#from IPython import display
import pyvirtualdisplay

pyvirtualdisplay.Display(visible=0,size=(1400,900)).start()

try:
    shutil.rmtree("videos")
except FileNotFoundError:
    pass
os.mkdir("videos")


env = gym.make('CarRacing-v0') # env_name = "Pendulum-v0"

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        #plt.imshow(os.path.join('videos', '-'.join(str(t)) + '.png'),
        #               env.render(mode='rgb_array'))
        #display.display(plt.gcf())
        #display.clear_output(wait=True)
        print(t)
        action = env.action_space.sample()  # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()