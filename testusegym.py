# see https://gym.openai.com/docs/
'''
import gym
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()'''

import time
import gym
env = gym.make('CarRacing-v0')
start_time = time.time()
for i_episode in range(2):
    observation = env.reset()
    som=0
    for t in range(100):
        env.render()     #pour que ce soit visible à l'écran il suffit de décommenter cette ligne -> ralentit tout considerablement. *4 computing time
        #print(i_episode,t,observation.shape)

        action = env.action_space.sample()
        #print(action)
        observation, reward, done, info = env.step(action)
       # print(reward)
        som+=reward
        print(som)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
print(time.time()-start_time)
env.close()
