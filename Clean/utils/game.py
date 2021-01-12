import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
import os






#from network import initnet 
#net=initnet(0.9,dtype)
#----------------------
#REAAAAAAAAAAAAAAADMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
## 1)  First choose your dtype
## 2)  TO RUN THIS PROGRAM ALONE DECOMMENT THE  2 LINES BEFORE
## 3)  YOU WILL HAVE TO COMMENT THEM AGAIN TO RUN MAIN
#----------------------

show=0
toshow=5


def launch_scenarios(Wout):
    global dtype
    global show
    Wout=np.reshape(Wout,(3,1025))
    Wout=torch.from_numpy(Wout)
    Wout.type(dtype)
    reward_list=[]
    start_time = time.time()
    nbep=5
    max_reward=0
    for i_episode in range(nbep):
        observation = env.reset()
        #env.viewer.close()
        reward_sum=0
        feature=torch.ones(1025,dtype=torch.float64)
        for t in range(500):
            '''
            if  show==toshow:
                env.render()
                show=0 #pour que ce soit visible à l'écran il suffit de décommenter cette ligne -> ralentit tout considerablement. *4 computing time
            else:
                show+=1

            '''
            action=torch.clip(torch.matmul(Wout,feature),-1,1)
            action=action.detach().numpy()
            observation, reward, done, info = env.step(action)
            obs=np.array(observation)
            obs=np.moveaxis(obs,[2],[0])
            obs=np.array([obs])
            obs=torch.from_numpy(obs)
            obs.type(dtype)
            feature[:-1]=net.RCstep(obs.float(),0.5,1e-6)
            reward_sum+=reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                reward_sum+=50000000
                break
            if reward_sum > max_reward:
                max_reward = reward_sum
            elif max_reward-reward_sum > 15:
                break
        print("sum reward:",reward_sum)
        reward_list.append(reward_sum)
    #env.close()
    return -sum(reward_list)/nbep    #CMA es minimzes


if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    launch_scenarios.dtype=dtype
    try:
        W=np.load('W.npy',allow_pickle=True)
        print('loaded W')
    except FileNotFoundError:
        print('not found creating W')
        W=sc.random(Nr,Nr,density=float(D/Nr))
        W=rho/max(abs(np.linalg.eigvals(W.A)))*W
        W=(2*W-(W!=0))
        W=W.A
        np.save('W.npy',W)
    env = gym.make('CarRacing-v0')
    from network import initnet
    import network
    network.W=W
    network.dtype=dtype
    net=initnet(0.9,dtype)
    launch_scenarios(np.random.random(3*1025))
