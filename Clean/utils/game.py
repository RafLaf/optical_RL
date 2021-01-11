import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc

dtype = torch.long
#dtype = torch.cuda.FloatTensor

#from network import initnet 
#net=initnet(0.9,dtype)
#----------------------
#REAAAAAAAAAAAAAAADMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
## 1)  First choose your dtype
## 2)  TO RUN THIS PROGRAM ALONE DECOMMENT THE  2 LINES BEFORE
## 3)  YOU WILL HAVE TO COMMENT THEM AGAIN TO RUN MAIN
#----------------------

 
show=0
toshow=1000

def launch_scenarios(Wout,net):
    Wout=np.reshape(Wout,(3,int(Wout.size/3)))
    Wout=torch.from_numpy(Wout)
    Wout.type(dtype)
    global show
    reward_list=[]
    env = gym.make('CarRacing-v0')
    start_time = time.time()
    nbep=5
    for i_episode in range(nbep):
        observation = env.reset()
        reward_sum=0
        feature=torch.ones(1025,dtype=torch.float64)
        for t in range(500):
            if  show==toshow:
                env.render()
                show =0 #pour que ce soit visible à l'écran il suffit de décommenter cette ligne -> ralentit tout considerablement. *4 computing time
            else:
                show+=1
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
                break
        print("sum reward:",reward_sum)
        reward_list.append(reward_sum)
    env.close()
    return -sum(reward_list)/nbep    #CMA es minimzes


if __name__ == "__main__":
    launch_scenarios(np.random.random(3*1025),net)
