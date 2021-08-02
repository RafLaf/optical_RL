
import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
import os
import scipy.sparse as sc






#from network import initnet 
#net=initnet(0.9,dtype)
#----------------------
#REAAAAAAAAAAAAAAADMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
## 1)  First choose your dtype
## 2)  TO RUN THIS PROGRAM ALONE DECOMMENT THE  2 LINES BEFORE
## 3)  YOU WILL HAVE TO COMMENT THEM AGAIN TO RUN MAIN
#----------------------



def launch_scenarios(Wout,display=0):
    global dtype
    global show
    Wout=np.reshape(Wout,(3,1025))
    Wout=torch.from_numpy(Wout)
    Wout=typedevice(Wout,device)
    Wout=Wout.to(dtype=torch.float32)
    reward_list=[]
    start_time = time.time()
    nbep=3
    max_reward=0
    
    net.r=torch.zeros(512).to(device=torch.device("cuda:0"))
    #env = gym.make('CarRacing-v0')
    for i_episode in range(nbep):
        net.r=torch.zeros(512).to(device=torch.device("cuda:0"))
        observation = env.reset()
        #env.viewer.close()
        reward_sum=0
        feature=torch.zeros(1025,dtype=torch.float32).to(device=device)
        feature[-1]=1
        #feature=torch.from_numpy(np.array(1024))
        for t in range(5000000):
            #print('game timestep',t, 'of episode', i_episode)
            if display==2:
                env.render('human')
            if display==1:
                if i_episode%3==0:
                    env.render('human')
            if display==0:
                pass
            #Wout=Wout.to(dtype=torch.torch.float64)
            action=torch.clip(torch.matmul(Wout,feature),-1,1)
            #erf ou 2sigmoid-1 ou tanh 
            #print(action)
            action=action.detach().cpu().numpy()*100
            #action=[0,1,0]
            observation, reward, done, info = env.step(action)
            obs=np.array(observation)
            obs=np.moveaxis(obs,[2],[0])
            obs=np.array([obs])
            obs=torch.from_numpy(obs)
            obs=typedevice(obs,device)
            feature[:-1]=net.RCstep(obs.float(),0.9,1e-6)
            #feature=net.RCstep(obs.float(),0.5,1e-6)
            reward_sum+=reward
            #print('sum reward',reward_sum)
            #stop conditions
            if done and t<998 and reward_sum > 900:
                print("Episode finished after {} timesteps".format(t+1))
                reward_sum+=50000000
                break
            if reward_sum>max_reward:
                max_reward=reward_sum
            
            #if reward_sum > max_reward:
            #    max_reward = reward_sum
            elif max_reward-reward_sum > 5:
                #reward_sum=reward_sum-(1000-t*0.1)
                break
        print("step number:",t)
        print("sum reward:",reward_sum)
        reward_list.append(max_reward)
        max_reward=0
    print(reward_list)
    time.sleep(3)
    return -sum(reward_list)/nbep    #CMA es minimzes

def typedevice(tensor,devi):
    return tensor.to(device=devi)


if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    device = torch.device('cpu')
    #device= 'cuda'
    launch_scenarios.dtype=dtype
    try:
        W=np.load('W.npy',allow_pickle=True)
        print('loaded W')
    except FileNotFoundError:
        Nr,D,rho=512,10,0.9
        print('not found creating W')
        W=sc.random(Nr,Nr,density=float(D/Nr))
        W=(2*W-(W!=0))
        W=rho/max(abs(np.linalg.eigvals(W.A)))*W
        W=W.A
        np.save('W.npy',W)
    env = gym.make('CarRacing-v0')
    #env = gym.wrappers.Monitor(env, "./vid", force=True)
    from network import initnet
    import network
    network.device=device
    net=initnet(0.9,dtype,W)
    launch_scenarios(np.random.random(3*1025))
