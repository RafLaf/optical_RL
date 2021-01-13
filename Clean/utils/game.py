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
    Wout=typedevice(Wout,dtype,device)
    reward_list=[]
    start_time = time.time()
    nbep=5
    max_reward=0
    display=True
    env = gym.make('CarRacing-v0')
    for i_episode in range(nbep):
        observation = env.reset()
        #env.viewer.close()
        reward_sum=0
        feature=torch.zeros(1025,dtype=dtype,device=device)
        feature[1024]=1
        #feature=torch.from_numpy(np.array(1024))
        for t in range(5000000):
            '''
            if  show==toshow:
                env.render()
                show=0 #pour que ce soit visible à l'écran il suffit de décommenter cette ligne -> ralentit tout considerablement. *4 computing time
            else:
                show+=1
            '''
            
            if  i_episode==nbep-1 and display==True:
                env.render()
                
            
            
            action=torch.clip(torch.matmul(Wout,feature),-1,1)
            action=action.detach().numpy()
            
            #a1=max(min((np.sum(np.array(feature.detach().numpy())*Wout[0:1024])+Wout[1024]),1),-1)
            #a2=max(min((np.sum(np.array(feature.detach().numpy())*Wout[1025:2049])+Wout[2049]),1),-1)
            #a3=max(min((np.sum(np.array(feature.detach().numpy())*Wout[2050:3074])+Wout[3074]),1),-1)
            #action=[a1,a2,a3]
            
            observation, reward, done, info = env.step(action)
            obs=np.array(observation)
            obs=np.moveaxis(obs,[2],[0])
            obs=np.array([obs])
            obs=torch.from_numpy(obs)
            obs=typedevice(obs,dtype,device)
            feature[:-1]=net.RCstep(obs.float(),0.5,1e-6)
            feature=net.RCstep(obs.float(),0.5,1e-6)
            reward_sum+=reward
            if done and t<998 and reward_sum > 900:
                print("Episode finished after {} timesteps".format(t+1))
                reward_sum+=50000000
                break
            if reward_sum > max_reward:
                max_reward = reward_sum
            elif max_reward-reward_sum > 15:
                reward_sum=reward_sum-(1000-t*0.1)
                break
        print("step number:",t)
        print("sum reward:",reward_sum)
        reward_list.append(reward_sum)
    env.close()
    return -sum(reward_list)/nbep    #CMA es minimzes

def typedevice(tensor,typ,devi):
    return tensor.to(device=devi,dtype=typ)


if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    device = 'cpu'
    #device= 'cuda'
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
    network.device=device
    network.W=W
    network.dtype=dtype
    net=initnet(0.9,dtype)
    launch_scenarios(np.random.random(3*1025))
