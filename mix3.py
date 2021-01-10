
import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
from scipy.stats import multivariate_normal
import cma

class Net(nn.Module):
	def __init__(self,rho):
		super(Net, self).__init__()
		self.pool3 = nn.MaxPool2d(2, 2)
		self.conv1 = nn.Conv2d(3, 32, 5)			  #convolution avec 3 channel entrée (RVB); 32 channel de sortie, kernel de 5*5 (pas sûr de moi pour le 5*5)
		nn.init.normal_(self.conv1.weight)	#normal distribution
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 5)
		nn.init.normal_(self.conv2.weight)
		self.conv3 = nn.Conv2d(64, 128, 5)
		nn.init.normal_(self.conv3.weight)
		#self.conv4 = nn.Conv2d(128, 256, 5)
		#nn.init.normal_(self.conv4.weight)
		self.Win=nn.Linear(512,512)
		Nr,D=512,15
		self.W=sc.random(Nr,Nr,density=float(D/Nr))
		self.W=rho/max(abs(np.linalg.eigvals(self.W.A)))*self.W
		self.W=(2*self.W-(self.W!=0))
		self.W=torch.from_numpy(self.W.A)
		self.r=torch.zeros(512)
	def forward(self, x):
		#print(x.shape)
		x=self.pool3(x)
		#print(x.shape)
		x = self.pool(self.conv1(x))
		#print(x.shape)
		x = self.pool(self.conv2(x))
		#print(x.shape)
		x = self.pool(self.conv3(x))
		#print(x.shape)
		#x = self.pool(self.conv4(x))
		#print(torch.min(x),torch.max(x))
		s=x.size()
		x=torch.reshape(x,(s[0],s[1]*s[2]*s[3]))
		return x
	def RCstep(self,x,aleak,gamma):
		a=self.forward(x)
		#print('ok')
		#print(a.size(),self.r.size())
		v1=torch.matmul(self.r,self.W.float())
		v2=gamma*self.Win(a)
		#print(v1.size(),v2.size(),(v1+v2).size())
		self.r=(1-aleak)*self.r+aleak*torch.tanh(v1+v2)
		self.r=torch.reshape(self.r,(512,))
		a=torch.reshape(a,(512,))
		#print(self.r.size(),a.size)
		return torch.cat((self.r,a))

	
net = Net(0.9)
def launch_scenarios(Wout):
	#Wout=np.reshape(Wout,(3,int(Wout.size/3)))
	reward_list=[]
	env = gym.make('CarRacing-v0')
	start_time = time.time()
	for i_episode in range(1):
		observation = env.reset()
		reward_sum=0
		feature=torch.from_numpy(np.array(1024))
		#feature=np.ones(1025)
		#feature[-1]=1
		
		for t in range(100):
			env.render()	 #pour que ce soit visible à l'écran il suffit de décommenter cette ligne -> ralentit tout considerablement. *4 computing time
			#print(i_episode,t,observation.shape)
			
			#generation d'action par WOUT et features
			a1=max(min((np.sum(np.array(feature.detach().numpy())*Wout[0:1024])+Wout[1024])/1025,1),-1)
			a2=max(min((np.sum(np.array(feature.detach().numpy())*Wout[1025:2049])+Wout[2049])/1025,1),-1)
			a3=max(min((np.sum(np.array(feature.detach().numpy())*Wout[2050:3074])+Wout[3074])/1025,1),-1)
			action=[a1,a2,a3]
			
			#print(feature.shape,Wout.shape)
			#action=np.clip(Wout@feature,-1,1)
			#print(action.shape)
			#action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			obs=np.array(observation)
			obs=np.moveaxis(obs,[2],[0])
			obs=np.array([obs])
			obs=torch.from_numpy(obs)
			#feature2=net.RCstep(obs.float(),0.5,1e-6)
			#feature[:-1]=feature2.detach().numpy()
			feature=net.RCstep(obs.float(),0.5)
			reward_sum+=reward
			
			#print("len,",len(feature))
			print("action: ",action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
		print("sum reward:",reward_sum)
		reward_list.append(reward_sum)
	#print(time.time()-start_time)
	env.close()
	return -sum(reward_list)   #CMA es minimzes



def progtot():
	for iterate in range(1):
		mu=np.random.random(3*1025)
		C=np.identity(1025*3)
		sig=0.5
		#rng = np.random.default_rng()
		#Wout=rng.multivariate_normal(mu,C*sig)
		Wout=np.random.multivariate_normal(mu,sig*C)
		print('ready to launch')
		#launch_scenarios(Wout)
		#cma.evolution_strategy(launch_scenarios,mu,5,options=cma.evolution_strategy.cma_default_options_(),args=())
		CMA=cma.evolution_strategy
		sol,es=CMA.fmin2(launch_scenarios,mu,5,options={'ftarget':-10})
		print(sol,es)
"""
A=np.random.random((1,3,96,96))
B = torch.from_numpy(A)
net = Net(0.9)
net.RCstep(B.float(),0.5)
"""