import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
from scipy.stats import multivariate_normal

dtype = torch.long
#dtype = torch.cuda.FloatTensor

def initnet(rho,dtype):
    return Net(rho,dtype)

class Net(nn.Module):
    def __init__(self,rho,dtype):
        super(Net, self).__init__()
        Nr,D=512,15
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool3.type(dtype)
        self.conv1 = nn.Conv2d(3, 32, 5)              #convolution avec 3 channel entrée (RVB); 32 channel de sortie, kernel de 5*5 (pas sûr de moi pour le 5*5)
        nn.init.normal_(self.conv1.weight)            #normal distribution
        self.pool = nn.MaxPool2d(2, 2)
        self.pool.type(dtype)
        self.conv2 = nn.Conv2d(32, 64, 5)
        nn.init.normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 5)
        nn.init.normal_(self.conv3.weight)
        try : 
            W=np.load('W.npy',allow_pickle=True)
        except:
            print('not found')
            self.Win=nn.Linear(512,512)
            self.W=sc.random(Nr,Nr,density=float(D/Nr))
            self.W=rho/max(abs(np.linalg.eigvals(self.W.A)))*self.W
            self.W=(2*self.W-(self.W!=0))
            np.save('W.npy',W)
        self.W=torch.from_numpy(self.W.A)
        self.W.type(dtype)
        self.r=torch.zeros(512)

    def forward(self, x):
        x=self.pool3(x)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        s=x.size()
        x=torch.reshape(x,(s[1]*s[2]*s[3],))
        return x
    def RCstep(self,x,aleak,gamma):
        a=gamma*self.forward(x)
        a.type(dtype)
        v1=torch.matmul(self.r,self.W.float())
        v2=self.Win(a)
        temp_r=(1-aleak)*self.r+aleak*torch.tanh(v1+v2)
        self.r=temp_r.detach().clone()
        self.r=torch.reshape(self.r,(512,))
        a=torch.reshape(a,(512,))
        return torch.cat((self.r,a))


if __name__ == "__main__":
    A=np.random.random((1,3,96,96))
    B = torch.from_numpy(A)
    net=Net(0.9,dtype)
    print(net.RCstep(B.float(),0.5,1e-6))