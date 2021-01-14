import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
import sys
import os

def typedevice(tensor,typ,devi):
    return tensor.to(device=devi,dtype=typ)

def initnet(rho,dtype,w):
    return Net(rho,dtype,w)

class Net(nn.Module):
    def __init__(self,rho,dtype,W):
        super(Net, self).__init__()
        Nr,D=512,15
        self.pool3 = nn.MaxPool2d(2, 2)
        #self.pool3=typedevice(self.pool3,dtype,device)
        self.conv1 = nn.Conv2d(3, 32, 5)              #convolution avec 3 channel entrée (RVB); 32 channel de sortie, kernel de 5*5 (pas sûr de moi pour le 5*5)
        nn.init.normal_(self.conv1.weight)            #normal distribution
        self.pool = nn.MaxPool2d(2, 2)
        #self.pool=typedevice(self.pool,dtype,device)
        self.conv2 = nn.Conv2d(32, 64, 5)
        nn.init.normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 5)
        nn.init.normal_(self.conv3.weight)
        self.Win=nn.Linear(512,512)
        #self.Win=typedevice(self.Win,dtype,device)
        self.W=W
        self.W=torch.from_numpy(self.W)
        self.W=typedevice(self.W,dtype,device)
        self.r=torch.zeros(512)
    def forward(self, x):
        x=self.pool3(x)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x=torch.reshape(x,(512,))
        return x
    def RCstep(self,x,aleak,gamma):
        a=gamma*self.forward(x)
        v1=torch.matmul(self.r,self.W.float())
        v2=self.Win(a)
        self.r=(1-aleak)*self.r+aleak*torch.tanh(v1+v2)
        return torch.cat((self.r,a))

if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    device = 'cpu'
    #device= 'cuda'
    try:
        W=np.load('W.npy',allow_pickle=True)
        print('loaded W')
        rho=0.9
        net=Net(rho,dtype,W)
    except FileNotFoundError:
        rho=0.9
        Nr,D=512,15
        W=sc.random(Nr,Nr,density=float(D/Nr))
        W=rho/max(abs(np.linalg.eigvals(W.A)))*W
        W=(2*W-(W!=0))
        W=W.A
        net=Net(rho,dtype,W)
        print('not found W creating W')
        np.save('W.npy',W)
    A=np.random.random((1,3,96,96))
    B = torch.from_numpy(A)
    print(net.RCstep(B.float(),0.5,1e-6))