import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
import sys
import os

def typedevice(tensor,devi):
    return tensor.to(device=devi)

def initnet(rho,dtype,w):
    net=Net(rho,dtype,w).to(device='cpu')
    return net.to(device=device)

class Net(nn.Module):
    def __init__(self,dtype,W):
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
        self.W=typedevice(self.W,device)
        self.r=torch.zeros(512).to(device=torch.device("cuda:0"))
    def forward(self, x):
        x=self.pool3(x)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x=torch.reshape(x,(512,))
        return x
    def RCstep(self,x,aleak,gamma):
        global a,v2
        a=gamma*self.forward(x)
        #print(self.r.device,self.W.device)
        v1=torch.matmul(self.W.float(),self.r)
        v2=self.Win(a)
        '''print('aleak',aleak)
        print('v1')
        print('\n',v1,'\n')
        print('v2')
        print('\n',v2,'\n')
        print('r')
        print('\n',v2,'\n')
        print('a')
        print('\n',a,'\n')'''
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