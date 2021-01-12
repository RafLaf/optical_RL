import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
from scipy.stats import multivariate_normal
import cma
import utils.network
import utils.game
import sys
import os


dtype = torch.long
#dtype = torch.cuda.FloatTensor




def progtot():
    for iterate in range(1):
        mu=np.zeros(3*1025)
        CMA=cma.evolution_strategy
        sol,es=CMA.fmin2(utils.game.launch_scenarios,mu,0.01,options={'ftarget':-50000,'maxiter':10,'popsize':2})
        print(sol,es)
        return(es)


if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
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
    try :
        net = torch.load('model.pt')
        net.eval()
        print('loaded net')
    except FileNotFoundError:
        print('creating net')
        net=initnet(0.9,dtype)
        torch.save(net, 'model.pt')
    utils.network.dtype=dtype
    utils.game.dtype=dtype
    utils.network.W=W
    utils.game.env=env
    utils.game.net=net
    progtot()
