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

#dtype = torch.long
dtype = torch.cuda.FloatTensor
torch.device('cuda')


def typedevice(tensor,typ,devi):
    return tensor.to(device=devi,dtype=typ)


def progtot():
    global dtype
    global es
    mu=np.zeros(3*1025)
    es = cma.CMAEvolutionStrategy(mu, 0.2)
    #es.optimize(utils.game.launch_scenarios)
    #res = es.result
    while not es.stop():
        print('hey')
        solutions = es.ask()
        es.tell(solutions, [utils.game.launch_scenarios(s) for s in solutions])
        es.disp()
    return(es.result_pretty())

if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    device = 'cpu'
    #device= 'cuda'
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
    utils.network.dtype=dtype
    utils.game.dtype=dtype
    utils.network.W=W
    utils.game.env=env
    utils.network.device=device
    utils.game.device=device
    try :
        net = torch.load('model.pt')
        net.eval()
        print('loaded net')
    except FileNotFoundError:
        print('creating net')
        net=utils.network.initnet(0.9,dtype)
        torch.save(net, 'model.pt')
    utils.game.net=net
    progtot()
    env.close()


