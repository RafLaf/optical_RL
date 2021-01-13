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
#dtype = torch.cuda.FloatTensor
#torch.device('cuda')


def typedevice(tensor,typ,devi):
    return tensor.to(device=devi,dtype=typ)


def progtot():
    global dtype
    global mu
    CMA=cma.evolution_strategy
    
    es = cma.CMAEvolutionStrategy(mu, 0.5,{'popsize':5,'ftarget':-50000,'maxiter':100000})
    #es.opts.set({'popsize':5,'ftarget':-50000,'maxiter':100000})
    iteration_number=0
    while not es.stop():
        iteration_number+=1
        if iteration_number==1:
            try:
                C=np.load('C.npy',allow_pickle=True)
                print('loaded C')
                es.C=C
            except FileNotFoundError:
                C=es.C
                np.save('C.npy',C)
        Wout = es.ask()
        print("len",len(Wout))
        mu=es.mean
        print("C",C)
        print("mu",mu)
        es.tell(Wout, [utils.game.launch_scenarios(Wouti) for Wouti in Wout])
        es.disp()
        np.save('W.npy',W)
        np.save('mu.npy',mu)
        np.save('C.npy',C)
        print("\n \n NEW ITERATION \n \n")
    #sol,es=CMA.fmin2(utils."game.launch_scenarios,mu,0.5,options={'ftarget':-50000,'maxiter':100000,'popsize':5})
    #print(sol,es)
    env.close()
    return(es)

if __name__ == "__main__":
    dtype = torch.long
    #dtype = torch.cuda.FloatTensor
    device = 'cpu'
    #device= torch.device('cuda')
    try:
        W=np.load('W.npy',allow_pickle=True)
        print('loaded W')
    except FileNotFoundError:
        print('not found creating W')
        Nr,D=512,15
        W=sc.random(Nr,Nr,density=float(D/Nr))
        W=0.9/max(abs(np.linalg.eigvals(W.A)))*W
        W=(2*W-(W!=0))
        W=W.A
        np.save('W.npy',W)
    utils.network.dtype=dtype
    utils.game.dtype=dtype
    utils.network.W=W
    utils.network.device=device
    utils.game.device=device
    try :
        net = torch.load('model.pt')
        net.eval()
        print('loaded net')
    except FileNotFoundError:
        print('creating net')
        net=utils.network.initnet(0.9,dtype,W)
        torch.save(net, 'model.pt')
    try:
        mu=np.load('mu.npy',allow_pickle=True)
        print('loaded mu')
    except FileNotFoundError:
        print('not found creating mu')
        mu=np.zeros(3*1025)
        np.save('mu.npy',mu)
    utils.network.dtype=dtype
    utils.game.dtype=dtype
    utils.network.W=W
    #utils.game.env=env
    utils.game.net=net
    progtot()


