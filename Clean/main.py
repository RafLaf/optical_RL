import time
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sc
from scipy.stats import multivariate_normal
import cma
from utils.network import initnet
from utils.game import launch_scenarios

dtype = torch.long
#dtype = torch.cuda.FloatTensor

net=initnet(0.9,dtype)

def progtot():
    for iterate in range(1):
        mu=np.zeros(3*1025)
        CMA=cma.evolution_strategy
        sol,es=CMA.fmin2(launch_scenarios,mu,0.3,args={net,},options={'ftarget':-50000,'maxiter':10000,'popsize':15})
        #env.close()
        #sol,es=CMA.fmin2(launch_scenarios,es.result[5],es.result[6],options={'ftarget':-10,'maxiter':1})
        print(sol,es)
        return(es)


if __name__ == "__main__":
    progtot()
