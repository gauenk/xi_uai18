import pickle,os
from os import path as osp
import _init_paths
import pandas as pd
import numpy as np
import numpy.random as npr
import mnist_loader
from distributions import WeibullDistribution as Weibull
from distributions import MultinomialDistribution as Multinomial
from processes import PoissonProcess
import matplotlib.pyplot as plt

def const_rate(tau):
    return 2*tau
def const_hazard(tau):
    return npr.uniform(0,tau)
def weibull_rate(tau,shape=1,scale=1):
    mean = (tau/scale)**shape
    return const*tau
def weibull_hazard(tau,shape=1,scale=1):
    rv = Weibull({'shape':shape,'scale':scale},
                 is_hazard_rate=True)
    return rv

class MarkedPoint():

    def __init__(self,value,level,pid):
        self.value = value
        self.level = level
        self.pid = pid
        self.children = []
        
    def __str__(self):
        rstr = '(v,l,p): ({:.3f},{:d},{:d})'.format(self.value,self.level,self.pid)
        for child in self.children:
            rstr += ', '+ str(child)
        #rstr += '\n'
        return rstr

    def get_tree(self):
        x,y = [self.value],[self.level]
        for child in self.children:
            c_x,c_y = child.get_tree()
            x.extend(c_x)
            y.extend(c_y)
        return x,y

def generate_hawkes_process_wrong():

    tau = 3
    p = PoissonProcess(const_rate,const_hazard)
    points = p.sample(tau)
    sample = [MarkedPoint(p,0,-1) for i,p in enumerate(points)]
    print("gen_hawks")
    for idx,point in enumerate(points):
        children = spawn_children(tau,point,idx,1)
        if len(children) > 0:
            sample[idx].children.extend(children)
    print(sample)
    return sample

def spawn_children(t_end,t_start,pid,level):
    tau = t_end - t_start
    p = PoissonProcess(const_rate,const_hazard)
    points = p.sample(tau) + t_start
    print(points)
    if len(points) == 0:
        return []

    sample = [MarkedPoint(p,level,pid) for i,p in enumerate(points)]
    #print("N: {:d} child: {:.3f}".format(len(points),tau))
    for idx,point in enumerate(points):
        # print("N: {:d} child: {:.3f} point: {:.3f}".format(len(points),tau,point))
        children = spawn_children(t_end,point,idx,level+1)
        if len(children) > 0:
            sample[idx].children.extend(children)
    return sample
    
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def gen_next_sk(lamTk,a,delta):
    u1 = npr.uniform(0,1)
    u2 = npr.uniform(0,1)
    D_k1 = 1 + delta * np_loge(u1) / (lamTk - a)
    S1_k1 = np.inf
    if D_k1 > 0:
        S1_k1 = -1/delta * np_loge(D_k1)
    S2_k2 = -1/a * np_loge(u2)
    S_k = np.min(S1_k1,S2_k2)
    return S_k

def gen_next_lam(lamTk,a,delta,Sk1):
    lamTk1_minus = (lamTk - a) * np.exp(-delta * Sk1) + a
    return lamTk1_minus

def generate_hawkes_process(G,a=1,delta=.5,lam0=2,time_f=3):
    
    # init first event time
    S_1 = gen_next_sk(lam0,a,delta)
    events = [S_1]
    lamTk_minus = gen_next_lam(lam0,a,delta,S_1)
    lamTk = lamTk_minus + G.sample(1)

    # get all the rest
    time = events[-1]
    while (time < time_f):
        # sample inter-event time
        S_k = gen_next_sk(lamTk,a,delta)
        events.append(events[-1] + S_k)
        # update rate constant
        lamTk_minus = gen_next_lam(lam0,a,delta,S_k)
        lamTk = lamTk_minus + G.sample(1)
    return events


if __name__ == "__main__":

    G = Multinomial({'prob_vector':prob})
    import seaborn as sns

    plt.show()
    sample = generate_hawkes_process()
    sns.distplot(sample,hist=True,rug=False,\
                 label='G').set(xlim=(0,3))


    # xval,yval,zval = [],[],[]
    # for i,s in enumerate(sample):
    #     x,y = s.get_tree()
    #     xval.extend(x)
    #     yval.extend(y)
    #     zval.extend([i for _ in range(len(x))])
    # zval = np.array(zval,dtype=np.float)
    # zval = zval / np.max(zval)
    # print(xval,yval,zval)
    # print(len(xval) == len(zval))
    # plt.scatter(xval,yval,c=zval)
    # plt.show()
