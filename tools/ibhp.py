import pickle,os,re
from easydict import EasyDict as edict
from functools import partial
from os import path as osp
import _init_paths
from base_utils import *
import pandas as pd
import numpy as np
import numpy.random as npr
import mnist_loader
from distributions import Gamma
from distributions import WeibullDistribution as Weibull
from distributions import MultinomialDistribution as Multinomial
from processes import PoissonProcess,CRP
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
eng_stopwords = set(stopwords.words('english'))

class Kappa():
    """
    Abstraction of a 1 x K IBHP Matrix
    """
    betas = None
    taus = None
    def __init__(self,w,c):
        self.w = w
        self.c = [c]
        self.K = [len(c)]

    def nz(self,i):
        """
        number of non-zero elements across k;
        same as l0 norm
        """
        nonzero = np.count_nonzero(self.c[i])
        return nonzero

    def __call__(self,delta,i,k):
        # TODO: test me.
        gammas = Kappa.betas*np.exp(-delta/Kappa.taus)
        masked_w = self.w[k,:] * self.c[i][k]
        return np.sum(masked_w * gammas)
    
    def inz(self,i):
        # index of nonzero rows; index 0 -> K - 1
        indices = self.c[i].nonzero()[0]
        return indices

    def update(self,w,c):
        self.w = w
        self.c.append(c)
        self.K.append(len(c))

def append_sample_f(c,w,v,t,T,ci,wi,vi,ti,Ti):
    c.append(ci)
    w.append(wi)
    v.append(vi)
    t.append(ti)
    T.append(Ti)

def sample_wv(c,w,v,K,Kplus,pi,M):
    # sample the new plus
    if Kplus == 0:
        return w,v
    wp = np.zeros( (Kplus,M.L) )
    vp = np.zeros( (Kplus,len(M.S)) )
    for k in range(Kplus):
        wp[k,:] = npr.dirichlet(pi.w0)
        vp[k,:] = npr.dirichlet(pi.v0)

    # append new samples
    w = np.vstack( (w,wp) )
    v = np.vstack( (v,vp) )
    return w,v
        
def sample_text(D,v,kappa):
    try:
        text_probs = np.sum(v[kappa.inz(-1),:],axis=0)/float(kappa.nz(-1))
        Ti = npr.multinomial(D,text_probs)
    except:
        print(kappa.inz(-1))
        print(np.sum(v[kappa.inz(-1),:],axis=0))
        print(np.sum(v[kappa.inz(-1),:],axis=0)/float(kappa.nz(-1)))
    return Ti
    
def sample_c(kappa,tp,t_hist,lambda0):
    # sample first K
    K = kappa.K[-1]
    cK = np.zeros(K)
    lambdas = np.zeros(K)
    for i,time in enumerate(t_hist):
        for k in range(kappa.K[i]):
            delta = tp - time
            if delta < 0:
                raise ValueError("Can't have negative wait time!")
            lambda_ki = kappa(delta,i,k)
            lambdas[k] += lambda_ki / kappa.nz(i)
    for k in range(K):
        pk = lambdas[k] / ( lambda0/K + lambdas[k] )
        cK[k] = npr.binomial(1,pk)

    # sample K+
    lambdaK = np.sum(lambdas)
    Kp_rate = lambda0 / ( lambda0 + lambdaK )
    Kp = npr.poisson(Kp_rate)
    cP = np.ones(Kp)

    c = np.hstack( (cK,cP) )
    return c,K+Kp,Kp
    
def sample_time(kappa,t_hist,K):
    # TODO: actually write this function; its wrong
    # lambdak_list = np.zeros(K)
    # for k in range(K):
    #     lambdak = 0
    #     for i,time in enumerate(t_hist):
    #         delta = tp - time
    #         lambdak += kappa(delta,i,k)
    #     lambdak /= kappa.nz(k)
    #     lambdak_list[k] = lambdak
    rate = 2 # np.sum(lambdak_list)
    time = PoissonProcess(rate,'c').sample_tau(1)[0]
    return time

def ibhp_prior(N,M,pi,theta,alpha0):

    """
    Known issues:
    -> the "sample_time" function is wrong
    -> the lambda0 sample time must be small since pk can result in all 0 c_ik
    -> sometimes the sample_text fails due to 
    """

    lambda0 = theta.lambda0 # unpack lambda0
    c_hist = []
    w_hist = []
    v_hist = []
    t_hist = []
    T_hist = []
    append_sample = partial(append_sample_f,
                            c_hist,w_hist,v_hist,
                            t_hist,T_hist)

    # 2. Generate First Event
    K = 0
    while K == 0:
        K = npr.poisson(alpha0) # should we require K > 0?
    c = np.ones(K)
    w = np.zeros( (K,M.L) )
    v = np.zeros( (K,len(M.S)) )
    for i in range(K):
        w[i,:] = npr.dirichlet(pi.w0)
        v[i,:] = npr.dirichlet(pi.v0)
    kappa = Kappa(w,c)
    time = PoissonProcess(lambda0,'c').sample_tau(1)[0]
    T = sample_text(M.D,v,kappa)
    append_sample(c,w,v,time,T)

    # 3. Generate Followup Events
    for i in range(1,N):
        print(time,t_hist)
        c,K,Kp = sample_c(kappa,time,t_hist,lambda0)
        w,v = sample_wv(c,w,v,K,Kp,pi,M)
        kappa.update(w,c)
        hold_time = sample_time(kappa,t_hist,K)
        time = hold_time + t_hist[-1]
        T = sample_text(M.D,v,kappa)
        append_sample(c,w,v,time,T)

    return c_hist,w_hist,v_hist,t_hist,T_hist

def ibhp_intensity(c,w,v,t):
    pass
