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

import networkx as nx

def complete_c(c):
    N = len(c)
    Kmax = len(c[N-1])
    cc = np.zeros( (N,Kmax) )
    for i,ci in enumerate(c):
        ci = np.array(ci,dtype=np.int)
        print(ci)
        print(np.bincount(ci))
        print(ci.nonzero()[0])
        cc[i,ci.nonzero()[0]] = ci.nonzero()[0]+1
    return cc

def view_ib(c,w,v,t,T):

    # see the evolution of topics

    N = len(c)
    Kmax = len(c[N-1])
    print("Kmax: {}".format(Kmax))
    x = []
    y = []
    for i in range(N):
        nzi = c[i].nonzero()[0]
        for k_value in nzi:
            x.append(t[i])
            y.append(k_value)
    ax = plt.subplot(1,1,1)
    ax.plot(x,y,'o')
    ax.set_title("Latent IB Topics")
    plt.show()

    # inspect topic content
    ntopics = 3 # we can't see all the topic info
    
