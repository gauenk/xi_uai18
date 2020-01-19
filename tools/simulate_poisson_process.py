import pickle,os
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



if __name__ == "__main__":

    pp = PoissonProcess(3,'c')
    sample1 = pp.sample(100)
    sample2 = pp.sample(100)
    sample3 = pp.sample(100)
    sample4 = npr.uniform(0,100,300)
    print(len(sample1))
    # T = 10000
    # for i in range(T):
    #     sample = crp.sample(1)

    # print(np.bincount(crp.c))
    # # plt.plot(crp.sizes)
    fig,ax = plt.subplots(1,2)
    ax[0].set_title("Plotting (Number of Points) [xaxis] v.s. (Time) [yaxis]")
    ax[0].plot(sample1,'+')
    ax[0].plot(sample2,'+')
    ax[0].plot(sample3,'+')

    ax[1].set_title("Comparing with Uniform")
    ax[1].plot(sample1,sample1,'+')
    ax[1].plot(sample2,sample2,'+')
    ax[1].plot(sample3,sample3,'+')
    ax[1].plot(sample4,sample4,'^')
    # x = np.linspace(1.,T)
    # y = crp.alpha * np_loge(x)
    # plt.plot(x,y)
    plt.show()
    print("HI")
