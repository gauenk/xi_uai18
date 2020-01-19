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
from processes import CRP
import matplotlib.pyplot as plt


if __name__ == "__main__":

    crp = CRP()
    T = 10000
    for i in range(T):
        sample = crp.sample(1)
    print(crp.n,crp.c,crp.sizes)
    print(np.bincount(crp.c))
    # plt.plot(crp.sizes)
    plt.plot(crp.c,'+')
    x = np.linspace(1.,T)
    y = crp.alpha * np_loge(x)
    plt.plot(x,y)
    plt.show()
    print("HI")
