import pickle,os
from os import path as osp
import _init_paths
import pandas as pd
import numpy as np
import numpy.random as npr
import mnist_loader
from distributions import WeibullDistribution as Weibull
from processes import PoissonProcess
import matplotlib.pyplot as plt

def const_rate(tau):
    return 100*tau
def const_hazard(tau):
    return npr.uniform(0,tau)
def weibull_rate(tau,shape=1,scale=1):
    mean = (tau/scale)**shape
    return const*tau
def weibull_hazard(tau,shape=1,scale=1):
    rv = Weibull({'shape':shape,'scale':scale},
                 is_hazard_rate=True)
    return rv


def verify_conditional_weibull():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as sss
    p = Weibull({'shape':1.0,'scale':1.0})
    regular_samples = p.sample(1000)
    conditional_samples_A = np.array(p.sample(1000,hold_time=.5))
    conditional_samples_B = np.array(p.sample(1000,hold_time=10))
    sns.distplot(regular_samples,hist=True,rug=False,\
                 label='P( \tau )').set(xlim=(0))
    sns.distplot(conditional_samples_A,hist=True,rug=False,\
                 label='P( \tau | \tau > 0.5)').set(xlim=(0))
    sns.distplot(conditional_samples_B,hist=True,rug=False,\
                 label='P( \tau | \tau > 10)').set(xlim=(0,20))

    ks_result = sss.ks_2samp(conditional_samples_A - .5,regular_samples)
    print(ks_result)
    ks_result = sss.ks_2samp(conditional_samples_B - 10,regular_samples)
    print(ks_result)
    ks_result = sss.ks_2samp(conditional_samples_B,regular_samples)
    print(ks_result)
    plt.show()

def compute_td_info(times):
    if len(times) == 0:
        return 0,0
    td_diffs = [times[0]]
    for i in range(len(times)-1):
        td_diffs.append(times[i+1] - times[i])
    return np.mean(td_diffs),np.std(td_diffs)

def verify_const_poisson_process(ntrials,tau_list):
    results = {'tau':[],'times':[],'n':[],'td_mean':[],'td_std':[]}
    for tau in tau_list:
        for idx in range(ntrials):
            p = PoissonProcess(const_rate,const_hazard)
            times = p.sample(tau)
            results['n'].append(len(times))
            results['tau'].append(tau)
            results['times'].append(times)
            td_mean,td_std = compute_td_info(times)
            results['td_mean'].append(td_mean)
            results['td_std'].append(td_std)

    return results

def generate_hawkes_process():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as sss

    tau = 3
    p = PoissonProcess(const_rate,const_hazard)
    points = p.sample(tau)
    print(points)
    sns.distplot(points,hist=True,rug=False,\
                 label='sim1').set(xlim=(0,tau))

    # mean = const_rate(tau)
    # N = npr.poisson( lam = mean )
    # shapes = np.repeat(const_rate(1),N)
    # points = sss.erlang(shapes).rvs()
    # print(points)
    # sns.distplot(points,hist=True,rug=False,\
    #              label='sim2').set(xlim=(0))

    plt.show()

if __name__ == "__main__":

    ntrials = 300
    tau_list = [1,2,3,4,5,6,7,10,50]
    save_fn = "./output/const_pp_check.pkl"
    if not osp.exists(save_fn):
        with open(save_fn,'rb') as f:
            results = pickle.load(f)
    else:
        print("ASDF")
        results = verify_const_poisson_process(ntrials,tau_list)
        with open(save_fn,'wb') as f:
            pickle.dump(results,f)
    
    
    data = pd.DataFrame(results)
    print(data)
    print(data.groupby(['tau']).mean())
    exit()
    print(results['tau'])
    fig,ax = plt.subplots(1,2)
    ax[0].plot(results['tau'],results['td_mean'])
    ax[1].plot(results['tau'],results['td_std'])
    plt.show()
    # generate_hawkes_process()
    # verify_conditional_weibull()
    
    # mnist = mnist_loader.mnist()
    # print(mnist.train_samples[0:10])

    # num_iters = 100
