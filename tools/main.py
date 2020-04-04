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
from ibhp import *
from ibhp_viz import *

def write_pickle_vocab(fn,D,S):
    data = {'D':D,'S':S}
    with open(fn,"wb") as f:
        pickle.dump(data,f)
    return 0

def read_pickle_vocab(fn):
    with open(fn,"rb") as f:
        data = pickle.load(f)
    D,S = data['D'],data['S']
    return D,S

def load_data():
    """
    Loads data in format:
      L :: number of basis kernels
      D :: length of document
      S :: vocab
    """
    #text = load_sb_data()
    # D,S = load_facebook_data()
    D,S = load_nips_data()
    return D,S

def load_sb_data():
    fn = "/home/kent/Documents/data/sb/TRN/SBC033.trn"
    text = pd.read_csv(fn)
    print(text)

def load_facebook_data():
    pass

def load_nips_data():
    cache = "/home/kent/Documents/data/nips_citations/parse_cache.pkl"
    if osp.exists(cache):
        D,S = read_pickle_vocab(cache)
        return D,S
    base = "/home/kent/Documents/data/nips_citations/"
    authors = pd.read_csv(osp.join(base,"authors.csv"))
    paper_authors = pd.read_csv(osp.join(base,"paper_authors.csv"))
    papers = pd.read_csv(osp.join(base,"papers.csv"))
    D = []
    S = []
    for paper in papers['paper_text']:
        s = extract_vocab(paper)
        S.extend(s)
        D.append(len(paper))
    S = list(np.unique(S))
    write_pickle_vocab(cache,D,S)
    return D,S

def extract_vocab(paper):
    paper_words = list(np.unique(re.sub("[^\w]", " ",  paper).lower().split()))
    for word in eng_stopwords:
        if word in paper_words:
            paper_words.remove(word)
    return paper_words


if __name__ == "__main__":

    # ? how to stop "zero event" events

    # experiment parameters
    N = 20
    alpha0 = 2 #?

    # 1. Initialize
    L = 3
    D,S = load_data()
    D = 1000
    M = edict({'L':L,'D':D,'S':S})
    w0 = npr.uniform(0,1,L)
    v0 = npr.uniform(0,1,len(S))
    pi = edict({'w0':w0,'v0':v0})
    lambda0 = 0.1 
    betas = npr.uniform(0,1,L)
    taus = npr.uniform(0,1,L)
    theta = edict({'lambda0':lambda0,
                   'betas':betas,
                   'taus':taus})
    Kappa.betas = betas
    Kappa.taus = taus
    c,w,v,t,T = ibhp_prior(N,M,pi,theta,alpha0)
    # view_ib(c,w,v,t,T)
    
    print("Done sampling generative text. Let's see what we have.")
    print(t)
    print(T)
