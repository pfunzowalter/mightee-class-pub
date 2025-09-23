import itertools
import numpy as np
import sys

def norm(x):
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    # x_trans =(x-x_min)/(x_max-x_min) #min-max normalisation
    x_trans = x/(x_max-x_min)
        
    return np.array(x_trans)

def norm_arr(lst):
    lst = np.array(lst)
    lst_trans = []
    for i in range(len(lst)):
        lst_trans.append(norm(lst[i]))
        
    return lst_trans


def feat_multiplication (x, par):
    x_scaled = []
    
    for i, j in zip(range(len(par)), range(len(x))):
        x_scaled.append(x[i] * par[j])
        
    return x_scaled

def generate_combinations(arrays):
    combinations = list(itertools.product(*arrays))
    return combinations

