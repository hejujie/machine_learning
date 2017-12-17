# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:19:21 2017

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def read_data(filename = 'data/testSet.csv'):
    df1 = pd.read_csv(filename, header = None)
    return np.array(df1)
    
    
def init_core(X, eps, min_pts):
    m = X.shape[0]
    core_neibour = []
    core_list = []

    for i in range(m):
        distance = np.sqrt(np.sum(np.square(X - X[i]), axis = 1))
        eps_neibour_i = (distance < eps)        
        if np.sum(eps_neibour_i) > min_pts:
            core_list.append(i)
            core_neibour.append(set(np.where(eps_neibour_i != 0)[0].tolist()))

    return core_list, core_neibour
    
def expand_cluster(X, core_list, core_neibour):
    m = X.shape[0]
    k = 0
    divide_index = np.ones((m)) * -1
    unvisit = set(range(m))
    copy_core = core_list.copy()
    
    while len(copy_core) != 0:
        unvisit_old = unvisit.copy()
        
        i = random.randint(0, len(copy_core)-1)
        core = copy_core.pop(i)
        reach_set = set([core])
    
        while len(reach_set) != 0:
            q = reach_set.pop()
            
            if q in core_list:
                index = core_list.index(q)
                delta = unvisit.intersection(core_neibour[index])
                reach_set = reach_set.union(delta)
                unvisit = unvisit.difference(delta)
                
        classes_k = list(unvisit_old.difference(unvisit))
        divide_index[classes_k] = k
        copy_core = list(set(copy_core).difference(classes_k))
        k = k + 1
            
    return divide_index

def plot(X, divide_index):
    unique = np.unique(divide_index)
    color = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
    color_num = 0
    for i in unique:
        class_index = np.where(divide_index == i)
        class_data = X[class_index]
        plt.plot(class_data[:, 0], class_data[:, 1], color[color_num % len(color)])
        color_num += 1
    plt.show()    
    
if __name__ == "__main__":
    X = read_data('data/testSet.csv')
    # divide_index with value is outlier, it will be plot as bule color
    for i in range(23, 27):
        print(i)
        core_list, core_neibour = init_core(X, i * 0.1, 4)
        divide_index = expand_cluster(X, core_list, core_neibour)
        plot(X, divide_index)
        
    
    
    
    
    