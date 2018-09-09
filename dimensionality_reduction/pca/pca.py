# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:03:48 2017

@author: jie
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import leastsq 

def read_data(filename = 'data/testSet.csv'):
    df1 = pd.read_csv(filename, header = None)
    return np.array(df1)

def pca(X, d):
    mean_value = np.mean(X, axis = 0)
    X_fix = X - mean_value
    cov_matrix = np.cov(X_fix.T)
    eig_value, eig_vector = np.linalg.eig(cov_matrix)

    # default , with negative, descent
    sort_index = np.argsort(-eig_value)
    choose_index = sort_index[0:d]

    eig_value = eig_value[choose_index]
    eig_vector = eig_vector[:, choose_index]
    print(eig_vector.shape)

    X_project = np.dot(X_fix, eig_vector.T)
    
    return X_project

def plot(X, X_project):
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.plot(X[:, 1], X_project[:, 0], 'ro')

    
if __name__ == "__main__":
    X = read_data('data/testSet.csv')
    d = 1
    X_project = pca(X, d)
    
    if d == 1:
        plot(X, X_project)
    
