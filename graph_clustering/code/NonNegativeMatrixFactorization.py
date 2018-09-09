# -*- coding: utf-8 -*-
"""
Created on Fri May  4 00:01:13 2018

@author: jie
"""

import numpy as np
from sklearn.preprocessing import Normalizer

class NonNegativeMatrixFactorization(object):
    def __init__(self, n_iterations = 1000, n_clusters = 5):
        self.n_iterations = n_iterations
        self.n_clusters = n_clusters
        
    def update_matrix(self, A):
        U = np.random.normal(size = (A.shape[0], self.n_clusters))
        V = np.random.normal(size = (A.shape[0], self.n_clusters))
        for i in range(self.n_iterations):
            numerator_U = np.multiply(U, np.dot(A, V))
            denominator_U = np.dot(U, np.dot(np.transpose(V), V)) + 1e-5
            U = np.divide(numerator_U, denominator_U)
            
            numerator_V = np.multiply(V, np.dot(np.transpose(A), U))
            denominator_V = np.dot(V, np.dot(np.transpose(U), U)) + 1e-5
            V = np.divide(numerator_V, denominator_V)
            
#            Normalizer(U)
#            Normalizer(V)
            
        return U, V
        
    def fit(self, data):
        U, V = self.update_matrix(data)
        labels = np.argmax(V, axis = 1)
        return labels
            



