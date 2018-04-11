# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:31:10 2018

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel


class SpectralClustering(object):
    
    
    def __init__(self, n_clusters, n_reductions, 
                 metric = 'rbf_kernel', models = 'ncut'):
        self.n_clusters = n_clusters
        self.n_reductions = n_reductions
        self.metric = metric
        self.models = models
        
    def vector_vector_distance(self, vector_i, vector_j):
        if self.metric == "euclidean":
            return np.linalg.norm(vector_i - vector_j)
        if self.metric == "rbf_kernel":
            pass
        
    def vector_matrix_distance(self, vector, matrix):
        diff_matrix = matrix - vector
        if self.metric == "euclidean":
            return np.linalg.norm(diff_matrix, axis = 1)
        
    def matrix_matrix_similiar(self, matrix):
        if self.metric == "euclidean":
            n = matrix.shape[0]
            distance_matrix = np.zeros((n, n))
            # how to both make use of the attribute of symmetry matrix and the parallel
            for i in range(n):
                distance_matrix[i, :] = self.vector_matrix_distance(matrix[i], matrix)
            return distance_matrix
            
        elif self.metric == 'rbf_kernel':
            return rbf_kernel(matrix)

    def compute_degree_matrix(self, matrix):
        # symmetry matrix, so the axis doesn't matter
        diag_value = np.sum(matrix, axis = 1)
        return np.diag(diag_value)

    def compute_laplace_matrix(self, similiar_matrix, degree_matrix):
        if self.models == 'ncut':
            degree_negsqrt = np.power(np.linalg.matrix_power(degree_matrix, -1), 0.5)
            res_temp = np.dot(np.dot(degree_negsqrt, similiar_matrix), degree_negsqrt)
            return np.identity(degree_negsqrt.shape[0]) - res_temp
            
        else:
            return degree_matrix - similiar_matrix
   
    def compute_k_eigenvector(self, laplace_matrix):
        eigen_values, eigen_vectors = np.linalg.eig(laplace_matrix)
        choice_index = np.argsort(eigen_values)[0:self.n_reductions]
        feature_data = eigen_vectors[:, choice_index]
        return feature_data
    
    def fit(self, data):
        similiar_matrix = self.matrix_matrix_similiar(data)
        print(similiar_matrix.shape)
        
        degree_matrix = self.compute_degree_matrix(similiar_matrix)
        
        laplace_matrix = self.compute_laplace_matrix(similiar_matrix, degree_matrix)
        
        feature_data = normalize(self.compute_k_eigenvector(laplace_matrix))
        
        kmeans = KMeans(n_clusters = self.n_clusters).fit(feature_data)
        return kmeans.labels_


    
    
