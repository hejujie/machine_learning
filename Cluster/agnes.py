# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:06:09 2017

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def read_data(filename = 'data/testSet.csv'):
    df1 = pd.read_csv(filename, header = None)
    return np.array(df1)
    
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

    
def distance_matrix_matrix(X1, X2):
    m1, m2 = X1.shape[0], X2.shape[0]
    distance = np.zeros((m1, m2))
    for i in range(m1):
        distance[i] = np.linalg.norm((X2 - X1[i]), ord = 2)
    return distance
    
def init_cluster(m):
    return np.array(range(m))
    
    
def expand_cluster(X, k, linkage = 'average'):
    
    m = X.shape[0]
    if k > m:
        print("Error happen because the cluster is larger than dataset")
        return 
        
    distance_matrix = np.zeros((m, m))
    
    divide_index = init_cluster(m)
    # all cluster only have one element
    for i in range(m):
        for j in range(i+1, m):
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    
    # if we have not this, the value of i will equal to j, 
    # then the shape of distance_matrix will smaller than clusters number
    max_distance = np.max(distance_matrix)
    for i in range(m):
        distance_matrix[i, i] = max_distance

    for i in range(m-k):
        # find the smallest distance, if have more than one, get the first one
        i, j = np.where(distance_matrix == np.min(distance_matrix))
        i, j = i[0], j[0]
        
        # change the cluster index and delete j in distance matrix
        index_cluster_j = np.where(divide_index == j)
        divide_index[index_cluster_j] = i
        divide_index[divide_index > j] -= 1
        distance_matrix = np.delete(distance_matrix, j, axis = 0)
        distance_matrix = np.delete(distance_matrix, j, axis = 1)
        
        index_cluster_i = np.where(divide_index == i)
        for j in range(distance_matrix.shape[0]):
            index_cluster_j = np.where(divide_index == j)
            distance = distance_matrix_matrix(X[index_cluster_i], X[index_cluster_j])
            if linkage == 'average':
                if j != i:
                    distance_matrix[i, j] = np.mean(distance)
                    distance_matrix[j, i] = distance_matrix[i, j]
    
    return divide_index
            
    
if __name__ == "__main__":
    X = read_data('data/testSet.csv')
    m = X.shape[0]
    for i in range(3, 10):
        divide_index = expand_cluster(X, i)
        print(" ")
        plt.title("cluster number is : {}".format(i))
        plot(X, divide_index)
    
    