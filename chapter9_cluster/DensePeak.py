# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:47:27 2018

@author: jie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances


t = 0.02

#
#class DensePeak(object):
#    def __init__():
 

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
       
def read_data(filename = '../input/testSet.csv'):
    df1 = pd.read_csv(filename, sep = '\t')
    return np.array(df1)    
    
def calculate_distance(matrix , metric = "euclidean"):
    if metric == "euclidean":
        return pairwise_distances(matrix)
    else:
        pass


def calculate_cutoff(distance_matrix, cut_ratio = 0.02):
    sort_vector = np.sort(distance_matrix, axis = None)
    index = 2 * int(sort_vector.shape[0] * cut_ratio) + distance_matrix.shape[0]
    
    if index > sort_vector.shape[0]:
        index = sort_vector.shape[0] - 1
    elif index < 0:
        index = 0
        
    return sort_vector[index]
        

def calculate_rho(distance_matrix, cutoff_distance, kernel = "gaussian"):
    if kernel == "gaussian":
        dis_exp = np.exp(-np.sqrt(np.divide(distance_matrix, cutoff_distance)))
        return np.sum(dis_exp, axis = 1)
        
    else:
        pass
    
def calculate_delta(rho_vector, distance_matrix):
    max_distance = np.max(distance_matrix)
    rho_argsort = np.argsort(-rho_vector)

    delta_vector = np.zeros_like(rho_vector)
    for i in range(rho_vector.shape[0]):
        index = rho_argsort[i]
        delta_vector[index] = max_distance
        
        for j in range(i):
            if distance_matrix[i, j] < delta_vector[index]:
                delta_vector[index] = distance_matrix[i, j]

    return delta_vector
    
def choice_center(rho_vector, delta_vector, method = "ratio", 
                  ratio = 0.01, n_centers = None):
    
    gamma_vector = np.multiply(rho_vector, delta_vector)
#    gamma_vector = np.multiply(normalize(rho_vector.reshape(-1, 1)), 
#                               normalize(delta_vector.reshape(-1, 1))).flatten()
    
    print("gamma", gamma_vector.shape)
    if method == "ratio":
        n_centers = int(np.ceil(ratio * rho_vector.shape[0]))
        center_index = np.argsort(-gamma_vector)[0:n_centers]
        
    elif method == "fixed":
        center_index = np.argsort(-gamma_vector)[0:n_centers]

    else:
        pass
    return center_index

        
#==============================================================================
# TODO: @hejujie divide class by different layer
#==============================================================================
def divide_data(data, center_index):
    print(center_index.shape[0], data.shape[0])
    distance = np.zeros((center_index.shape[0], data.shape[0]))
    
    # the same with k-means
    for i in range(center_index.shape[0]):
        distance[i] = np.sum(np.square(data - data[center_index[i]]), axis = 1)
    
#    print(distance)
    divide_index = np.argmin(distance, axis = 0)
    
    return divide_index
            
            
        
def main():
    data = read_data("../input/Jain_cluster=2.txt")
#    data = [[1,2], [3,4], [5, 6]]
    a = (pairwise_distances(data))
    
    cut = (calculate_cutoff(a))
    
    rho = (calculate_rho(a, cut))
    
#    print(rho)
    
    delta = calculate_delta(rho, a)
    
#    plt.plot(rho, delta, 'o')
#    plt.show()
#    print(delta)

    center_index = choice_center(rho, delta, method='fixed', n_centers=2)
    print(center_index.shape)
#    print(center_index)
    
#    plt.ylim(0, 1000)
    print(data[center_index].shape)
    plt.plot(data[center_index][:, 0], data[center_index][:, 1], 'ro')
    
    plt.show()
    
    a = divide_data(data, center_index)
    
#    print(a)
    plot(data, a)
    
if __name__ == "__main__":
    main()