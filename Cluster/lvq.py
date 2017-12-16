# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:12:16 2017

@author: jie
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(filename = 'data/lvq_test.csv'):
    df1 = pd.read_csv(filename)
    return np.array(df1)
    
def lvq(X, Y, q, max_iter = 100, eta = 0.1):
    """
    Input:  
        X: dataset
        Y: label
        q: number of vector
        max_iter: the iteration number
        eta: the learning rate
    Output: 
        X_vec: the vector after max_iteration
    """
    
    # Initial Vector
    n = X.shape[0]
    sample = random.sample(range(n), q)
    X_vec = X[sample]
    Y_vec = Y[sample]

    # Update Vector base on the label of random_choice data and vector
    # if label is the same, the closest vector_point will be colser to data, otherwise: farther
    for iters in range(max_iter):
        index = random.randint(0, n-1)
        
        distance = np.sqrt(np.sum(np.square(X_vec - X[index]), axis = 1))
        min_q = np.argmin(distance, axis = 0)
        
        if Y_vec[min_q] == Y[index]:
            X_vec[min_q] = X_vec[min_q] + eta * (X[index] - X_vec[min_q])
        else:
            X_vec[min_q] = X_vec[min_q] - eta * (X[index] - X_vec[min_q])
            
    return X_vec
    
def divide(X, X_vec):
    """
    Input:
        X: the primer data
        X_vec: the value of vector
    Output:
        divide_index: the index of data base on the closest distance to vector
    """
    n = X.shape[0]
    q = X_vec.shape[0]
    
    # distance is an [q, n] matrix, means the distance between every data point to vector point
    distance = np.zeros((q, n))
    for i in range(q):
        distance[i] = np.sqrt(np.sum(np.square(X - X_vec[i]), axis = 1))
    divide_index = np.argmin(distance, axis = 0)
    return divide_index
    
def plot(X, X_vec, divide_index):
    q = X_vec.shape[0]
    color = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
    for i in range(q):
        class_index = np.where(divide_index == i)
        class_data = X[class_index]
        plt.plot(class_data[:, 0], class_data[:, 1], color[i % len(color)])
    plt.plot(X_vec[:, 0], X_vec[:, 1], 'r', linewidth = 2)
    plt.show()

if __name__ == "__main__":
    filename = 'data/lvq_test.csv'
    data = read_data(filename = filename)
    X = data[:, 0:-1]
    Y = data[:, -1]
    for i in range(10000):
        print(i)
        X_vec = lvq(X, Y, q = 8, max_iter = 20, eta = 0.0001)
        divide_index = divide(X, X_vec)
        plot(X, X_vec, divide_index)
        