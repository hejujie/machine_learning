# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:04:59 2017

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(filename = 'data/testSet.csv'):
    df1 = pd.read_csv(filename)
    return np.array(df1)
    
def shuffle_sample(data, ratio):
    '''
    Input: data to be divide, and the ratio of dev data
    Output: train data and dev data
    '''
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
    random.seed(10)
    index_valid = random.sample(range(population), int(ratio * population))
    index_train = np.delete(index_all, index_valid, axis = 0)
    return data[index_train], data[index_valid]
    
#data = read_data('data/16qam_snr13.csv')
#X, _ = shuffle_sample(data, 0.99)

X = read_data()
N = X.shape[0]

# parameter
k = 4
delta = 0.01
max_iter = 5000

# Initial
initial_index = np.array(random.sample(range(N), k))
center = X[initial_index]
plt.plot(center[:, 0], center[:, 1])
plt.show()


# K-means
for iters in range(max_iter):
    # compute distance
    distance = np.zeros((k, N))
    for i in range(k):
        distance[i] = np.sqrt(np.sum(np.square(X - center[i]), axis = 1))
        
    # divide class
    divide = np.argmin(distance, axis = 0)    
    
    # compute center
    change_flag = 0
    for i in range(k):
        data_index = np.where(divide == i)
        center_value = np.mean(X[data_index], axis = 0)  
        if np.all(center_value != center[i]):
            change_flag = 1
            center[i] = center_value
    
    if change_flag == 0:
        print("k-means didn't change after iters: ", iters)
        break
    
    iters += 1
    

# test in 2D data
color = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
for i in range(k):
    data_index = np.where(divide == i)
    class_data = X[data_index]
    plt.plot(class_data[:, 0], class_data[:, 1], color[i % len(color)])
    

