# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 01:07:59 2017

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
    
data = read_data('data/16qam_snr13.csv')
X, _ = shuffle_sample(data, 0.9996)

#X = read_data()
N = X.shape[0]
M = X.shape[1]

# parameter
k = 16
delta = 0.01
max_iter = 5000

# Initial
"""
随机选择第一个中心点。
    计算所有样本到中心点的距离的最小值。[N, 1]
    对样本值进行求和。[1,1]
    在[0, sum]之间对样本值进行抽样。
    计算这个样本落在和对应的哪个位置，则选择那个位置作为作为下一个中心点。
    直到最后取到K个中心点。
"""
center = np.zeros((k, M))
index = np.random.randint(0, N)
center[0] = X[index]
for choice in range(1, k):
    distance = np.zeros((choice, N))
    for i in range(choice):
#        print(center.shape, i)
        distance[i] = np.sqrt(np.sum(np.square(X - center[i]), axis = 1))
        
    min_value = np.min(distance, axis = 0)
    sum_value = np.sum(min_value)
    sample_value = np.random.randint(0, sum_value)
    
    for i in range(min_value.shape[0]):
        sample_value -= min_value[i]
        if sample_value <= 0:
            index = i
            break
        
    center[choice] = X[index]
    choice += 1

print("\n\nfinish initial, the center is ")
plt.plot(center[:, 0], center[:, 1], 'ro')
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
        print("\n\nk-means++ didn't change after iters: ", iters)
        break
    

# test in 2D data
color = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
for i in range(k):
    data_index = np.where(divide == i)
    class_data = X[data_index]
    plt.plot(class_data[:, 0], class_data[:, 1], color[i % len(color)])
    
plt.show()