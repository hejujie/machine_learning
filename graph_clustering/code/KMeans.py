# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:04:59 2017

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans(object):
    
    
    def __init__(self, n_clusters, max_iters = 5000, init_method = "KMeans++"):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method

    def init_center(self, data):
        centers = np.zeros((self.n_clusters, data.shape[1]))
        if self.init_method == "random":
            centers_index = np.array(random.sample(range(data.shape[0]), self.n_clusters))
            centers = data[centers_index]

        elif self.init_method == "KMeans++":
            index = np.random.randint(0, data.shape[0])
            centers[0] = data[index]

            for choice in range(1, self.n_clusters):
                distance = np.zeros((choice, data.shape[0]))
                for i in range(choice):
                    distance[i] = np.sqrt(np.sum(np.square(data - centers[i]), axis = 1))
                    
                min_value = np.min(distance, axis = 0)
                sum_value = np.sum(min_value)
                sample_value = np.random.randint(0, sum_value)
                
                for i in range(min_value.shape[0]):
                    sample_value -= min_value[i]
                    if sample_value <= 0:
                        index = i
                        break
                    
                centers[choice] = data[index]
                choice += 1
        return centers
        
    def divide_data(self, data, centers):
        distance = np.zeros((self.n_clusters, data.shape[0]))
        for i in range(self.n_clusters):
            distance[i] = np.sqrt(np.sum(np.square(data - centers[i]), axis = 1))
            
        divide = np.argmin(distance, axis = 0)    
        return divide
        
    def compute_centers(self, data, divide, center):
        change_flag = 0
        centers = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            data_index = np.where(divide == i)
            center_value = np.mean(data[data_index], axis = 0)  
            if np.all(center_value != centers[i]):
                change_flag = 1
                centers[i] = center_value
        return centers, change_flag
        
    def fit(self, data):
        centers = self.init_center(data)
        for i in range(self.max_iters):
            divide = self.divide_data(data, centers)
            centers, change_flag = self.compute_centers(data, divide, centers)
            if change_flag == 0:
                break
        return divide
    


    


