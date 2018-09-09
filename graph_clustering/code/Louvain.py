# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:39:50 2018

@author: jie
"""

import numpy as np
import pandas as pd


def read_data(file = "../input/washington/washington_adj.txt", sep = '\t', describe = False):
    df1 = pd.read_csv(file, sep = sep, header = None)
    if describe == True:
        print("------------------------------------------------")
        print(df1.describe())
        print("-------------------------------------------------")
    return np.array(df1)


class Louvain(object):
    def __init__(self, n_iterations = 1000):
        self.n_iterations = n_iterations
        self.weight_sum = None
        
    def init_community(self, data):
        self.weight_sum = np.sum(data) / 2
        labels = np.arange(data.shape[0])
        return labels
        
        
    def calculate_modularity(self, data, labels):
        
        unique_labels = set(labels)
        sum_tot = np.zeros(data.shape[0])
        sum_in = np.zeros(data.shape[0])
        
        
        for i in unique_labels:
            sample_label = np.where(labels == i)[0]
            sum_tot[sample_label] = np.sum(data[sample_label])
            sum_in[sample_label] = np.sum(data[sample_label][:, sample_label])
        return sum_tot, sum_in
        
            
    def calculate_new_community(self, data, labels):
        sum_tot, sum_in = self.calculate_modularity(data, labels)
        unique_labels = set(labels)
        
        change_flag = 0
        new_labels = np.copy(labels)
        
        for i in unique_labels:
            delta_q_i = np.zeros(data.shape[0])
            sample_i = np.where(labels == i)[0]
            for j in unique_labels:
                sample_j = np.where(labels == j)[0]
                if i == j or np.sum(data[sample_i][:, sample_j]) == 0:
                    delta_q_i[j] = 0
                else:
                    k_i_in = np.array(np.sum(data[sample_i][:, sample_j])) * 2
                                             
                    add_modularity = ((sum_in[j] + k_i_in) / (2 * self.weight_sum) 
                                       - np.square(sum_tot[j] / (2*self.weight_sum)))
                    isoloate_modularity = (sum_in[j] / (2 * self.weight_sum) 
                                          - np.square(sum_tot[j] / (2 * self.weight_sum))
                                          - (sum_tot[i] / (2 * self.weight_sum)))
                    delta_q_i[j] = add_modularity - isoloate_modularity
            new_index_i = np.argmax(delta_q_i)
            if new_index_i != labels[sample_i][0]:
                change_flag = 1
                new_labels[sample_i] = new_index_i
            
        return new_labels, change_flag
        
    def fit(self, data):
        
        labels = self.init_community(data)
        
        for i in range(self.n_iterations):
            labels, change_flag = self.calculate_new_community(data, labels)
            if change_flag == 0:
                break
        return labels
        
    
        