# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:37:54 2018

@author: jie
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class NMF(object):
    def __init__(self, n_community = 9, n_embedding = 99, max_iterations = 10000, 
                 alpha = 5, beta = 5, lambd = 0.8):
        self.n_community = n_community
        self.n_embedding = n_embedding
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
    
    def calc_similiar(self, data):
        matrix_s1 = data
        matrix_s2 = np.zeros_like(matrix_s1)
        for i in range(matrix_s2.shape[0]):
            for j in range(matrix_s2.shape[1]):
                numerator = np.dot(matrix_s1[i], matrix_s1[j])
                denominator = np.linalg.norm(matrix_s1[i], ord = 2) * np.linalg.norm(matrix_s1[j], ord = 2)
                matrix_s2[i][j] = numerator / denominator
        S = matrix_s1 + 5 * matrix_s2
        return S
                
    def calc_matrix_b(self, data):
        vector_k = np.sum(data, axis = 1)
        scale_2e = np.sum(vector_k)
        matrix_b = data - np.outer(vector_k, vector_k) / scale_2e
        return matrix_b
        
    def init_matrix_h(self, data):
        matrix_h = np.zeros((data.shape[0], self.n_community))
        zipped = zip(range(matrix_h.shape[0]), 
                           np.random.randint(0, matrix_h.shape[1], matrix_h.shape[0]))
        for tuples in zipped:
            matrix_h[tuples] = 1
        return matrix_h
        
    def fit(self, data):
        matrix_m = np.random.random([ data.shape[0], self.n_embedding])
        matrix_u = np.random.random([data.shape[0], self.n_embedding])
        matrix_c = np.random.random([self.n_community, self.n_embedding])
        matrix_h = self.init_matrix_h(data)
        matrix_b = self.calc_matrix_b(data)
        matrix_s = self.calc_similiar(data)
        matrix_delta = np.zeros_like(matrix_h)

        
        epsilon = 1e-5
        for iters in range(self.max_iterations):
            matrix_m = np.multiply(matrix_m, np.divide(np.dot(matrix_s, matrix_u),
                                                  np.dot(matrix_m, 
                                                         np.dot(matrix_u.T, matrix_u)) + epsilon))
            
            matrix_u = np.multiply(matrix_u, np.divide((np.dot(matrix_s.T, matrix_m) + self.alpha * np.dot(matrix_h, matrix_c)), 
                                                  np.dot(matrix_u, (np.dot(matrix_m.T, matrix_m) + 
                                                                    self.alpha * np.dot(matrix_c.T, matrix_c))) + epsilon))
            
            matrix_c = np.multiply(matrix_c, np.divide(np.dot(matrix_h.T, matrix_u), 
                                                  np.dot(matrix_c, np.dot(matrix_u.T, matrix_u)) + epsilon))
            
            numerator_h = 2 * self.beta * np.dot(matrix_b, matrix_h)
            denominator_h = 8 * self.lambd * np.dot(matrix_h, np.dot(matrix_h.T, matrix_h))
            matrix_delta = numerator_h + denominator_h
            matrix_h = np.multiply(matrix_h, np.sqrt(np.divide((-numerator_h + matrix_delta), denominator_h + epsilon)))
        
        return matrix_u
    
    

def main():
    pass

def read_data(file = "../input/washington/washington_adj.txt", sep = '\t', describe = False):
    df1 = pd.read_csv(file, sep = sep, header = None)
    if describe == True:
        print("------------------------------------------------")
        print(df1.describe())
        print("-------------------------------------------------")
    return np.array(df1)


if __name__ == "__main__":
    root_path = "../input/"
    nmi_list = []
    label_list = os.listdir(root_path)
    for folder in os.listdir(root_path):
        file_path = os.path.join(root_path, folder)
        print("\nfile is: \t", folder)
        
        labels = read_data(file = file_path + "/" + folder + "_label.txt").flatten()
        data1 = read_data(file = file_path + "/" + folder + "_adj.txt")
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.figure()
        for i, alpha in enumerate([0.1, 0.5, 1, 5, 10]):
            acc_list = []
            for belta in [0, 2, 4, 6, 8, 10]:
                embed = NMF(max_iterations=5)
                embedding = embed.fit(data1, )
                X_train, X_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.2, random_state = 666)
        #        clf = SVC()
        #        clf = LogisticRegression()
                clf = KNeighborsClassifier(n_neighbors=10)
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                accuracy = clf.score(X_test, y_test)
                acc_list.append(clf.score(X_test, y_test))
            plt.plot(range(len(acc_list)), acc_list, color = color[i], label = "alpha = {}".format(alpha))
        plt.legend(loc='upper right')
        plt.xlabel("belta")
        plt.ylabel("accuracy")
        plt.show()
                
                    
        