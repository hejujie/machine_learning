# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:02:05 2018

@author: jie
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class LINE(object):
    def __init__(self, n_embedding, epislon = 0.1, max_iteration = 1000, 
                 alpha_1 = 10, alpha_2 = 10):
        self.n_embedding = n_embedding
        self.max_iterations = max_iteration
        self.epislon = epislon
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        
    def calc_laplace_matrix(self, matrix):
        matrix_s = np.dot(matrix, matrix.T)
        matrix_d = np.diag(np.sum(matrix_s, axis = 1))
        matrix_d_neg_sqrt = np.power(matrix_d, -1/2)
        matrix_d_neg_sqrt[np.isinf(matrix_d_neg_sqrt)] = 0
#        matrix_d_neg_sqrt = np.power(np.linalg.matrix_power(matrix_d, -1), 0.5)

        matrix_laplace = np.dot(matrix_d_neg_sqrt, 
                                  np.dot(matrix_s, matrix_d_neg_sqrt))
        
        return matrix_laplace
        
    def get_k_eigen_vector(self, eigen_value, eigen_vector, k):
        index_choice = np.argsort(-eigen_value)[0:k]
        return eigen_vector[:, index_choice]
        
    def solve_formula(self, variable, matrix_laplace,
                matrix_u_a, matrix_u_g, matrix_u_y, matrix_h):
        

        if variable == 'g':
            matrix_sum = (matrix_laplace + 
                                  self.alpha_1 * np.dot(matrix_u_a, matrix_u_a.T) + 
                                  self.alpha_2 * np.dot(matrix_u_y, matrix_u_y.T) + 
                                                 np.dot(matrix_h, matrix_h.T))
            
        elif variable == 'a':
            matrix_sum = (self.alpha_1 * matrix_laplace + 
                          self.alpha_1 * np.dot(matrix_u_g, matrix_u_g.T) + 
                                         np.dot(matrix_u_y, matrix_u_y.T) + 
                                         np.dot(matrix_h, matrix_h.T))
            
        elif variable == 'y':
            matrix_sum = (self.alpha_2 * matrix_laplace + 
                          self.alpha_2 * np.dot(matrix_u_g, matrix_u_g.T) + 
                                         np.dot(matrix_u_y, matrix_u_y.T) + 
                                         np.dot(matrix_h, matrix_h.T))
            
        elif variable == 'h':
            matrix_sum = (np.dot(matrix_u_g, matrix_u_g.T) +
                          np.dot(matrix_u_a, matrix_u_a.T) + 
                          np.dot(matrix_u_y, matrix_u_y.T))
        else:
            print("Variable error")
            
        eigen_value, eigen_vector = np.linalg.eig(matrix_sum)
        return self.get_k_eigen_vector(eigen_value, eigen_vector, self.n_embedding)
        
    
        
    def fit(self, data, attribute, labels):
        laplace_g = self.calc_laplace_matrix(data)
        laplace_a = self.calc_laplace_matrix(attribute)
        laplace_y = self.calc_laplace_matrix(labels)
        
        matrix_u_g = np.random.random([data.shape[0], self.n_embedding])
        matrix_u_a = np.random.random([data.shape[0], self.n_embedding])
        matrix_u_y = np.random.random([data.shape[0], self.n_embedding])
        matrix_h = np.random.random([data.shape[0], self.n_embedding])

        for i in range(self.max_iterations):
            matrix_u_g = self.solve_formula('g', laplace_g, matrix_u_a, matrix_u_g, matrix_u_y, matrix_h)
            matrix_u_a = self.solve_formula('a', laplace_a, matrix_u_a, matrix_u_g, matrix_u_y, matrix_h)
            matrix_u_y = self.solve_formula('y', laplace_y, matrix_u_a, matrix_u_g, matrix_u_y, matrix_h)
            matrix_h = self.solve_formula('h', None, matrix_u_a, matrix_u_g, matrix_u_y, matrix_h)
        
        return matrix_h
        
    
def main():
    pass


def shuffle_sample(data, ratio, seed = None):
    '''
    Input: data to be divide, and the ratio of dev data
    Output: train data and dev data
    '''
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
    random.seed(10)
    index_valid = random.sample(range(population), int(ratio * population))
    index_train = np.delete(index_all, index_valid, axis = 0)
    return index_train, index_valid

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
    for k, folder in enumerate(os.listdir(root_path)):
        file_path = os.path.join(root_path, folder)
        print("\nfile is: \t", folder)
        
        labels_init = read_data(file = file_path + "/" + folder + "_label.txt").flatten()
        data1 = read_data(file = file_path + "/" + folder + "_adj.txt")
        attrib1 = read_data(file = file_path + "/" + folder + "_feature.txt")

        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.figure()
        
        labels1 = labels_init
        
        labels1 = labels1.reshape(labels1.shape[0], -1)
        enc = OneHotEncoder()
        enc.fit(labels1)
        labels_onehot = enc.transform(labels1).toarray()
        
        (G_1, G_2, attr_train, attr_test, y_train, y_test, 
         labels_onehot_train, labels_onehot_test) = train_test_split(data1, attrib1, labels_init, labels_onehot, test_size=0.2, random_state = 666)
    
        X_train = G_1[0:G_1.shape[0], 0:G_1.shape[0]]
        X_test = G_2[0:G_2.shape[0], 0:G_2.shape[0]]
        

        plt.figure()
        for i, alpha_1 in enumerate([0.01, 0.5, 2, 5, 5, 10, 20]):
            acc_list = []
            for alpha_2 in [0.01, 0.1, 0.5, 2, 5, 10, 20]:
                embed = LINE(n_embedding=100, epislon=0.01, max_iteration=500, alpha_1 = alpha_1, alpha_2 = alpha_2)
                embedding_train = embed.fit(X_train, attr_train, labels_onehot_train)
        
                clf = KNeighborsClassifier(10)
                clf.fit(embedding_train, y_train)
                
                eta = 0.5
                embedding_test = (np.dot(G_2, np.linalg.pinv(np.dot(np.linalg.pinv(embedding_train), G_1))) + 
                                  eta * np.dot(attr_test, np.linalg.pinv(np.dot(np.linalg.pinv(embedding_train), attr_train))))
                predict = clf.predict(embedding_test)
                
                accuracy = clf.score(embedding_test, y_test)
                acc_list.append(accuracy)
            plt.plot([0.01, 0.1, 0.5, 2, 5, 10, 20], acc_list, color = color[i], label = "alpha_1 = {}".format(alpha_1))
        plt.legend(loc='upper right')
        plt.xlabel("alpha_2")
        plt.ylabel("accuracy")
        plt.show()