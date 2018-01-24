# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:22:53 2017

@author: jie
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
#from sklearn import cross_validation,metrics
        
def read_data(filename):
    df1 = pd.read_csv(filename, sep = ' ', header = None)
    return np.array(df1)
    
def get_label(data):
    return data.T[data.shape[1]-1].reshape(-1, 1)

def get_X(data):
    argu_X = np.ones(data.shape[0]).reshape(1, -1)
    primer_X = data.T[0:data.shape[1]-1]
    argu_X = np.concatenate((argu_X, primer_X))
    return argu_X.T

def normalize(data):
    max1 = np.max(data, axis = 0)
    min1 = np.min(data, axis = 0)
    return (data - min1) / (max1 - min1)


      
    
def logistic_regression(X, Y, valid_X, valid_Y, learning_rate = 1, belt1 = 0.9, decay_rate = 0.99, 
                        belt2 = 0.99, reg_num = 0.01, batch_size = 128, opt = 'grad', max_iteration = 30000):
    W = np.zeros((X.shape[1]))
    
    
    '''
    Parameter for Adam
    '''
    mot = np.random.random((W.shape))
    vec = np.random.random((W.shape))
#    mot = np.zeros((W.shape))
#    vec = np.zeros((W.shape))
    eps = 1e-8
    
    '''
    Parameter for RMSprop
    '''
    cache_rms = np.random.random((W.shape))
#    cache_rms = np.zeros((W.shape))

    
    '''
    Parameter for Adagrad
    '''
    cache_ada = np.random.random((W.shape))
#    cache_ada = np.zeros((W.shape))

    accuracy_list = []
    accuracy_max = 0

    for i in range(max_iteration):
        if batch_size == 'all':
            X_batch = X
            Y_batch = Y
                    
        else:
            index_batch = random.sample(range(X.shape[0]), int(batch_size))
            X_batch = X[index_batch]
            Y_batch = Y[index_batch]
        
        predict = np.dot(X_batch, W).reshape(-1, 1) 
        gradient = np.sum((((1 / (1 + np.exp(-predict))) - Y_batch) * \
                            X_batch + reg_num * W), axis = 0)

        if opt == 'grad':
            W = W - learning_rate * gradient
            
        elif opt == 'momentum':
            vec = belt1 * vec - learning_rate * gradient
            W = W + vec
            
        elif opt == 'nesterov':
            vec_prev = vec
            vec = belt1 * vec - learning_rate * gradient
            W = W - belt1 * vec_prev + (1 + belt1) * vec

        elif opt == 'adagrad':
            cache_ada += (gradient ** 2)
            W = W - learning_rate * gradient / (np.sqrt(cache_ada) + eps)
            
        elif opt == 'rmsprop':
            cache_rms = decay_rate * cache_rms + (1 - decay_rate) * (gradient**2)
            W = W - learning_rate * gradient / (np.sqrt(cache_rms) + eps)
    
        elif opt == 'adam':
            mot = belt1 * mot + (1-belt1) * gradient
            vec = belt2 * vec + (1-belt2) * (gradient**2)
            W = W - learning_rate * mot / (np.sqrt(vec) + eps)
#                    if i % 1000 == 0:
##            learning_rate = learning_rate * 0.9
##            print(learning_rate)
##            accuracy = test_logistic(W, valid_X, valid_Y)
#            print(accuracy)
#            if accuracy > accuracy_max:
#                accuracy_max = accuracy
#                W_best = W
#            print(accuracy)
        accuracy_list.append(np.sum(gradient))


#            if accuracy > 0.77:
#                break
         
    return W, accuracy_list
     
def test_logistic(W, valid_X, valid_Y):
    predict_valid = np.dot(valid_X, W).reshape(-1, 1)
    predict_label = np.sign(1/(1+np.exp(-predict_valid)) - 0.5)
    predict_label[predict_label < 0] = 0
    accuracy = (predict_label == valid_Y)
    accuracy = np.sum(accuracy) / valid_X.shape[0]
    return accuracy
    
def test_label(W, test_X):
    predict_valid = np.dot(test_X, W).reshape(-1, 1)
    predict_label = np.sign(1/(1+np.exp(-predict_valid)) - 0.5)
    predict_label[predict_label < 0] = 0
    return predict_label

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

def data_from_index(data, index):
    '''
    Input: data to be divide, and the index of dev data
    Output: train data and dev data
    '''
    index_all = np.array(range(data.shape[0]))
    index_valid = index
    index_train = np.delete(index_all, index_valid, axis = 0)
    return data[index_train], data[index_valid] 
    
    
if __name__ == "__main__":
    
    # get train data
    file_train = '../test/train.csv'
    train = read_data(file_train)
    train_size = train.shape[0]
#    print(train.shape)
    # get test data
    file_test = '../test/test.csv'
    test = read_data(file_test)
    test[:, -1] = 1.0
    test_size = test.shape[0]

    # the same normalize
#    test[:, -1] = 1.0
#    all_data = np.vstack((train, test)).astype('float64')
#    all_data = normalize(all_data)
#    
#    train = all_data[0: train_size]
#    test = all_data[-test_size-1: -1]
#    print(type(train))
#
#    test_X = get_X(test)
#    print(test_X.shape)
    
#    sample_train, sample_valid = shuffle_sample(train, ratio = 0.3)
    train_X = get_X(train).astype('float64')
#    valid_X = get_X(sample_valid)
    tlabels = get_label(train).astype('float64')
#    vlabels = get_label(sample_valid)
    test_X = get_X(test).astype('float64')


#    k = 5
#    fold_data_num = int(train.shape[0] / k)
#    random.seed(10)
#    shuffle_index = random.sample(range(train.shape[0]), int(train.shape[0]))
#    for i in range(k):
#        valid_index = np.array(shuffle_index[i*fold_data_num: (i+1)*fold_data_num-1])
#        sample_train, sample_valid = data_from_index(train, valid_index)
#        train_X = get_X(sample_train)
#        valid_X = get_X(sample_valid)
#        tlabels = get_label(sample_train)
#        vlabels = get_label(sample_valid)
#
#        opt = 'grad'
#        begin_time = time.time()
#        reg_num = 0
#        W, accuracy_lists = logistic_regression(train_X, tlabels, valid_X, vlabels, opt = opt,learning_rate=1,
#                                               batch_size = 'all', reg_num = 0.01, max_iteration = 1)
#        if i == 0:
#            W_sum = W
#        else:
#            W_sum += W
#        plt.figure()
#        plt.plot(range(len(accuracy_lists)), accuracy_lists, 'b')
#        end_time = time.time()
#
#                
#    W = W_sum / k
#    train_X = get_X(train)
#    tlabels = get_label(train)
#    accuracy_total = test_logistic(W, train_X, tlabels)
#    print("the accuracy is", accuracy_total)

    opt = 'grad'
    W, accuracy_lists = logistic_regression(train_X, tlabels, 1, 1, opt = opt,learning_rate=1,
                                               batch_size = 'all', reg_num = 0, max_iteration = 1)
    print(W)
    label = test_label(W, test_X)
    output = pd.DataFrame(label)
    print(output)
#    output.to_csv('../output/test_output.txt', index = False, header = None)