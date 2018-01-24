# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:41:19 2017

@author: jie
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

    
# str is use to point the line, np is all of the data
def npstr_to_np(data):
    lists = []
    for i in range(data.shape[0]):
        lists.extend(np.array(data[i].split()))
    lists = list(dedupe(lists))
    result = np.array(lists)
    return result

def res_matrix(Numline, data_str, data_np):
    result = np.zeros([Numline, data_np.shape[0]])
    data_list = list(data_np)
    for i in range(Numline):
        split = data_str[i].split()
        for j in range(len(split)):
            result[i][data_list.index(split[j])] = 1
    return result

def one_hot(Numline, data_str, data_np):
    data_onehot = res_matrix(Numline=Numline, data_str=data_str, data_np=data_np)
    return data_onehot

def compute_label(norm, train, test, train_labels, k):
    line_test = test.shape[0]
    label = np.zeros([test.shape[0], train_labels.shape[1]])
    for i in range(line_test):
        if norm == 2:
            distance = np.sqrt(np.sum((np.square((train - test[i]))), axis = 1))
        if norm == 1:
            distance = np.sum((np.abs(train - test[i])), axis = 1)
        sortdistance = distance.argsort()
        for j in range(k):
#            print(train_labels[sortdistance[j]])
            label[i] += np.divide(train_labels[sortdistance[j]]+0.0001, distance[sortdistance[j]]+0.0001)
    label = np.divide(label, np.sum(label, axis = 1, keepdims = True))
    return label
        
        
def compute_correlation(data1, data2):
    correlation = 0
    for i in range(data1.shape[0]):
        correlation_matrix = np.corrcoef(data1[i], data2[i])
        correlation += correlation_matrix[0][1]
    correlation /= data1.shape[0]
    return correlation

def naive_bayes(train, tlabels, vlabels, smooth = "epislon"):
    p_labels = np.mean(tlabels, axis = 0)
    print(tlabels.T.shape, train.shape)
    p_labels_values = np.dot(tlabels.T, train)
    print(p_labels_values.shape)
    if smooth == "theory":
        smooth_values = p_labels_values.copy()
        smooth_values[smooth_values != 0] = 1
        smooth_values = np.sum(smooth_values, axis = 1)
        p_labels_values = np.divide(p_labels_values + 1, smooth_values.reshape(-1, 1) \
                                    + np.sum(p_labels_values, axis = 1, keepdims = True))
    elif smooth == "epislon":
        p_labels_values = np.divide(p_labels_values + 0.000001, np.sum(p_labels_values, axis = 1, keepdims = True))
    else:
        p_labels_values = np.divide(p_labels_values, np.sum(p_labels_values, axis = 1, keepdims = True))
    
    mat_labels = np.dot(valid, p_labels_values.T) * p_labels
    correlation = compute_correlation(mat_labels, vlabels)
    return correlation 
    
if __name__ == '__main__':
    
    columns = ['anger','disgust','fear','joy','sad','surprise']
    
    df1 = pd.read_csv('../DATA/regression_dataset/train_set.csv')
    line_train = len(df1)
    train_str = np.array(df1['Words (split by space)'])
    tlabels_str = np.array(df1[columns])

    
    df2 = pd.read_csv('../DATA/regression_dataset/test_set.csv')
    line_test = len(df2)
    test_str = np.array(df2['Words (split by space)'])
    
    
    df3 = pd.read_csv('../DATA/regression_dataset/validation_set.csv')
    line_valid = len(df3)
    valid_str = np.array(df3['Words (split by space)'])
    vlabels_str = np.array(df3[columns])
    
    data_str = np.concatenate((train_str, test_str, valid_str), axis = 0)
    data_np = npstr_to_np(data_str)

    train = one_hot(line_train, train_str, data_np)
    tlabels = tlabels_str
    test = one_hot(line_test, test_str, data_np)
    valid = one_hot(line_valid, valid_str, data_np)
    vlabels = vlabels_str
    
# 不同平滑方式
#    correlation_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "n")
#    print("correlation without smooth: ", correlation_nb)
    
#    correlation_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "theory")
#    print("correlation with theory smooth: ", correlation_nb)
    
    correlation_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "epislon")
    print("correlation with epsilon smooth:", correlation_nb)

#  range中设置范围
#    import time
#    accur = []
#    begin_time = time.time()
#    for i in range(100):
#        label_mat = compute_label(2, train, valid, tlabels, i+1)
#        correlation = compute_correlation(label_mat, vlabels)
#        accur.append(correlation)
##        print(np.argmax(vlabels, axis = 1))
##        labels = np.argmax(label_mat, axis = 1)
##        accuracy = np.sum(((labels) == np.argmax(vlabels, axis = 1))) / labels.shape[0]
##        distance = (np.sum((np.square((label_mat - vlabels)))))
#        print(i+1, correlation)
#    middle_time = time.time()
#    print(middle_time - begin_time)
#    plt.plot(range(100), accur, 'ro')
