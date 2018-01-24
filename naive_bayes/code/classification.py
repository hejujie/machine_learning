# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:41:19 2017

@author: jie
"""


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math 

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
    counter = Counter(lists)
    lists = list(dedupe(lists))
    result = np.array(lists)
    return result, counter

def tran_matrix(Numline, data_str, data_np, counter, types):
    result = np.zeros([Numline, data_np.shape[0]])
    data_list = list(data_np)
    for i in range(Numline):
        split = data_str[i].split()
        if types == 'onehot':
            for j in range(len(split)):
                result[i][data_list.index(split[j])] = 1
        elif types == 'tfidf':
            for j in range(len(split)):
                result[i][data_list.index(split[j])] += (1/len(split)) * \
                np.log2((Numline/(np.array(min(Numline, counter.get(split[j]))))))
    return result
    
    

def compute_dis(norm, train, test, train_labels, k):
    line_test = test.shape[0]
    label_mat = np.zeros([test.shape[0], train_labels.shape[1]])
    for i in range(line_test):
        if norm == 2:
            distance = np.sqrt(np.sum((np.square((train - test[i]))), axis = 1))
        if norm == 1:
            distance = np.sum((np.abs(train - test[i])), axis = 1)
        sortdistance = distance.argsort()
        res = 1
        decay = 1
        for j in range(k):
            label_mat[i] += train_labels[sortdistance[j]] * res
            res = res * decay
    return label_mat
        
    
    
def naive_bayes(train, tlabels, vlabels, smooth = True, alpha = 1):
    p_labels = (np.mean(tlabels, axis = 0))
    p_labels_values = np.dot(tlabels.T, train)
#    p_labels_values = np.log(p_labels_values)
    if smooth == "theory":
        smooth_values = p_labels_values.copy()
        smooth_values[smooth_values != 0] = 1
        smooth_values = alpha * np.sum(smooth_values, axis = 1)
        p_labels_values = np.divide(p_labels_values + alpha, smooth_values.reshape(-1, 1) \
                                    + np.sum(p_labels_values, axis = 1, keepdims = True))
    elif smooth == "epislon":
        p_labels_values = np.divide(p_labels_values + 0.000001, np.sum(p_labels_values, axis = 1, keepdims = True))
    else:
        p_labels_values = np.divide(p_labels_values, np.sum(p_labels_values, axis = 1, keepdims = True))
    
#    p_labels_values = np.log(p_labels_values)
    mat_labels = np.dot(valid, p_labels_values.T) * p_labels
    label_pred = np.argmax(mat_labels, axis = 1)
    
    accuracy = np.sum((label_pred == np.argmax(vlabels, axis = 1))) / label_pred.shape[0]
    return accuracy 

if __name__ == '__main__':
    
    df1 = pd.read_csv('../DATA/classification_dataset/train_set.csv')
#    df1 = add_data(df1, 1, 2, 1, 0, 1, 1)
    line_train = len(df1)
    train_str = np.array(df1['Words (split by space)'])
    tlabels_str = np.array(df1['label'])
    
    
    df2 = pd.read_csv('../DATA/classification_dataset/test_set.csv')
    line_test = len(df2)
    test_str = np.array(df2['Words (split by space)'])
    
    
    df3 = pd.read_csv('../DATA/classification_dataset/validation_set.csv')
    line_valid = len(df3)
    valid_str = np.array(df3['Words (split by space)'])
    vlabels_str = np.array(df3['label'])
    
    data_str = np.concatenate((train_str,test_str, valid_str, tlabels_str), axis = 0)
    data_np, counter = npstr_to_np(data_str)
    train_np, _ = npstr_to_np(train_str)
    label_np, _ = npstr_to_np(tlabels_str)

    train = tran_matrix(line_train, train_str, data_np, counter, 'tfidf')
    tlabels = tran_matrix(line_train, tlabels_str, label_np, counter, 'onehot')
    test = tran_matrix(line_test, test_str, data_np, counter, 'tfidf')
    valid = tran_matrix(line_valid, valid_str, data_np, counter, 'tfidf')
    vlabels = tran_matrix(line_valid, vlabels_str, label_np, counter, 'onehot')

# 不同平滑    
    accuracy_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "none")
    print("accuracy without smooth: ", accuracy_nb)
    
    accuracy_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "epislon")
    print("accuracy with epsilon smooth:", accuracy_nb)

    accuracy_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "theory")
    print("alpha = 1", "accuracy with theory smooth: ", accuracy_nb)
    
    accuracy_nb = naive_bayes(train = train, tlabels = tlabels, vlabels = vlabels, smooth = "theory", alpha=0.05)
    print("alpha = 0.05", "accuracy with theory smooth: ", accuracy_nb)
    
## sklearn结果对比
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    tlabels_tran = np.sum(tlabels * np.array([1, 2, 3, 4, 5, 6]), axis = 1)
    vlabels_tran = np.sum(vlabels * np.array([1, 2, 3, 4, 5, 6]), axis = 1)
    clf.fit(train, tlabels_tran)
    print("accuracy with sklearn_MultinomialNB:", clf.score(valid, vlabels_tran))
    
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(train, tlabels_tran)
    print("accuracy with sklearn_Gaussian:", clf.score(valid, vlabels_tran))
    
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(train, tlabels_tran)
    print("accuracy with sklearn_BernoulliNB:", clf.score(valid, vlabels_tran))
    
    
# range中设置范围
#    import time
#    accur = []
#    begin_time = time.time()
#    for i in range(1):
##        label_mat = compute_dis(2, train, valid, tlabels, i+ 1)
#        label_mat = compute_dis(2, train, valid, tlabels, 1)
#        labels = np.argmax(label_mat, axis = 1)
#        print(labels)
#        accuracy = np.sum((labels == np.argmax(vlabels, axis = 1))) / labels.shape[0]
#        print(i+1, accuracy)
#        accur.append(accuracy)
#    middle_time = time.time()
#    print(middle_time - begin_time)
#    plt.plot(range(100), accur, 'ro')