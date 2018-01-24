# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:21:25 2018

@author: jie
"""

from lstm import LstmParameter, LstmNetwork
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def softmax(score):
    max_score = np.max(score, axis = 0, keepdims = True)
    score -= max_score
    score_exp = np.exp(score)
    poss = score_exp / np.sum(score_exp, axis = 0, keepdims = True)
    return poss

def predict_label(pred, parameter):
    value = np.dot(parameter.wr, pred) + parameter.br
    output = softmax(value)
    predict = np.argmax(output)
    return predict

# must have @classmethod
class ToyLossLayer:
    @classmethod
    def __init__(self, y_dim = 3):
        self.output = np.zeros((y_dim))
        self.output_loss = 0
        self.mark
    
    @classmethod
    def loss(self, pred, label, parameter):
        value = np.dot(parameter.wr, pred) + parameter.br
        self.output = softmax(value)
#        print(self.output)
        self.mask = np.zeros_like(self.output)
        self.mask[label] = 1
        self.output_loss = -(np.dot(np.log(self.output).T, self.mask))
        return self.output_loss
    
    @classmethod
    def bottom_diff(self, pred, label, parameter):
        parameter.wr_diff += np.dot((self.output - self.mask).reshape(-1, 1), pred.reshape(1, -1))
        parameter.br_diff += (self.output - self.mask)
        diff = np.dot(self.output - self.mask, parameter.wr)
        return diff
        
if __name__ == "__main__":
    np.random.seed(0)
    x_dim = 300
    cell_count = 50
    y_dim = 3


    
    df1 = pd.read_csv("../../input/lstm/simple_train.csv", header = None, nrows = 9000)
    label = df1.pop(0)
    label = np.array(label)
    for i, x in enumerate(['LOW', 'MID', 'HIG']):
        label[np.where(label == x)] = i
    data = np.array(df1)
    data_size = int(data.shape[0] * 0.7)
    valid_size = int(data.shape[0] * 0.3)
    print(data_size)
    


    lstm_parameter = LstmParameter(cell_count, x_dim, y_dim)
    lstm_network = LstmNetwork(lstm_parameter)

    loss_list = []
    batch_size = 128
    accuracy_best = 0.4
    for iters in range(1000000): 
        ind = random.randint(0, data_size-1)
        train_ind = data[ind].reshape((cell_count, x_dim))  
        label_ind = label[ind]

#        print("iters is" , iters)
        for index in range(cell_count):
            lstm_network.forward_compute(train_ind[index])
        
        if iters % batch_size == 0:
            if iters == 0:
                loss  = lstm_network.backward_compute(label_ind, ToyLossLayer)
            else:
                loss = loss / batch_size
#                print("predict is", lstm_network.lstm_cell_list[cell_count-1].state.h.shape)
#                print("iter is {} loss is {}".format(iters, loss))
                loss_list.append(loss)
                lstm_parameter.update_parameter(lr = 0.3, batch_size = batch_size)
            loss = lstm_network.backward_compute(label_ind, ToyLossLayer)
             
        else:
            loss_temp = lstm_network.backward_compute(label_ind, ToyLossLayer)
            loss += loss_temp
        
        
        if iters % (batch_size * 100) == 0 and iters > 0:
            if iters % (batch_size * 200) == 0:
                plt.plot(range(len(loss_list)), loss_list, 'r')
                plt.show()
            accuracy = 0
#            for j in range(0, data_size):
            for j in range(data_size, data_size + valid_size):
                train_ind = data[j].reshape((cell_count, x_dim))  
                label_ind = label[j]
        #        print("iters is" , iters)
                for index in range(cell_count):
                    lstm_network.forward_compute(train_ind[index])  
                    
                predict = predict_label(lstm_network.lstm_cell_list[len(lstm_network.x_list) - 1].state.h, lstm_parameter)
                accuracy += (predict == label_ind)
            print("accurary in valid", accuracy / valid_size)
            if accuracy - accuracy_best > 0.05:
                accuracy = 0.95 * accuracy
                accuracy_best = accuracy

        lstm_network.x_list_clear()

