# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:21:25 2018

@author: jie
"""

import numpy as np

def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))

def sigmoid_derivative(sigmoid_value):
    return np.multiply(sigmoid_value, (1- sigmoid_value))

def tanh_derivative(tanh_value):
    return (1 - tanh_value ** 2)
    
class LstmParameter:
    def __init__(self, cell_count, x_dim, y_dim, factor=0.1):
        # the dimension of x + h, as the dimension of h is the dimension of cell
        concat_len = x_dim + cell_count

        self.cell_count = cell_count
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        self.wf = np.random.uniform(-1, 1, (cell_count, concat_len)) * factor
        self.wi = np.random.uniform(-1, 1, (cell_count, concat_len)) * factor
        self.wg = np.random.uniform(-1, 1, (cell_count, concat_len)) * factor
        self.wo = np.random.uniform(-1, 1, (cell_count, concat_len)) * factor
        self.wr = np.random.uniform(-1, 1, (y_dim, cell_count)) * factor

        self.bf = np.random.uniform(-1, 1, (cell_count)) * factor
        self.bi = np.random.uniform(-1, 1, (cell_count)) * factor
        self.bg = np.random.uniform(-1, 1, (cell_count)) * factor
        self.bo = np.random.uniform(-1, 1, (cell_count)) * factor
        self.br = np.random.uniform(-1, 1, (y_dim))    

        self.wf_diff = np.zeros((cell_count, concat_len))
        self.wi_diff = np.zeros((cell_count, concat_len))
        self.wg_diff = np.zeros((cell_count, concat_len))
        self.wo_diff = np.zeros((cell_count, concat_len))
        self.wr_diff = np.zeros((y_dim, cell_count))
        self.bf_diff = np.zeros((cell_count))
        self.bi_diff = np.zeros((cell_count))
        self.bg_diff = np.zeros((cell_count))
        self.bo_diff = np.zeros((cell_count))
        self.br_diff = np.zeros((y_dim))
            
        
    def update_parameter(self, lr = 0.01, batch_size = 1):
        self.wf -= lr * self.wf_diff / batch_size
        self.wi -= lr * self.wi_diff / batch_size
        self.wg -= lr * self.wg_diff / batch_size
        self.wo -= lr * self.wo_diff / batch_size
        self.wr -= lr * self.wr_diff / batch_size
        self.bg -= lr * self.bg_diff / batch_size
        self.bi -= lr * self.bi_diff / batch_size
        self.bf -= lr * self.bf_diff / batch_size
        self.bo -= lr * self.bo_diff / batch_size
        self.br -= lr * self.br_diff / batch_size
        
        self.wf_diff = np.zeros_like(self.wf)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wg_diff = np.zeros_like(self.wg) 
        self.wo_diff = np.zeros_like(self.wo)
        self.wr_diff = np.zeros_like(self.wr)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 
        self.br_diff = np.zeros_like(self.br)
        
        
class LstmState:
    def __init__(self, cell_count, x_dim):
        self.f = np.zeros(cell_count)
        self.i = np.zeros(cell_count)
        self.g = np.zeros(cell_count)
        self.o = np.zeros(cell_count)
        self.s = np.zeros(cell_count)
        self.h = np.zeros(cell_count)
        self.button_diff_h = np.zeros_like(self.h)
        self.button_diff_s = np.zeros_like(self.s)
        
        
class LstmCell:
    def __init__(self, lstm_parameter, lstm_state):
        self.state = lstm_state
        self.parameter = lstm_parameter
        self.xc = None
        
    def forward_prop(self, x, s_prev=None, h_prev=None):
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        
        self.s_prev = s_prev
        self.h_prev = h_prev
        
        xc = np.hstack((x, h_prev))
        self.xc = xc
        
        self.state.f = sigmoid(np.dot(self.parameter.wf, xc) + self.parameter.bf)
        self.state.i = sigmoid(np.dot(self.parameter.wi, xc) + self.parameter.bi)
        self.state.o = sigmoid(np.dot(self.parameter.wo, xc) + self.parameter.bo)
        self.state.g = np.tanh(np.dot(self.parameter.wg, xc) + self.parameter.bg)
        self.state.s = np.multiply(self.state.g, self.state.i) + np.multiply(s_prev, self.state.f)
        self.state.h = np.multiply(self.state.s, self.state.o)
        
    def backward_prop(self, back_ward_h, back_ward_s):
        ds = np.multiply(self.state.o, back_ward_h) + back_ward_s
        do = np.multiply(self.state.s, back_ward_h)
        di = np.multiply(self.state.g, ds)
        dg = np.multiply(self.state.i, ds)
        df = np.multiply(self.s_prev, ds)
        
        di_input = np.multiply(sigmoid_derivative(self.state.i), di)
        df_input = np.multiply(sigmoid_derivative(self.state.f), df)
        do_input = np.multiply(sigmoid_derivative(self.state.o), do)
        dg_input = np.multiply(tanh_derivative(self.state.g), dg)
        
        self.parameter.wi_diff += np.outer(di_input, self.xc)
        self.parameter.wf_diff += np.outer(df_input, self.xc)
        self.parameter.wo_diff += np.outer(do_input, self.xc)
        self.parameter.wg_diff += np.outer(dg_input, self.xc)
        self.parameter.bi_diff += di_input
        self.parameter.bf_diff += df_input       
        self.parameter.bo_diff += do_input
        self.parameter.bg_diff += dg_input    
        
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.parameter.wi.T, di_input)
        dxc += np.dot(self.parameter.wf.T, df_input)
        dxc += np.dot(self.parameter.wo.T, do_input)
        dxc += np.dot(self.parameter.wg.T, dg_input)
        
        self.state.buttom_diff_h = dxc[self.parameter.x_dim:]
        self.state.buttom_diff_s = np.multiply(ds, self.state.f)
        

class LstmNetwork:
    
    
    def __init__(self, lstm_parameter):
        self.lstm_parameter = lstm_parameter
        self.lstm_cell_list = []
        self.x_list = []

    def forward_compute(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_cell_list):
            lstm_state = LstmState(self.lstm_parameter.cell_count, self.lstm_parameter.x_dim)
            self.lstm_cell_list.append(LstmCell(self.lstm_parameter, lstm_state))
            
        idx = len(self.x_list) - 1

        if idx == 0:
            self.lstm_cell_list[idx].forward_prop(x)
            
        else:
            s_prev = self.lstm_cell_list[idx - 1].state.s
            h_prev = self.lstm_cell_list[idx - 1].state.h
            self.lstm_cell_list[idx].forward_prop(x, s_prev, h_prev)
        
        
    def backward_compute(self, label, loss_layer):
        
#        assert(len(y_list) == len(self.x_list))
        idx = len(self.x_list) - 1
        
        loss = loss_layer.loss(self.lstm_cell_list[idx].state.h, label, self.lstm_parameter)
        diff_h = loss_layer.bottom_diff(self.lstm_cell_list[idx].state.h, label, self.lstm_parameter)
        diff_s = np.zeros(self.lstm_parameter.cell_count)
        self.lstm_cell_list[idx].backward_prop(diff_h, diff_s)
        idx -= 1
        
#==============================================================================
#         TODO: determine how to BP with softmax
#==============================================================================
        while idx >= 0:
#            loss += loss_layer.loss(self.lstm_cell_list[idx].state.h, label, self.lstm_parameter)
#            diff_h = loss_layer.bottom_diff(self.lstm_cell_list[idx].state.h, y_list[idx], self.lstm_parameter)
#            diff_h = np.zeros_like(diff_h)
            diff_h = self.lstm_cell_list[idx + 1].state.buttom_diff_h
            diff_s = self.lstm_cell_list[idx + 1].state.buttom_diff_s
            self.lstm_cell_list[idx].backward_prop(diff_h, diff_s)
            idx -= 1
            
#        print(loss_layer.output)
#        label = np.argmax(loss_layer.output)
#        return loss, label
        return loss
        

    def x_list_clear(self):
        self.x_list = []

    


