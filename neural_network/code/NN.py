import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt

# eps=1e-8, beta1=0.9, beta2=0.999

def network_initial(architect, initial_func = 'he_init'):
    '''
    Input: the number of node in every layer
    Output: a dictionary which every element is {parameter_name, parameter}, 
            which begin from W1, ie: {W1: Weight_Of_W1, W2: Weight_Of_W2}
    '''
    np.random.seed(10)
    
    parameter = {}
    adam_para = {}
    layer_len = architect.shape[0]

    if initial_func == 'zeros_init':
        for i in range(1, layer_len):
            parameter['W' + str(i)] = (np.zeros((architect[i], architect[i-1])))
            parameter['b' + str(i)] = (np.zeros((architect[i], 1)))

    if initial_func == 'random_init':
        for i in range(1, layer_len):
            parameter['W' + str(i)] = (np.random.random((architect[i], architect[i-1]))) * 0.1
            parameter['b' + str(i)] = (np.random.random((architect[i], 1))) * 0.1

    if initial_func == 'he_init':
        for i in range(1, layer_len):
            parameter['W' + str(i)] = (np.random.random((architect[i], architect[i-1]))) * np.sqrt(2 / architect[i-1]) * 0.1
            parameter['b' + str(i)] = (np.random.random((architect[i], 1))) * np.sqrt(2 / architect[i-1]) * 0.1
    
    for i in range(1, layer_len):
        adam_para['MW' + str(i)] = (np.zeros((architect[i], architect[i-1])))
        adam_para['Mb' + str(i)] = (np.zeros((architect[i], 1)))
        adam_para['VW' + str(i)] = (np.zeros((architect[i], architect[i-1])))
        adam_para['Vb' + str(i)] = (np.zeros((architect[i], 1)))
                      
    return parameter, adam_para

def linear_forward(A, W, b):
     ''' 
     Input:  A is the value of node in this layer.
             W, b is the parameter of this layer
     Output: Z——the value of W*A + b
     '''
     Z = np.dot(W, A) + b
     return Z
     
     
def activate_forward(Z, activation):
    '''
    Input: the value compute by linear_forard function, and the type of activation funciton
    Output: the output after go through activation funcion
    '''
    if activation == 'sigmoid':
        A = np.divide(1, (1 + np.exp(-Z.astype("float64"))))
        
    elif activation == 'relu':
        A = np.maximum(0, Z)
    return A
    
def forward_propagation(X, parameter, activation = 'relu', dropout = 0.7):
    '''
    Input: X is the 
    '''
    cache = {}
    cache['A0'] = X
    A_prev = X
    L = len(parameter) // 2

    for i in range(1, L):
        Z = linear_forward(A_prev, parameter.get('W' + str(i)), parameter.get('b' + str(i)))
        A = activate_forward(Z, activation)
        # batch_normalize
#        A = normalize(A)
        
        # dropout
        mask = np.random.random((A.shape))
        mask = (mask < dropout)
        A = np.multiply(A, mask) / dropout
        cache['Z' + str(i)] = Z
        cache['A' + str(i)] = A
        cache['M' + str(i)] = mask
        A_prev = A

    Z = linear_forward(A_prev, parameter.get('W' + str(L)), parameter.get('b' + str(L)))
    A = Z
    cache['Z' + str(L)] = Z
    cache['A' + str(L)] = A
              
    return A, cache
    
           
def compute_cost(Predict, Y, function = 'cross_entropy'):
    ''' 
    Input: 
        Predict: the compute value of output layer
        Y: the value of the true label 
        Function: different type of cost function
    
    Output:
        the value of cost function
    '''
    Y = Y.reshape((Predict.shape))
    if function == 'cross_entropy':
        cost = np.mean(np.multiply(Y, np.log(Predict)) + np.multiply((1-Y), np.log(1-Predict)), axis = 0)
        
    elif function == 'square_error':
        cost = (np.mean(np.square(Predict - Y)))
    return cost
    
    
def activation_backward(dA, Z, activation = 'relu'):
    if activation == 'relu':
#        relu_value = np.maximum(0, Z)
#        relu_value[relu_value > 0] = 1
#        dZ = np.multiply(dA, relu_value)
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0 
        
    elif activation == 'sigmoid':
        sigmoid_value = np.divide(1, (1+np.exp(-Z.astype("float64"))))
        dZ = np.multiply(dA, (sigmoid_value * (1-sigmoid_value)))
    return dZ
    
def linear_backward(dZ, A_minus, W, reg_num):
    m = A_minus.shape[1]
    dW = (np.dot(dZ, A_minus.T) + reg_num * W) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_minus = np.dot(W.T, dZ)
    return dW, db, dA_minus
    
def backward_propagation(parameter, cache, Y, reg_num = 0.1, function = 'square_error', activation = 'relu', dropout = 0.7):
    
    grad = {}
    L = len(parameter) // 2
#    print(L)
    AL = cache.get('A' + str(L))  
    if function == 'square_error':
        grad['dA' + str(L)] = (AL - Y) 
    
    if function == 'cross_entropy':
        grad['dA' + str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
    grad['dZ' + str(L)] = grad.get('dA' + str(L)) 
    grad['dW' + str(L)], grad['db' + str(L)], grad['dA' + str(L-1)] = \
         linear_backward(grad.get('dZ' + str(L)), cache.get('A' + str(L-1)), parameter.get('W' + str(L)), reg_num = reg_num)

    for i in reversed(range(1, L)):
        grad['dZ' + str(i)] = activation_backward(grad.get('dA' + str(i)),
                                 cache.get('Z' + str(i)), activation = activation)
        grad['dW' + str(i)], grad['db' + str(i)], grad['dA' + str(i-1)] = \
             linear_backward(grad.get('dZ' + str(i)), cache.get('A' + str(i-1)), parameter.get('W' + str(i)), reg_num = reg_num)
        
        # dropout 
        if i != 1:
            grad['dA' + str(i-1)] = np.multiply(grad['dA' + str(i-1)], cache['M' + str(i-1)]) / dropout            
             
    return grad
             
def update_parameter(adam_para, parameter, grad, learning_rate, update_function = 'gradient'):
 #eps=1e-8, beta1=0.9, beta2=0.999

    L = len(parameter) // 2
    if update_function == 'gradient':
        for i in range(1, L+1):
            parameter['W' + str(i)] = parameter['W' + str(i)] - learning_rate * (grad.get('dW' + str(i)))
            parameter['b' + str(i)] = parameter['b' + str(i)] - learning_rate * (grad.get('db' + str(i)))
    elif update_function == 'adam':
        for i in range(1, L+1):
            adam_para['MW' + str(i)] = 0.9 * adam_para['MW' + str(i)] + \
                      (1 - 0.9) * (grad.get('dW' + str(i)))
            adam_para['Mb' + str(i)] = 0.9 * adam_para['Mb' + str(i)] + \
                      (1 - 0.9) * (grad.get('db' + str(i)))
            adam_para['VW' + str(i)] = 0.99 * adam_para['VW' + str(i)] + \
                      (1 - 0.99) * np.square((grad.get('dW' + str(i))))
            adam_para['Vb' + str(i)] = 0.99 * adam_para['Vb' + str(i)] + \
                      (1 - 0.99) * np.square((grad.get('db' + str(i))))
            parameter['W' + str(i)] = parameter['W' + str(i)] - learning_rate * \
                      adam_para['MW' + str(i)] / (np.sqrt(adam_para['VW' + str(i)]) + 1e-8)
            parameter['b' + str(i)] = parameter['b' + str(i)] - learning_rate * \
                      adam_para['Mb' + str(i)] / (np.sqrt(adam_para['Vb' + str(i)]) + 1e-8)
                      
    return parameter, adam_para
    
    
def read_data(filename):
    df1 = pd.read_csv(filename)
    return np.array(df1)

def shuffle_sample(data, ratio):
    '''
    Input: data to be divide, and the ratio of dev data
    Output: train data and dev data
    '''
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
#    random.seed(10)
    index_valid = random.sample(range(population), int(ratio * population))
    index_train = np.delete(index_all, index_valid, axis = 0)
    return data[index_train], data[index_valid] 
  
def normalize(data):
    if data.shape[0] < 2:
        return data
    std1 = np.std(data, axis = 0)
    mean1 = np.mean(data, axis = 0)
    return (data - mean1) / std1

if __name__ == "__main__":
#    from sklearn import datasets
#    iris=datasets.load_iris()
#    X_train = np.array(iris.data).T
#    Y_train = np.array(iris.target)


    file_train = '../input/train1new.csv'
    train = read_data(file_train)
    
#    sample_train = train[0:-1500]
#    sample_valid = train[-1500:-1]
    sample_train, sample_valid = shuffle_sample(train, ratio = 0.3)
#    sample_train = train[0:-715]
#    sample_valid = train[-715:-1]
#    sample_valid, _ = shuffle_sample(sample_valid, ratio = 0.95)
    print(sample_train.shape, sample_valid.shape)
    
    X_train = sample_train[:, 0:-1]
    X_train = X_train.T
    Y_train = sample_train[:, -1]

    X_valid = sample_valid[:, 0:-1]
    X_valid = X_valid.T
    Y_valid = sample_valid[:, -1]
    
    print(X_train.shape)
    batch_size = 1280
    iteration = 50000
    learning_rate = 0.001
    init_func = 'he_init'
    reg_num = 0.1
    dropout = 0.85
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_num = 0
#    for batch_size in [256, 2048, 0.8 * X_train.shape[1],'all']:
#    for batch_size in ['all']:
    for func in ['adam']:
        print(func)
        cost_list = []
        valid_list = []
        parameter, adam_para = network_initial(np.array([X_train.shape[0], 40, 30, 1]), initial_func = init_func)
        for i in range(iteration):
            
            if batch_size == "all":
                sample_X = X_train
                sample_Y = Y_train
            else:
                index_batch = random.sample(range(X_train.shape[1]), int(batch_size))
                sample_X = X_train[:, index_batch]
                sample_Y = Y_train[index_batch]
    
            output, cache = forward_propagation(sample_X, parameter, activation='relu', dropout = dropout)
            grads = backward_propagation(parameter, cache, sample_Y, activation='relu', reg_num = reg_num, dropout = dropout)
            parameter, adam_para = update_parameter(adam_para, parameter, grads, learning_rate = learning_rate, update_function = func)
            
            if i % 100 == 0 and i > 0:
                cost = compute_cost(output, sample_Y, function = 'square_error')
                cost_list.append(cost)
                output_valid, cache = forward_propagation(X_valid, parameter, activation='relu', dropout = 1)
                cost_valid = compute_cost(output_valid, Y_valid, function = 'square_error')
                valid_list.append(cost_valid)
                print(i,':\t', cost, ':\t', cost_valid)
        
#        plt.plot(range(len(cost_list)), np.array(cost_valid)-np.array(cost_list), color[color_num],
#                 label = "dropout_prob is:{}".format(dropout), linewidth = 2)
        plt.plot(range(len(cost_list)), cost_list, color[color_num],
                 label = func, linewidth = 2)
        plt.plot(range(len(valid_list)), valid_list, color[color_num+1],
                 label = "valid_set", linewidth = 2)
            
        color_num += 1
    plt.xlabel("different network result after iteration : {} ".format(iteration))
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
