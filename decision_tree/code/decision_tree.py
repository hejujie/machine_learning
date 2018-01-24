import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import build_tree
import random

#random.sample(population,k)  
#
#class node:
#    def __init__(self)

change_sum = 0

def information_entropy(label):
    class_count = np.array(pd.value_counts(label))
    class_prob = class_count / np.sum(class_count)
    entropy = -np.sum(class_prob * np.log2(class_prob))
    return entropy
    
def get_unique_label(data, attribute):
    attrubute_column = data[:, attribute]
    unique_value = pd.unique(pd.DataFrame(attrubute_column)[0])
    return attrubute_column, unique_value

    
def information_gain(data, attribute):
    entropy_gain = information_entropy(data[:, data.shape[1]-1])
    data_attribute, unique_attribute = get_unique_label(data, attribute)
    
    for i in unique_attribute:
        index = np.where(data_attribute == i)[0]
        prob = index.shape[0] / (data.shape[0])
        data_divide = data[index]    
        entropy_divide = information_entropy(data_divide[:, data_divide.shape[1]-1])
        entropy_gain -= prob * entropy_divide
    return entropy_gain
    
def information_gain_ratio(data, attribute):
    data_attribute = data[:, attribute]
    intrinsic_value = information_entropy(data_attribute)
    
    gain_value = information_gain(data, attribute)
    gain_ratio = gain_value / intrinsic_value
    return gain_ratio
 
def gini_value(label):
    class_count = np.array(pd.value_counts(label))
    class_prob = class_count / np.sum(class_count)
    gini_value = 1 - np.sum(np.square(class_prob))
    return gini_value

# 其实这个完全可以跟计算information统一起来。
def gini_index(data, attribute):
    data_attribute, unique_attribute = get_unique_label(data, attribute)
    gini_indexs = 0
    
    for i in unique_attribute:
        index = np.where(data_attribute == i)[0]
        prob = index.shape[0] / (data.shape[0])
        data_divide = data[index]    
        gini_values = gini_value(data_divide[:, data_divide.shape[1]-1])
        gini_indexs += prob * gini_values
    return -gini_indexs


def construct_tree(data, tree_node, from_value, depth, types = 'gini'):
    '''
        Input:
            data is divide by attribute (tree_node) with value (from_value)
        TO DO:
            mark the best choice of value
            the data is the best attribte of data divide from (from_value)
            
    '''
    depth += 1
    # check whether is pure:
    labels, unique_label = get_unique_label(data, data.shape[1]-1)
    label = np.sign(np.sum(labels) + 0.1)
    if unique_label.shape[0] == 1:
        tuple1 = (from_value, -1)
        new_node = build_tree.node(tuple1)
        new_node._label = label
        tree_node.add_child(new_node)
        return 
   
    # gain divide gain and choice the best divide
    gain = []
    if types == 'gain':
        for i in range(data.shape[1]-1):
            gain.append(information_gain(data, i))
    elif types == 'gain_ratio':
        for i in range(data.shape[1]-1):
            gain.append(information_gain_ratio(data, i))
    else:
        for i in range(data.shape[1]-1):
            gain.append(gini_index(data, i))
    attribute = np.argmax(gain)
    
#    if depth == 1:
#        print(gain)
    
    tuple1 = (from_value, attribute)
    new_node = build_tree.node(tuple1)
    tree_node._label = label
    tree_node.add_child(new_node)
    
    
    # divide data
    attrubute_column, unique_value = get_unique_label(data, attribute)
    for i in unique_value:
        index = np.where(attrubute_column == i)[0]
        data_divide = data[index]
        data_divide = np.delete(data_divide, attribute, axis = 1)
        construct_tree(data_divide, new_node, from_value = i, depth = depth, types = types)
        
def predict_tree(data, tree_node, index_begin, predict, depth, max_depth, attribute):
    '''
    Input: 
        data: the data which divide by tree_node
        tree_node: the father node which use to divide data
        index_begin: the index of those data
        predict: the predict result
        
    To Do:
        get child_node from tree_node
        decision whether 
    '''
    
    # check whether is pure:
    depth += 1
    if len(tree_node.getchild()) == 0 or depth > max_depth :
#    if len(tree_node.getchild()) == 0:
        label = tree_node.getlabel()
#        print("depht", depth)
        predict[index_begin] = label
        return 
    
#    print("depth = {}, attribute = {}".format(depth, attribute))
        
    next_node_list = tree_node.getchild()
    attrubute_column, unique_value = get_unique_label(data, attribute)
    attribute_choice = 0
    next_node = next_node_list[0]

    for i in unique_value:
        index = np.where(attrubute_column == i)[0]
        data_divide = data[index]
        data_divide = np.delete(data_divide, attribute, axis = 1)
        
        for j in range(len(next_node_list)):
            tuple1 = next_node_list[j].getdata()
            if tuple1[0] == i:
                attribute_choice = tuple1[1]
                next_node = next_node_list[j]
        predict_tree(data_divide, next_node, index_begin[index],
                     predict, depth, max_depth, attribute_choice)

            
def shuffle_sample(data, ratio, seed = 10):
#    random.seed(seed)
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
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
    
if __name__ == '__main__':
    accuracy_list = []


    train_file = '../input/train.csv'
    df_train = pd.read_csv(train_file, header = None)
    train_numpy = np.array(df_train)
#        train, valid = shuffle_sample(train_numpy, 0.3)
#        valid_labels = valid[:, -1]
    train_labels = train_numpy[:, -1]


    test_file = '../input/test.csv'
    df_test = pd.read_csv(test_file, header = None)
    test = np.array(df_test)
    
    k = 5
    fold_data_num = int(train_numpy.shape[0] / k)
    shuffle_index = random.sample(range(train_numpy.shape[0]), int(train_numpy.shape[0]))
    for i in range(k):
        valid_index = np.array(shuffle_index[i*fold_data_num: (i+1)*fold_data_num-1])
        train, valid = data_from_index(train_numpy, valid_index)
        valid_labels = valid[:, -1]

    


#    for j in ['gain', 'gain_ratio', 'gini']:
        j = 'gini'
        decision_tree = build_tree.node(-1)
        construct_tree(train, decision_tree, from_value = -1, depth = 0, types = j)
    ###
    ###   
    #    def test_tree(node):
    #        child = node.getchild()
    ##        print("label", node.getlabel())
    #
    #        if len(child) == 0:
    ##            print("label", node.getlabel())
    #            return 
    #        else:
    #            for i in child:
    #                test_tree(i)        
    #    test_tree(decision_tree)
        max_depth = 3
        index_begin = np.array(range(test.shape[0]))
        predict = np.zeros((test.shape[0]))
        first_node = decision_tree._child[0]
        attribute_init = first_node.getdata()[1]
        predict_tree(test, first_node, index_begin, 
                     predict, depth = 0, max_depth = max_depth, attribute = attribute_init)
        if i == 0:
            predict_sum = predict
        else:
            predict_sum += predict
    
    predict = np.sign(predict_sum)
    print(predict)
    df1 = pd.DataFrame(predict)
    df1.to_csv('../output/15352104_hejujie.txt', index = False, header = None)
#        accuracy = np.sum(predict == train_labels) / train_numpy.shape[0]
#        accuracy_list.append(accuracy)
#        print("max_depth = {}, accuracy is:{}".format(max_depth, accuracy))
#    if j == 'gain':
#        plt.plot(range(len(accuracy_list)), accuracy_list, 'r')
#    if j == 'gain_ratio':
#        plt.plot(range(len(accuracy_list)), accuracy_list, 'g')
#    if j == 'gini':
#        plt.plot(range(len(accuracy_list)), accuracy_list, 'b')