# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:04:59 2017

@author: jie
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(filename = 'data/testSet.csv'):
    df1 = pd.read_csv(filename, '\t')
    return np.array(df1)
    
#image = read_data('data/data_image.txt')
#real = read_data('data/data_real.txt')
#data = np.hstack((real, image))
#print(data.shape)
data = read_data()

output = pd.DataFrame(data)

output.to_csv('data/testSet.csv', index = False, header = None)

    

