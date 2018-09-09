# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:34 2018

@author: jie
"""

import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from SpectralClustering import SpectralClustering
from NormalizedCuts import NormalizedCuts
from NonNegativeMatrixFactorization import NonNegativeMatrixFactorization
from Louvain import Louvain
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score


def purity_score(y_true, y_pred):
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner

    return accuracy_score(y_true, y_labeled_voted)
    

def read_data(file = "../input/washington/washington_adj.txt", sep = '\t', describe = False):
    df1 = pd.read_csv(file, sep = sep, header = None)
    if describe == True:
        print("------------------------------------------------")
        print(df1.describe())
        print("-------------------------------------------------")
    return np.array(df1)
    
    
    
def main():
    root_path = "../input/"
    nmi_list = []
    label_list = os.listdir(root_path)
    for folder in os.listdir(root_path):
        file_path = os.path.join(root_path, folder)
        print("\nfile is: \t", folder)
        
        labels = read_data(file = file_path + "/" + folder + "_label.txt").flatten()
        data = read_data(file = file_path + "/" + folder + "_adj.txt")
        
#        clf = NormalizedCuts(n_reductions=5, n_processes=5, n_clusters=len(set(labels)))
#        predicts = clf.base_spectral(data)

    
#        clf = Louvain(n_iterations=5000)
#        predicts = clf.fit(data)

        score_best = 0
        for i in range(1, 100):
            print(i)
            clf = NonNegativeMatrixFactorization(n_iterations=i*10, n_clusters=5)
            predicts = clf.fit(data) 
            score = ([normalized_mutual_info_score(labels, predicts), 
                         purity_score(labels, predicts),
                        adjusted_rand_score(labels, predicts)])
            if score[0] > score_best:
                score_best = score[0]
        nmi_list.append(score_best)
        plt.show()
        print("nmi: \t\t", score[0])
        print("purity: \t", score[1])
        print("accuracy: \t", score[2])
    print(nmi_list, label_list)
    plt.bar(range(len(nmi_list)), nmi_list, color='rgbc', tick_label=label_list) 
    plt.show()

if __name__ == "__main__":
    main()
    