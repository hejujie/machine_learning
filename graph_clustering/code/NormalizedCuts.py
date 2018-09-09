# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:30:18 2018

@author: jie
"""

import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


from sklearn.metrics import accuracy_score
import numpy as np

def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
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
    



class NormalizedCuts(object):
    def __init__(self, n_reductions, n_processes, n_clusters):
        self.n_reductions = n_reductions
        self.n_processes = n_processes
        self.n_clusters = n_clusters
        
    def base_spectral(self, data):
        spectral_clf = SpectralClustering(n_clusters=self.n_processes, n_reductions=self.n_reductions, 
                                          models='ncut')
        labels = spectral_clf.fit(data)
        return labels 
        
    def fit(self, data):
        labels = self.base_spectral(data)
#==============================================================================
# TODO: @jujie save the community doesn't change 
#==============================================================================
        while (len(set(labels))) > self.n_clusters:
            note_tuple = (0, 0)          
            min_ncut = 0
            for i in set(labels):
                sample_i = np.where(labels == i)[0]
                for j in set(labels):
                    sample_j = np.where(labels == j)[0]
                    if i == j:
                        continue
                    else:
                        sample_add = np.append(sample_i, sample_j)
                        cut_add = np.sum(data[sample_add]) - np.sum(data[sample_add][:, sample_add])
                        assoc_add = np.sum(data[sample_add])
                        cut_i_j = np.sum(data[sample_i][:, sample_j])
                        assoc_i = np.sum(data[sample_i])
                        assoc_j = np.sum(data[sample_j])
                        ncut = (cut_add / assoc_add) - (cut_i_j / assoc_i) - (cut_i_j / assoc_j)
                        if ncut < min_ncut:
                            min_ncut = ncut
                            note_tuple = (i, j)
            if note_tuple[0] == note_tuple[1]:
                break
            else:
                labels[np.where(labels == note_tuple[0])[0]] = note_tuple[1]
        return labels

                                    
def main():
    labels = read_data(file = "../input/washington/washington_label.txt").flatten()
    print(set(labels))
    data = read_data(file = "../input/washington/washington_adj.txt")
    print(data.shape)
    clf = NormalizedCuts(n_reductions=50, n_processes=50, n_clusters=5)
    predicts = clf.fit(data)
#    predicts = clf.base_spectral(data)
    print((predicts))
    print("nmi", normalized_mutual_info_score(labels, predicts))
    print("purity", purity_score(labels, predicts))
    print("accuracy", adjusted_rand_score(labels, predicts))
    
if __name__ == "__main__":
    main()
    
    
    