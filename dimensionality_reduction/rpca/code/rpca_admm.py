# -*- coding: utf-8 -*-

import numpy as np

class RpcaAdmm(object):
    
    def __init__(self, n_iterations = 1000, stop_tol = 1e-6, lambd = 0.1, rho = 1):
        self.n_iterations = n_iterations
        self.stop_tol = stop_tol
        self.lambd = lambd
        self.rho = rho
    
    def shrinkage(self, X, tua):
        shrink = np.maximum((np.abs(X) - tua), 0)
        return np.sign(X) * shrink    
    
    def svd_thres(self, X, tau):
        U_matrix, sigma, VT_matrix = np.linalg.svd(X)
        sigma_fix = np.zeros_like(X)
        for i in range(sigma.shape[0]):
            sigma_fix[i][i] = sigma[i]
        # because value in sigma is all positive, so we can just use shrinkage
        return np.dot(np.dot(U_matrix, self.shrinkage(sigma_fix, tau)), VT_matrix)
    
    def fit(self, data):
        data[np.isnan(data)] = 0
        low_rank_recovery = np.random.random((data.shape))
        sparse_recovery = np.random.random((data.shape))
        dual_matrix = np.random.random((data.shape))
        
        for i in range(self.n_iterations):
            low_rank_recovery = self.svd_thres(data - sparse_recovery - 1/self.rho * dual_matrix, 
                                               1 / self.rho)
            sparse_recovery = self.shrinkage(data - low_rank_recovery - 1/self.rho * dual_matrix, 
                                             self.lambd / self.rho)
            dual_matrix = dual_matrix + self.rho * (low_rank_recovery + sparse_recovery - data)
            
            if np.abs(np.sum(data - low_rank_recovery - sparse_recovery)) < self.stop_tol:
                print("finish iteration", self.stop_tol)
                break
        print("the recovery error is", np.sum(data - low_rank_recovery - sparse_recovery))
        return low_rank_recovery + sparse_recovery, low_rank_recovery, sparse_recovery



    