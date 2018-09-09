# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from rpca_admm import RpcaAdmm

def gen_sparse_noise(M, N, card):
    sign = np.sign(np.random.random((M, N)) - 0.5)
    sparse_noise = sign * (np.random.random((M, N)) < card)
    return sparse_noise

def gen_low_rank_noise(M, N, rank):
    rank_data = np.random.random((rank, N))
    low_rank_noise = np.zeros((M, N))
    for i in range(M):
        index = np.random.choice(range(10), (1))
        low_rank_noise[i] = rank_data[index]
    return low_rank_noise - np.mean(low_rank_noise, axis = 1, keepdims = True)
#    return low_rank_noise - np.mean(low_rank_noise)

def data_generation(M, N, rank, card):
    low_rank_noise = gen_low_rank_noise(M, N, rank)
    sparse_noise = gen_sparse_noise(M, N, card)
    return low_rank_noise + sparse_noise, low_rank_noise, sparse_noise

def plot_picture(noise, low_rank_noise, sparse_noise):
    plt.imshow(noise)
    plt.title("Noise")
    plt.show()
    plt.imshow(low_rank_noise)
    plt.title("Low rank noise")
    plt.show()
    plt.imshow(sparse_noise)
    plt.title("Sparse Noise")
    plt.show()


def main():
    rank = 10
    card = 0.2
    M = 100
    N = 50
    noise, low_rank_noise, sparse_noise = data_generation(M, N, rank, card)
    plot_picture(noise, low_rank_noise, sparse_noise)
    data = noise
    
    
    max_iterations = 1000
    stop_tol = 1e-10
    lambd = 1 / np.sqrt(np.maximum(M, N))
    rho = 10 * lambd
    rpca = RpcaAdmm(n_iterations=max_iterations, stop_tol=stop_tol, lambd=lambd, rho=rho)
    noise_recovery, low_rank_recovery, sparse_recovery = rpca.fit(data)
    print("\n\n-----------------------------------------------------------------------\n")
    print("picture after recovery")
    plot_picture(noise_recovery, low_rank_recovery, sparse_recovery)
    
if __name__ == "__main__":
    main()