import matplotlib.pyplot as plt
from fbm_use_def import *
import numpy as np
from wigner_def import *
from scipy.linalg import toeplitz
import math
import pandas as pd

def compute_eigenvalues(N, H):
    """
    根据显式余弦展开公式计算循环矩阵的特征值序列 lambda_j。
    
    参数:
        N: 矩阵维度
        H: Hurst 指数 (0 < H < 1)
    
    返回:
        lambdas: numpy 数组，长度为 N，对应特征值 lambda_0, ..., lambda_{N-1}
    """
    alpha = 2 - 2 * H
    lambdas = np.zeros(N)
    for j in range(N):
        # 第一个项
        lambda_j = 1 / (1 ** alpha)
        # 累加余弦项
        for k in range(1, N):
            tk = 1 / ((k + 1) ** alpha)
            tnk = 1 / ((N - k + 1) ** alpha)
            lambda_j += 2 * (tk + tnk) * np.cos(2 * np.pi * j * k / N)
        lambdas[j] = lambda_j
    return lambdas

def compute_eigenvalues_T(N, H):
    t = []
    alpha = 2 - 2 * H
    for j in range(N):
        tk = 1/((j+1)**alpha)
        t.append(tk)
    T = toeplitz(t)  # 默认对称
    eig = np.linalg.eigvals(T)

    return eig

#test for
H = np.arange(0.50,0.901,0.05)
for h in H:
    print("H=", h)
    n = 1024
    dimension = 1024
    simulation_times = 10

    file_name = f"D:\\Data\\pu\\toe_{h:.2f}_{dimension}_{simulation_times}.csv" # toe_0.90_1024_10.csv
    df = pd.read_csv(file_name,header=None)
    matrix = df.values
    # matrix_1 = matrix[:,1]

    eig = compute_eigenvalues(n, h)
    eig_T = compute_eigenvalues_T(n, h)
    print(f"sum of eig: {sum(eig)}   {sum(eig_T)}")
    error = sum(abs(a-b) for a,b in zip(eig,eig_T))
    print('误差：', error)
    error_real = []
    sum_of = []
    for i in range(simulation_times):
        # print(i)
        matrix_1 = matrix[:,i]
        sum_of.append(sum(matrix_1))
        error_real_ = sum(abs(a-b) for a,b in zip(matrix_1,eig_T))
        error_real.append(error_real_)
    # print(f'real_true：, {sum_of}')
    print("real_true：[" + ", ".join(f"{x:.6f}" for x in sum_of) + "]")

    # print(f'真实误差：, {error_real:.4f}')
    print("真实误差：：[" + ", ".join(f"{x:.6f}" for x in error_real) + "]")
    print("*************")
#理论相关系数