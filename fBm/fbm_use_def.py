import numpy as np
import scipy.linalg as la
import pandas as pd
from fbm import FBM
import sys
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy.stats import linregress
from numpy.linalg import eigvals
from numpy.fft import fft


def get_hurst_series(n, hurst):
    """
    生成具有指定 Hurst 指数的时间序列。
    参数:
    - n: 序列长度
    - hurst: Hurst 指数 (0 < hurst < 1)
    
    返回:
    - 一个具有指定 Hurst 指数的时间序列
    """
    # 使用分数布朗运动 (fBm) 生成时间序列
    f = FBM(n=n+1, hurst=hurst, length=1, method='daviesharte')
    series = f.fbm()

    return series
def create_toeplitz_error_matrix(matrix):
    """生成Toeplitz矩阵并计算误差矩阵"""
    N = matrix.shape[0]
    toeplitz_matrix = np.zeros((N, N))
    
    for k in range(N):
        avg_val = np.mean(np.diag(matrix, k))
        for i in range(N - k):
            toeplitz_matrix[i, i + k] = avg_val
            if k > 0:
                toeplitz_matrix[i + k, i] = avg_val
    
    return toeplitz_matrix, matrix - toeplitz_matrix
# matrix = np.array([[3,4,5],[5,7,7],[6,7,8]])
# toe,random = create_toeplitz_error_matrix(matrix)
# print(matrix)
# print(toe)
# print(random)

def get_tau_lag(hurst, n, S, L, start_l,tau):
    """计算相关系数矩阵及Toeplitz变换"""
    data = get_hurst_series(n+1,hurst)
    data = np.diff(data)
    start_tra = round((n - (S + L)) / 2)
    data = data[start_tra:start_tra + L + S + 1]

    data_mean,data_std = np.mean(data), np.std(data)
    data = (data - data_mean) / data_std
    
    matrix_tra = np.array([data[t:t + S] for t in range(L)])
    matrix_corr = np.corrcoef(matrix_tra)
    matrix_corr = abs(matrix_corr)
    matrix_corr = (matrix_corr + matrix_corr.T) / 2
    toe_corr, error_corr = create_toeplitz_error_matrix(matrix_corr)
    return matrix_corr, toe_corr, error_corr



def get_Circulant_matrix_from_toeplitz_matrix(matrix):
    c_1 = matrix[:, 0]
    c_2= [c_1[0]]
    N = len(c_1)

    for k in range(1, len(c_1)):
        flag = c_1[k] +c_1[N-k]
        c_2.append(flag)
    # matrix_C = np.vstack((c_2,)*dimension)
    n = len(c_2)
    # 初始化一个n x n的矩阵
    circulant = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # 每行的元素是前一行循环右移一位的结果
            circulant[i][j] = c_2[(j - i) % n]
    return circulant

import numpy as np

def get_middle_60_percent_np(arr):
    lower = int(len(arr)*0.2)
    upper = int(len(arr)*0.8)
    return arr[lower:upper]