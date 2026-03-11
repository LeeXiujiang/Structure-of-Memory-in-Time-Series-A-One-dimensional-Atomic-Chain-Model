#fbm _get toeplitz_matrix   and    random matrix(随机涨落矩阵)
try:  # just a hack to make running these in development easier
    import empyricalRMT
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import scipy.linalg as la
import pandas as pd
from fbm import FBM
import gc
import sys
import time
import os
import psutil
import numba
from numba import njit,prange

# class prange(object):
#     """ Provides a 1D parallel iterator that generates a sequence of integers.
#     In non-parallel contexts, prange is identical to range.
#     """
#     def __new__(cls, *args):
#         return range(*args)

@njit(parallel=True)
def pearson_corr_parallel(trajectorys):
    S, L = trajectorys.shape
    corr = np.zeros((L, L))
    
    for i in prange(L):
        corr[i, i] = 1.0
        for j in range(i + 1, L):
            x = trajectorys[:, i]
            y = trajectorys[:, j]
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            std_x = np.std(x)
            std_y = np.std(y)
            cov = np.mean((x - mean_x) * (y - mean_y))
            corr_val = cov / (std_x * std_y)
            corr[i, j] = corr_val
            corr[j, i] = corr_val
    return corr   
# @njit
# def pearson_corr_loop(trajectorys):
#     S, L = trajectorys.shape
#     corr = np.zeros((L, L))
#     for i in range(L):
#         corr[i, i] = 1.0
#         for j in range(i + 1, L):
#             x = trajectorys[:, i]
#             y = trajectorys[:, j]
#             mean_x = np.mean(x)
#             mean_y = np.mean(y)
#             std_x = np.std(x)
#             std_y = np.std(y)
#             cov = np.mean((x - mean_x) * (y - mean_y))
#             corr_val = cov / (std_x * std_y)
#             corr[i, j] = corr_val
#             corr[j, i] = corr_val
#     return corr
def free_memory():
    process = psutil.Process(os.getpid())
    print(f"当前内存占用: {process.memory_info().rss / 1024**3:.2f} GB")
    gc.collect()
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
    f = FBM(n+1, hurst=hurst, length=1, method='daviesharte')
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


def non_diag_moments(matrix):
    """计算矩阵的非对角元素的一阶矩，二阶矩，标准差"""
    mask = np.eye(matrix.shape[0], dtype=bool)
    non_diag_elements = matrix[~mask]
    return np.mean(non_diag_elements), np.mean(non_diag_elements**2), np.std(non_diag_elements)
def matrix_std(matrix,std=False,scale=False):
    """计算矩阵的标准差"""
    if std:
        mean_value, _, std_dev = non_diag_moments(matrix)
        matrix = (matrix - mean_value) / std_dev
        np.fill_diagonal(matrix, 1)
    if scale:
        np.fill_diagonal(matrix, 0)
    return matrix
def get_windows_from_ts(ts, S, L):
    return np.array([ts[i:i+S] for i in range(L)]).T

# def get_windows_from_ts(ts,S,L):
#     trajectorys = np.zeros((S, L))
#     for i in range(L):
#         trajectorys[:,i] = ts[i:i+S]
#     return trajectorys
def get_corr_from_trajectory(trajectorys, method):
    """计算相关系数矩阵"""
    L = trajectorys.shape[1]
    
    if method == 0:
        # Pearson correlation
        corr = np.corrcoef(trajectorys.T)
        corr = abs(corr)
        corr = (corr + corr.T) / 2  # 保对称性
    elif method == 1:
        # Dot method
        norms = np.einsum('ij,ij->j', trajectorys, trajectorys)  # 每列的平方和
        corr = trajectorys.T @ trajectorys
        corr = corr / norms[:, None]
        corr = (corr + corr.T) / 2  # 保对称性
    
    else:
        # Cosine similarity
        norms = np.linalg.norm(trajectorys, axis=0)
        norm_matrix = np.outer(norms, norms)
        dot_product = trajectorys.T @ trajectorys
        corr = dot_product / norm_matrix
    
    toe_corr, error_corr = create_toeplitz_error_matrix(corr)
    return corr, toe_corr, error_corr

    # return corr

def get_tau_lag_1(hurst, n, S, L, start_l,method=0):
    """计算相关系数矩阵及Toeplitz变换"""
    data = get_hurst_series(n+1,hurst)
    data = np.diff(data)
    start_tra = round((n - (S + L)) / 2)
    data = data[start_tra:start_tra + L + S + 1]

    data_mean,data_std = np.mean(data), np.std(data)
    data = (data - data_mean) / data_std
    
    trajectorys = get_windows_from_ts(data,S,L)
    corr, toe_corr, error_corr = get_corr_from_trajectory(trajectorys,method = method)

    # matrix_tra = np.array([data[t:t + S] for t in range(L)])
    # matrix_corr = np.corrcoef(matrix_tra)
    
    # toe_corr, error_corr = create_toeplitz_error_matrix(matrix_corr)
    # print(toe_corr)
    # print(error_corr)
    return corr, toe_corr, error_corr
def get_tau_lag_2(hurst, n, S, L, start_l):
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
def cal_error_eigenvalue_1(h, S,dimension, simulation_times,method=0):
    """计算误差矩阵的特征值"""
    # E1, E2, E3 = [], [], []
    E1, E2, E3 = np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times))
    E_vectors_toe = {}
    E_vactors_random = {}
    for t in prange(simulation_times):
        percent = (t + 1) / simulation_times * 100
        sys.stdout.write(f"\r进度 :{percent:.2f}%")
        sys.stdout.flush()
        # sys.stdout.write(f"\r进度 :{percent:.2f}%")  # 使用 `\r` 让光标回到行首，覆盖前面内容
        # sys.stdout.flush()  # 刷新输出，确保立即更新
        n, L, start_l = 600000,  dimension, 300000
        corr_use, toe_corr_use, error_corr_use = get_tau_lag_1(h, n, S, L, start_l,method)
        
        # corr_use = matrix_std(corr_use,std=False,scale=True)
        # toe_corr_use = matrix_std(toe_corr_use,std=False,scale=True)
        # error_corr_use = matrix_std(error_corr_use,std=False,scale=True)
        # E1[:,t] = (np.linalg.eigvals(corr_use))
        eigenvalues, eigenvectors = np.linalg.eigh(toe_corr_use)
        E2[:,t] = eigenvalues
        E_vectors_toe[t] = eigenvectors

        # E2[:,t] = (np.linalg.eigvals(toe_corr_use))
        # eigenvalues, eigenvectors = np.linalg.eigh(error_corr_use)
        # E3[:,t] = (np.linalg.eigvals(error_corr_use))
        
    print('\n')
    return E2,E_vectors_toe
def cal_error_eigenvalue_2(h,S, dimension, simulation_times,menthod=0):
    """计算误差矩阵的特征值"""
    # E1, E2, E3 = [], [], []
    E1, E2, E3 = np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times))
    
    for t in prange(simulation_times):
        percent = (t + 1) / simulation_times * 100
        if t % 10 == 0:
            sys.stdout.write(f"\r进度 :{percent:.2f}%")
            sys.stdout.flush()
        # sys.stdout.write(f"\r进度 :{percent:.2f}%")  # 使用 `\r` 让光标回到行首，覆盖前面内容
        # sys.stdout.flush()  # 刷新输出，确保立即更新
        n,  L, start_l = 600000,  dimension, 300000
        corr_use, toe_corr_use, error_corr_use = get_tau_lag_2(h, n, S, L, start_l)
        
        # corr_use = matrix_std(corr_use,std=False,scale=True)
        # toe_corr_use = matrix_std(toe_corr_use,std=False,scale=True)
        # error_corr_use = matrix_std(error_corr_use,std=False,scale=True)
        E1[:,t] = (np.linalg.eigvals(corr_use))
        E2[:,t] = (np.linalg.eigvals(toe_corr_use))
        E3[:,t] = (np.linalg.eigvals(error_corr_use))
        
    print('\n')
    return E2, E3, E1
def get_eigenvector_eigenvalue(H,S,dimension, simulation_times,method=0):
    E_H = {}
    E_vectors_H = {}
    menthof_print = {'p','dot','cos'}
    for h in H:
        print(f"Processing h={h:.2f}")
        E2,E_vectors_toe = cal_error_eigenvalue_1(h, S,dimension, simulation_times,method)
        E_H[f"H={h:.2f}"] = E2
        E_vectors_H[f"H={h:.2f}"] = E_vectors_toe
    return E_H, E_vectors_H
# def main():
#     dimension, simulation_times = 2000, 100
#     # H = [0.70,0.75,0.80,0.85,0.90]
#     H = np.arange(0.50,0.901,0.05)
#     S = 40000
#     E_h = {}
#     E_vectors_H = {}
#     for h in H:
#         print(f"Processing h={h:.2f}")
#         E2,E_vectors_toe = cal_error_eigenvalue_1(h, S,dimension, simulation_times,method=0)
#         E_h[f"H={h:.2f}"] = E2
#         E_vectors_H[f"H={h:.2f}"] = E_vectors_toe
#         # pd.DataFrame(E_1).to_csv(f"D:\\Data\\pu\\toe_{h:.2f}_{dimension}_{simulation_times}_p.csv", index=False, header=False)
#         # pd.DataFrame(E_2).to_csv(f"D:\\Data\\pu\\error_{h:.2f}_{dimension}_{simulation_times}_p.csv", index=False, header=False)
#         # pd.DataFrame(E_3).to_csv(f"D:\\Data\\pu_python\\midu_corr_{h:.2f}_{dimension}_{simulation_times}_std_1.csv", index=False, header=False)
        
#         # free_memory()  # 释放内存
# if __name__ == "__main__":
#     main()
    