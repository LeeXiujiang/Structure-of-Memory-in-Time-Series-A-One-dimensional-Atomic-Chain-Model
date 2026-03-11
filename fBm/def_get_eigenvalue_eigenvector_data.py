#fbm _get toeplitz_matrix   and    random matrix(随机涨落矩阵)

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
import numpy as np

import numpy as np

def truncate_toeplitz_by_energy(matrix, threshold=0.99,print_cut_flag=True):
    """
    根据 Toeplitz 矩阵的首行提取相关序列，
    按能量累计比例进行截断（将距离主对角线超过阈值的元素置零）。

    参数：
        matrix: 对称 Toeplitz 矩阵 (N x N)
        threshold: 能量保留比例（如 0.99）

    返回：
        truncated_matrix: 截断后的矩阵
        cutoff_index: 截断使用的最大 |i-j| 值
    """
    N = matrix.shape[0]

    # 从首行提取 Toeplitz 的相关函数序列
    C = np.abs(matrix[:,0])  # 或者不用 abs，视你的数据而定
    energy = C ** 2
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]
    ratio = cumulative_energy / total_energy

    # 确定能量累计达到阈值的截断索引
    cutoff_index = np.searchsorted(ratio, threshold)
    if print_cut_flag:
        print(f"{threshold*100}% 能量保留后，需要截断到第 {cutoff_index} 个元素")
    # 构造截断后的矩阵
    i, j = np.indices((N, N))
    mask = np.abs(i - j) >= cutoff_index
    truncated_matrix = matrix.copy()
    truncated_matrix[mask] = 0

    return truncated_matrix, cutoff_index

# 示例：H = 0.8, N = 1000
# cut_k = find_cutoff_index(H=0.8, N=1000, threshold=0.99)
# print(f"H=0.8 时，建议在第 {cut_k} 项之后截断")

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
def split_into_four_blocks(A):
    """
    将一个 N x N 矩阵划分为四个 N/2 x N/2 的块。
    返回顺序为：左上（A11），右上（A12），左下（A21），右下（A22）
    """
    N = A.shape[0]
    if A.shape[0] != A.shape[1] or N % 2 != 0:
        raise ValueError("输入必须是 N x N 的矩阵，且 N 必须为偶数。")
    
    half = N // 2
    A11 = A[:half, :half]
    A12 = A[:half, half:]
    A21 = A[half:, :half]
    A22 = A[half:, half:]
    
    return A11, A12, A21, A22
def toe_zipu(toeplitz_matrix):
    A11, A12, A21, A22 = split_into_four_blocks(toeplitz_matrix)
    m = A12.shape[0]

    J = np.fliplr(np.eye(m))
    H = J @ A12
    toe_duichen = A11 + H
    toe_xieduichen = A11 -H
    # toe_xieduichen = np.vstack((A21, H))
    return toe_duichen,toe_xieduichen
# def toe_zipu(toeplitz_matrix):
#     A_m_n = {}
#     M, N = toeplitz_matrix.shape
#     for i in range(M):
#         A_m_n[i] = toeplitz_matrix[0][i]
#     toe_duichen = np.zeros((int(M/2),int(N/2)))
#     toe_xieduichen = np.zeros((int(M/2),int(N/2)))
#     for i in range(int(M/2)):
#         for j in range(int(N/2)):
#             toe_duichen[i,j] = A_m_n[abs(i-j)] + A_m_n[abs(i+j-1)]
#             toe_xieduichen[i,j] = A_m_n[abs(i-j)] - A_m_n[abs(i+j-1)]


#     return toe_duichen,toe_xieduichen

def non_diag_moments(matrix):
    """计算矩阵的非对角元素的一阶矩，二阶矩，标准差"""
    mask = np.eye(matrix.shape[0], dtype=bool)
    non_diag_elements = matrix[~mask]
    return np.mean(non_diag_elements), np.mean(non_diag_elements**2), np.std(non_diag_elements)
def matrix_std(matrix,std=False,scale=False):
    """计算矩阵的标准差"""
    if std:
        mean_value, _, std_dev = non_diag_moments(matrix)
        print(f"mean: {mean_value:.7f} std: {std_dev:.7f}")
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

    #矩阵操作 带状



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

from def_eig_thoery import *
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
def fit_power_decay(h,t, y, constant_a=False,delta = 0.04):
    t = np.asarray(t)
    y = np.asarray(y)
    slope = 2-2*h
    # 定义拟合函数（两种情况）
    if constant_a:
        def func(t, a, alpha):
            return a / ((t + 1) ** alpha)
        p0 = [1.0, 2-2*h]  # 初始猜测 a, alpha
        bounds = ([0.1, slope-delta], [2, slope+delta])  # 设置 a 和 alpha 的范围
        popt, _ = curve_fit(func, t[:50], y[:50], p0=p0, bounds=bounds)
        a, alpha = popt
        y_fit = func(t, a, alpha)
        return alpha, a,y_fit
    else:
        def func(t, alpha):
            return 1 / ((t + 1) ** alpha)
        p0 = [2-2*h]  # 初始猜测 alpha
        bounds = (slope-delta, slope+delta)  # 给 alpha 设置合理范围
        popt, _ = curve_fit(func, t[:50], y[:50], p0=p0, bounds=bounds)
        alpha = popt[0]
        y_fit = func(t, alpha)
        return alpha,y_fit

        return alpha, a, y_fit  # 返回顺序为 alpha, a  y_fit
def cal_error_eigenvalue_1(h, S,dimension, simulation_times,method=0,func = False,constant_a=False,zipu = 0,cut=False):
    """计算误差矩阵的特征值"""
    # E1, E2, E3 = [], [], []
    if zipu != 0 :
        E1, E2, E3 = np.zeros((int(dimension/2),simulation_times)),np.zeros((int(dimension/2),simulation_times)),np.zeros((int(dimension/2),simulation_times))
    else:
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

        Tk = toe_corr_use[:,0]
        if func:
            tau = range(len(Tk))
            if constant_a:
                alpha, a ,Tk_fit= fit_power_decay(h,tau, Tk, constant_a=True)
                print(f"alpha = {alpha}, a = {a}")

            else:    
                alpha,Tk_fit = fit_power_decay(h,tau, Tk, constant_a=False)
                print(f"alpha = {alpha}")
            # toe_corr = toeplitz(Tk_fit)
            toe_corr_use = toeplitz(Tk_fit)


        # 子谱  对称子谱 与 斜对称子谱
        toe_duichen,toe_xieduichen = toe_zipu(toe_corr_use)
        if zipu == 1:
            toe_corr_use = toe_duichen
        if zipu == 2:
            toe_corr_use = toe_xieduichen
        # eigenvalues, eigenvectors = np.linalg.eigh(toe_duichen)
        if cut:
            toe_corr_use,_ = truncate_toeplitz_by_energy(toe_corr_use,0.90)
        stan = True
        if stan:
            toe_corr_use = matrix_std(toe_corr_use,std=True,scale=False)
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
def get_eigenvector_eigenvalue(H,S,dimension, simulation_times,method=0,func = False,constant_a=False,zipu = 0,cut=False):
    E_H = {}
    E_vectors_H = {}
    menthof_print = {'p','dot','cos'}
    for h in H:
        print(f"Processing h={h:.2f}")
        E2,E_vectors_toe = cal_error_eigenvalue_1(h, S,dimension, simulation_times,method,func,constant_a,zipu = zipu,cut=cut)
        E_H[f"H={h:.2f}"] = E2
        E_vectors_H[f"H={h:.2f}"] = E_vectors_toe
    return E_H, E_vectors_H
import numpy as np

def zipu_duichen_eigenvectors(x):
    """
    输入：归一化的列向量 x，形状为 (m, 1)
    输出：归一化后的列向量 [Jx + x; x]，其中 J 为反对角线为 1 的对称矩阵，输出形状为 (2m, 1)
    """
    # 检查 x 是否为列向量
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("输入 x 必须是形状为 (m, 1) 的列向量。")
    
    m = x.shape[0]
    
    # 构造 J：反对角线为 1，其余为 0
    J = np.fliplr(np.eye(m))

    # 计算 Jx + x
    Jx_plus_x = J @ x 

    # 拼接 [Jx + x; x]
    result = np.vstack((Jx_plus_x, x))

    # 归一化
    result = result / np.sqrt(2)

    return result
def zipu_xieduichen_eigenvectors(x):
    """
    输入：归一化的列向量 x，形状为 (m, 1)
    输出：归一化后的列向量 [Jx + x; x]，其中 J 为反对角线为 1 的对称矩阵，输出形状为 (2m, 1)
    """
    # 检查 x 是否为列向量
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("输入 x 必须是形状为 (m, 1) 的列向量。")
    
    m = x.shape[0]
    
    # 构造 J：反对角线为 -1，其余为 0
    J = -np.fliplr(np.eye(m))

    # 计算 Jx
    Jx_plus_x = J @ x

    # 拼接 [Jx + x; x]
    result = np.vstack((Jx_plus_x, x))

    # 归一化
    result = result / np.sqrt(2)

    return result
def batch_process_eigenvectors(X, func=1):
    """
    输入：
        X: 一个 (m, n) 的矩阵，每一列是一个归一化特征向量
        func: 函数，形如 (=1)zipu_duichen_eigenvectors 或 (=2)zipu_xieduichen_eigenvectors
    输出：
        Y: 一个 (2m, n) 的矩阵，每一列是变换后的结果
    """
    if X.ndim != 2:
        raise ValueError("输入 X 必须是二维数组，每列为一个特征向量")
    
    m, n = X.shape
    Y = np.zeros((2 * m, n))
    
    for i in range(n):
        x_i = X[:, i].reshape(-1, 1)  # 取第 i 个列向量，形状为 (m, 1)
        if func == 1:
            y_i = zipu_duichen_eigenvectors(x_i)
        elif func == 2:
            y_i = zipu_xieduichen_eigenvectors(x_i)
        # y_i = func(x_i)              # 应用 zipu_duichen_eigenvectors 或 zipu_xieduichen_eigenvectors
        Y[:, i] = y_i.ravel()        # 放入结果矩阵中
    
    return Y

def cal_error_eigenvalue_zipu(h, S,dimension, simulation_times,method=0,func = False,constant_a=False,zipu = 0):
    """计算误差矩阵的特征值"""
    # E1, E2, E3 = [], [], []
    # if zipu != 0 :
    #     E1, E2, E3 = np.zeros((int(dimension/2),simulation_times)),np.zeros((int(dimension/2),simulation_times)),np.zeros((int(dimension/2),simulation_times))
    # else:
    #     E1, E2, E3 = np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times)),np.zeros((dimension,simulation_times))
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

        Tk = toe_corr_use[:,0]
        if func:
            tau = range(len(Tk))
            if constant_a:
                alpha, a ,Tk_fit= fit_power_decay(h,tau, Tk, constant_a=True)
                print(f"alpha = {alpha}, a = {a}")

            else:    
                alpha,Tk_fit = fit_power_decay(h,tau, Tk, constant_a=False)
                print(f"alpha = {alpha}")
            # toe_corr = toeplitz(Tk_fit)
            toe_corr_use = toeplitz(Tk_fit)

        toe_corr_use = (toe_corr_use + toe_corr_use.T) / 2
        # 子谱  对称子谱 与 斜对称子谱
        toe_duichen,toe_xieduichen = toe_zipu(toe_corr_use)
        # if zipu == 1:
        #     toe_corr_use = toe_duichen
        # if zipu == 2:
        #     toe_corr_use = toe_xieduichen
        
        # 全谱
        # scipy.linalg.eig
        eigenvalues, eigenvectors = np.linalg.eigh(toe_duichen)
        # 对称子谱

        eigenvalues_duichen, eigenvectors_duichen = np.linalg.eigh(toe_duichen)
        eigenvalues_xieduichen, eigenvectors_xieduichen = np.linalg.eigh(toe_xieduichen)

        # 特征向量  from zipu to
        eigenvectors_duichen = batch_process_eigenvectors(eigenvectors_duichen, 1)
        eigenvectors_xieduichen = batch_process_eigenvectors(eigenvectors_xieduichen, 2)

        # 对称 特征向量  与 斜对称 特征向量
        eigenvalues_all = np.hstack((eigenvalues_duichen, eigenvalues_xieduichen))  # 一维数组，长度是2m
        eigenvectors_all = np.hstack((eigenvectors_duichen, eigenvectors_xieduichen))
        
        # 对特征值排序，并对特征向量做相应排列
        sorted_indices = np.argsort(eigenvalues_all)  # 升序索引
        eigenvalues_all = eigenvalues_all[sorted_indices]
        eigenvectors_all = eigenvectors_all[:, sorted_indices]
        # 对称 特征向量  与 斜对称 特征向量
        # eigenvalues_all = np.column_stack((eigenvalues_xieduichen, eigenvalues_duichen)).reshape(-1)
        
        # eigenvectors_all = np.column_stack((eigenvectors_xieduichen, eigenvectors_duichen)).reshape(eigenvectors_xieduichen.shape[0], -1, order='F')
    

        #
        # error = 0
        print(f"全谱 ： {(eigenvalues)[:10]}")
        print(f"子谱 ： {(eigenvalues_all)[:10]}")
        print(f"max of 全谱 ： {max(np.sort(eigenvalues))} max of 子谱 ： {max(np.sort(eigenvalues_all))}")
        print(f"min of 全谱 ： {min(np.sort(eigenvalues))} min of 子谱 ： {min(np.sort(eigenvalues_all))}")



        error = sum(abs(i-j) for i,j in zip(np.sort(eigenvalues), np.sort(eigenvalues_all)))
        print('error: ',error)


        E2[:,t] = eigenvalues_all
        E_vectors_toe[t] = eigenvectors_all

        # E2[:,t] = (np.linalg.eigvals(toe_corr_use))
        # eigenvalues, eigenvectors = np.linalg.eigh(error_corr_use)
        # E3[:,t] = (np.linalg.eigvals(error_corr_use))
        
    print('\n')
    return E2,E_vectors_toe
def cal_error_eigenvalue_zipu_average(h, S, dimension, simulation_times, method=0, func=False, constant_a=False, zipu=0):
    """计算 Toeplitz 误差矩阵的平均特征值与平均特征向量"""

    E2_mean = np.zeros(dimension)
    E_vectors_sum = np.zeros((dimension, dimension))

    for t in prange(simulation_times):
        percent = (t + 1) / simulation_times * 100
        sys.stdout.write(f"\r进度 :{percent:.2f}%")
        sys.stdout.flush()

        n, L, start_l = 600000, dimension, 300000
        corr_use, toe_corr_use, error_corr_use = get_tau_lag_1(h, n, S, L, start_l, method)

        Tk = toe_corr_use[:, 0]
        if func:
            tau = range(len(Tk))
            if constant_a:
                alpha, a, Tk_fit = fit_power_decay(h, tau, Tk, constant_a=True)
                print(f"alpha = {alpha}, a = {a}")
            else:
                alpha, Tk_fit = fit_power_decay(h, tau, Tk, constant_a=False)
                print(f"alpha = {alpha}")
            toe_corr_use = toeplitz(Tk_fit)

        toe_corr_use = (toe_corr_use + toe_corr_use.T) / 2

        # 构造对称子谱与斜对称子谱
        toe_duichen, toe_xieduichen = toe_zipu(toe_corr_use)

        # 分别计算两个谱的特征值和特征向量
        eigenvalues_duichen, eigenvectors_duichen = np.linalg.eigh(toe_duichen)
        eigenvalues_xieduichen, eigenvectors_xieduichen = np.linalg.eigh(toe_xieduichen)

        # 处理特征向量
        eigenvectors_duichen = batch_process_eigenvectors(eigenvectors_duichen, func=1)
        eigenvectors_xieduichen = batch_process_eigenvectors(eigenvectors_xieduichen, func=2)

        # 合并谱与特征向量
        eigenvalues_all = np.hstack((eigenvalues_duichen, eigenvalues_xieduichen))
        eigenvectors_all = np.hstack((eigenvectors_duichen, eigenvectors_xieduichen))

        # 打印调试信息
        print(f"全谱 ： {np.sort(np.linalg.eigvalsh(toe_duichen))[:10]}")
        print(f"子谱 ： {np.sort(eigenvalues_all)[:10]}")
        print(f"max of 全谱 ： {max(np.sort(np.linalg.eigvalsh(toe_duichen)))} max of 子谱 ： {max(np.sort(eigenvalues_all))}")
        print(f"min of 全谱 ： {min(np.sort(np.linalg.eigvalsh(toe_duichen)))} min of 子谱 ： {min(np.sort(eigenvalues_all))}")

        error = sum(abs(i - j) for i, j in zip(np.sort(np.linalg.eigvalsh(toe_duichen)), np.sort(eigenvalues_all)))
        print('error: ', error)

        # 累加特征值与特征向量
        E2_mean += eigenvalues_all
        E_vectors_sum += eigenvectors_all

    print('\n')
    E2_mean /= simulation_times
    E_vectors_avg = E_vectors_sum / simulation_times

    return E2_mean, E_vectors_avg

def get_eigenvector_eigenvalue_zipu(H,S,dimension, simulation_times,method=0,func = False,constant_a=False,zipu = 0):
    E_H = {}
    E_vectors_H = {}
    menthof_print = {'p','dot','cos'}
    for h in H:
        print(f"Processing h={h:.2f}")
        E2,E_vectors_toe = cal_error_eigenvalue_zipu(h, S,dimension, simulation_times,method,func,constant_a,zipu = zipu)
        E_H[f"H={h:.2f}"] = E2
        E_vectors_H[f"H={h:.2f}"] = E_vectors_toe
    return E_H, E_vectors_H
def get_eigenvector_eigenvalue_zipu_average(H,S,dimension, simulation_times,method=0,func = False,constant_a=False,zipu = 0):
    E_H = {}
    E_vectors_H = {}
    menthof_print = {'p','dot','cos'}
    for h in H:
        print(f"Processing h={h:.2f}")
        E2,E_vectors_toe = cal_error_eigenvalue_zipu_average(h, S,dimension, simulation_times,method,func,constant_a,zipu = zipu)
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
    