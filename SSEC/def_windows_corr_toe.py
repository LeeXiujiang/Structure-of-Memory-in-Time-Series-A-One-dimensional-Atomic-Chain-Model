from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np


#
def operation1_vectorized(log_returns_clean, L,ts = False):
    """
    第一个操作的高效向量化版本
    """
    N = len(log_returns_clean)
    if ts:
        data = log_returns_clean.values
    
    else:
        data = log_returns_clean
    
    # 使用滑动窗口视图（避免内存复制）
    
    windows = sliding_window_view(data, L)  # shape: (N-L+1, L)
    
    # 标准化数据
    windows_standardized = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)
    
    # 计算相关系数矩阵
    corr_matrix = abs(windows_standardized @ windows_standardized.T / L)
    
    return (corr_matrix + corr_matrix.T)/2

def operation2_vectorized(log_returns_clean, L,ts = False):
    """
    第二个操作的高效向量化版本
    """
    if ts:
        data = log_returns_clean.values
    else:
        data = log_returns_clean
    N = len(log_returns_clean)
    circular_data = np.concatenate([data, data])
    
    # 使用滑动窗口视图
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(circular_data, L)  # shape: (2N-L+1, L)
    
    # 标准化数据
    windows_standardized = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)
    
    # 计算相关系数矩阵
    corr_matrix = abs( windows_standardized @ windows_standardized.T / L)
    
    return (corr_matrix + corr_matrix.T)/2

from scipy.linalg import toeplitz

def toe_and_random_from_matrix(matrix):
    """
    简洁版本：从矩阵斜行平均创建Toeplitz矩阵
    """
    m, n = matrix.shape
    
    # 收集所有对角线的平均值
    diag_means = []
    for k in range(-m+1, n):
        diag = np.diag(matrix, k)
        if len(diag) > 0:
            diag_means.append(np.mean(diag))
    
    # 构建Toeplitz矩阵
    size = len(diag_means)
    center = size // 2
    
    # 第一列（从中心向下）
    first_col = diag_means[center:]
    # 第一行（从中心向左，反转）
    first_row = diag_means[center::-1]
    toeplitz_matrix = toeplitz(first_col, first_row)
    random_matrix = matrix - toeplitz_matrix
    return toeplitz_matrix, random_matrix