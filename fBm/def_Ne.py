import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_density(vals,bin_size=100,lower = -1,upper = 1.5,print_curvce = False):
    lower = min(vals)

    N = bin_size
    dx = (upper - lower) / float(N)  #带宽
    unit = 1.0 / (float(len(vals)) * dx)
    # lower = 0
    density = dict()   # 存储每个区间内的密度
    for i in range(N):
        lower_bound = lower + i * dx
        upper_bound = lower + (i + 1) * dx
        # 统计当前区间内的数据个数
        count = sum(unit for val in vals if lower_bound <= val < upper_bound)
        
        # 将结果存入字典，key为区间列表
        density[tuple([lower_bound, upper_bound])] = count    
    x,y=[],[]
    s = 0
    for k in range(len(density)):
        key,value = list(density.items())[k]
        # 计算 bim = (left + right) / 2
        left, right = key
        bim = (left + right) / 2
        s += value * (right - left)
        x.append(bim)
        y.append(value)
    if print_curvce:
        print(f"Area under curve = {s}")
    
    #返回密度值
    return x,y


def get_Ne_from_eig_list(list_val,bin_size,lower = -3,upper = 3,scale = False,print_curvce = False): 
    """

    # 返回值
    list_vals: 一维数组，包含所有特征值
    x: 一维数组，包含bin_mindle vals
    y: 一维数组，包含特征值 x 对应的 的密度
    """
    list_val = np.sort(list_val)

    sum_c = np.sum(list_val)
    print (f"sum of eig_vals = {sum_c:.0f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    #缩放
    L = len(list_val)
    if L == 0:
        print("list is empty")
    #缩放
    if scale:
        list_val = list_val / np.sqrt(L)
    if bin_size <= 0:
        raise ValueError("bin size must be positive")

    x,y = get_density(list_val,bin_size,lower,upper,print_curvce)

    return list_val,x,y


def get_Ne_from_eig_matrix(matrix_val,bin_size,lower = -1,upper = 1.5,scale = False,print_curvce=False): 

    eig_vals = matrix_val.flatten()
    eig_vals = np.sort(eig_vals)
    eig_vals = [i-1 for i in eig_vals]
    L = matrix_val.shape[0]
    sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    #缩放
    # if scale:
    #     eig_vals = eig_vals / np.sqrt(L)
    lower = min(eig_vals)
    print(f'lower bound = {lower}')
    # lower = -3
    x,y = get_density(eig_vals,bin_size,-1,upper,print_curvce)

    return eig_vals,x,y
def get_density_2(vals, bin_size=100, lower=None, upper=None, print_curve=False):
    """
    计算数据的概率密度函数
    
    参数:
    vals: 输入数据列表/数组
    bin_size: 直方图的箱子数量
    lower: 直方图下限，如果为None则使用数据最小值
    upper: 直方图上限，如果为None则使用数据最大值
    print_curve: 是否打印曲线下面积信息
    
    返回:
    x: 每个区间的中点
    y: 对应区间的概率密度值
    """
    
    # 如果没有指定边界，使用数据的实际范围
    if lower is None:
        lower = np.min(vals)
    if upper is None:
        upper = np.max(vals)
    
    N = bin_size
    dx = (upper - lower) / float(N)  # 每个箱子的宽度
    
    # 方法1: 使用numpy的直方图函数（推荐）
    # 将数据值限制在[lower, upper]范围内
    vals_clipped = np.clip(vals, lower, upper)
    
    # 计算直方图
    counts, bin_edges = np.histogram(vals_clipped, bins=N, range=(lower, upper), density=False)
    
    # 转换为概率密度（总面积=1）
    probability_density = counts / (len(vals) * dx)
    
    # 计算每个箱子的中点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算曲线下面积
    area = np.sum(probability_density * dx)
    
    if print_curve:
        print(f"Area under probability density curve = {area:.6f}")
        print(f"Expected area for PDF = 1.0")
        print(f"Data range: [{lower:.4f}, {upper:.4f}]")
        print(f"Bin width: {dx:.6f}")
    
    return bin_centers.tolist(), probability_density.tolist()
def get_density_3(vals, bin_size=100, lower=None, upper=None, print_sum=False):
    """
    计算数据在每个区间的概率分布（概率之和为1）
    
    参数:
    vals: 输入数据列表/数组
    bin_size: 直方图的箱子数量
    lower: 直方图下限，如果为None则使用数据最小值
    upper: 直方图上限，如果为None则使用数据最大值
    print_sum: 是否打印概率总和信息
    
    返回:
    x: 每个区间的中点
    y: 对应区间的概率值（概率之和为1）
    """
    
    # 如果没有指定边界，使用数据的实际范围
    if lower is None:
        lower = np.min(vals)
    if upper is None:
        upper = np.max(vals)
    
    N = bin_size
    dx = (upper - lower) / float(N)  # 每个箱子的宽度
    
    # 将数据值限制在[lower, upper]范围内
    vals_clipped = np.clip(vals, lower, upper)
    
    # 计算直方图（统计每个区间的数据点个数）
    counts, bin_edges = np.histogram(vals_clipped, bins=N, range=(lower, upper), density=False)
    
    # 转换为概率：每个区间的点数除以总点数
    probabilities = counts / len(vals)
    
    # 计算每个箱子的中点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算概率总和（应该等于1）
    total_prob = np.sum(probabilities)
    
    if print_sum:
        print(f"Total probability = {total_prob:.6f}")
        print(f"Expected total = 1.0")
        print(f"Data range: [{lower:.4f}, {upper:.4f}]")
        print(f"Number of bins: {N}")
        print(f"Bin width: {dx:.6f}")
        print(f"Total data points: {len(vals)}")
        # 显示每个区间的概率范围
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities[np.nonzero(probabilities)])
        print(f"Max probability in a bin: {max_prob:.6f}")
        print(f"Min non-zero probability: {min_prob:.6f}")
    
    return bin_centers.tolist(), probabilities.tolist()
def get_Ne_from_eig_matrix_2(matrix_val,bin_size,lower = -1,upper = 1.5,scale = False,print_curvce=False): 

    eig_vals = matrix_val.flatten()
    eig_vals = np.sort(eig_vals)
    eig_vals = [i-1 for i in eig_vals]
    L = matrix_val.shape[0]
    sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    #缩放
    # if scale:
    #     eig_vals = eig_vals / np.sqrt(L)
    lower = min(eig_vals)
    print(f'lower bound = {lower}')
    # lower = -3
    x,y = get_density_3(eig_vals,bin_size,lower,upper,print_curvce)

    return eig_vals,x,y