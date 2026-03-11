import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from def_unfolding_menthod import *
def get_density(vals,bin_size=100,lower = -3,upper = 3,print_curvce = False):
    
    N = bin_size
    dx = (upper - lower) / float(N)  #带宽
    unit = 1.0 / (float(len(vals)) * dx)

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


def get_spacings_from_eig_list(list_val,bin_size,n = 0,lower = -3,upper = 3,scale = False,print_curvce = False): 
    """

    # 返回值
    list_vals: 一维数组，包含所有特征值
    x: 一维数组，包含bin_mindle vals
    y: 一维数组，包含特征值 x 对应的 的密度
    """
    list_val = np.sort(list_val)

    # sum_c = np.sum(list_val)
    # print (f"sum of eig_vals = {sum_c:.0f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    #缩放
    L = len(list_val)
    if L == 0:
        print("list is empty")
    #缩放
    # if scale:
    #     list_val = list_val / np.sqrt(L)
    # if bin_size <= 0:
    #     raise ValueError("bin size must be positive")

    x,y = get_density(list_val,bin_size,lower,upper,print_curvce)

    return list_val,x,y


def get_spacings_from_eig_matrix(matrix_val,bin_size,n=0,menthod = 'Ploy',lower = 0,upper = 3,print_curvce=False): 

    eig_vals = matrix_val.flatten()
    eig_vals = np.sort(eig_vals)

    #特征值展开

    L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    spacings = []
    for i in range(num_of_system):
        eig_val =eig_vals,matrix_val[:,i]
        spacing = [eig_val[i+(n+1)] - eig_val[i] for i in range(L-n-1)]
        spacings.append(spacing)

    x,y = get_density(eig_vals,bin_size,lower,upper,print_curvce)

    return eig_vals,x,y