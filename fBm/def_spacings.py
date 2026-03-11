import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from def_unfolding_menthod import *
from typing import List  


def unfolding_method_choose(eig_vals,menthod= 'Poly',degree = 5):
    #特征值展开 方法选择
    if menthod == 'Poly':
        eig_unfloded,stps,func = unflod_poly(eig_vals,degree=5)

    if menthod == 'Spline':
        eig_unfloded,stps,func = unflod_spline(eig_vals)
   
    if menthod == 'Gaussian':
        eig_unfloded,stps = unflod_gaussian(eig_vals)
    
    if menthod == 'Exp':
        eig_unfloded,stps = unflod_exp_smooth(eig_vals)
    return eig_unfloded
    
def get_spacings_from_eig_list(list_val,n = 0,menthod= 'Poly',degree = 5,percent = True,lower = 0.2,upper = 0.8,average_spacing = True,print_curvce = False): 
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
    
    
    if percent:
        list_val = list_val[int(L*lower): int(L*upper)]
    #特征值 展开
    eig_unfloded = unfolding_method_choose(list_val,menthod=menthod,degree=degree)
    eig_unfloded = np.sort(eig_unfloded)

    spacings = []
    for i in range(L-n-1):
        spacing = eig_unfloded[i+(n+1)] - eig_unfloded[i]
        spacings.append(spacing)
    if average_spacing:
        avg_spacing = np.mean(spacings)
        spacings = spacings / avg_spacing
    #缩放
    # if scale:
    #     list_val = list_val / np.sqrt(L)
    # if bin_size <= 0:
    #     raise ValueError("bin size must be positive")



    return spacings


def get_spacings_from_eig_matrix(matrix_val,n=0,menthod = 'Ploy',degree = 5,percent = True,lower = 0.2,upper = 0.8,average_sapings = True,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    spacings = []
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded = unfolding_method_choose(eig_val,menthod=menthod,degree=degree)
        eig_unfloded = np.sort(eig_unfloded)
        L = len(eig_unfloded)
        if percent:
            eig_unfloded = eig_unfloded[int(L*lower): int(L*upper)]
        L = len(eig_unfloded)
        spacing = [eig_unfloded[i+(n+1)] - eig_unfloded[i] for i in range(L-n-1)]
        if average_sapings:
            avg_spacing = np.mean(spacing)
            print(f"min of spacing = {np.min(spacing):.4f},max of spacing = {np.max(spacing):.4f}")
            print(f"avg_spacing = {avg_spacing:.4f}")
            spacing = spacing / avg_spacing
            print(f"averahe lafter : min of spacing = {np.min(spacing):.4f},max of spacing = {np.max(spacing):.4f}")

        spacings.extend(spacing)

    

    return spacings
from def_unfolding_menthod import Li_fit
from numpy.polynomial import Polynomial
def unfold_spectrum(eigenvalues, deg=5):
    """对升序排列的特征值进行 unfolding"""
    eigenvalues = np.sort(eigenvalues)
    N = np.arange(1, len(eigenvalues) + 1)
    
    # 多项式拟合经验累计分布
    p = Polynomial.fit(eigenvalues, N, deg)
    unfolded = p(eigenvalues)  # 得到 unfolded 值 xi_i = N̄(λ_i)
    
    return unfolded
def get_spacings_from_eig_matrix_fit(matrix_val,n=0,method = 'Ploy',degree = 5,percent = True,lower = 0.2,upper = 0.8,average_sapings = True,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    # unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    spacings = []
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        # eig_unfloded,_,_ = Li_fit(eig_val,method=method,degree=degree)
        eig_unfloded = unfold_spectrum(eig_val,deg=degree)
        # eig_unfloded = 
        if percent:
            eig_unfloded = eig_unfloded[int(L*lower): int(L*upper)]
        L = len(eig_unfloded)
        eig_unfloded = np.sort(eig_unfloded)
        spacing = [eig_unfloded[i+(n+1)] - eig_unfloded[i] for i in range(L-n-1)]
        if average_sapings:
            avg_spacing = np.mean(spacing)
            print(f"min of spacing = {np.min(spacing):.4f},max of spacing = {np.max(spacing):.4f}")
            print(f"avg_spacing = {avg_spacing:.4f}")
            spacing = spacing / avg_spacing
            print(f"averahe lafter : min of spacing = {np.min(spacing):.4f},max of spacing = {np.max(spacing):.4f}")
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        spacings.extend(spacing)
        

        # unfloded.extend(eig_unfloded)

    

    return spacings