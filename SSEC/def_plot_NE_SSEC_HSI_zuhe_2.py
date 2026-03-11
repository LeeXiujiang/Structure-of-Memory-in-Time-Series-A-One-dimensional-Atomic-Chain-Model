import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from typing import List
import seaborn as sns
from scipy.stats import beta, skew
#add beta distribution fit
def fit_beta_from_data(data, bins=100, lower=0,upper=4):
    """
    拟合 Beta 分布，绘制拟合曲线，并输出偏度

    Parameters
    ----------
    data : array-like
        输入序列（一维）
    bins : int
        直方图分箱数
    lower : int/float
        下界
    upper : int/float
        上界
    Returns
    -------
    params : dict
        Beta 分布参数与偏度信息
    """
    data = np.asarray(data, dtype=float)
    data = data[(data > lower) & (data <= upper)]

    

    # ---------- 1. 原始范围
    s_min, s_max = data.min(), data.max()
    eps = 1e-6

    # ---------- 2. 映射到 (0,1)
    x = (data - s_min) / (s_max - s_min)
    x = np.clip(x, eps, 1 - eps)

    # ---------- 3. Beta 拟合
    a, b, _, _ = beta.fit(x, floc=0, fscale=1)

    # ---------- 4. 原始变量下的 Beta 拟合曲线
    s_plot = np.linspace(s_min, s_max, 500)
    x_plot = (s_plot - s_min) / (s_max - s_min)

    pdf_original = beta.pdf(x_plot, a, b) / (s_max - s_min)

    # ---------------------------
    # 3. 理论 Beta 偏度
    # ---------------------------
    beta_skew = (
        2 * (b - a) * np.sqrt(a + b + 1)
        / ((a + b + 2) * np.sqrt(a * b))
    )

    # ---------------------------
    # 4. 样本偏度
    # ---------------------------
    sample_skew = skew(x)



    # ---------------------------
    # 6. 输出结果
    # ---------------------------
    params = {
        "alpha": a,
        "beta": b,
        "sample_skew": sample_skew,
        "beta_theoretical_skew": beta_skew
    }

    print("Beta fit results:")
    print(f"  alpha = {a:.4f}")
    print(f"  beta  = {b:.4f}")
    print(f"  sample skewness       = {sample_skew:.4f}")
    print(f"  beta theoretical skew = {beta_skew:.4f}")

    return params, s_plot, pdf_original,beta_skew




#源代码
def get_density(vals, bin_size=100, lower=-3, upper=4, print_curvce=False):
    """
    计算数据的密度分布
    
    参数:
    vals: 输入数据
    bin_size: 分箱数量
    lower: 下限
    upper: 上限
    print_curvce: 是否打印曲线下面积
    
    返回:
    x: 分箱中心值
    y: 密度值
    """
    N = bin_size
    # lower = 0
    dx = (upper - lower) / float(N)  # 带宽
    unit = 1.0 / (float(len(vals)) * dx)
    density = dict()   # 存储每个区间内的密度
    
    for i in range(N):
        lower_bound = lower + i * dx
        upper_bound = lower + (i + 1) * dx
        # 统计当前区间内的数据个数
        count = sum(unit for val in vals if lower_bound <= val < upper_bound)
        
        # 将结果存入字典，key为区间列表
        density[tuple([lower_bound, upper_bound])] = count
    
    x, y = [], []
    s = 0
    for k in range(len(density)):
        key, value = list(density.items())[k]
        # 计算 bim = (left + right) / 2
        left, right = key
        bim = (left + right) / 2
        s += value * (right - left)
        x.append(bim)
        y.append(value)
    
    if print_curvce:
        print(f"Area under curve = {s}")
    
    # 返回密度值
    return x, y

def get_Ne_from_eig_list(list_val, bin_size, lower=-3, upper=3, scale=False, print_curvce=False):
    """
    从特征值列表计算谱密度
    
    参数:
    list_val: 一维数组，包含所有特征值
    bin_size: 分箱数量
    lower: 下限
    upper: 上限
    scale: 是否缩放
    print_curvce: 是否打印曲线下面积
    
    返回:
    list_val: 排序后的特征值
    x: 一维数组，包含bin_middle vals
    y: 一维数组，包含特征值x对应的密度
    """
    # list_val = np.sort(list_val)

    sum_c = np.sum(list_val)
    L = len(list_val)
    
    if L == 0:
        print("list is empty")
    
    # 缩放
    if scale:
        list_val = list_val / np.sqrt(L)
    
    if bin_size <= 0:
        raise ValueError("bin size must be positive")

    x, y = get_density(list_val, bin_size, lower, upper, print_curvce)

    return list_val, x, y

def get_Ne_from_eig_matrix(matrix_val, bin_size, lower=0, upper=3, move = 0,scale=False, print_curvce=False):
    """
    从特征值矩阵计算谱密度
    
    参数:
    matrix_val: 特征值矩阵
    bin_size: 分箱数量
    lower: 下限
    upper: 上限
    scale: 是否缩放
    print_curvce: 是否打印曲线下面积
    
    返回:
    eig_vals: 展平后的特征值
    x: 分箱中心值
    y: 密度值
    """
    eig_vals = []
    for i in range(matrix_val.shape[1]):
        a = matrix_val[:, i]
        a = a[a > 0]
        eig_vals.extend(a)
    
    #特征值平移
    if move != 0:
        # eig_vals = eig_vals + move
        eig_vals = [x + move for x in eig_vals]
        upper = upper + move
    # eig_vals = np.sort(eig_vals)
    L = matrix_val.shape[0]
    # sum_c = np.sum(eig_vals) / matrix_val.shape[1]
    
    # 缩放
    if scale:
        eig_vals = eig_vals / np.sqrt(L)
    
    lower = min(eig_vals)

    x, y = get_density(eig_vals, bin_size, lower, upper, print_curvce)

    return eig_vals, x, y

def find_subplot_layout(T):
    """
    计算子图布局
    
    参数:
    T: 子图数量
    
    返回:
    m, n: 行数和列数
    """
    if T == 1:
        return 1, 1
    m = math.ceil(math.sqrt(T))
    n = math.ceil(T / m)
    # 尝试使 m 和 n 更接近
    while abs(m - n) > 1 and (m - 1) * n >= T:
        m -= 1
    return m, n

def generate_gradient_two_colors(colorA, colorB, N=256):
    """
    生成 colorA 到 colorB 的渐变颜色列表（仅支持 2 色）
    
    参数:
    colorA: 起始颜色（名称或十六进制，如 'blue' 或 '#0000FF'）
    colorB: 结束颜色
    N: 返回的颜色数量
    
    返回:
    渐变颜色列表，格式为 [(R1,G1,B1), (R2,G2,B2), ...]
    """
    # 将颜色名称转换为 RGB（范围 0-1）
    rgbA = mcolors.to_rgb(colorA)
    rgbB = mcolors.to_rgb(colorB)
    
    # 线性插值
    gradient = np.linspace(0, 1, N)
    colors = []
    for t in gradient:
        r = rgbA[0] + (rgbB[0] - rgbA[0]) * t
        g = rgbA[1] + (rgbB[1] - rgbA[1]) * t
        b = rgbA[2] + (rgbB[2] - rgbA[2]) * t
        colors.append((r, g, b))
    return colors

def generate_gradient_seaborn(color_list, N=256):
    """
    使用 Seaborn 生成渐变颜色
    
    参数:
    color_list: 颜色列表（支持名称或十六进制）
    N: 颜色数量
    
    返回:
    RGB 元组列表
    """
    palette = sns.color_palette(color_list, n_colors=N)
    return palette

def plot_Ne(plot_eig, Ne, alpha_1_list, skew_dict,plot_one=True,ax=None):
    """
    绘制谱密度图
    
    参数:
    plot_eig: 特征值数据
    Ne: 谱密度数据
    alpha_1_list: 参数列表
    color_gradient: 是否使用渐变色
    plot_one: 是否在单个图中绘制
    """
    scale_change = 2.54
    scale_change_2 = 2.54*2.54
    len_H = len(alpha_1_list)
    list_year_end = [f"{i}" for i in range(2019, 2025, 1)]
    # skew_plot = skew_dict.value()
    # skew_plot = skew_dict.values()
    skew_plot = list(skew_dict.values())

    # colors = plt.cm.get_cmap('coolwarm', (len_H-1))
    # color_list = [colors(i) for i in range(len_H)]
    # color_list = color_list[-len_H:]
    # # 定义颜色渐变范围（从蓝色到红色）
    colors = plt.cm.get_cmap('coolwarm', len_H)  # 'coolwarm' 是 matplotlib 内置的渐变色
    color_list = [colors(i) for i in range(len_H)]
    if plot_one:
        # 若未提供外部子图，则新建 figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        for i, alpha_1 in enumerate(alpha_1_list):
            color = color_list[i]
            # ax.plot(plot_eig[f"alpha_1={alpha_1:.2f}"], 
            #         Ne[f"alpha_1={alpha_1:.2f}"], 
            #         c=color)
            ax.scatter(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                       Ne[f"alpha_1={alpha_1:.2f}"], 
                       c=[color], 
                       s=20/scale_change_2, 
                       label=fr"$Year$ = {list_year_end[i]}, <Skew>={skew_plot[i]:.2f}")
        
        # ax.set_xlabel(r"$\lambda$",fontsize =12)
        # ax.set_ylabel(r'$\bar{\rho}(\lambda)$',fontsize =12)
        # ax.legend(frameon=False, fontsize=8)
        # ax.set_title("Spectral density", fontsize=12)
        
        if ax is None:
            plt.tight_layout()
            plt.show()

        
        # ax.set_xlabel(r"$\lambda$")
        # ax.set_ylabel(r"$\bar{\rho(\lambda)}$")
        # ax.legend(frameon=False, fontsize=9)
        # ax.set_title("Spectral density", fontsize=12)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    else:
        plt.close('all')
        m, n = find_subplot_layout(len(alpha_1_list))
        fig, axes = plt.subplots(m, n, figsize=(10, 8))
        axes = axes.flatten()
        for i, alpha_1 in enumerate(alpha_1_list):
            ax = axes[i]
            ax.plot(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                    Ne[f"alpha_1={alpha_1:.2f}"], 
                    c=color_list[i])
            ax.scatter(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                       Ne[f"alpha_1={alpha_1:.2f}"], 
                       c=[color_list[i]], s=20/scale_change_2)
            ax.set_title(fr"$Year$ = {list_year_end[i]}")
        plt.tight_layout()
        plt.show()
def plot_Beat_fit(plot_eig, Ne, alpha_1_list,skew, plot_one=True,ax=None):
    """
    绘制谱密度图
    
    参数:
    plot_eig: 特征值数据
    Ne: 谱密度数据
    alpha_1_list: 参数列表
    color_gradient: 是否使用渐变色
    plot_one: 是否在单个图中绘制
    """
    scale_change = 2.54
    scale_change_2 = 2.54*2.54
    len_H = len(alpha_1_list)
    list_year_end = [f"{i}" for i in range(2019, 2025, 1)]
    # colors = plt.cm.get_cmap('coolwarm', (len_H-1))
    # color_list = [colors(i) for i in range(len_H)]
    # color_list = color_list[-len_H:]
    # # 定义颜色渐变范围（从蓝色到红色）
    colors = plt.cm.get_cmap('coolwarm', len_H)  # 'coolwarm' 是 matplotlib 内置的渐变色
    color_list = [colors(i) for i in range(len_H)]
    if plot_one:
        # 若未提供外部子图，则新建 figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        for i, alpha_1 in enumerate(alpha_1_list):
            color = color_list[i]
            ax.plot(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                    Ne[f"alpha_1={alpha_1:.2f}"], 
                    c=color,linewidth=2/scale_change)
            # ax.scatter(plot_eig[f"alpha_1={alpha_1:.2f}"], 
            #            Ne[f"alpha_1={alpha_1:.2f}"], 
            #            c=[color], 
            #            s=10, 
            #            label=fr"$Year$ = {list_year_end[i]}")
        #刻度尺与标签
        ax.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
        ax.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
        ax.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.set_xlabel(r"$\lambda$",fontsize =12)
        ax.set_ylabel(r'$\bar{\rho}(\lambda)$',fontsize =12)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(right=4.5)
        # ax.set_title("Spectral density", fontsize=12)
        
        if ax is None:
            plt.tight_layout()
            plt.show()

        
        # ax.set_xlabel(r"$\lambda$")
        # ax.set_ylabel(r"$\bar{\rho(\lambda)}$")
        # ax.legend(frameon=False, fontsize=9)
        # ax.set_title("Spectral density", fontsize=12)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    else:
        plt.close('all')
        m, n = find_subplot_layout(len(alpha_1_list))
        fig, axes = plt.subplots(m, n, figsize=(10, 8))
        axes = axes.flatten()
        for i, alpha_1 in enumerate(alpha_1_list):
            ax = axes[i]
            ax.plot(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                    Ne[f"alpha_1={alpha_1:.2f}"], 
                    c=color_list[i])
            ax.scatter(plot_eig[f"alpha_1={alpha_1:.2f}"], 
                       Ne[f"alpha_1={alpha_1:.2f}"], 
                       c=[color_list[i]], s=10)
            ax.set_title(fr"$Year$ = {list_year_end[i]}")
        plt.tight_layout()
        plt.show()

def read_data_SSEC():
    """
    读取数据
    
    返回:
    alpha_1_list: 参数列表
    eig_vals: 特征值字典
    """
    eig_vals = {}
    
    # SSEC 数据配置
    list_year_start = ['2008', '2010', '2012', '2014', '2016', '2018', '2020', '2022']
    list_year_end = ['2009', '2011', '2013', '2015', '2017', '2019', '2021', '2023']
    
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    
    # 或者使用另一组年份
    list_year_start = [f"{i}" for i in range(2019, 2025, 1)]
    list_year_end = [f"{i}" for i in range(2019, 2025, 1)]
    
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    
    for i, alhpa_1 in enumerate(alpha_1_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_SSEC_{start_}_{end_}_50000.csv"
        
        try:
            df = pd.read_csv(file_name, header=None)
            eig_matrix = df.values
            eig_vals[f"alpha_1={alhpa_1:.2f}"] = eig_matrix
        except FileNotFoundError:
            print(f"文件未找到: {file_name}")
            # 创建模拟数据用于演示
            np.random.seed(42)
            eig_matrix = np.random.rand(100, 50)  # 模拟数据
            eig_vals[f"alpha_1={alhpa_1:.2f}"] = eig_matrix
    
    data_out = [f"{list_start_time[i]}--{list_end_time[i]}" for i in range(L)]
    
    return alpha_1_list, eig_vals
def read_data_HSI():
    """
    读取数据
    
    返回:
    alpha_1_list: 参数列表
    eig_vals: 特征值字典
    """
    eig_vals = {}
    
    # SSEC 数据配置
    list_year_start = ['2008', '2010', '2012', '2014', '2016', '2018', '2020', '2022']
    list_year_end = ['2009', '2011', '2013', '2015', '2017', '2019', '2021', '2023']
    
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    
    # 或者使用另一组年份
    list_year_start = [f"{i}" for i in range(2019, 2025, 1)]
    list_year_end = [f"{i}" for i in range(2019, 2025, 1)]
    
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    
    for i, alhpa_1 in enumerate(alpha_1_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_HSI_{start_}_{end_}_72000.csv"
        
        try:
            df = pd.read_csv(file_name, header=None)
            eig_matrix = df.values
            eig_vals[f"alpha_1={alhpa_1:.2f}"] = eig_matrix
        except FileNotFoundError:
            print(f"文件未找到: {file_name}")
            # 创建模拟数据用于演示
            np.random.seed(42)
            eig_matrix = np.random.rand(100, 50)  # 模拟数据
            eig_vals[f"alpha_1={alhpa_1:.2f}"] = eig_matrix
    
    data_out = [f"{list_start_time[i]}--{list_end_time[i]}" for i in range(L)]
    
    return alpha_1_list, eig_vals
def create_Ne(alpha_1_list, eig_vals,move=0):
    """
    计算谱密度
    
    参数:
    alpha_1_list: 参数列表
    eig_vals: 特征值字典
    
    返回:
    plot_eig: 绘图用的特征值
    Ne: 谱密度值
    alpha_1_list: 参数列表
    """
    plot_eig = {}
    Ne = {}
    move_list = [-1,-0.6,-0.2,0.2,0.6,1]
    for i, alpha_1 in enumerate(alpha_1_list):
        eig_ = eig_vals[f"alpha_1={alpha_1:.2f}"]

        _, x, y = get_Ne_from_eig_matrix(eig_, bin_size=60, upper=3, move= move_list[i],scale=False, print_curvce=True)
        plot_eig[f"alpha_1={alpha_1:.2f}"] = x
        Ne[f"alpha_1={alpha_1:.2f}"] = y
    
    return plot_eig, Ne, alpha_1_list
def get_beta_dis_from_eig_matrix(matrix_val, bin_size, lower=0, upper=3, move = 0,scale=False, print_curvce=False):
    """
    从特征值矩阵计算谱密度
    
    参数:
    matrix_val: 特征值矩阵
    bin_size: 分箱数量
    lower: 下限
    upper: 上限
    scale: 是否缩放
    print_curvce: 是否打印曲线下面积
    
    返回:
    eig_vals: 展平后的特征值
    x: 分箱中心值
    y: 密度值
    """
    eig_vals = []
    for i in range(matrix_val.shape[1]):
        a = matrix_val[:, i]
        a = a[a > 0]
        eig_vals.extend(a)
    
    #特征值平移
    if move != 0:
        # eig_vals = eig_vals + move
        eig_vals = [x + move for x in eig_vals]
        upper = upper + move
    # eig_vals = np.sort(eig_vals)
    L = matrix_val.shape[0]
    # sum_c = np.sum(eig_vals) / matrix_val.shape[1]
    
    # 缩放
    if scale:
        eig_vals = eig_vals / np.sqrt(L)
    
    lower = min(eig_vals)
    params, s_plot, pdf_original,beta_skew = fit_beta_from_data(eig_vals,bin_size,lower,upper)
    # x, y = get_density(eig_vals, bin_size, lower, upper, print_curvce)

    return params, s_plot, pdf_original,beta_skew
def cerat_beta_dis_SSEC(alpha_1_list, eig_vals):
    """
    计算beta fenbu
    
    参数:
    alpha_1_list: 参数列表
    eig_vals: 特征值字典
    
    返回:
    plot_eig: 绘图用的特征值
    Ne: 谱密度值
    alpha_1_list: 参数列表
    """
    plot_eig = {}
    Y = {}
    params_dict = {}
    skew_dict = {}
    move_list = [-1,-0.6,-0.2,0.2,0.6,1]
    for i, alpha_1 in enumerate(alpha_1_list):
        eig_ = eig_vals[f"alpha_1={alpha_1:.2f}"]
        params, s_plot, pdf_original,beta_skew = get_beta_dis_from_eig_matrix(eig_, bin_size=60, upper=3, move= move_list[i],scale=False, print_curvce=True)
        # _, x, y = get_Ne_from_eig_matrix(eig_, bin_size=60, upper=3, move= move_list[i],scale=False, print_curvce=True)
        plot_eig[f"alpha_1={alpha_1:.2f}"] = s_plot
        Y[f"alpha_1={alpha_1:.2f}"] = pdf_original
        skew_dict[f"alpha_1={alpha_1:.2f}"] = beta_skew
        params_dict[f"alpha_1={alpha_1:.2f}"] = params
    
    return plot_eig, Y, alpha_1_list,params_dict,skew_dict
def spectrum_density_plot_SSEC(ax=None):
    """
    主函数：绘制谱密度图
    """
    scale_change = 2.54
    scale_change_2 = 2.54*2.54
    alpha_1_list, eig_vals = read_data_SSEC()
    plot_eig, Ne, alpha_1_list = create_Ne(alpha_1_list, eig_vals)
    plot_eig_2, Y, alpha_1_list,params_dict,skew_dict = cerat_beta_dis_SSEC(alpha_1_list, eig_vals)
    # print
    plot_Ne(plot_eig, Ne, alpha_1_list, skew_dict,plot_one=True,ax=ax)

    # plot_eig, Y, alpha_1_list,params_dict,skew_dict = cerat_beta_dis_SSEC(alpha_1_list, eig_vals)
    plot_Beat_fit(plot_eig_2, Y, alpha_1_list, skew_dict, plot_one=True,ax=ax)
def spectrum_density_plot_HSI(ax=None):
    """
    主函数：绘制谱密度图
    """
    alpha_1_list, eig_vals = read_data_HSI()
    plot_eig, Ne, alpha_1_list = create_Ne(alpha_1_list, eig_vals)
    plot_eig_2, Y, alpha_1_list,params_dict,skew_dict = cerat_beta_dis_SSEC(alpha_1_list, eig_vals)
    plot_Ne(plot_eig, Ne, alpha_1_list, skew_dict,plot_one=True,ax=ax)

    
    plot_Beat_fit(plot_eig_2, Y, alpha_1_list, skew_dict, plot_one=True,ax=ax)