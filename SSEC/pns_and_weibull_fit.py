import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial import Polynomial
import os
from def_spacings import *
# from def_spacings import *
from def_Ne import *
# from def_nnsd import find_subplot_layout
def read_data():
    eig_vals = {} 
    # H = np.arange(0.50,0.901,0.05)

    alpha_1_list = np.arange(0.1,0.901,0.1)
    alpha_1_list = [0.1,0.4,0.7,0.9]
    year_list = [2019,2020,2021,2022,2023,2024]
    dimension ,simulation_times = 2000 , 100
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_SSEC_{str(year_)}_{year_}_50000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        eig_vals[f"year={year_}"] = eig_matrix
    
    #THEORY
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='toe')

    return alpha_1_list,eig_vals


def create_spacings(year_list,eig_vals,n=0):

    spaings_dict = {}

    for i,year_ in enumerate(year_list):

        eig_ = eig_vals[f"year={year_}"]
        spaings = get_spacings_from_eig_matrix_fit(eig_,n,method='Poly',degree=10,percent=False,average_sapings=True)

        spaings_dict[f"year={year_}"] = spaings


    return spaings_dict,alpha_1_list

alpha_1_list, eig_vals = read_data()
from numpy.polynomial import Polynomial

def unfold_spectrum(eigenvalues, deg=5):
    """对升序排列的特征值进行 unfolding"""
    eigenvalues = np.sort(eigenvalues)
    N = np.arange(1, len(eigenvalues) + 1)
    
    # 多项式拟合经验累计分布
    p = Polynomial.fit(eigenvalues, N, deg)
    unfolded = p(eigenvalues)  # 得到 unfolded 值 xi_i = N̄(λ_i)
    
    return unfolded
def generate_eigenvalues(matrix_size, num_matrices):
    # eigenvalues = []
    matrix_out = np.zeros((matrix_size, num_matrices))
    for i in range(num_matrices):
        M = np.random.randn(matrix_size, matrix_size)  # 随机矩阵非对角元素均值0方差1
        np.fill_diagonal(M, 0)  # 对角元素设置为0
        M = (M + M.T) /2  # 确保矩阵对称
        eigvals = np.linalg.eigvalsh(M)  # 计算特征值
        # 
        matrix_out[:, i] = eigvals
        # eigenvalues.append(np.sort(eigvals))  # 确保特征值排序
    return matrix_out  # 每行为一个矩阵的特征值
def normalize_and_compute_Pn(vals, n, num_bins=50):
    """
    将特征值序列归一化为单位密度，并计算 P_n(s) 的概率密度函数。
    
    参数:
        vals (list or numpy array): 原始特征值序列。
        n (int): 间隔中包含的特征值个数。
        num_bins (int): 用于直方图的分区数量（默认50）。
    
    返回:
        normalized_vals (numpy array): 调整后的单位密度特征值序列。
        s_vals (numpy array): 间隔值 (s) 的中心点。
        Pn (numpy array): 概率密度值。
    """
    # Step 1: 归一化特征值到单位密度
    vals = np.sort(vals)  # 确保顺序
    N = len(vals)
    min_val, max_val = np.min(vals), np.max(vals)
    normalized_vals = (vals - min_val) / (max_val - min_val) * N  # 归一化
    print(sum(normalized_vals))
    # Step 2: 计算 P_n(s)
    intervals = []  # 存储满足条件的间隔
    for i in range(N):
        for j in range(i + 1, N):
            s = normalized_vals[j] - normalized_vals[i]  # 间隔长度
            count = np.sum((normalized_vals > normalized_vals[i]) & (normalized_vals < normalized_vals[j]))
            if count == n:  # 确保间隔内有 n 个特征值
                intervals.append(s)
    
    # 构建直方图，统计 P_n(s)
    hist, bin_edges = np.histogram(intervals, bins=num_bins, range=(0, 8),density=True)
    s_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 每个区间的中心点
    print(sum(hist))
    print(np.dot(s_vals,hist))
    return normalized_vals, s_vals, hist
def normalize_Pns(vals, n, num_bins=50):
    """
    将特征值序列归一化为单位密度，并计算 P_n(s) 的概率密度函数。
    
    参数:
        vals (list or numpy array): 原始特征值序列。
        n (int): 间隔中包含的特征值个数。
        num_bins (int): 用于直方图的分区数量（默认50）。
    
    返回:
        normalized_vals (numpy array): 调整后的单位密度特征值序列。
        s_vals (numpy array): 间隔值 (s) 的中心点。
        Pn (numpy array): 概率密度值。
    """
    # Step 1: 归一化特征值到单位密度
    vals = np.sort(vals)  # 确保顺序
    N = len(vals)
    min_val, max_val = np.min(vals), np.max(vals)
    normalized_vals = (vals - min_val) / (max_val - min_val) * N  # 归一化
    # print(sum(normalized_vals))
    # Step 2: 计算 P_n(s)
    intervals = []  # 存储满足条件的间隔
    for i in range(len(normalized_vals)-n-1):
        s = normalized_vals[i+n+1] - normalized_vals[i]  # 间隔长度
        intervals.append(s)
    # for i in range(N):
    #     for j in range(i + 1, N):
    #         s = normalized_vals[j] - normalized_vals[i]  # 间隔长度
    #         count = np.sum((normalized_vals > normalized_vals[i]) & (normalized_vals < normalized_vals[j]))
    #         if count == n:  # 确保间隔内有 n 个特征值
    #             intervals.append(s)
    
    # 构建直方图，统计 P_n(s)
    hist, bin_edges = np.histogram(intervals, bins=num_bins, range=(0, 8),density=True)
    s_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 每个区间的中心点
    print(sum(hist))
    print(np.dot(s_vals,hist))
    return normalized_vals, s_vals, hist

num_bins = 50
def calculate_Pn(matrix_out, n, num_bins):
    s_vals = []
    # ave_space_all = []
    for i in range(matrix_out.shape[1]):
        intervals = []
        vals_cal = sorted(matrix_out[:,i])
        
        #percent
        start = int(len(vals_cal)*2/8)
        end = int(len(vals_cal)*6/8)
        vals_cal = vals_cal[start:end]

        N = len(vals_cal)
        # min_val, max_val = np.min(vals_cal), np.max(vals_cal)
        # normalized_vals = (vals_cal - min_val) / (max_val - min_val) * N  # 归一化
        #特征值展开
        normalized_vals = unfold_spectrum(vals_cal)
        # normalized_vals = np.sort(normalized_vals)
        # normalized_vals = unfold_spectrum(normalized_vals)
        for i in range(len(normalized_vals)-n-1):
            s = normalized_vals[i+n+1] - normalized_vals[i]  # 间隔长度
            intervals.append(s)
        intervals = [i for i in intervals if i > 0]
        # ave_space_all.append(np.mean(intervals))
        # print(ave_space)
        intervals = intervals / np.average(intervals)
        s_vals.extend(intervals)
    print(f"average spacings of {n} : {np.average(s_vals)}")
    # 构建直方图，统计 P_n(s)
    vals_out = s_vals
    hist, bin_edges = np.histogram(s_vals, bins=num_bins, range=(0, 4),density=True)
    s_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 每个区间的中心点
    print(sum(hist))
    print(np.dot(s_vals,hist))
    return vals_out, s_vals, hist
    # return 
def plot_different_file(file_name):
    # file_name = 'D:\\Matlab_out_data\\test_p_log\\toe_0.7_1024_10.csv'
    df = pd.read_csv(file_name, header=None)
    matrix_out = df.values
    _,s_vals,p0_s = calculate_Pn(matrix_out, 0, num_bins)
    _,s_vals_1,p1_s = calculate_Pn(matrix_out, 1, num_bins)
    _,s_vals_2,p2_s = calculate_Pn(matrix_out, 2, num_bins)
    _,s_vals_3,p3_s = calculate_Pn(matrix_out, 3, num_bins)
    _,s_vals_4,p4_s = calculate_Pn(matrix_out, 4, num_bins)
    return  s_vals,p0_s,s_vals_1,p1_s,s_vals_2,p2_s,s_vals_3,p3_s,s_vals_4,p4_s

# 为每个 n 构建 spacings_dict_n，并计算谱密度
all_plot_eig = dict()
all_Ne = dict()
n_list=[0,1,2]
for n in n_list:
    spacings_dict, alpha_1_list = create_spacings(alpha_1_list, eig_vals, n=n)
    plot_eig = dict()
    Ne = dict()
    for alpha_1 in alpha_1_list:
        eigs = spacings_dict[f"alpha_1={alpha_1:.1f}"]
        x, y = get_density(eigs, bin_size=40, upper=4, print_curvce=False)
        plot_eig[f"alpha_1={alpha_1:.1f}"] = x
        Ne[f"alpha_1={alpha_1:.1f}"] = y
    all_plot_eig[n] = plot_eig
    all_Ne[n] = Ne

#plot# 主绘图逻辑
alpha_1_list_list = [0.1,0.4,0.7,0.9]
year_List = np.arange(2019,2025,1)
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  # 显式指定衬线字体




for i, alpha_1 in enumerate(alpha_1_list):
    # 第一行：Python 原始谱n P0s.p1s,p2s

    # 颜色映射
    len_n = len(n_list)
    colors = plt.cm.get_cmap('coolwarm', (len_n))  # 'coolwarm' 是 matplotlib 内置的渐变色
    # # 生成 20 个渐变色
    color_list = [colors(i) for i in range(len_n)]
    color_list = ["#304D9B", "#EA514B", "#F6A64D"]
    color_list = ["#0000FF", "#FF0000", "#FFA500"]
    # color_map = color_list[:len_n]
    # color_map = plt.cm.get_cmap('tab10', len(n_list))
    line_styles = ['-', '--', '-.', ':']

    for j, n in enumerate(n_list):
        plot_eig = all_plot_eig[n]
        Ne = all_Ne[n]
        color = color_list[j]
        linestyle = line_styles[j % len(line_styles)]
        label = fr"$P_{n}(s)$"
        # if j== 0:
        #     axes[1, i].scatter(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
        #                 color=color, s=10,label ='_nolegend_')  
        axes[0, i].plot(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
                    color=color, linestyle=linestyle, label=label)
        axes[0, i].scatter(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
                    color=color, s=10,label ='_nolegend_')
    axes[0, i].set_title(fr"$\alpha_1$={alpha_1:.1f}",fontsize=16)
    if i == 0:
        axes[0, i].set_ylabel(r"$P_n(s)$", fontsize=16)
        axes[0, i].legend(fontsize=16)
    # axes[0, i].set_xlabel(r"$s$", fontsize=10)
    axes[0,i].set_xlim(0, 4)
    axes[0, i].text(0.02, 0.02, f"(a{i+1})", transform=axes[0, i].transAxes,
                    fontsize=16, va='bottom', ha='left')
    # 第二行：Python 对称+斜对称谱
    file_name_1 = f"D:\\Data\\plot_zuhe\\ARCH\\P0_data_alpha_1{alpha_1_list[i]:.2f}.csv"
    file_name_2 = f"D:\\Data\\plot_zuhe\\ARCH\\P0_fit_alpha_1{alpha_1_list[i]:.2f}.csv"
    # 读取数据
    data = pd.read_csv(file_name_1, header=None)
    fit = pd.read_csv(file_name_2, header=None)
    s_vals = data[0]
    p0_s = data[1]
    x_fit = fit[0]
    y_fit = fit[1]
    # 绘图
    plot_eig = all_plot_eig[0]
    Ne = all_Ne[0]
    # axes[1, i].scatter(s_vals, p0_s, color='blue', label=r'$P_0(s)$', s=10)
    axes[1, i].scatter(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
                 color='blue', label=r'$P_0(s)$', s=16)  
    axes[1, i].plot(x_fit, y_fit, 'r-', label='Weibull fit', linewidth=2)
    
    if i == 0:
        axes[1, i].set_ylabel(r'$P_0(s)$', fontsize=16)
        axes[1, i].legend(fontsize=16)
    # axes[1, i].set_xlabel(r'$s$', fontsize=10)
    axes[1, i].text(0.02, 0.02, f"(b{i+1})", transform=axes[1, i].transAxes,
                    fontsize=16, va='bottom', ha='left')
    # axes[1, i].tick_params(labelsize=8)

    # 第三行：Python 对称+斜对称谱
    # s_val_file_out = f"{h[i]}"
    file_name = f"D:\\Data\\pu\\ARCH_toe_{(alpha_1_list[i]):.2f}_2000_100_p.csv"
    s_vals, p0_s, *_ = plot_different_file(file_name)
    plot_eig = all_plot_eig[0]
    Ne = all_Ne[0]
    # 原始谱 - 蓝色
    axes[2, i].scatter(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
                 color='blue') 
    axes[2, i].plot(plot_eig[f"alpha_1={alpha_1:.1f}"], Ne[f"alpha_1={alpha_1:.1f}"], 
                 'b-') 
    # # 原始谱 - 蓝色
    # axes[2, i].scatter(s_vals, p0_s, color='blue')
    # axes[2, i].plot(s_vals, p0_s, 'b-')

    # 对称子谱 - 实心黑圆
    file_name = f"D:\\Data\\pu\\ARCH_toe_duichen_{(alpha_1_list[i]):.1f}_2000_100_p.csv"
    s_vals, p0_s, *_ = plot_different_file(file_name)
    axes[2, i].scatter(s_vals, p0_s, color='black', marker='o')  # 实心黑圆
    axes[2, i].plot(s_vals, p0_s, color='black', linestyle='-')

    # 斜对称子谱 - 空心黑圆
    file_name = f"D:\\Data\\pu\\ARCH_toe_xieduichen_{(alpha_1_list[i]):.1f}_2000_100_p.csv"
    s_vals, p0_s, *_ = plot_different_file(file_name)
    axes[2, i].scatter(s_vals, p0_s, facecolors='none', edgecolors='black', marker='o')  # 空心黑圆
    axes[2, i].plot(s_vals, p0_s, color='black', linestyle='--')

    # plt.text(0.5, 0.90, f"H = {h[i]}", ha='center', va='bottom',
    #          transform=plt.gca().transAxes, fontsize=10)
    # plt.legend()
    if i == 0:
        axes[2, i].set_ylabel(fr'$P_0(s)$', fontsize=16)
    axes[2, i].set_xlabel(fr'$s$', fontsize=16)
    axes[2, i].text(0.02, 0.02, f"(c{i+1})", transform=axes[2, i].transAxes,
                    fontsize=16, va='bottom', ha='left')
fig.subplots_adjust(left=0.04, right=0.98,top=0.96, bottom=0.06)
plt.savefig("figure12_ARCH_Pns.png", dpi=600, bbox_inches='tight')
plt.savefig("figure12ARCH_Pns.pdf", dpi=600, bbox_inches='tight')
plt.savefig("figure12_ARCH_Pns.eps", dpi=600,format= 'eps', bbox_inches='tight')

plt.show()

