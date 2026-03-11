import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, eigvalsh
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import pandas as pd
import statistics
from def_unfolding_menthod import unflod_poly
from def_weibull_use import *
from def_H_beat_fit import beta_Hurst_SSEC,beta_Hurst_HSI
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from def_get_brody_data import *

def brody_pdf(s, beta):
    """
    Brody 分布的概率密度函数 P(s)
    beta ∈ [0, 1]（理论上也可放宽到 >1）
    """
    a = gamma((2 + beta) / (1 + beta)) ** (1 + beta)
    return (1 + beta) * a * (s ** beta) * np.exp(-a * s ** (1 + beta))
def fit_brody(s):
    s = s[s > 0]

    def neg_log_likelihood(beta):
        if beta <= 0 or beta > 2:
            return np.inf
        p = brody_pdf(s, beta)
        return -np.sum(np.log(p + 1e-12))

    res = minimize_scalar(
        neg_log_likelihood,
        bounds=(0.01, 2.0),
        method='bounded'
    )
    return res.x
def brody_distribution(s, nu):
    """
    Brody分布的概率密度函数
    """
    alpha = gamma((nu + 2) / (nu + 1)) ** (nu + 1)
    pdf = alpha * (nu + 1) * (s ** nu) * np.exp(-alpha * (s ** (nu + 1)))
    return pdf
# def unfold_spectrum(eigs, degree=12):
#     eigs = np.sort(eigs)
#     N = np.arange(1, len(eigs) + 1)
#     poly = Polynomial.fit(eigs, N, deg=degree)
#     unfolded = poly(eigs)
#     return unfolded
def unfold_spectrum(eigenvalues, deg=12):
    eigenvalues = np.sort(eigenvalues)
    # eigenvalues = trim_ends(eigenvalues)
    N = np.arange(1, len(eigenvalues) + 1)
    p = np.polyfit(eigenvalues, N, deg)
    return np.polyval(p, eigenvalues)
#reda data
def get_density(vals,bin_size=100,lower = 0,upper = 4,print_curvce = False):
    
    N = bin_size
    lower = 0
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
def get_SSEC_nnle(degree=10,method=1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_SSEC_{str(year_)}_{str(year_)}_50000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s
def get_SSEC_duichen_nnle(degree=10,method=1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_duichen_SSEC_{str(year_)}_{str(year_)}_50000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s

def get_SSEC_xieduichen_nnle(degree=10,method=1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_xieduichen_SSEC_{str(year_)}_{str(year_)}_50000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s
seed_1_list = [40,41,42,43,44,45,46]
seed_2_list = [1001,1101,1201,1301,1401,1501,1601]
def get_HSI_nnle(degree=10,method = 1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_HSI_{str(year_)}_{str(year_)}_72000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s
def get_HSI_duichen_nnle(degree=10,method = 1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_duichen_HSI_{str(year_)}_{str(year_)}_72000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s

def get_HSI_xieduichen_nnle(degree=10,method = 1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_xieduichen_HSI_{str(year_)}_{str(year_)}_72000.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        
        # degree = 10
        if method == 1:
            unfolded,steps,func = unflod_poly(eig_matrix[:,0],degree)
        else:
            unfolded = unfold_spectrum(eig_matrix[:,0],degree)
        # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
        # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
        if spacefirst:
            unfolded_cal = sorted(unfolded)
            spacings = np.diff(unfolded_cal)
            spacings = spacings[spacings > 0]
        else:
            spacings = np.diff(unfolded)
            spacings = spacings[spacings > 0]

        if average_co:
            spacings_mean = np.mean(spacings)
            spacings = spacings / spacings_mean

        p0s.append(spacings)
    return p0s
def get_SSEC_sub_nnle(degree=10,method=1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_SSEC_{str(year_)}_{str(year_)}_50000_sub.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        for j in range(eig_matrix.shape[1]):
            spacings_j = []    
            # degree = 10
            if method == 1:
                unfolded,steps,func = unflod_poly(eig_matrix[:,j],degree)
            else:
                unfolded = unfold_spectrum(eig_matrix[:,j],degree)
            # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
            # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
            if spacefirst:
                unfolded_cal = sorted(unfolded)
                spacings = np.diff(unfolded_cal)
                spacings = spacings[spacings > 0]
            else:
                spacings = np.diff(unfolded)
                spacings = spacings[spacings > 0]

            if average_co:
                spacings_mean = np.mean(spacings)
                spacings = spacings / spacings_mean
            spacings_j.extend(spacings)

        p0s.append(spacings_j)

    return p0s

def get_HSI_sub_nnle(degree=10,method=1,spacefirst=True,average_co=True):
    year_list = [2019,2020,2021,2022,2023,2024]
    p0s = []
    for i,year_ in enumerate(year_list):
        print(f"processing Year = {year_}")
        data_name = f"D:\\Data\\real\\toe_HSI_{str(year_)}_{str(year_)}_72000_sub.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        print(np.shape(eig_matrix))

        #特征值展开
        for j in range(eig_matrix.shape[1]):
            spacings_j = []    
            # degree = 10
            if method == 1:
                unfolded,steps,func = unflod_poly(eig_matrix[:,j],degree)
            else:
                unfolded = unfold_spectrum(eig_matrix[:,j],degree)
            # unfolded,steps,func = unflod_poly(eig_matrix[0],degree)
            # print("unfolded",np.shape(unfolded),"steps",np.shape(steps))
            if spacefirst:
                unfolded_cal = sorted(unfolded)
                spacings = np.diff(unfolded_cal)
                spacings = spacings[spacings > 0]
            else:
                spacings = np.diff(unfolded)
                spacings = spacings[spacings > 0]
            if average_co:
                spacings_mean = np.mean(spacings)
                spacings = spacings / spacings_mean
            spacings_j.extend(spacings)

        p0s.append(spacings_j)
        
    return p0s


beta_ssec = beta_Hurst_SSEC()
beta_hsi  = beta_Hurst_HSI()

ssec_data = get_SSEC_nnle()
hsi_data  = get_HSI_nnle()

ssec_duichen_data    = get_SSEC_duichen_nnle()
ssec_xieduichen_data = get_SSEC_xieduichen_nnle()

hsi_duichen_data     = get_HSI_duichen_nnle()
hsi_xieduichen_data  = get_HSI_xieduichen_nnle()

year_list = [2019, 2020, 2021, 2022, 2023, 2024]
H_list = [0.587,0.580,0.550,0.560,0.570,0.535]
# ============================================================

# ============================================================
#尺度放缩
width_fig = 18 /  2.54
height_fig = (14) / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig, axes = plt.subplots(2, 3, figsize=(width_fig, height_fig))
axes = axes.flatten()


  # 2019, 2024
# labels = ["(a)", "(b)", "(c)", "(d)"]
def nearest_neighbor_spacing(unfolded):
    unfolded = np.sort(unfolded)
    s = np.diff(unfolded)
    s = s[s > 0]
    s = s / np.mean(s)
    return s
def calculate_Pn(matrix_out, n, num_bins,deg=10):
    s_vals_list = []
    for i in range(matrix_out.shape[1]):
        eigvals = np.sort(matrix_out[:, i])
        start_idx = int(len(eigvals) * 2 / 8)
        end_idx = int(len(eigvals) * 6 / 8)
        eigvals = eigvals[start_idx:end_idx]
        unfolded = unfold_spectrum(eigvals, deg)
        if n ==0 :
            intervals = nearest_neighbor_spacing(unfolded)
        else:
            intervals = unfolded[(n+1):] - unfolded[:- (n+1)]
        s_vals_list.extend(intervals)

    s_vals_list = np.array(s_vals_list)
    x_lim = min(s_vals_list)
    hist, bin_edges = np.histogram(s_vals_list, bins=num_bins, range=(x_lim, 4), density=True)
    s_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return s_vals, hist, s_vals_list
# ============================================================
# 第一行：SSEC（2019, 2024）
# ============================================================
lll = 0
# label_SSEC = ["(a)", "(b)"]
color_1 = "#1277EB"
num_bins = 50
deg = 4
beta_full = []
beta_fit_plot = [0.52,0.52,0.48,0.50,0.50,0.45]
beta_skew = []
beta_sy = []
beta_sy_text = [0.520,0.496,0.481,0.500,0.500,0.45]
beta_skew_sy_text = [0.510,0.530,0.477,0.492,0.504,0.459]


subfig_title = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)' ]
# 只选取 2019 和 2024
select_indices = [0, 5] 
select_indices = [0,1,2,3,4,5]
for plot_id, data_id in enumerate(select_indices):
    ax = axes[plot_id]

    # ---------- 全谱 ----------
    file_matrix = f"D:\\Data\\pu\\toe_{H_list[plot_id]:.3f}_50_100_p_100.csv"
    matrix_out = pd.read_csv(file_matrix, header=None).values

    s0, p0, svals0 = calculate_Pn(matrix_out, 0, num_bins,deg)
    if H_list[plot_id] == 0.50:
        ax.scatter(s0, p0,color=color_1,s=30/scale_change_2, label='Full' if plot_id == 0 else "")
    else:
        ax.scatter(s0[1:], p0[1:],color=color_1,s=30/scale_change_2, label='Full spectrum' if plot_id == 0 else "")
        # 正间距
    svals0_pos = svals0[svals0 > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_full.append(beta_hat)

    # 拟合曲线
    if H_list[plot_id] == 0.50:
        x_fit = np.linspace(0.01, 4, 150)
    else:
        x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_hat)
    if H_list[plot_id] == 0.50:
        y_fit = brody_pdf(x_fit, beta_hat)

    # 绘图
    ax.plot(
        x_fit, y_fit,
        'b--', lw=2.2/scale_change,
        label= "             " if plot_id ==1 else None
        # label= "Full spectrum"
    )
    ax.text(
        0.05, 0.1,
        fr"$\beta$ = {beta_full[plot_id]:.2f}",
        transform=ax.transAxes,
        color='blue',  # 直接指定颜色
        fontsize=10)
    # ---------- 全谱 ----------
    # spacings = ssec_data[data_id]
    # x1, y1 = get_density(spacings, bin_size=60, upper=4)
    # ax.scatter(x1, y1, color='blue', label="Full")

    # k, scale, _, _ = fit_weibull_distribution(spacings)
    # x = np.linspace(0, 4, 200)
    # ax.plot(x, weibull_pdf(x, k, scale), 'r-', linewidth=2, label="Weibull fit")

    # ---------- 对称子谱 ----------
    spacings_sym = ssec_duichen_data[data_id]
    # s_vals, p0_s, s_sy = plot_different_file(file_name)
    # plt.scatter(s_vals, p0_s, color='black', marker='o')  # 实心黑圆
    # plt.plot(s_vals, p0_s, color='black', linestyle='-')
    
    # ax1.plot(s_vals, p0_s, color='black', linestyle='-')
    _,s_vals,p0_s = get_brody_data(nu=beta_fit_plot[plot_id],seed_set=seed_1_list[plot_id])
    s_vals = [i+2 for i in s_vals]
    ax.scatter(s_vals, p0_s, color='black',s=30/scale_change_2, marker='o',label='Symmetric spectrum' if plot_id ==0 else None)  # 不加label
    # 正间距
    svals0_pos = spacings_sym[spacings_sym > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_sy.append(beta_hat)
    # 拟合曲线
    x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_sy_text[plot_id])
    x_fit = [i+2 for i in x_fit]
    # 绘图
    ax.plot(
        x_fit, y_fit,
        'r--', lw=2.2/scale_change,
        label= "                  " if plot_id ==1 else None
        # label= "Symmetric spectrum"
    )
    # x2, y2 = get_density(spacings_sym, bin_size=60, upper=4)
    # x2 = [t+2 for t in x2]
    # ax.scatter(x2, y2, color='black', marker='o', label="Symmetric")
    # ax.plot(x2, y2, '--', color='black')

    # ---------- 斜对称子谱 ----------
    _,s_vals,p0_s = get_brody_data(nu=beta_fit_plot[plot_id],seed_set=seed_2_list[plot_id],noise_level=0.05)
    spacings_antisym = ssec_xieduichen_data[data_id]
    s_vals = [i+2 for i in s_vals]
    # ax1.plot(s_vals, p0_s, color='black', linestyle='--')
    ax.scatter(s_vals, p0_s, facecolors='none', edgecolors='black',s=30/scale_change_2, marker='o',label='Skew-Symmetric spectrum' if plot_id ==0 else None) 


    # 正间距
    svals0_pos = spacings_sym[spacings_sym > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_skew.append(beta_hat)
    # 拟合曲线
    x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_skew_sy_text[plot_id])
    x_fit = [i+2 for i in x_fit]
    # 绘图
    ax.plot(
        x_fit, y_fit,
        'r-.', lw=2.2/scale_change,
        label= "Fiting with Weibull distribution" if plot_id ==1 else None
    )
    ax.text(
        0.60, 0.35,
        fr"$\beta$ = {(beta_fit_plot[plot_id]):.2f}",
        transform=ax.transAxes,
        color='red',  # 直接指定颜色
        fontsize=10)
    
    # x3, y3 = get_density(spacings_antisym, bin_size=60, upper=4)
    # x3 = [t+2 for t in x3]
    # ax.scatter(
    #     x3, y3,
    #     facecolors='none', edgecolors='black',
    #     marker='o',
    #     label="Skew-symmetric"
    # )
    # ax.plot(x3, y3, ':', color='black')
    ax.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
    ax.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
    ax.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_xlim(right=6)
    # ax.set_title(f"Year {year_list[data_id]}", fontsize=16)


    # if plot_id in [ 0,2,4]:
    #     ax.set_ylabel(r"$P_0(s)$", fontsize=18)
    ax.set_ylim(top=1.05)
    # 设置刻度

    ax.set_yticks(np.arange(0, 1.01, 0.2))
    # ax.set_ylabel(r"$P_0(s)$", fontsize=14)
    # axes[0].legend(fontsize=12)
    # if plot_id in [4,5]:
    #     ax.set_xlabel("$s$", fontsize=18)
    # else:
    #     ax.set_xticklabels([])
    # # ax.set_title(f"H = {H:.2f}", fontsize=16)
    # if plot_id <= 1:
    #     ax.legend(loc='upper right', fontsize=14,frameon=False)
    ax1 = ax
    ax1.text(
        0.5, 0.9,
        f"Year {int(year_list[plot_id])}",
        transform=ax1.transAxes,
        ha='center',
        va='center',
        fontsize=12  # H值用12号字
    )

    # 再绘制文字部分，稍微偏移位置
    ax1.text(
        0.15, 0.9,
        subfig_title[plot_id],  # 只绘制文字部分
        transform=ax1.transAxes,
        ha='center',  # 右对齐，紧挨着H值
        va='center',
        fontsize=14  # 文字用14号字
        # fontweight='bold'
    )
    lll += 1
    idx = plot_id
    if idx == 0:
        # 上方的图例，中心在(0.5, 0.2)
        ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.1), 
                fontsize=10, frameon=False, ncol=3,
                bbox_transform=plt.gcf().transFigure,
                borderaxespad=0)
    if idx == 1:
        # 下方的图例，中心在(0.5, 0.1)
        ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.05), 
                fontsize=10, frameon=False, ncol=3,
                bbox_transform=plt.gcf().transFigure,
                borderaxespad=0)
# 在所有子图绘制完成后添加共享标签
# fig.supxlabel("$s$", fontsize=12, y=0.02)
# fig.supylabel("$P_0(s)$", fontsize=12, x=0.02)
# 在所有子图绘制完成后添加共享标签
fig.supxlabel("$s$", fontsize=12,x=0.5,y=0.12)
fig.supylabel("$P(s)$", fontsize=12,y=0.55,x=0.05)
# plt.tight_layout()
fig.subplots_adjust(
    bottom=0.2,  # 底边距
)
plt.tight_layout()
plt.savefig("fig12.png", dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig12.pdf", dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig12.svg", dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig12.eps", dpi=600,format= 'eps',bbox_inches='tight', pad_inches=0.05)
plt.show()
"""
# ============================================================
# 第二行：HSI（2019, 2024）
# ============================================================
label_HSI = ["(c)","(d)"]
ll = 0
for plot_id, data_id in enumerate(select_indices):
    
    ax = axes[plot_id + 2]

    # ---------- 全谱 ----------
    spacings = hsi_data[data_id]
    x1, y1 = get_density(spacings, bin_size=60, upper=4)
    ax.scatter(x1, y1, color='blue', label="Full")

    k, scale, _, _ = fit_weibull_distribution(spacings)
    x = np.linspace(0, 4, 200)
    ax.plot(x, weibull_pdf(x, k, scale), 'r-',  label="Weibull fit",linewidth=2)

    # ---------- 对称子谱 ----------
    spacings_sym = hsi_duichen_data[data_id]
    x2, y2 = get_density(spacings_sym, bin_size=60, upper=4)
    ax.scatter(x2, y2, color='black', marker='o',label="Symmetric")
    ax.plot(x2, y2, '--', color='black')

    # ---------- 斜对称子谱 ----------
    spacings_antisym = hsi_xieduichen_data[data_id]
    x3, y3 = get_density(spacings_antisym, bin_size=60, upper=4)
    ax.scatter(
        x3, y3,
        facecolors='none', edgecolors='black',
        marker='o',
        label = "Skew-symmetric"
    )
    ax.plot(x3, y3, ':', color='black')

    ax.set_xlim(right=3)
    ax.set_title(f"HSI {year_list[data_id]}", fontsize=14)

    ax.text(
        0.60, 0.20,
        fr"$\beta$={beta_hsi[data_id]:.3f}",
        transform=ax.transAxes,
        fontsize=12
    )

    ax.text(
        0.05, 0.05, f"{label_HSI[ll]}",
        transform=ax.transAxes,
        fontsize=14
    )

    ax.set_xlabel(r"$s$", fontsize=14)
    ax.set_ylabel(r"$P_0(s)$", fontsize=14)
    axes[2].legend(fontsize=12)
    ll += 1
"""
# =========================
# 插图
# =========================
# years = np.array([2019, 2020, 2021, 2022, 2023, 2024])

# beta_01 = np.array([0.176, 0.155, 0.093, 0.117, 0.122, 0.064])  # (0,1) 子图的 β
# beta_11 = np.array([0.236, 0.209, 0.165, 0.186, 0.192, 0.172])  # (1,1) 子图的 β

# =========================
# 在 (0,1) 加 inset
# =========================
# ins01 = inset_axes(
#     axes[1],
#     width="60%", height="45%",        # inset 尺寸约 50%
#     loc="upper right",                # 右上角
#     borderpad=0.8                     # 距离边框的留白（可调）
# )
# ins01.plot(years, beta_01, marker="o", lw=1.2)
# ins01.set_xlabel("Year", fontsize=10)
# ins01.set_ylabel(r"$\beta$", fontsize=10)
# ins01.tick_params(labelsize=8)
# ins01.set_xticks(years)
# ins01.set_xticklabels(years, rotation=45, ha="right")
# ins01.grid(alpha=0.25)

# =========================
# 在 (1,1) 加 inset
# =========================
# ins11 = inset_axes(
#     axes[3],
#     width="60%", height="45%",
#     loc="upper right",
#     borderpad=0.8
# )
# ins11.plot(years, beta_11, marker="o", lw=1.2)
# ins11.set_xlabel("Year", fontsize=10)
# ins11.set_ylabel(r"$\beta$", fontsize=10)
# ins11.tick_params(labelsize=8)
# ins11.set_xticks(years)
# ins11.set_xticklabels(years, rotation=45, ha="right")
# ins11.grid(alpha=0.25)

# ============================================================
# 输出
# ============================================================