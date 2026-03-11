"""
这是计算谱密度程序,并绘制谱密度图
date： 2025.05.15   USST
author： LiXiujiang
"""
#导库
from def_Ne import *
from scipy import integrate
from scipy.stats import gamma
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta
from def_eig_thoery import *
from typing import Tuple
from scipy import stats, optimize
from scipy.optimize import curve_fit
# import numpy as np
import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
# import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from def_plot import plot_Ne
import seaborn as sns
import math
from typing import List  
from def_beat_dis_fit import *
def beta_pdf(x, a, b, loc, scale):
    # 归一化到 (0,1)
    z = (x - loc) / scale
    z = np.clip(z, 1e-12, 1 - 1e-12)  # 防止 log(0)
    # 标准 beta pdf
    return beta.pdf(z, a, b) / scale  # 注意要除以 scale！

def beta_pdf(x, a, b, loc, scale):
    # 归一化到 (0,1)
    z = (x - loc) / scale
    z = np.clip(z, 1e-12, 1 - 1e-12)  # 防止 log(0)
    # 标准 beta pdf
    return beta.pdf(z, a, b) / scale  # 注意要除以 scale！

# ------------------------------
# 稳定版 Beta 拟合
# ------------------------------
def fit_weighted_beta(x, y):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    # 初值估计
    # 用第一个显著点作为 loc 初值
    thresh = y.max()*1e-3
    nz = np.where(y > thresh)[0]
    if len(nz)==0:
        nz = np.arange(len(x))
    loc0 = x[nz[0]]
    scale0 = max(1e-3, x[nz[-1]] - x[nz[0]])
    # 方法矩估计先在标准化区间上
    x_std = (x - loc0) / scale0
    x_std = np.clip(x_std, 1e-9, 1-1e-9)
    mean_ = np.sum(x_std * y) / np.sum(y)  # approximate
    var_  = np.sum((x_std-mean_)**2 * y) / np.sum(y)
    # method-of-moments if var>0
    if var_>0:
        common = mean_*(1-mean_)/var_ - 1
        a0 = max(1e-3, mean_*common)
        b0 = max(1e-3, (1-mean_)*common)
    else:
        a0, b0 = 2.0, 2.0
    p0 = [a0, b0, loc0, scale0]

    # weights: 给高密度点更高权重。curve_fit sigma -> weight = 1/sigma
    # 我们希望 weight ∝ y  -> sigma ∝ 1/y
    sigma = 1.0 / (y + 1e-6)
    # bounds: a,b>0, scale>0
    lower = [1e-6, 1e-6, np.min(x)-1.0, 1e-6]
    upper = [1e6, 1e6, np.max(x)+1.0, (np.max(x)-np.min(x))*5]

    popt, pcov = curve_fit(beta_pdf, x, y, p0=p0, sigma=sigma, bounds=(lower,upper), maxfev=20000)
    # return popt, pcov
    # 解包拟合结果参数
    a, b, loc, scale = popt
    print(f"alpha_1 = {a:.4f}, alpha_2 = {b:.4f}, loc = {loc:.4f}, scale = {scale:.4f}")
    # ====== 构建平滑拟合曲线 ======
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = beta_pdf(x_fit, a, b, loc, scale)

    # x_fit = np.linspace(min(x), max(x), 1000)                   # 拟合结果的 x 就是原来的 x
    # y_fit = beta_pdf(x_fit, a, b, loc, scale)
    return x_fit, y_fit
def mix_beta_pdf(x, params):
    # params = [a1,b1,a2,b2, loc, scale, c]
    a1,b1,a2,b2,loc,scale,c = params
    p1 = beta.pdf(x, a1, b1, loc=loc, scale=scale)
    p2 = beta.pdf(x, a2, b2, loc=loc, scale=scale)
    return c*p1 + (1-c)*p2
def fit_mixture_beta(x, y):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    # initial guesses
    loc0 = x[np.argmax(y > (y.max() * 1e-3))] if np.any(y > (y.max() * 1e-3)) else x.min()
    scale0 = (x.max() - x.min()) * 0.8
    p0 = [2.0, 5.0, 5.0, 2.0, loc0, scale0, 0.5]

    bounds = [
        (1e-3, 1e6), (1e-3, 1e6),
        (1e-3, 1e6), (1e-3, 1e6),
        (x.min() - 1, x.max() + 1),
        (1e-6, (x.max() - x.min()) * 5),
        (0.0, 1.0)
    ]

    def loss(params):
        model = mix_beta_pdf(x, params)
        w = y + 1e-6
        return np.sum(w * (model - y) ** 2)

    res = minimize(
        loss, p0, bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 5000}
    )

    # -------- 新增：返回拟合曲线 x_fit, y_fit --------
    # x_fit = x
    x_fit = np.linspace(min(x), max(x), 300)
    y_fit = mix_beta_pdf(x_fit, res.x)
    print(res.x)
    return x_fit, y_fit
def find_subplot_layout(T):
    if T == 1:
        return 1, 1
    m = math.ceil(math.sqrt(T))
    n = math.ceil(T / m)
    # 尝试使 m 和 n 更接近
    while abs(m - n) > 1 and (m - 1) * n >= T:
        m -= 1
    return m, n
def fit_gamma_from_xy(x, y, assume_loc_zero=True, use_mle=False, x_fit_count=200):
    """
    从离散 (x, y)（y 为对应 x 的密度估计）估计 Gamma 分布参数并返回拟合曲线。
    
    参数:
      x: 1D array, 横坐标（应单调或至少相对有序）
      y: 1D array, 与 x 同长，代表在这些 x 上的密度值（可能未归一化）
      assume_loc_zero: 如果 True 使用 loc=0 的两参数 Gamma（shape, scale）
                       否则允许 loc != 0（使用 MLE 时可考虑）
      use_mle: 如果 True，用 scipy 的 MLE（三参数）或自定义优化来拟合（会覆盖矩估计）
      x_fit_count: 输出 x_fit 的点数（默认 200）
    
    返回:
      params: dict, 包含估计的参数（shape, scale, loc）
      x_fit: 1D array, 用于绘图的 x 值
      y_fit: 1D array, 对应 x_fit 的 Gamma pdf 值
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # 确保 x 单调递增 —— 若不是，按 x 排序
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # 数值积分归一化 y（梯形规则）
    area = np.trapz(y, x)
    if area <= 0:
        raise ValueError("y 的数值积分 <= 0，无法归一化。")
    y_norm = y / area
    
    # 计算均值和方差（基于归一化的密度）
    mu = np.trapz(x * y_norm, x)
    second = np.trapz((x**2) * y_norm, x)
    var = second - mu**2
    if var <= 0:
        raise ValueError("计算得到的方差 <= 0，不能用矩方法估计 Gamma。")
    
    # 矩估计（假设 loc=0）
    k_mom = mu**2 / var
    theta_mom = var / mu
    loc_mom = 0.0
    
    params = {"shape": k_mom, "scale": theta_mom, "loc": loc_mom, "method": "moment"}
    
    # 可选：用 MLE 拟合（允许 loc 非零或者覆盖 moment）
    if use_mle:
        # 初始值来自矩估计（loc 初值设为最小 x 的小一点）
        loc0 = np.minimum(0.0, x.min() - 1e-6)
        a0 = max(1e-6, k_mom)
        scale0 = max(1e-6, theta_mom)
        # 使用 scipy.stats.gamma.fit（它返回 a, loc, scale）
        try:
            a_fit, loc_fit, scale_fit = gamma.fit(x, floc=None, a=a0, scale=scale0)
            params = {"shape": a_fit, "scale": scale_fit, "loc": loc_fit, "method": "mle_scipy"}
        except Exception as e:
            # 如果 scipy 拟合失败，保留矩估计并返回警告
            params["mle_error"] = str(e)
            params["method"] = "moment"
    
    # 如果用户不想允许 loc != 0，但启用了 use_mle，会用 floc=0 强制 loc=0：
    if use_mle and assume_loc_zero:
        try:
            a_fit, loc_fit, scale_fit = gamma.fit(x, floc=0)  # 强制 loc=0
            params = {"shape": a_fit, "scale": scale_fit, "loc": 0.0, "method": "mle_scipy_floc0"}
        except Exception as e:
            params["mle_error"] = str(e)
            params["method"] = "moment"
    
    # 构造 x_fit（以支持范围为主，稍微扩展一点便于可视化）
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    if span <= 0:
        span = max(1.0, abs(xmin))
    x_fit = np.linspace(max(0, xmin - 0.1*span), xmax + 0.1*span, x_fit_count)
    
    # 计算 y_fit（使用 scipy.stats.gamma.pdf）
    shape = params["shape"]
    scale = params["scale"]
    loc = params.get("loc", 0.0)
    y_fit = gamma.pdf(x_fit, a=shape, loc=loc, scale=scale)
    
    # 可选：归一化 y_fit（保证数值上是 pdf）
    area_fit = np.trapz(y_fit, x_fit)
    if area_fit > 0:
        y_fit = y_fit / area_fit  # 这样 x_fit,y_fit 也严格是归一化的 pdf
    
    return params, x_fit, y_fit
def fit_gamma_distribution(x: np.ndarray, y: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    根据盒数记数密度数据拟合Gamma分布
    
    参数:
    ----------
    x : np.ndarray
        盒数序列（通常表示大小或强度）
    y : np.ndarray
        对应的记数密度
    
    返回:
    ----------
    params : dict
        Gamma分布参数字典，包含:
        - 'a': shape参数
        - 'loc': 位置参数
        - 'scale': 尺度参数
        - 'mean': 均值 = a * scale
        - 'variance': 方差 = a * scale^2
    x_fit : np.ndarray
        拟合用的x值（通常比原始x更密集）
    y_fit : np.ndarray
        拟合的Gamma分布概率密度函数值
    
    注意:
    ----------
    假设y已经归一化或可以视为概率密度
    """
    
    # 确保输入为numpy数组
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 方法1: 使用矩估计法（Method of Moments）获取初始参数
    # 计算数据的均值和方差（假设y是概率密度）
    if np.sum(y) > 0:
        # 如果y是计数，可以归一化
        y_normalized = y / np.sum(y)
    else:
        y_normalized = y.copy()
    
    # 使用加权均值和方差
    data_mean = np.average(x, weights=y_normalized)
    data_variance = np.average((x - data_mean)**2, weights=y_normalized)
    
    # Gamma分布的矩估计
    # 对于Gamma分布: mean = a * scale, variance = a * scale^2
    if data_variance > 0 and data_mean > 0:
        # 方法1: 矩估计
        a_initial = data_mean**2 / data_variance  # shape参数
        scale_initial = data_variance / data_mean  # scale参数
        
        # 方法2: 使用MLE（最大似然估计）进行更精确的拟合
        try:
            # 使用scipy的fit方法进行最大似然估计
            # 注意：这里假设x可以重复采样，权重为y
            # 创建一个伪样本，其中每个x值根据y重复多次
            samples = []
            max_samples = 10000  # 最大样本数
            
            # 根据y的权重生成样本
            total_weight = np.sum(y)
            if total_weight > 0:
                for xi, yi in zip(x, y):
                    n_samples = int(yi / total_weight * max_samples)
                    samples.extend([xi] * n_samples)
            
            if len(samples) > 10:
                samples_array = np.array(samples)
                # 使用MLE拟合Gamma分布
                a_mle, loc_mle, scale_mle = stats.gamma.fit(samples_array, floc=0)
            else:
                # 如果样本太少，使用矩估计
                a_mle, loc_mle, scale_mle = a_initial, 0, scale_initial
                
        except Exception as e:
            print(f"MLE拟合失败: {e}，使用矩估计")
            a_mle, loc_mle, scale_mle = a_initial, 0, scale_initial
    
    else:
        # 如果方差为0或均值为0，使用默认值
        a_mle, loc_mle, scale_mle = 1.0, 0, 1.0
    
    # 方法3: 使用非线性最小二乘法进一步优化拟合
    def gamma_pdf(x, a, scale):
        """Gamma分布的概率密度函数"""
        return stats.gamma.pdf(x, a, loc=0, scale=scale)
    
    try:
        # 准备数据用于拟合（排除y=0的点）
        mask = y > 0
        x_fit_data = x[mask]
        y_fit_data = y[mask]
        
        # 如果需要归一化
        if np.sum(y_fit_data) > 0:
            y_fit_data_normalized = y_fit_data / np.sum(y_fit_data)
        else:
            y_fit_data_normalized = y_fit_data
        
        # 非线性最小二乘拟合
        popt, _ = optimize.curve_fit(
            gamma_pdf,
            x_fit_data,
            y_fit_data_normalized,
            p0=[a_mle, scale_mle],
            bounds=(0, [np.inf, np.inf])
        )
        
        a_final, scale_final = popt
        loc_final = 0  # 假设位置参数为0
        
    except Exception as e:
        print(f"曲线拟合失败: {e}，使用MLE估计")
        a_final, scale_final, loc_final = a_mle, scale_mle, loc_mle
    
    # 计算拟合曲线
    x_min, x_max = np.min(x), np.max(x)
    x_fit = np.linspace(max(0, x_min), x_max * 1.5, 200)
    y_fit = stats.gamma.pdf(x_fit, a_final, loc=loc_final, scale=scale_final)
    
    # 计算分布的均值和方差
    mean = a_final * scale_final
    variance = a_final * scale_final**2
    
    # 整理参数
    params = {
        'a': a_final,
        'loc': loc_final,
        'scale': scale_final,
        'mean': mean,
        'variance': variance,
        'shape': a_final,  # 别名
        'rate': 1.0 / scale_final if scale_final > 0 else 0  # rate参数
    }
    # print(f"gamma = {}")
    
    return  params,x_fit, y_fit
def generate_gradient_two_colors(colorA, colorB, N=256):
    """
    生成 colorA 到 colorB 的渐变颜色列表（仅支持 2 色）
    :param colorA: 起始颜色（名称或十六进制，如 'blue' 或 '#0000FF'）
    :param colorB: 结束颜色
    :param N: 返回的颜色数量
    :return: 渐变颜色列表，格式为 [(R1,G1,B1), (R2,G2,B2), ...]
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
    :param color_list: 颜色列表（支持名称或十六进制）
    :param N: 颜色数量
    :return: RGB 元组列表
    """
    palette = sns.color_palette(color_list, n_colors=N)
    return palette
########## 1-读取数据 ##########
#①读取数据 from data.csv
def read_data():
    eig_vals = {}
    H = np.arange(0.50,0.901,0.05)
    dimension ,simulation_times = 2000 , 100
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        data_name = f"D:\\Data\\pu\\toe_{h:.2f}_{dimension}_{simulation_times}_p.csv"
        data_name = f"D:\\Data\\pu\\toe_{h:.2f}_2000_100_p.csv"
        # data_name = f"D:\\Data\\pu\\toe_{h:.2f}_2000_100_p_100.csv"
        df = pd.read_csv(data_name,header=None)
        eig_matrix = df.values
        eig_vals[f"H={h:.2f}"] = eig_matrix
    

    #理论测试
    
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return H,eig_vals
def fit_gamma_from_samples(x, use_mle=True, x_fit_count=200):

    x = np.asarray(x)
    
    # 如果数据不全为正，Gamma 分布不适用
    if np.any(x <= 0):
        raise ValueError("Gamma 分布要求 x > 0")

    # 默认：最大似然估计
    if use_mle:
        a, loc, scale = gamma.fit(x, floc=0)  # 强制 loc=0
        method = "MLE (scipy.stats.gamma.fit)"
    else:
        # 矩估计
        mu = np.mean(x)
        var = np.var(x, ddof=1)
        a = mu**2 / var
        scale = var / mu
        loc = 0
        method = "Method of Moments"
    
    params = {"shape": a, "scale": scale, "loc": loc, "method": method}

    # 生成用于绘图的 x_fit
    xmin, xmax = x.min(), x.max()
    x_fit = np.linspace(xmin, xmax, x_fit_count)

    # 计算 y_fit
    y_fit = gamma.pdf(x_fit, a=a, loc=loc, scale=scale)

    return params, x_fit, y_fit
def get_a_b_loc_scale_from_NE(eig_vals,H,everone=False):

    a_all = []
    b_all = []
    x_fit = {}
    y_fit = {}
    skew = []
    for i,h in enumerate(H):
        # print(f"processing H = {h:.2f}") 
        data = eig_vals[f"H={h:.2f}"]
        data_all = (data.flatten())
        data_all = data_all[data_all < 2]
        

        params, x_fit_h, y_fit_h = fit_gamma_from_samples(data_all)
        # print(params)
        print(x_fit_h[:10])
        print(y_fit_h[:10])
        # a_all.append(result['alpha'])
        # b_all.append(result['beta'])
        # skew.append(result['skew'])
        x_fit_h = [ts-1 for ts in x_fit_h]
        x_fit[f"H={h:.2f}"] = x_fit_h
        y_fit[f"H={h:.2f}"] = y_fit_h

    return x_fit,y_fit
from def_GB2_fit import *
def get_a_b_loc_scale_from_NE_2(eig_vals,H,everone=False):

    a_all = []
    b_all = []
    x_fit = {}
    y_fit = {}
    skew = []
    for i,h in enumerate(H):
        # print(f"processing H = {h:.2f}") 
        data = eig_vals[f"H={h:.2f}"]
        data_all = (data.flatten())
        data_all = data_all[data_all < 2]
        

        params, x_fit_h, y_fit_h = fit_gamma_from_samples(data_all)
        # print(params)
        print(x_fit_h[:10])
        print(y_fit_h[:10])
        # a_all.append(result['alpha'])
        # b_all.append(result['beta'])
        # skew.append(result['skew'])
        x_fit_h = [ts-1 for ts in x_fit_h]
        x_fit[f"H={h:.2f}"] = x_fit_h
        y_fit[f"H={h:.2f}"] = y_fit_h

    return x_fit,y_fit
def get_a_b_loc_scale_from_NE_GB2(plot_eig,Ne,H,everone=False):

    a_all = []
    b_all = []
    x_fit = {}
    y_fit = {}
    skew = []
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")    
        result = fit_generalized_beta_simple(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])
        a_all.append(result['alpha'])
        b_all.append(result['beta'])
        skew.append(result['skew'])
        x_fit[f"H={h:.2f}"] = result['x_fit']
        y_fit[f"H={h:.2f}"] = result['y_fit']

        print("拟合参数：")
        print(f"a={result['alpha']:.4f}, b={result['beta']:.4f}")
        print(f"skew = {result['skew']:.4f}")


    return a_all,b_all,x_fit,y_fit,skew
def get_a_b_loc_scale_from_NE_GB2_2(plot_eig,Ne,H,everone=False):

    a_all = []
    b_all = []
    x_fit = {}
    y_fit = {}
    skew = []
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")    
        x_2,y_2 = fit_weighted_beta(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])
        # a_all.append(result['alpha'])
        # b_all.append(result['beta'])
        # skew.append(result['skew'])
        x_fit[f"H={h:.2f}"] =x_2
        y_fit[f"H={h:.2f}"] = y_2

    return x_fit,y_fit
def get_a_b_loc_scale_from_NE_GB2_3(plot_eig,Ne,H,everone=False):

    a_all = []
    b_all = []
    x_fit = {}
    y_fit = {}
    skew = []
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")    
        x_2,y_2 = fit_mixture_beta(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])

        x_fit[f"H={h:.2f}"] =x_2
        y_fit[f"H={h:.2f}"] = y_2

    return x_fit,y_fit
#计算谱密度
def create_Ne(H,eig_vals):
    plot_eig = {}
    Ne = {}
    for i,h in enumerate(H):
        eig_ = eig_vals[f"H={h:.2f}"]
        _,x,y = get_Ne_from_eig_matrix(eig_,bin_size=80,upper= 1.5,scale=False,print_curvce=True)   #lower=-0.1,upper=0.1
        plot_eig[f"H={h:.2f}"] = x
        Ne[f"H={h:.2f}"] = y
    return plot_eig,Ne,H
# plot_Ne()

# def spectrum_density_plot():

H,eig_vals = read_data()
plot_eig,Ne,H = create_Ne(H,eig_vals)
a_all,b_all,x_fit,y_fit,skew= get_a_b_loc_scale_from_NE_GB2(plot_eig,Ne,H)
# x_fit_2,y_fit_2 = get_a_b_loc_scale_from_NE(eig_vals,H)
x_fit_2,y_fit_2 = get_a_b_loc_scale_from_NE_GB2_3(plot_eig,Ne,H)
# plot_Ne(plot_eig,Ne,H)
len_H = len(H)
# plot_one = True
colors = plt.cm.get_cmap('coolwarm', (len_H))  # 'coolwarm' 是 matplotlib 内置的渐变色
# # 生成 20 个渐变色
color_list = [colors(i) for i in range(len_H)]
color_list = color_list[-len_H:]
width_fig = 14 /  2.54
height_fig = (14/8*6.5) / 2.54
scale_change = 2.54
scale_change_2 = (2.54)*(2.54)
fig = plt.figure(figsize=(width_fig, height_fig))
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 第一行：跨越两列的大子图
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
# 第二行左子图
ax2 = plt.subplot2grid((2, 2), (1, 0))
# ========== (a)图：谱密度 ==========
# fig, ax = plt.subplots(figsize=(14, 7))
for i, h in enumerate(H):
    color = color_list[i]
    # ax1.plot(plot_eig[f"H={h:.2f}"], Ne[f"H={h:.2f}"], c=color)
    ax1.scatter(plot_eig[f"H={h:.2f}"], Ne[f"H={h:.2f}"], c=[color], s=30/scale_change_2, label=rf"H={h:.2f}")
    if h < 0.65:
        ax1.plot(x_fit[f"H={h:.2f}"], y_fit[f"H={h:.2f}"], '--',c=color,linewidth=1.5 / scale_change)
    else:
        ax1.plot(x_fit_2[f"H={h:.2f}"], y_fit_2[f"H={h:.2f}"], '--',c=color,linewidth=1.5 / scale_change)

ax1.legend(fontsize=8,frameon=False)
# ax1.legend(loc='upper left', bbox_to_anchor=(0.18, 0.95))



ax1.set_ylim(top=4)
ax1.set_ylim(bottom=0)
ax1.set_xlim(left=-1)
ax1.set_xlim(right=1.5)
ax1.set_xlabel(r'$\lambda$', fontsize=12)
# ax1.tick_params(labelsize=12)
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax1.xaxis.labelpad = 2
ax1.yaxis.labelpad = 2
# ax.set_ylabel(r'$$\rho(\bar{\lambda})$$', fontsize=14)
ax1.set_ylabel(r'$\bar{\rho}(\lambda)$', fontsize=12)
ax1.text(0.02, 0.12, '(a)', transform=ax1.transAxes,
            fontsize=14,  va='bottom', ha='left')

alpha_1 = []
alpha_2 = []
skew = [0.084,0.200,0.543,0.944,1.265,1.541,1.769,1.986,2.179]
ax2.scatter(H,skew,c='k',s=30/scale_change_2)
ax2.plot(H,skew,c='k',linewidth=1.5 / scale_change)

# ax1.scatter(plot_eig[f"H={h:.2f}"], Ne[f"H={h:.2f}"], c=[color], s=20/scale_change_2
ax2.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax2.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax2.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax2.xaxis.labelpad = 0
ax2.yaxis.labelpad = 0
ax2.set_xlabel(r'$H$', fontsize=12)
ax2.set_ylabel(r'$\langle Skew \rangle$', fontsize=12,labelpad=0)
# ax2.tick_params(labelsize=12)
ax2.text(0.02, 0.12, '(b)', transform=ax2.transAxes,
            fontsize=14,  va='bottom', ha='left')


# 第二行右子图
inset_ax = plt.subplot2grid((2, 2), (1, 1))
omega_bins = np.arange(0, 4.0001, 0.01)
omega_centers = (omega_bins[:-1] + omega_bins[1:]) / 2
P_values = np.arange(0.50, 0.901, 0.05)

g_list = []
R_list = []
for P in P_values:
    filename = f"D:\\Data\\pu\\toe_{P:.2f}_50_1000_p_10.csv"
    df = pd.read_csv(filename, header=None)
    e = -np.sort(-(df.values), axis=0)
    n_rows, n_cols = e.shape
    cumsum_matrix = np.cumsum(e, axis=0)
    col_sums = np.sum(e, axis=0)
    R_values = np.sum(cumsum_matrix / col_sums, axis=0) / n_rows
    R_average = np.mean(R_values) 
    R_list.append(R_average)
    omega_data = np.sqrt(df.values.flatten())
    hist, _ = np.histogram(omega_data, bins=omega_bins, density=False)
    g_list.append(hist / 100)

# 插图
# inset_ax = inset_axes(ax, width="55%", height="55%", loc='upper right', borderpad=0.4)
inset_ax.plot(P_values, R_list, c='k', linewidth=1.5 / scale_change)
inset_ax.scatter(P_values, R_list, c='k', s= 30/scale_change_2)

# inset_ax.tick_params(labelsize=12)
inset_ax.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
inset_ax.tick_params(axis='x', pad=0.5)   # x 轴刻度标签与轴的距离
inset_ax.tick_params(axis='y', pad=0.5)   # y 轴刻度标签与轴的距离
inset_ax.xaxis.labelpad = 1
inset_ax.yaxis.labelpad = 1
inset_ax.set_xlabel(r'$H$', fontsize=12)
inset_ax.set_ylabel(r'$\langle R \rangle$', fontsize=12, labelpad=-1)
inset_ax.set_xlim(0.48, 0.92)
inset_ax.set_ylim(min(R_list) * 0.95, max(R_list) * 1.05)
inset_ax.text(0.02, 0.12, '(c)', transform=inset_ax.transAxes,
            fontsize=14,  va='bottom', ha='left')
# inset_ax.grid(True, alpha=0.3)

# 保存
# plt.tight_layout()
# fig.savefig('SpectralDensity_with_Rinset.png', dpi=600, bbox_inches='tight')
# fig.savefig('SpectralDensity_with_Rinset.eps', dpi=300, format='eps', bbox_inches='tight')
plt.savefig("fig02.png", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig02.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig('fig02.svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig02.eps", dpi=600,format= 'eps', bbox_inches='tight',pad_inches=0.05)
plt.show()
# print(P_values)
# print(R_list)

# # plot_eig,Ne
# x_put = []
# y_put = []
# for i,h in enumerate(H):
#     x = plot_eig[f"H={h:.2f}"]
#     y = Ne[f"H={h:.2f}"]
#     x_put.append(x)
#     y_put.append(y)
# x_matrix = np.column_stack(x_put)
# y_matrix = np.column_stack(y_put)
# merged_data = []
# for i in range(x_matrix.shape[1]):
#     # 添加x_matrix的第i列
#     merged_data.append(x_matrix[:, i])
#     # 添加y_matrix的第i列
#     merged_data.append(y_matrix[:, i])

# # 转换为DataFrame
# merged_df = pd.DataFrame(np.column_stack(merged_data))

# # 设置列名（可选）
# column_names = []
# for i in range(x_matrix.shape[1]):
#     column_names.append(f'X_H{i+1}')
#     column_names.append(f'Y_H{i+1}')
# merged_df.columns = column_names

# 保存为CSV文件
