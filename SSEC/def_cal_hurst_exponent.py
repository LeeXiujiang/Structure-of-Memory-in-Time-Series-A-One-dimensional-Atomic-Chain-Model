import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 生成fbm series
def generate_fbm(hurst, n=1000,diff = True):
    """生成分数布朗运动"""
    t = np.arange(n)
    cov = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.5 * (abs(t[i])**(2*hurst) + abs(t[j])**(2*hurst) - abs(t[i]-t[j])**(2*hurst))
    
    # 使用Cholesky分解生成相关序列
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    fbm = L @ np.random.randn(n)
    if diff:
        fbm_diff = np.diff(fbm)
    # fbm_diff = np.diff(fbm)
        return fbm_diff
    else:
        return fbm


def hurst_rs(time_series):
    """
    R/S分析法计算Hurst指数
    """
    n = len(time_series)
    max_k = int(np.floor(n / 2))
    
    # 计算不同时间窗口的R/S值
    rs_values = []
    window_sizes = []
    
    for k in range(10, max_k + 1):
        # 将序列分成m个长度为k的子序列
        m = int(np.floor(n / k))
        rs_k = []
        
        for i in range(m):
            # 每个子序列
            sub_series = time_series[i*k:(i+1)*k]
            if len(sub_series) < 2:
                continue
                
            # 计算累积离差
            mean_val = np.mean(sub_series)
            deviations = sub_series - mean_val
            cumulative_deviations = np.cumsum(deviations)
            
            # 计算极差R
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # 计算标准差S
            S = np.std(sub_series, ddof=1)
            
            if S > 0:
                rs_k.append(R / S)
        
        if len(rs_k) > 0:
            rs_values.append(np.mean(rs_k))
            window_sizes.append(k)
    
    # 对数变换后线性回归
    log_rs = np.log(rs_values)
    log_n = np.log(window_sizes)
    
    # 计算Hurst指数
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
    hurst = slope
    
    return hurst, (log_n, log_rs), (slope, intercept)

# DFA 去趋势波动分析法计算Hurst指数
def hurst_dfa(time_series, min_box_size=10, max_box_size=None, trend_order=3):
    """
    DFA方法计算Hurst指数
    改进版：支持p阶多项式去趋势
    
    参数:
    time_series: 输入时间序列
    min_box_size: 最小窗口大小
    max_box_size: 最大窗口大小  
    trend_order: 去趋势多项式阶数 (1=线性, 2=二次, 3=三次, 等)
    """
    n = len(time_series)
    if max_box_size is None:
        max_box_size = n // 4
    
    # 步骤1: 计算累积离差序列
    y = np.cumsum(time_series - np.mean(time_series))
    
    # 步骤2: 不同窗口大小的波动分析
    window_sizes = []
    fluctuations = []
    
    # 使用几何序列作为窗口大小
    box_sizes = []
    current_size = min_box_size
    while current_size <= max_box_size:
        box_sizes.append(current_size)
        current_size = int(current_size * 1.2)
    
    for box_size in box_sizes:
        if box_size > len(y) or box_size < trend_order + 2:
            # 窗口大小必须大于多项式阶数+1
            continue
            
        # 将序列分成多个不重叠的窗口
        n_boxes = len(y) // box_size
        if n_boxes < 1:
            continue
            
        f_n = []
        
        for i in range(n_boxes):
            # 每个窗口的数据
            start_idx = i * box_size
            end_idx = start_idx + box_size
            segment = y[start_idx:end_idx]
            
            # 去趋势 - p阶多项式拟合
            x = np.arange(box_size)
            
            try:
                # 多项式拟合
                coefficients = np.polyfit(x, segment, trend_order)
                # 计算趋势
                trend = np.polyval(coefficients, x)
                
                # 计算去趋势后的波动
                detrended = segment - trend
                f_n.append(np.sqrt(np.mean(detrended**2)))
                
            except (np.linalg.LinAlgError, ValueError):
                # 如果拟合失败，跳过该窗口
                continue
        
        # 平均波动（至少需要一些有效的窗口）
        if len(f_n) > 0:
            fluctuations.append(np.mean(f_n))
            window_sizes.append(box_size)
    
    if len(fluctuations) < 3:
        raise ValueError(f"有效窗口数量不足 ({len(fluctuations)})，无法进行回归分析。请调整参数。")
    
    # 步骤3: 对数线性回归
    log_f = np.log(fluctuations)
    log_n = np.log(window_sizes)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_f)
    hurst = slope
    
    return hurst, (log_n, log_f), (slope, intercept)

def hurst_dfa_advanced(time_series, min_box_size=10, max_box_size=None, trend_order=1, 
                      box_size_method='geometric', overlap=False):
    """
    增强版DFA方法，提供更多选项
    
    参数:
    time_series: 输入时间序列
    min_box_size: 最小窗口大小
    max_box_size: 最大窗口大小
    trend_order: 去趋势多项式阶数
    box_size_method: 窗口大小生成方法 ('geometric', 'linear', 'log')
    overlap: 是否使用重叠窗口
    """
    n = len(time_series)
    if max_box_size is None:
        max_box_size = n // 4
    
    # 计算累积离差序列
    y = np.cumsum(time_series - np.mean(time_series))
    
    # 生成窗口大小序列
    if box_size_method == 'geometric':
        box_sizes = []
        current_size = min_box_size
        while current_size <= max_box_size:
            if current_size >= trend_order + 2:  # 确保窗口足够大
                box_sizes.append(current_size)
            current_size = int(current_size * 1.2)
            
    elif box_size_method == 'linear':
        box_sizes = np.linspace(min_box_size, max_box_size, 20, dtype=int)
        box_sizes = [bs for bs in box_sizes if bs >= trend_order + 2]
        
    elif box_size_method == 'log':
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), 20, dtype=int)
        box_sizes = [bs for bs in box_sizes if bs >= trend_order + 2]
    
    box_sizes = np.unique(box_sizes)  # 去重
    
    window_sizes = []
    fluctuations = []
    
    for box_size in box_sizes:
        if box_size > len(y):
            continue
            
        if overlap:
            # 重叠窗口
            n_boxes = len(y) - box_size + 1
        else:
            # 不重叠窗口
            n_boxes = len(y) // box_size
        
        if n_boxes < 1:
            continue
            
        f_n = []
        
        for i in range(n_boxes):
            if overlap:
                start_idx = i
                end_idx = i + box_size
            else:
                start_idx = i * box_size
                end_idx = start_idx + box_size
            
            if end_idx > len(y):
                break
                
            segment = y[start_idx:end_idx]
            x = np.arange(len(segment))
            
            try:
                # 多项式去趋势
                coefficients = np.polyfit(x, segment, trend_order)
                trend = np.polyval(coefficients, x)
                detrended = segment - trend
                
                # 计算均方根波动
                rms_fluctuation = np.sqrt(np.mean(detrended**2))
                f_n.append(rms_fluctuation)
                
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if len(f_n) > 0:
            fluctuations.append(np.mean(f_n))
            window_sizes.append(box_size)
    
    if len(fluctuations) < 3:
        raise ValueError(f"有效数据点不足 ({len(fluctuations)})，请调整参数")
    
    # 对数线性回归
    log_f = np.log(fluctuations)
    log_n = np.log(window_sizes)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_f)
    hurst = slope
    
    results = {
        'hurst': hurst,
        'window_sizes': window_sizes,
        'fluctuations': fluctuations,
        'log_n': log_n,
        'log_f': log_f,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'trend_order': trend_order
    }
    
    return hurst, results


# 方差法计算Hurst指数
def hurst_variance(time_series):
    """
    方差法计算Hurst指数
    """
    n = len(time_series)
    max_m = int(np.floor(n / 10))
    
    variances = []
    window_sizes = []
    
    for m in range(2, max_m + 1):
        # 计算聚合序列的方差
        k = n // m
        aggregated = []
        
        for i in range(k):
            segment = time_series[i*m:(i+1)*m]
            aggregated.append(np.mean(segment))
        
        if len(aggregated) > 1:
            var_m = np.var(aggregated, ddof=1)
            variances.append(var_m)
            window_sizes.append(m)
    
    # 对数线性回归
    log_var = np.log(variances)
    log_m = np.log(window_sizes)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m, log_var)
    hurst = 1 + slope / 2
    
    return hurst, (log_m, log_var), (slope, intercept)






# 测试数据 - 生成分数布朗运动

from hurst import compute_Hc, random_walk
"""
# 测试R/S方法
fbm_05 = generate_fbm(0.5, 1000)  # 随机游走
fbm_07 = generate_fbm(0.7, 1000)  # 持久性序列
fbm_03 = generate_fbm(0.3, 1000)  # 反持久性序列

hurst_05, data_05, reg_05 = hurst_rs(fbm_05)
hurst_07, data_07, reg_07 = hurst_rs(fbm_07)
hurst_03, data_03, reg_03 = hurst_rs(fbm_03)

print(f"Hurst指数 (理论0.5): {hurst_05:.3f}")
print(f"Hurst指数 (理论0.7): {hurst_07:.3f}")
print(f"Hurst指数 (理论0.3): {hurst_03:.3f}")

h_3,_,_ = compute_Hc( generate_fbm(0.3, 1000,False), kind="random_walk")
print(f"\nR/S Hurst指数 (理论0.5): {h_3:.3f}")
h_5,_,_ = compute_Hc(generate_fbm(0.5, 1000,False), kind="random_walk")
print(f"R/S Hurst指数 (理论0.7): {h_5:.3f}")
h_3,_,_ = compute_Hc(generate_fbm(0.7, 1000,False), kind="random_walk")
print(f"R/S Hurst指数 (理论0.3): {h_3:.3f}")



# 测试DFA方法
hurst_dfa_05, dfa_data_05, dfa_reg_05 = hurst_dfa(fbm_05)
hurst_dfa_07, dfa_data_07, dfa_reg_07 = hurst_dfa(fbm_07)
hurst_dfa_03, dfa_data_03, dfa_reg_03 = hurst_dfa(fbm_03)

print(f"\nDFA Hurst指数 (理论0.5): {hurst_dfa_05:.3f}")
print(f"DFA Hurst指数 (理论0.7): {hurst_dfa_07:.3f}")
print(f"DFA Hurst指数 (理论0.3): {hurst_dfa_03:.3f}")



# 测试方差法
hurst_var_05, var_data_05, var_reg_05 = hurst_variance(fbm_05)
hurst_var_07, var_data_07, var_reg_07 = hurst_variance(fbm_07)
hurst_var_03, var_data_03, var_reg_03 = hurst_variance(fbm_03)

print(f"\n方差法 Hurst指数 (理论0.5): {hurst_var_05:.3f}")
print(f"方差法 Hurst指数 (理论0.7): {hurst_var_07:.3f}")
print(f"方差法 Hurst指数 (理论0.3): {hurst_var_03:.3f}")
"""

def hurst_aggregation(time_series):
    """
    聚合方法计算Hurst指数
    """
    n = len(time_series)
    max_agg = int(np.floor(np.log2(n))) - 2
    
    std_devs = []
    agg_levels = []
    
    for k in range(1, max_agg + 1):
        agg_size = 2**k
        n_agg = n // agg_size
        
        # 创建聚合序列
        aggregated = []
        for i in range(n_agg):
            start_idx = i * agg_size
            end_idx = start_idx + agg_size
            aggregated.append(np.mean(time_series[start_idx:end_idx]))
        
        # 计算聚合序列的标准差
        if len(aggregated) > 1:
            std_dev = np.std(aggregated, ddof=1)
            std_devs.append(std_dev)
            agg_levels.append(agg_size)
    
    # 对数线性回归
    log_std = np.log(std_devs)
    log_agg = np.log(agg_levels)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_agg, log_std)
    hurst = 1 + slope
    
    return hurst, (log_agg, log_std), (slope, intercept)

"""
# 测试聚合方法
hurst_agg_05, agg_data_05, agg_reg_05 = hurst_aggregation(fbm_05)
hurst_agg_07, agg_data_07, agg_reg_07 = hurst_aggregation(fbm_07)
hurst_agg_03, agg_data_03, agg_reg_03 = hurst_aggregation(fbm_03)

print(f"\n聚合方法 Hurst指数 (理论0.5): {hurst_agg_05:.3f}")
print(f"聚合方法 Hurst指数 (理论0.7): {hurst_agg_07:.3f}")
print(f"聚合方法 Hurst指数 (理论0.3): {hurst_agg_03:.3f}")

def plot_hurst_comparison():
    # 比较不同方法的Hurst指数计算结果
    methods = ['R/S', 'DFA', 'Variance', 'Aggregation']
    hurst_05_all = [hurst_05, hurst_dfa_05, hurst_var_05, hurst_agg_05]
    hurst_07_all = [hurst_07, hurst_dfa_07, hurst_var_07, hurst_agg_07]
    hurst_03_all = [hurst_03, hurst_dfa_03, hurst_var_03, hurst_agg_03]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, hurst_05_all, width, label='理论H=0.5', alpha=0.8)
    bars2 = ax.bar(x, hurst_07_all, width, label='理论H=0.7', alpha=0.8)
    bars3 = ax.bar(x + width, hurst_03_all, width, label='理论H=0.3', alpha=0.8)
    
    # 添加理论值参考线
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('计算方法')
    ax.set_ylabel('Hurst指数')
    ax.set_title('不同方法计算的Hurst指数比较')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体 SimHei
    plt.tight_layout()
    plt.show()

# 绘制比较图
plot_hurst_comparison()

def calculate_all_hurst(time_series, method='all'):
    
    # 综合计算Hurst指数的函数
    
    results = {}
    
    if method in ['all', 'rs']:
        results['rs'] = hurst_rs(time_series)[0]
    
    if method in ['all', 'dfa']:
        results['dfa'] = hurst_dfa(time_series)[0]
    
    if method in ['all', 'variance']:
        results['variance'] = hurst_variance(time_series)[0]
    
    if method in ['all', 'aggregation']:
        results['aggregation'] = hurst_aggregation(time_series)[0]
    
    return results

# 使用示例
test_data = generate_fbm(0.6, 2000)  # 生成H=0.6的分数布朗运动
all_hurst = calculate_all_hurst(test_data)

print("\n所有方法的Hurst指数计算结果:")
for method, value in all_hurst.items():
    print(f"{method.upper()}: {value:.4f}")

print(f"\n平均Hurst指数: {np.mean(list(all_hurst.values())):.4f}")
"""

