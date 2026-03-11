H = [0.5 , 0.55 ,0.6,  0.65, 0.7 , 0.75, 0.8,  0.85, 0.9 ,1]
H_beat = [0.08052694112156167, 0.09908370582849098, 0.15694565940203675, 0.2253259575424634, 0.34037245422889434, 0.48147881528421266, 0.5802854945238303, 0.6655905888266207, 0.7462572135118211]
# print(len(H))
# print(len(H_beat))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
# 数据
H = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,1]
H_beat = [0.08052694112156167, 0.09908370582849098, 0.15694565940203675, 
          0.2253259575424634, 0.34037245422889434, 0.48147881528421266, 
          0.5802854945238303, 0.6655905888266207, 0.7462572135118211,1]

# 定义几种可能的拟合函数
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def power_func(x, a, b):
    return a * x**b

def sigmoid_func(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d
def H_beta_linear_func(x, a=1.9134, b=-0.9592):
    return a * x + b

def H_beta_quadratic_func(x, a=1.2846, b=0.0047, c=-0.2801):
    return a * x**2 + b * x + c

def H_beta_exponential_func(x, a=0.6070, b=1.2403, c=-1.0907):
    return a * np.exp(b * x) + c

def H_beta_power_func(x, a=1.0449, b=3.1601):
    return a * x**b

def H_beta_sigmoid_func(x, a=1.4756, b=6.2479, c=0.8323, d=-0.1070):
    return a / (1 + np.exp(-b * (x - c))) + d
def Beta_H_fit(x, n=3):
    """
    随机抽取n种拟合方法，计算x的拟合结果并取平均
    
    参数:
    x: 输入值或数组
    n: 随机抽取的方法数量 (1 <= n <= 5)
    
    返回:
    tuple: (平均值结果, 使用的函数列表, 各函数结果列表)
    """
    if n < 1 or n > 5:
        raise ValueError("n必须在1到5之间")
    
    # 所有拟合函数的列表
    all_functions = [
        ("线性拟合", H_beta_linear_func),
        ("二次拟合", H_beta_quadratic_func),
        ("指数拟合", H_beta_exponential_func),
        ("幂函数拟合", H_beta_power_func),
        ("S型函数拟合", H_beta_sigmoid_func)
    ]
    
    # 随机选择n个函数
    selected_functions = random.sample(all_functions, n)
    
    # print(f"随机选择了 {n} 种拟合方法:")
    # for i, (name, func) in enumerate(selected_functions, 1):
        # print(f"  {i}. {name}")
    
    # 计算每个函数的结果
    results = []
    function_names = []
    
    for name, func in selected_functions:
        try:
            result = func(x)
            results.append(result)
            function_names.append(name)
            # print(f"  {name}: {result}")
        except Exception as e:
            print(f"  {name} 计算失败: {e}")
            # 如果某个函数计算失败，移除它
            continue
    
    if not results:
        raise ValueError("所有选择的拟合方法都计算失败")
    
    # 计算平均值
    if isinstance(x, (list, np.ndarray)):
        # 对于数组输入，按元素求平均
        avg_result = np.mean(results, axis=0)
    else:
        # 对于单个数值输入
        avg_result = np.mean(results)
    
    # print(f"平均值结果: {avg_result}")
    
    return avg_result, function_names, results
    

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    data = np.array(data)
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]
def beta_Hurst_SSEC(RS=False,quchu=True):
    years = np.arange(2019, 2025, 1)
    beat_SSEC = []
    dfa_data, rs_data = [], []
    # H_list = [0.587,0.580,0.550,0.560,0.570,0.535]
    for i, year_val in enumerate(years):
        data_name = f"D:\\Data\\real\\Hurst_SSEC_{str(year_val)}_{str(year_val)}.csv"
        
        df = pd.read_csv(data_name, index_col=None, parse_dates=False)
        hurst_year_matrix = df.values
        hurst_year_matrix = np.sort(hurst_year_matrix,axis=0)
        # quchu = True
        if quchu:            
            # 对hurst_year_matrix的两列分别处理
            col0_filtered = remove_outliers(hurst_year_matrix[:, 0],lower_percentile=10)
            col1_filtered = remove_outliers(hurst_year_matrix[:, 1],lower_percentile=10)
            
            dfa_data.append(col0_filtered)
            rs_data.append(col1_filtered)   
        else:
            dfa_data.append(hurst_year_matrix[:, 0])
            rs_data.append(hurst_year_matrix[:, 1])


    for i, y in enumerate(years):
        dfa_hurst = dfa_data[i]
        beat_ = []
        for j in range(len(dfa_hurst)):
            beta_j = H_beta_linear_func(dfa_hurst[j])
            beat_.append(beta_j)
        # beat_ = Beta_H_fit(dfa_hurst,3)

        beat_SSEC.append(np.mean(beat_))
    return beat_SSEC

def beta_Hurst_HSI(RS = False,quchu=True):
    years = np.arange(2019, 2025, 1)
    beat_HSI = []
    dfa_data, rs_data = [], []

    for i, year_val in enumerate(years):
        data_name = f"D:\\Data\\real\\Hurst_HSI_{str(year_val)}_{str(year_val)}.csv"
        df = pd.read_csv(data_name, index_col=None, parse_dates=False)
        hurst_year_matrix = df.values
        hurst_year_matrix = np.sort(hurst_year_matrix,axis=0)
        # quchu = True
        if quchu:            
            # 对hurst_year_matrix的两列分别处理
            col0_filtered = remove_outliers(hurst_year_matrix[:, 0],lower_percentile=10)
            col1_filtered = remove_outliers(hurst_year_matrix[:, 1],lower_percentile=10)
            
            dfa_data.append(col0_filtered)
            rs_data.append(col1_filtered)   
        else:
            dfa_data.append(hurst_year_matrix[:, 0])
            rs_data.append(hurst_year_matrix[:, 1])
    for i, y in enumerate(years):
        dfa_hurst = dfa_data[i]
        beat_ = []
        for j in range(len(dfa_hurst)):
            beta_j = H_beta_linear_func(dfa_hurst[j])
            beat_.append(beta_j)
        # beat_ = Beta_H_fit(dfa_hurst,3)
        
        beat_HSI.append(np.mean(beat_))
    # for i, y in enumerate(years):
    #     dfa_hurst = dfa_data[i]
    #     beat_ = Beta_H_fit(dfa_hurst,3)
    #     beat_HSI.append(np.mean(beat_))
    return beat_HSI


















        # all_data.append(dfa_data[i])
        # if RS:
        #     all_data.append(rs_data[i])
        #     positions.extend([i*2 + 0.8, i*2 + 1.2])
        # else:
        #     positions = np.arange(len(years))  # 位置从 0 到 5
        #     labels = [str(year) for year in years]  # 标签为年份
        # labels.extend([f'{y}\nDFA', f'{y}\nRS'])

    # box_plot = ax.boxplot(all_data, positions=positions, widths=0.3, patch_artist=True,showfliers=False)

    # 设置颜色（交替颜色）
    # if RS:
    #     colors = ['lightblue', 'lightcoral'] * len(years)
    # else:
    #     color_a = "#6387BB"
    #     color_b = "#DA635C"
    #     colors = [color_a, color_a] *len(years)
    # for patch, color in zip(box_plot['boxes'], colors):
    #     patch.set_facecolor(color)

    # # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    # ax.set_xlabel('Year',fontsize =14)
    # ax.set_ylabel('Hurst Exponent')
    # ax.set_xticks(positions, labels, rotation=45)
    # ax.grid(True, alpha=0.3)
        # for j in range(len(dfa_hurst)):
        #     all_data.append(dfa_hurst[j])
        # positions.extend([i*2 + 0.8, i*2 + 1.2])
        # all_data.append(dfa_data[i])
        # if RS:
        #     all_data.append(rs_data[i])
        #     positions.extend([i*2 + 0.8, i*2 + 1.2])
        # else:
        #     positions = np.arange(len(years))  # 位置从 0 到 5
        #     labels = [str(year) for year in years]  # 标签为年份
        # labels.extend([f'{y}\nDFA', f'{y}\nRS'])

    # box_plot = ax.boxplot(all_data, positions=positions, widths=0.3, patch_artist=True,showfliers=False)

    # 设置颜色（交替颜色）
    # colors = ['lightblue', 'lightcoral'] * len(years)
    # if RS:
    #     colors = ['lightblue', 'lightcoral'] * len(years)
    # else:
    #     color_a = "#6387BB"
    #     color_b = "#DA635C"
    #     colors = [color_a, color_a] *len(years)
    # for patch, color in zip(box_plot['boxes'], colors):
    #     patch.set_facecolor(color)

    # # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    # ax.set_xlabel('Year',fontsize =14)
    # ax.set_ylabel('Hurst Exponent',fontsize =14)
    # ax.set_xticks(positions, labels, rotation=45)
    # ax.grid(True, alpha=0.3)

#线性拟合
# 线性拟合
# try:
#     popt_linear, pcov_linear = curve_fit(linear_func, H, H_beat)
#     H_fit_linear = linear_func(np.array(H), *popt_linear)
#     # plt.subplot(2, 3, 2)
#     # plt.plot(H, H_beat, 'bo', label='原始数据')
#     # plt.plot(H, H_fit_linear, 'r-', label=f'线性拟合: y={popt_linear[0]:.3f}x+{popt_linear[1]:.3f}')
#     # plt.xlabel('H')
#     # plt.ylabel('H_beat')
#     # plt.title('线性拟合')
#     # plt.legend()
#     # plt.grid(True)
#     # print(f"线性拟合: y = {popt_linear[0]:.4f} * x + {popt_linear[1]:.4f}")
# except:
#     pass
    # print("线性拟合失败")
# 尝试不同的拟合
"""
plt.figure(figsize=(12, 8))
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体 SimHei
# 原始数据
plt.subplot(2, 3, 1)
plt.plot(H, H_beat, 'bo-', label='原始数据')
plt.xlabel('H')
plt.ylabel('H_beat')
plt.title('原始数据')
plt.legend()
plt.grid(True)

# 线性拟合
try:
    popt_linear, pcov_linear = curve_fit(linear_func, H, H_beat)
    H_fit_linear = linear_func(np.array(H), *popt_linear)
    plt.subplot(2, 3, 2)
    plt.plot(H, H_beat, 'bo', label='原始数据')
    plt.plot(H, H_fit_linear, 'r-', label=f'线性拟合: y={popt_linear[0]:.3f}x+{popt_linear[1]:.3f}')
    plt.xlabel('H')
    plt.ylabel('H_beat')
    plt.title('线性拟合')
    plt.legend()
    plt.grid(True)
    print(f"线性拟合: y = {popt_linear[0]:.4f} * x + {popt_linear[1]:.4f}")
except:
    print("线性拟合失败")

# 二次拟合
try:
    popt_quad, pcov_quad = curve_fit(quadratic_func, H, H_beat)
    H_fit_quad = quadratic_func(np.array(H), *popt_quad)
    plt.subplot(2, 3, 3)
    plt.plot(H, H_beat, 'bo', label='原始数据')
    plt.plot(H, H_fit_quad, 'r-', label=f'二次拟合: y={popt_quad[0]:.3f}x²+{popt_quad[1]:.3f}x+{popt_quad[2]:.3f}')
    plt.xlabel('H')
    plt.ylabel('H_beat')
    plt.title('二次拟合')
    plt.legend()
    plt.grid(True)
    print(f"二次拟合: y = {popt_quad[0]:.4f} * x² + {popt_quad[1]:.4f} * x + {popt_quad[2]:.4f}")
except:
    print("二次拟合失败")

# 指数拟合
try:
    popt_exp, pcov_exp = curve_fit(exponential_func, H, H_beat, maxfev=5000)
    H_fit_exp = exponential_func(np.array(H), *popt_exp)
    plt.subplot(2, 3, 4)
    plt.plot(H, H_beat, 'bo', label='原始数据')
    plt.plot(H, H_fit_exp, 'r-', label=f'指数拟合: y={popt_exp[0]:.3f}exp({popt_exp[1]:.3f}x)+{popt_exp[2]:.3f}')
    plt.xlabel('H')
    plt.ylabel('H_beat')
    plt.title('指数拟合')
    plt.legend()
    plt.grid(True)
    print(f"指数拟合: y = {popt_exp[0]:.4f} * exp({popt_exp[1]:.4f} * x) + {popt_exp[2]:.4f}")
except:
    print("指数拟合失败")

# 幂函数拟合
try:
    popt_power, pcov_power = curve_fit(power_func, H, H_beat, maxfev=5000)
    H_fit_power = power_func(np.array(H), *popt_power)
    plt.subplot(2, 3, 5)
    plt.plot(H, H_beat, 'bo', label='原始数据')
    plt.plot(H, H_fit_power, 'r-', label=f'幂函数拟合: y={popt_power[0]:.3f}x^{popt_power[1]:.3f}')
    plt.xlabel('H')
    plt.ylabel('H_beat')
    plt.title('幂函数拟合')
    plt.legend()
    plt.grid(True)
    print(f"幂函数拟合: y = {popt_power[0]:.4f} * x^{popt_power[1]:.4f}")
except:
    print("幂函数拟合失败")

# S型函数拟合
try:
    popt_sigmoid, pcov_sigmoid = curve_fit(sigmoid_func, H, H_beat, maxfev=5000)
    H_fit_sigmoid = sigmoid_func(np.array(H), *popt_sigmoid)
    plt.subplot(2, 3, 6)
    plt.plot(H, H_beat, 'bo', label='原始数据')
    plt.plot(H, H_fit_sigmoid, 'r-', label='S型函数拟合')
    plt.xlabel('H')
    plt.ylabel('H_beat')
    plt.title('S型函数拟合')
    plt.legend()
    plt.grid(True)
    print(f"S型函数拟合: y = {popt_sigmoid[0]:.4f} / (1 + exp(-{popt_sigmoid[1]:.4f}(x-{popt_sigmoid[2]:.4f}))) + {popt_sigmoid[3]:.4f}")
except:
    print("S型函数拟合失败")

plt.tight_layout()
plt.show()

# 计算拟合优度 R²
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

print("\n=== 拟合优度比较 ===")
H_array = np.array(H)
H_beat_array = np.array(H_beat)

fits = []
try:
    y_pred = linear_func(H_array, *popt_linear)
    r2 = calculate_r_squared(H_beat_array, y_pred)
    fits.append(("线性拟合", r2))
except: pass

try:
    y_pred = quadratic_func(H_array, *popt_quad)
    r2 = calculate_r_squared(H_beat_array, y_pred)
    fits.append(("二次拟合", r2))
except: pass

try:
    y_pred = exponential_func(H_array, *popt_exp)
    r2 = calculate_r_squared(H_beat_array, y_pred)
    fits.append(("指数拟合", r2))
except: pass

try:
    y_pred = power_func(H_array, *popt_power)
    r2 = calculate_r_squared(H_beat_array, y_pred)
    fits.append(("幂函数拟合", r2))
except: pass

try:
    y_pred = sigmoid_func(H_array, *popt_sigmoid)
    r2 = calculate_r_squared(H_beat_array, y_pred)
    fits.append(("S型函数拟合", r2))
except: pass

# 按R²值排序
fits.sort(key=lambda x: x[1], reverse=True)
for name, r2 in fits:
    print(f"{name}: R² = {r2:.6f}")
    """