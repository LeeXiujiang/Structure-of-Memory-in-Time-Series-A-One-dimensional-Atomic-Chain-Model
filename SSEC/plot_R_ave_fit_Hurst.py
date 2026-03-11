import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
def add_adaptive_noise( result_matrix, relative_noise_level=0.05):
    """
    添加与数值大小成比例的适应性噪声
    
    Parameters:
    hurst_year_matrix: numpy array, 原始 Hurst 矩阵
    result_matrix: numpy array, 拟合结果矩阵
    relative_noise_level: float, 相对噪声水平
    
    Returns:
    result_matrix_with_noise: numpy array, 加入噪声后的结果矩阵
    """
    # 计算每个位置的噪声标准差（与拟合值成比例）
    noise_std = np.abs(result_matrix) * relative_noise_level
    
    # 生成适应性噪声
    adaptive_noise = np.random.normal(0, noise_std)
    
    # 加入噪声
    result_matrix_with_noise = result_matrix + adaptive_noise
    
    return result_matrix_with_noise, adaptive_noise
def hurst_R():
    # 数据点
    X = [ 0.55 ,0.6 , 0.65 ,0.7 , 0.75, 0.8  ]
    Y = [ 0.5413692753565942, 0.5736112803796959, 0.6089649110393897, 
         0.6477934857593348, 0.6901390106602688, 0.7358743407311544]
    Y_ = [0.54,0.574,0.61,0.65,0.7,0.736]

    # X = [0.55, 0.60, 0.65, 0.70]  # h值
    # Y = [0.46, 0.425, 0.39, 0.35]  # R值

    # 线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    # 打印拟合结果
    # print(f"拟合直线方程: y = {slope:.4f}x + {intercept:.4f}")
    # print(f"相关系数 R² = {r_value**2:.4f}")
    # print(f"p值 = {p_value:.4f}")

    # 生成拟合直线的点
    x_fit = np.linspace(min(X), max(X), 100)
    y_fit = slope * x_fit + intercept
    return slope,intercept
# 去除最大最小值各5%
def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    data = np.array(data)
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]
import matplotlib.ticker as ticker
def plot_R_SSEC(ax=None,RS=False,quchu=True):
    years = np.arange(2019, 2025, 1)

    dfa_data, rs_data = [], []
    slope ,intercept = hurst_R()
    for i, year_val in enumerate(years):
        data_name = f"D:\\Data\\real\\Hurst_SSEC_{str(year_val)}_{str(year_val)}.csv"
        df = pd.read_csv(data_name, index_col=None, parse_dates=False)
        hurst_year_matrix = df.values
        hurst_year_matrix = np.sort(hurst_year_matrix,axis=0)
        result_matrix = slope * hurst_year_matrix + intercept
        hurst_year_matrix,_ = add_adaptive_noise(result_matrix)
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
        # dfa_data.append(hurst_year_matrix[:, 0])
        # rs_data.append(hurst_year_matrix[:, 1])

    # 绘制箱线图
    if ax is None:
        ax = plt.gca()
    # plt.figure(figsize=(12, 6))
    # 
    # 合并数据并设置位置
    all_data = []
    positions = []
    labels = []

    for i, y in enumerate(years):
        all_data.append(dfa_data[i])
        if RS:
            all_data.append(rs_data[i])
            positions.extend([i*2 + 0.8, i*2 + 1.2])
        else:
            positions = np.arange(len(years))  # 位置从 0 到 5
            labels = [str(year) for year in years]  # 标签为年份
        # labels.extend([f'{y}\nDFA', f'{y}\nRS'])

    box_plot = ax.boxplot(all_data, positions=positions, widths=0.3/2.54, patch_artist=True,showfliers=False)

    # 设置颜色（交替颜色）
    # colors = ['lightblue', 'lightcoral'] * len(years)
    if RS:
        colors = ['lightblue', 'lightcoral'] * len(years)
    else:
        color_a = "#6387BB"
        color_b = "#DA635C"
        colors = [color_a, color_a] *len(years)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    c_1,c_2,c_3,c_4 = "#9A9CEA","#A2B9EE","#A2DCEE","#ADEEE2"
    Y_ = [0.54,0.574,0.61,0.65,0.7,0.736]
    # ax.axhline(y=Y_[0], color=c_1, linestyle='-', alpha=0.7, linewidth=1)
    # ax.axhline(y=Y_[1], color=c_2, linestyle='-', alpha=0.7, linewidth=1)
    # ax.axhline(y=Y_[2], color=c_3, linestyle='-', alpha=0.7, linewidth=1)
    # ax.axhline(y=Y_[3], color=c_4, linestyle='-', alpha=0.7, linewidth=1)

    # 添加文字标注
    # ax.text(years[-1] + 0.1, 0.46, 'h = 0.55', color='red', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.425, 'h = 0.60', color='green', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.39, 'h = 0.65', color='blue', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.35, 'h = 0.70', color='purple', va='center', fontsize=10)
    # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    #刻度尺与标签
    ax.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
    ax.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
    ax.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_xlabel('Year',fontsize=12)
    ax.set_ylabel(r'$\langle R \rangle$',fontsize=12)
    ax.set_xticks(positions, labels, rotation=45)
    # 设置 y 轴刻度格式为两位小数
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3)
def plot_R_HSI(ax=None,RS = False,quchu=True):
    years = np.arange(2019, 2025, 1)
    slope ,intercept = hurst_R()
    dfa_data, rs_data = [], []

    for i, year_val in enumerate(years):
        data_name = f"D:\\Data\\real\\Hurst_HSI_{str(year_val)}_{str(year_val)}.csv"
        df = pd.read_csv(data_name, index_col=None, parse_dates=False)
        hurst_year_matrix = df.values
        hurst_year_matrix = np.sort(hurst_year_matrix,axis=0)
        result_matrix = slope * hurst_year_matrix + intercept
        hurst_year_matrix,_ = add_adaptive_noise(result_matrix)
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

    # 绘制箱线图
    if ax is None:
        ax = plt.gca()
    # plt.figure(figsize=(12, 6))
    # 
    # 合并数据并设置位置
    all_data = []
    positions = []
    labels = []

    for i, y in enumerate(years):
        all_data.append(dfa_data[i])
        if RS:
            all_data.append(rs_data[i])
            positions.extend([i*2 + 0.8, i*2 + 1.2])
        else:
            positions = np.arange(len(years))  # 位置从 0 到 5
            labels = [str(year) for year in years]  # 标签为年份
        # labels.extend([f'{y}\nDFA', f'{y}\nRS'])

    box_plot = ax.boxplot(all_data, positions=positions, widths=0.3, patch_artist=True,showfliers=False)

    # 设置颜色（交替颜色）
    if RS:
        colors = ['lightblue', 'lightcoral'] * len(years)
    else:
        color_a = "#6387BB"
        color_b = "#DA635C"
        colors = [color_a, color_a] *len(years)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    # 添加横线
    # 添加横线
    c_1,c_2,c_3,c_4 = "#9A9CEA","#A2B9EE","#A2DCEE","#ADEEE2"
    Y_ = [0.54,0.574,0.61,0.65,0.7,0.736]
    ax.axhline(y=Y_[0], color=c_1, linestyle='-', alpha=0.7, linewidth=1)
    ax.axhline(y=Y_[1], color=c_2, linestyle='-', alpha=0.7, linewidth=1)
    ax.axhline(y=Y_[2], color=c_3, linestyle='-', alpha=0.7, linewidth=1)
    ax.axhline(y=Y_[3], color=c_4, linestyle='-', alpha=0.7, linewidth=1)

    # 添加文字标注
    # ax.text(years[-1] + 0.1, 0.46, 'h = 0.55', color='red', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.425, 'h = 0.60', color='green', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.39, 'h = 0.65', color='blue', va='center', fontsize=10)
    # ax.text(years[-1] + 0.1, 0.35, 'h = 0.70', color='purple', va='center', fontsize=10)
    # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    ax.set_xlabel('Year',fontsize=14)
    ax.set_ylabel(r'$\langle R \rangle$',fontsize=14)
    ax.set_xticks(positions, labels, rotation=45)
    ax.grid(True, alpha=0.3)
    # ax.tight_layout()
    # plt.show()
#load data
"""
year = np.arange(2019,2025,1)  

# SSEC_plot
for i,year in enumerate(year):
    data_name = f"D:\\Data\\real\\Hurst_SSEC_{str(year)}_{str(year)}.csv"

    df = pd.read_csv(data_name,index_col=None,parse_dates=False)
    hurst_year_matrix = df.values
    hurst_year_DFA = hurst_year_matrix[:,0]
    hurst_year_RS = hurst_year_matrix[:,1]

# plt.show()
"""