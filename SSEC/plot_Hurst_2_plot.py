import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 去除最大最小值各5%
def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    data = np.array(data)
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]
def plot_Hurst_SSEC(ax=None,RS=False,quchu=True):
    years = np.arange(2019, 2025, 1)

    dfa_data, rs_data = [], []

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

    # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    #刻度尺与标签
    ax.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
    ax.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
    ax.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_xlabel('Year',fontsize =12)
    ax.set_ylabel('Hurst Exponent',fontsize =12)
    ax.set_xticks(positions, labels, rotation=45)
    ax.grid(True, alpha=0.3)
def plot_Hurst_HSI(ax=None,RS = False,quchu=True):
    years = np.arange(2019, 2025, 1)

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

    # plt.title('Hurst Exponent Boxplots (DFA vs RS) by Year')
    ax.set_xlabel('Year',fontsize =14)
    ax.set_ylabel('Hurst Exponent',fontsize =14)
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