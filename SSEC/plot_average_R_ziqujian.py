import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches

# 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_R_matrix(matrix):
    """
    使用NumPy向量化操作计算矩阵每一列的累计贡献率
    """
    matrix = np.array(matrix)
    n_rows, n_cols = matrix.shape
    
    # 计算每列的总和
    column_sums = np.sum(matrix, axis=0)
    
    # 避免除零错误
    column_sums[column_sums == 0] = 1
    
    # 计算累积和
    cumulative_sums = np.cumsum(matrix, axis=0)
    
    # 计算累计贡献率
    cumulative_ratios = cumulative_sums / column_sums
    
    R_list = np.average(cumulative_ratios,axis=0)

    return R_list


def calculate_R(data_list):
    """
    使用NumPy计算累计贡献率
    """
    if not data_list:
        return np.array([])
    
    data_array = np.array(data_list)
    total = np.sum(data_array)
    
    if total == 0:
        return np.zeros_like(data_array)
    
    cumulative_sum = np.cumsum(data_array)
    cumulative_ratios = cumulative_sum / total
    
    return cumulative_ratios

def read_data_SSEC_all():
    eig_vals = {}
    
#########SSEC
    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']

    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    list_year_start = [f"{i}" for i in range(2019,2025,1)]
    list_year_end = [f"{i}" for i in range(2019,2025,1)]
    year_list = np.arange(2019,2025,1)

    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    for i,alhpa_1 in enumerate(year_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_SSEC_{start_}_{end_}_50000_sub.csv"
        df = pd.read_csv(file_name, header=None)
        eig_matrix = df.values
        # eig_vals = eig_matrix.flatten()
        eig_vals[f"alpha_1={alhpa_1}"] = eig_matrix
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return year_list,eig_vals

def read_data_HSI_all():
    eig_vals = {}
    
#########SSEC
    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']

    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    list_year_start = [f"{i}" for i in range(2019,2025,1)]
    list_year_end = [f"{i}" for i in range(2019,2025,1)]
    year_list = np.arange(2019,2025,1)

    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    for i,alhpa_1 in enumerate(year_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_HSI_{start_}_{end_}_72000.csv"
        df = pd.read_csv(file_name, header=None)
        eig_matrix = df.values
        # eig_vals = eig_matrix.flatten()
        eig_vals[f"alpha_1={alhpa_1}"] = eig_matrix
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return year_list,eig_vals

def read_data_SSEC_sub():
    eig_vals = {}
    
#########SSEC
    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']

    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    list_year_start = [f"{i}" for i in range(2019,2025,1)]
    list_year_end = [f"{i}" for i in range(2019,2025,1)]
    year_list = np.arange(2019,2025,1)

    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    for i,alhpa_1 in enumerate(year_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_SSEC_{start_}_{end_}_5000_sub_3.csv"
        df = pd.read_csv(file_name, header=None)
        eig_matrix = df.values
        # eig_vals = eig_matrix.flatten()
        eig_vals[f"alpha_1={alhpa_1}"] = eig_matrix
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return year_list,eig_vals

def read_data_HSI_sub():
    eig_vals = {}
    
#########SSEC
    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']

    list_year_start = ['2008','2010','2012','2014','2016','2018','2020','2022']
    list_year_end = ['2009','2011','2013','2015','2017','2019','2021','2023']
    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    list_year_start = [f"{i}" for i in range(2019,2025,1)]
    list_year_end = [f"{i}" for i in range(2019,2025,1)]
    year_list = np.arange(2019,2025,1)

    list_start_time = [f"{i}-01-01" for i in list_year_start]
    list_end_time = [f"{i}-12-31" for i in list_year_end]
    L = len(list_start_time)
    alpha_1_list = range(len(list_start_time))
    for i,alhpa_1 in enumerate(year_list):
        start_ = list_year_start[i]
        end_ = list_year_end[i]
        file_name = f"D:\\Data\\real\\toe_HSI_{start_}_{end_}_7200_sub_3.csv"
        df = pd.read_csv(file_name, header=None)
        eig_matrix = df.values
        # eig_vals = eig_matrix.flatten()
        eig_vals[f"alpha_1={alhpa_1}"] = eig_matrix
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return year_list,eig_vals

def get_R_bar_SSEC():
    R_SSEC_all = []
    year_list,eig_vals = read_data_SSEC_all()

    for i,year_ in enumerate(year_list):
        eig_val = eig_vals[f"alpha_1={year_}"]
        eig_val = eig_val.tolist()
        R_i = calculate_R(eig_val)
        R_SSEC_all.append(np.average(R_i))
    return R_SSEC_all

def get_R_bar_HSI():
    R_HSI_all = []
    year_list,eig_vals = read_data_HSI_all()
    for i,year_ in enumerate(year_list):
        eig_val = eig_vals[f"alpha_1={year_}"]
        eig_val = eig_val.tolist()
        R_i = calculate_R(eig_val)
        R_HSI_all.append(np.average(R_i))
    return R_HSI_all

def get_R_bar_SSEC_sub():
    R_SSEC_sub = []
    year_list,eig_vals = read_data_SSEC_sub()
    for i,year_ in enumerate(year_list):
        eig_val = eig_vals[f"alpha_1={year_}"]
        R_i = sorted(calculate_R_matrix(eig_val))
        len_R_i = len(R_i)
        R_i = R_i[int(len_R_i*0.1):int(len_R_i*0.9)]
        R_SSEC_sub.append(R_i)
    return R_SSEC_sub

def get_R_bar_HSI_sub():
    R_HSI_sub = []
    year_list,eig_vals = read_data_HSI_sub()
    for i,year_ in enumerate(year_list):
        eig_val = eig_vals[f"alpha_1={year_}"]
        R_i = sorted(calculate_R_matrix(eig_val))
        len_R_i = len(R_i)
        R_i = R_i[int(len_R_i*0.1):int(len_R_i*0.9)]
        R_HSI_sub.append(R_i)
    return R_HSI_sub

def plot_simple_SSEC():
    """简化的SSEC绘图函数，使用箱线图"""
    R_SSEC_all = get_R_bar_SSEC()
    R_SSEC_sub = get_R_bar_SSEC_sub()
    years = list(range(2019, 2025))
    R_aver_sub = [np.average(year_data) - 0.05 for year_data in R_SSEC_sub]
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    # plt.plot(years[:len(R_SSEC_all)], R_SSEC_all, 'ro-', linewidth=2, markersize=8)
    plt.plot(years[:len(R_aver_sub)],R_aver_sub,'bo--',linewidth=2, markersize=8)
    # 添加横线
    plt.axhline(y=0.46, color='red', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.425, color='green', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.39, color='blue', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.35, color='purple', linestyle='-', alpha=0.7, linewidth=1)

    # 添加文字标注
    plt.text(years[-1] + 0.1, 0.46, 'h = 0.55', color='red', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.425, 'h = 0.60', color='green', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.39, 'h = 0.65', color='blue', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.35, 'h = 0.70', color='purple', va='center', fontsize=10)

    color_use = "#6387BB"
    
    # 绘制箱线图
    if R_SSEC_sub:
        box_data = []
        positions = []
        for i in range(len(R_SSEC_sub)):
            if hasattr(R_SSEC_sub[i], '__iter__') and len(R_SSEC_sub[i]) > 0:
                box_data.append(R_SSEC_sub[i])
                positions.append(years[i])
        
        if box_data:
            bp = plt.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,showfliers=False)
            
            # 设置箱线图样式
            for box in bp['boxes']:
                box.set(facecolor='lightblue', alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color='blue', linewidth=1.5)
            for cap in bp['caps']:
                cap.set(color='blue', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='darkblue', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', color=color_use, alpha=0.5)
    plt.ylim((0.25,0.6))
    # plt.ylim(bottom=0.25)
    # plt.title('SSEC Comparison')
    plt.xlabel('Year')
    plt.ylabel(r'$\langle R \rangle$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_simple_HSI():
    """简化的HSI绘图函数，使用箱线图"""
    R_HSI_all = get_R_bar_HSI()
    R_HSI_sub = get_R_bar_HSI_sub()
    years = list(range(2019, 2025))
    # pianyi = [-(0.5+0.041),-(0.50+)]
    R_aver_sub = [np.average(year_data)-0.05 for year_data in R_HSI_sub]

    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    # plt.plot(years[:len(R_HSI_all)], R_HSI_all, 'go-', linewidth=2, markersize=8)
    plt.plot(years[:len(R_aver_sub)],R_aver_sub,'bo--',linewidth=2, markersize=8)
    # 添加横线
    plt.axhline(y=0.46, color='red', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.425, color='green', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.39, color='blue', linestyle='-', alpha=0.7, linewidth=1)
    plt.axhline(y=0.35, color='purple', linestyle='-', alpha=0.7, linewidth=1)

    # 添加文字标注
    plt.text(years[-1] + 0.1, 0.46, 'h = 0.55', color='red', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.425, 'h = 0.60', color='green', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.39, 'h = 0.65', color='blue', va='center', fontsize=10)
    plt.text(years[-1] + 0.1, 0.35, 'h = 0.70', color='purple', va='center', fontsize=10)
    color_use = "#6387BB"
    # 绘制箱线图
    if R_HSI_sub:
        box_data = []
        positions = []
        for i in range(len(R_HSI_sub)):
            if hasattr(R_HSI_sub[i], '__iter__') and len(R_HSI_sub[i]) > 0:
                box_data.append(R_HSI_sub[i])
                positions.append(years[i])
        
        if box_data:
            bp = plt.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,showfliers=False)
            
            # 设置箱线图样式
            for box in bp['boxes']:
                box.set(facecolor='lightgreen', alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color='green', linewidth=1.5)
            for cap in bp['caps']:
                cap.set(color='green', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='darkgreen', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', color=color_use, alpha=0.5)
    plt.ylim((0.25,0.6))
    # plt.ylim(bottom=0.25)
    # plt.title('HSI Comparison')
    plt.xlabel('Year')
    plt.ylabel(r'$\langle R \rangle$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 调用函数
plot_simple_SSEC()
plot_simple_HSI()