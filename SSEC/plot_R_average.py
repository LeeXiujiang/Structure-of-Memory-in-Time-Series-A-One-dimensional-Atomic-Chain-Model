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
        file_name = f"D:\\Data\\real\\toe_SSEC_{start_}_{end_}_5000_sub.csv"
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
        file_name = f"D:\\Data\\real\\toe_HSI_{start_}_{end_}_72000_sub.csv"
        df = pd.read_csv(file_name, header=None)
        eig_matrix = df.values
        # eig_vals = eig_matrix.flatten()
        eig_vals[f"alpha_1={alhpa_1}"] = eig_matrix
    # eig_vals = create_fbm_theory_eigenvalue(H,dimension,simulation_times,method='cir')
    return year_list,eig_vals
#
def get_R_bar_SSEC():
    R_SSEC_all = []
    year_list,eig_vals = read_data_SSEC_all()

    for i,year_ in enumerate(year_list):
        eig_val = eig_vals[f"alpha_1={year_}"]
        eig_val = eig_val.tolist()
        R_i = calculate_R(eig_val)
        R_SSEC_all.append(np.average(R_i))
        # print(key,val)
    # R_bar = 0
    # for key,val in eig_vals.items():
    #     R_bar += val
    # R_bar /= len(eig_vals)
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
        # eig_val = eig_val.tolist()
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
        # eig_val = eig_val.tolist()
        R_i = sorted(calculate_R_matrix(eig_val))
        len_R_i = len(R_i)
        R_i = R_i[int(len_R_i*0.1):int(len_R_i*0.9)]
        R_HSI_sub.append(R_i)
    return R_HSI_sub

def get_SSEC_HSI_R():
    R_SSEC_all = get_R_bar_SSEC()
    R_HSI_all = get_R_bar_HSI()
    R_SSEC_sub = get_R_bar_SSEC_sub()
    R_HSI_sub = get_R_bar_HSI_sub()

# import matplotlib.pyplot as plt
# import numpy as np

def plot_simple_SSEC():
    """简化的SSEC绘图函数"""
    R_SSEC_all = get_R_bar_SSEC()
    R_SSEC_sub = get_R_bar_SSEC_sub()
    years = list(range(2019, 2025))
    
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(years[:len(R_SSEC_all)], R_SSEC_all, 'ro-', linewidth=2, markersize=8, label='SSEC All')
    
    # 绘制小提琴图
    if R_SSEC_sub:
        violin_data = []
        for i in range(len(R_SSEC_sub)):
            if hasattr(R_SSEC_sub[i], '__iter__'):
                violin_data.append(list(R_SSEC_sub[i]))
            else:
                violin_data.append([R_SSEC_sub[i]])
        
        vp = plt.violinplot(violin_data, positions=years[:len(violin_data)], showmeans=True)
        for body in vp['bodies']:
            body.set_facecolor('blue')
            body.set_alpha(0.5)
        vp['cmeans'].set_color('darkblue')
    
    # plt.title('SSEC Comparison')
    plt.xlabel('Year')
    plt.ylabel(r'$\langle R \rangle$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_simple_HSI():
    """简化的HSI绘图函数"""
    R_HSI_all = get_R_bar_HSI()
    R_HSI_sub = get_R_bar_HSI_sub()
    years = list(range(2019, 2025))
    
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(years[:len(R_HSI_all)], R_HSI_all, 'go-', linewidth=2, markersize=8, label='HSI All')
    
    # 绘制小提琴图
    if R_HSI_sub:
        violin_data = []
        for i in range(len(R_HSI_sub)):
            if hasattr(R_HSI_sub[i], '__iter__'):
                violin_data.append(list(R_HSI_sub[i]))
            else:
                violin_data.append([R_HSI_sub[i]])
        
        vp = plt.violinplot(violin_data, positions=years[:len(violin_data)], showmeans=True)
        for body in vp['bodies']:
            body.set_facecolor('orange')
            body.set_alpha(0.5)
        vp['cmeans'].set_color('darkorange')
    
    plt.title('HSI Comparison')
    plt.xlabel('Year')
    plt.ylabel(r'$\langle R \rangle$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 调用函数
plot_simple_SSEC()
plot_simple_HSI()