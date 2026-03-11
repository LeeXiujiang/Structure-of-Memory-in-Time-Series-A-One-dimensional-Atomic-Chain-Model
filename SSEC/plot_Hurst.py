##############1 import lib ##############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet  # 用于检测文件编码
from pathlib import Path
from def_cal_hurst_exponent import hurst_rs,hurst_dfa,hurst_variance,hurst_dfa_advanced
from def_windows_corr_toe import operation1_vectorized,operation2_vectorized,toe_and_random_from_matrix
# from def_cal_hurst_exponent import hurst_rs,hurst_dfa,hurst_variance,hurst_dfa_advanced
########################################
##############2 read data ##############
file_name_SSEC = "D:\\Data\\SSEC\\SSEC_1.csv"
file_name_HSI = "D:\\Data\\HSI\\HSI_1.csv"

# file_list = ['SSEC.xlsx', 'SZI.xlile_listsx', 'HSI.xlsx', 'cleaned_IXIC.xlsx', 'cleaned_DJI.xlsx']
file_list = [file_name_SSEC, file_name_HSI]
# file_list = [file_name_HSI]
log_returns_list = []
index_list = ['SSEC','HSI']


for i,file_name in enumerate(file_list):
    print(f"\n📘 Reading file: {file_name}")
    
    # 读取 Excel
    # df_file = pd.read_csv(file_name,header=None)
    df_file = pd.read_csv(file_name,index_col=0,parse_dates=["trade_time"] if i==0 else ["date"])
    # df_file = pd.read_csv(file_name,index_col=0,parse_dates=["date"] if i==0 else ["date"])
    print("列名为：", df_file.columns.tolist())
    
    # 统一列名
    if i == 0:
        df_ = df_file.rename(columns={'trade_time': 'trade_time', 'close': 'close'})
        print(df_.head(-10))
        # print("列名已修改为：", df_.columns.tolist())
    elif i == 1:
        df_ = df_file.rename(columns={'date': 'trade_time', 'close': 'close'})
        print("列名已修改为：", df_.columns.tolist())
        # print(df_.head(-10))

    ######提取时间段
    start_time_list = [2019,2020,2021,2022,2023,2024]
    end_time_list = [2019,2020,2021,2022,2023,2024]
    # start_time_list = [2019]
    # end_time_list = [2019]

    t=0
    for st,en in zip(start_time_list,end_time_list):
        start_time = f'{st}-01-01'
        end_time = f'{en}-12-31'
        
        close_data = df_.loc[start_time:end_time, 'close']
        print(f"head_10: \n",close_data.head(10),'\n')
        print(f"tail_10: \n",close_data.tail(10),'\n')

        # # ===== 计算对数收益率 =====
        log_returns = np.log(close_data / close_data.shift(1))
        log_returns_clean = log_returns.dropna()
        
        print(f"{start_time}--{end_time}:✅ len of log return: {len(log_returns_clean)}")

        Hurst_year_DFA = []
        Hurst_year_RS = []
        #计算hurst指数
        CAL_hurst = True
        if CAL_hurst:
            if i == 0:
                # h_v,h_v_data,h_v_params = hurst_variance(log_returns_clean)
                NNN = len(log_returns_clean)
                CH ,end=0,5000
                pp= 1
                while end < NNN:
                    print(f"第{pp}段")
                    sub_series = log_returns_clean[CH:end]
                    h_d,h_d_data,_ = hurst_dfa(sub_series)
                    h_r,h_r_data,h_r_params = hurst_rs(sub_series)
                    Hurst_year_DFA.append(h_d)
                    Hurst_year_RS.append(h_r)
                    CH += 2000
                    end += 2000
                    pp += 1
                # h_r,h_r_data,h_r_params = hurst_rs(log_returns_clea     

            elif i == 1:
                NNN = len(log_returns_clean)
                CH ,end=0,5000
                pp = 1
                while end < NNN:
                    print()
                    sub_series = log_returns_clean[CH:end]
                    h_d,h_d_data,_ = hurst_dfa(sub_series)
                    h_r,h_r_data,h_r_params = hurst_rs(sub_series)
                    Hurst_year_DFA.append(h_d)
                    Hurst_year_RS.append(h_r)
                    CH += 2000
                    end += 2000
                    pp += 1

                # h_v,h_v_data,h_v_params = hurst_variance(log_returns_clean)
                # h_r,h_r_data,h_r_params = hurst_rs(log_returns_clean)
                # h_d,h_d_data,_ =hurst_dfa(log_returns_clean)
                # # print(f"✅ Hurst exponent using variance method: {h_v:.4f}")
                # # print(f"✅ Hurst exponent using RS method: {h_r:.4f}")
                # print(f"✅ Hurst exponent using DFA method: {h_d:.4f}")
        Hurst_matrix = np.array([Hurst_year_DFA, Hurst_year_RS]).T
        df = pd.DataFrame(Hurst_matrix)

        #滑动窗口计算互相关依赖矩阵C
        # L = 50000
        # corr_C = operation1_vectorized(log_returns_clean,L=L)
        # print(f"✅ len of corr_C: {len(corr_C)}")
        # #计算互相关依赖矩阵的特征向量
        # toe_matrix,random_matrix = toe_and_random_from_matrix(corr_C)
        # print(f"✅ len of toe_matrix: {len(toe_matrix)}")
        # print(f"✅ len of random_matrix: {len(random_matrix)}")
        #计算特征值
        # eigenvals_toe = np.linalg.eigvals(toe_matrix)
        # 将特征值转换为单列矩阵（二维数组，只有一列）
        # eigenvals_column = eigenvals_toe.reshape(-1, 1)
        # 转换为DataFrame并保存为CSV
        # df = pd.DataFrame(eigenvals_column)
        
        # 保存为CSV文件
        # file_name_put = f"toe_{index_list[i]}_{start_time_list[i]}_{end_time_list[i]}_{st}{en}"
        output_file = f"D:\\Data\\real\\Hurst_{index_list[i]}_{start_time_list[t]}_{end_time_list[t]}.csv"
        df.to_csv(output_file, index=False,header=None)
        t += 1