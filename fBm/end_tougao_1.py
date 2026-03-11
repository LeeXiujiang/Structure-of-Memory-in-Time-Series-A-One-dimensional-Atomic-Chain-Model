

import numpy as np
import scipy.linalg as la
import pandas as pd
from fbm import FBM
import sys
import matplotlib.pyplot as plt
def series_list_to_csv(series_list, filename='fBm_series.csv'):
    """
    将series_list转换为矩阵并保存为CSV文件
    
    参数:
    series_list: 列表，包含L个长度为n的子列表
    filename: 输出文件名
    """
    # 确保所有子列表长度相同
    if len(set(len(sub_list) for sub_list in series_list)) > 1:
        raise ValueError("所有子列表必须具有相同的长度")
    
    # 将列表转换为numpy数组（n*L矩阵）
    # 注意：如果希望行是n，列是L，需要转置
    matrix = np.array(series_list).T
    
    # 将矩阵转换为DataFrame并保存为CSV
    df = pd.DataFrame(matrix)
    df.to_csv(filename, index=False, header=False)
    print(f"矩阵已保存为 {filename}")
    print(f"矩阵形状: {matrix.shape} (行×列 = {matrix.shape[0]}×{matrix.shape[1]})")
    
    return matrix
def load_fbm_data():
    """加载fBm分析数据（子图b）"""
    save_dir = "saved_data_200"
    params_df = pd.read_csv(f"D://learn_code//python//FX//{save_dir}//parameters.csv")
    H = np.array(eval(params_df['H'].iloc[0]))
    
    all_X, all_Y = [], []
    for h in H:
        df_xy = pd.read_csv(f"D://learn_code//python//FX//{save_dir}//toe_corr_H{h:.2f}.csv",header=None)
        matrix = df_xy.values
        y_1 = matrix[1:,0]
        x_1 = np.arange(1,len(y_1)+1)
        all_X.append(x_1)
        all_Y.append(y_1)
    
    return H, all_X, all_Y
def load_fbm_data_150():
    """加载fBm分析数据（子图b）"""
    save_dir = "saved_data_150"
    params_df = pd.read_csv(f"D://learn_code//python//FX//{save_dir}//parameters.csv")
    H = np.array(eval(params_df['H'].iloc[0]))
    
    all_X, all_Y = [], []
    for h in H:
        df_xy = pd.read_csv(f"D://learn_code//python//FX//{save_dir}//toe_corr_H{h:.2f}.csv",header=None)
        matrix = df_xy.values
        y_1 = matrix[1:,0]
        x_1 = np.arange(1,len(y_1)+1)
        all_X.append(x_1)
        all_Y.append(y_1)
    
    return H, all_X, all_Y
#创建图片
# 创建数据和子图
scale_change = 2.54
scale_change_2 = 2.54*2.54
width_fig = 8 /  2.54
height_fig = 12 / 2.54
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_fig, height_fig))
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.sans-serif'] = ['Arial']
# 全局设置刻度字体大小
plt.rcParams['xtick.labelsize'] = 7    # X轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 7    # Y轴刻度标签字体大小
H = np.arange(0.50, 0.901, 0.05)

#字体 set
label_fig = 14
legent_fig = 12
stick_fig = 10
title_fig = 16


len_H = len(H)
plot_one = True
colors = plt.cm.get_cmap('coolwarm', (len_H))  # 'coolwarm' 是 matplotlib 内置的渐变色
# # 生成 20 个渐变色
color_list = [colors(i) for i in range(len_H)]
color_list = color_list[-len_H:]
# (a) plot  ----------------ax1--------------------------
n = 1000
def get_hurst_series(n, hurst):
    """
    生成具有指定 Hurst 指数的时间序列。
    参数:
    - n: 序列长度
    - hurst: Hurst 指数 (0 < hurst < 1)
    
    返回:
    - 一个具有指定 Hurst 指数的时间序列
    """
    # 使用分数布朗运动 (fBm) 生成时间序列
    f = FBM(n=n+1, hurst=hurst, length=1, method='daviesharte')
    series = f.fbm()

    return series
# 生成不同Hurst指数的序列
H = np.arange(0.50, 0.901, 0.05)
colors = plt.cm.viridis(np.linspace(0, 1, len(H)))  # 使用viridis色彩映射

# 为每个Hurst指数生成序列并绘制
df = pd.read_csv("fBm_series(3).csv", header=None)
fbm_ts = df.values

series_list  = [] 
for i, hurst in enumerate(H):
    series = fbm_ts[:,i]

    # 计算垂直偏移量：每个序列比前一个高1.5个单位
    vertical_offset = i * 1.5
    
    # 添加偏移量
    shifted_series = series + vertical_offset
    
    # 获取序列的最后一个值
    last_value = shifted_series[-1]
    
    # 向左偏移50个数据点
    offset_from_end = 50

    # 绘制时间序列
    ax1.plot(shifted_series, 
             color=color_list[i], 
             linewidth=1.5 / scale_change, 
             alpha=0.8,
            )

    ax1.text(n - offset_from_end,  # x位置：总长度减去偏移量
             last_value,  # y位置：序列的最后一个值
             f'H={hurst:.2f}', 
             fontsize=8, 
             va='center',  # 垂直居中
             ha='right')
    # 添加偏移量标注线
    ax1.axhline(y=vertical_offset, color='gray', linestyle=':', alpha=0.3, linewidth=0.3)

# result_matrix = series_list_to_csv(series_list, 'fBm_series.csv')
# 在子图左下角添加(a)标签
ax1.text(0.02, 0.02, '(a)', 
         transform=ax1.transAxes,  # 使用坐标轴相对坐标
         fontsize=14, 
         verticalalignment='bottom',
         horizontalalignment='left')
# 设置子图(a)的属性
# ax1.set_title('(a) 不同Hurst指数的时间序列', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax1.xaxis.labelpad = 2
ax1.yaxis.labelpad = 2
ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel(r'$B_H(t)$', fontsize=12)
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.grid(True, alpha=0.3, linestyle='--')

# ==================== 子图 (b): fBm相关函数分析 ====================
# 加载fBm分析数据（子图b）
H_b, X_b_2, Y_b_2 = load_fbm_data()
H_b, X_b, Y_b = load_fbm_data_150()
colors_b = plt.cm.coolwarm(np.linspace(0, 1, len(H_b)))
colors_b = color_list
put_data = []
# H_b = np.arange(0.60, 0.901, 0.05)
for i, h in enumerate(H_b):
    if abs(h) <= 0.59:
        continue
    print(f"h = {h}")
    x = X_b[i]
    y = Y_b[i]
    len_x = len(x)
    len_y = len(y)
    x_2 = X_b_2[i]
    y_2 = Y_b_2[i]
    print(y[-1])
    x_2 = x_2[(len_x-1):]
    y_2 = y_2[(len_y-1):]
    print(y_2[0])
    delta_ = y[-1]-y_2[0]
    y_2 = [yy2+delta_ for yy2 in y_2]
    put_data.append(y_2)
    print("delta = ", y[-1]-y_2[-1])

    y_log = np.log(y)
    # 对数拟合
    if h >= 0.55:

        x_log = np.log(x[:50])
        y_log = np.log(y[:50])
    
    # 线性拟合
    from scipy.stats import linregress
    slope, intercept = linregress(x_log, y_log)[:2]
    y_fit_log = slope * np.log(x) + intercept
    y_fit = np.exp(y_fit_log)
    
    # 绘制
    ax2.plot(x, y_fit, color=colors_b[i], linestyle='--', linewidth=1.5 / scale_change, alpha=0.8)
    ax2.scatter(x, y, color=colors_b[i], alpha=0.6, s=30 / scale_change_2)
    ax2.scatter(x_2, y_2, color=colors_b[i], alpha=0.6, s=30 / scale_change_2)
    # 添加H值标签
    label_x = x[-16]
    label_y = y_fit[-16]
    # ax2.text(label_x, label_y, f'{h:.2f}', 
    #         fontsize=7, va='center', ha='left',
    #         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax2.text(label_x, label_y, f'{h:.2f}', 
            fontsize=8, va='center', ha='left')
# 转换为 DataFrame，每个 list 作为一列
df = pd.DataFrame(put_data).T  # .T 是转置，使每个 list 成为列

# 保存为 CSV
df.to_csv('output_1.csv', index=False, header=False)

# 添加竖直参考线
ax2.axvline(x=100, color='k', linestyle=':', linewidth=2 / scale_change  , alpha=0.8)
ax2.axvline(x=200, color='k', linestyle=':', linewidth=2 / scale_change, alpha=0.8)

# 子图(b)设置
ax2.text(0.02, 0.02, '(b)', transform=ax2.transAxes,
            fontsize=14,  va='bottom', ha='left')
# ax2.set_title('fBm相关函数衰减', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax2.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax2.xaxis.labelpad = 2
ax2.yaxis.labelpad = 2
ax2.set_xlabel(r'$\tau$', fontsize=12)
ax2.set_ylabel(r'$Acc_{\tau}$', fontsize=12)
ax2.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax2.set_xscale('log')
ax2.set_yscale('log')
# ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# plt.tight_layout()
# 保存
plt.savefig('fig01.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig('fig01.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig('fig01.svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig('fig01.eps', dpi=600,format= 'eps',  bbox_inches='tight', pad_inches=0.05)
plt.show()

