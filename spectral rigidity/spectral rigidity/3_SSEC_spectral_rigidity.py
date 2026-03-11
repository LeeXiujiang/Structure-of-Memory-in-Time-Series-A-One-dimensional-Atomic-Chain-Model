import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:  # just a hack to make running these in development easier
    import empyricalRMT
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.smoother import SmoothMethod
from empyricalRMT.unfold import Unfolded
from math import pi

# ================================================================
# 理论谱刚度
# ================================================================

def delta3_poisson(L):
    return L / 15.0

def delta3_goe(L):
    gamma = 0.5772156649
    return (1.0 / (pi**2)) * (np.log(2 * pi * L) + gamma - 5/4 - pi*pi/8)
def delta3_gue(L):
    gamma = 0.5772156649
    return (1.0 / (2*(pi**2))) * (np.log(2 * pi * L) + gamma - 5/4)
# Unfolded
# eigs = Eigenvalues.generate(1000, kind="goe")
# def combine_unfolded_instances(instances: list[Unfolded]) -> Unfolded:
#     combined_originals = np.concatenate([inst.originals for inst in instances])
#     combined_vals = np.concatenate([inst.values for inst in instances])
#     sorted_indices = np.argsort(combined_originals)
#     sorted_originals = combined_originals[sorted_indices]
#     sorted_vals = combined_vals[sorted_indices]
#     # originals=eigs, unfolded=unfolded
#     return Unfolded(originals = sorted_originals, unfolded= sorted_vals)
def combine_unfolded_instances(instances: list[Unfolded]) -> Unfolded:
    # 检查是否为空列表
    if not instances:
        raise ValueError("Input list of Unfolded instances is empty")

    # 检查所有实例的 originals 和 _vals 长度是否一致
    first_len = len(instances[0]._originals)
    for inst in instances:
        if len(inst._originals) != first_len or len(inst._vals) != first_len:
            raise ValueError("All Unfolded instances must have the same length")

    # 计算 originals 和 unfolded 的平均值（按索引位置）
    avg_originals = np.mean([inst._originals for inst in instances], axis=0)
    avg_unfolded = np.mean([inst._vals for inst in instances], axis=0)

    return Unfolded(originals=avg_originals, unfolded=avg_unfolded)
#get eigs 

H = [0.60,0.70,0.80,0.90]
H = [2019,2020,2021,2022,2023,2024]

def getunfoled_SSEC_form_matrix_quanpu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\SSEC\\toe_SSEC_{str(h)}_{str(h)}_50000.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_SSEC_form_matrix_duichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\SSEC\\toe_duichen_SSEC_{str(h)}_{str(h)}_50000.csv" #toe_duichen_SSEC_2019_2019_50000
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_SSEC_form_matrix_xieduichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\SSEC\\toe_xieduichen_SSEC_{str(h)}_{str(h)}_50000.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
#hsi
def getunfoled_HSI_form_matrix_quanpu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\SSEC\\toe_HSI_{str(h)}_{str(h)}_72000.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_HSI_form_matrix_duichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\real\\toe_duichen_HSI_{str(h)}_{str(h)}_72000.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_HSI_form_matrix_xieduichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:\\Data\\real\\toe_xieduichen_HSI_{str(h)}_{str(h)}_72000.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        # matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=10))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
# for i in range(len(eigs_out)):
#     # print(f"{i+1}/{len(eigs_out)}")
#     unfoldings[f"H = {H[i]:.2f}"] = eigs_out[i].unfold(smoother=SmoothMethod.Polynomial, degree=5)

# unfoldings = {
#     "H = 0.50": eigs_out[1].unfold(smoother=SmoothMethod.Polynomial, degree=5),    #指数
#     "H = 0.60": eigs_out[1].unfold(smoother=SmoothMethod.Polynomial, degree=5),    #多项式
#     "H = 0.70": eigs_out[1].unfold(smoother=SmoothMethod.Polynomial, degree=5),
#     "H = 0.80": eigs_out[1].unfold(smoother=SmoothMethod.Polynomial, degree=5),   #Goe
# }





ssec_unfoldings_quanpu = getunfoled_SSEC_form_matrix_quanpu(H)
ssec_unfoldings_duichenzipu = getunfoled_SSEC_form_matrix_duichenzipu(H)
ssec_unfoldings_xieduichenzipu = getunfoled_SSEC_form_matrix_xieduichenzipu(H)
color_1 = 'b'
color_2 = 'b'
N = len(ssec_unfoldings_quanpu)
# fig, axes = plt.subplots(nrows=3, ncols=N, figsize=(18, 16))
# a_3_2 = True
# if a_3_2:
#     fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
#     ax_list = [axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[2,0],axes[2,1]]
# else:
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
#     ax_list = axes.flatten()


#尺度放缩
width_fig = 10 /  2.54
height_fig = 10/8*6 / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig, ax1 = plt.subplots(figsize=(width_fig, height_fig))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 展开为四个子图
# ax2 ,ax3,ax4 = ax1,ax1,ax1

# 组为一个列表
ax_list = [ax1, ax1, ax1, ax1, ax1, ax1, ax1]
# ax_list = [axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[2,0],axes[2,1]]
# ax_list = axes.flatten()
L = np.arange(1,30,0.5)
p = delta3_poisson(L)
g = delta3_goe(L)
ax1.plot(L, p, color="green", linestyle="-", linewidth=2.5/scale_change, label="Poisson")
ax1.plot(L, g, color="orange", linestyle="-", linewidth=2.5/scale_change, label="GOE")
colors = plt.cm.get_cmap('coolwarm', (6))  # 'coolwarm' 是 matplotlib 内置的渐变色
# # 生成 20 个渐变色
color_list = [colors(i) for i in range(6)]
color_list = color_list[-6:]
# 第一行：全谱
for j, (label, unfolded) in enumerate(ssec_unfoldings_quanpu.items()):
    

    ax = ax_list[j]

    # ====================================================
    # 全谱（红色） + 误差带
    # ====================================================
    # ax.plot(L, m_full, color=color_a, linewidth=2.2, label="Full spectrum")
    
    # ax = ax_list[j]
    # title = f"{label}"
    title = f"SSEC {str(H[j])}"
    color_1 = color_list[j]
    # unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig, axes=ax,marker="D",
    #                                 color_a=color_1,
    #                                 s=30/scale_change_2,linewidth = 2.5/scale_change,
    #                                 label_add=None)
    unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig,
                                     axes=ax,marker="D",color_a=color_1,
                                     s=30,linewidths=1,label_add="No label")


# 第二行：对称子谱
for j, (label, unfolded) in enumerate(ssec_unfoldings_duichenzipu.items()):
    ax = ax_list[j]
    title = f"Year = {int(H[j])}"
    color_1 = color_list[j]
    unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig,
                                     axes=ax,marker="o",color_a=color_1,
                                     s=30,linewidths=1,label_add="No label")

# 第三行：斜对称子谱
for j, (label, unfolded) in enumerate(ssec_unfoldings_xieduichenzipu.items()):
    ax = ax_list[j]
    title = f"SSEC {int(H[j])}"
    color_2 = color_list[j]
    # unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig, axes=ax,marker="o",
    #                                 color_a = color_2,
    #                                 s=30/scale_change_2,linewidth = 2.5/scale_change,
    #                                 kongxin=True)
    unfolded.plot_spectral_rigidity(L=L, title="",ensembles=[], fig=fig, axes=ax, 
                                    marker="o", color_a=color_2,
                                    s=30,linewidths=1,
                                    kongxin=True)
    
    # ax.set_title(title, fontsize=14)

# for ax in ax_list:
#刻度尺与标签
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax.set_ylim(top=2.2)
ax1.xaxis.labelpad = 2
ax1.yaxis.labelpad = 2
ax1.set_yticks(np.arange(0, 2.01, 0.5))
ylabel = r'$\overline{\Delta_3(L)}$'
ax1.set_ylabel(ylabel, fontsize=12)

ax1.set_xlabel(r"$L$", fontsize=12)
ax1.legend(
    fontsize=8,
    frameon=False,
    ncol=2,                # 一行显示两个图例
    loc='upper left',      # 以左上角为锚点
    bbox_to_anchor=(0.01, 0.98)  # 图例左上角放在数据坐标(0.1, 0.9)附近
)
    # Add (c1), (c2), ...
    # axes[2, j].text(0.02, 0.15, f"(c{j+1})", transform=axes[2, j].transAxes,
    #                 fontsize=16, va='bottom', ha='left')

    # if j == 0:
    #     axes[2, j].set_ylabel(r"$\Delta_3(L)$ of Antisymmetric Subspectrum", fontsize=12, va='center')
    # else:
    #     axes[2, j].set_ylabel("")
    # axes[2, j].set_xlabel("")
    # # axes[2, j].set_xlabel("L",fontsize=16,va='center')
    # if j != 0:
    #     axes[2, j].legend().set_visible(False)
texts = [f'{yyy}' for yyy in [2019,2020,2021,2022,2023,2024]]
positions = [(2, 1.7), (2, 1.6), (2, 1.5), (2, 1.4), (2, 1.3),(2, 1.2)]
# 添加文字
for text, (x, y) in zip(texts, positions):
    ax1.text(
        x, y,           # 坐标位置
        text,           # 文字内容
        fontsize=8,    # 字体大小
        ha='center',    # 水平居中 (horizontal alignment)
        va='center',    # 垂直居中 (vertical alignment)
        transform=ax1.transData  # 使用数据坐标
    )
ax1.text(
    6, 1.85,           # 坐标位置
    "full spectrum",           # 文字内容
    fontsize=8,    # 字体大小
    ha='center',    # 水平居中 (horizontal alignment)
    va='center',    # 垂直居中 (vertical alignment)
    transform=ax1.transData  # 使用数据坐标
)
ax1.text(
    13, 1.85,           # 坐标位置
    "sub-spectra",           # 文字内容
    fontsize=8,    # 字体大小
    ha='center',    # 水平居中 (horizontal alignment)
    va='center',    # 垂直居中 (vertical alignment)
    transform=ax1.transData  # 使用数据坐标
)
positions_all = [(6, 1.7), (6, 1.6), (6, 1.5), (6, 1.4),(6,1.3),(6,1.2)]
positions_sub_1 = [(12.5, 1.7), (12.5, 1.6), (12.5, 1.5), (12.5, 1.4),(12.5,1.3),(12.5,1.2)]
positions_sub_2 = [(13.5, 1.7), (13.5, 1.6), (13.5, 1.5), (13.5, 1.4),(13.5,1.3),(13.5,1.2)]
# 1. 绘制 positions_all (菱形标记，不同颜色)
for i, (x, y) in enumerate(positions_all):
    ax1.scatter(
        x, y,
        marker="D",                    # 菱形标记
        s=30/scale_change_2,
        linewidths=1/scale_change,
        color=color_list[i],           # 每个点不同颜色
        edgecolors=   color_list[i]       # 边框颜色
        # label=f'Diamond {i+1}' if i == 0 else None  # 只添加一次图例
    )

# 2. 绘制 positions_sub_1 (实心圆，不同颜色)
for i, (x, y) in enumerate(positions_sub_1):
    ax1.scatter(
        x, y,
        marker="o",                    # 实心圆
        s=30/scale_change_2,
        linewidths=1/scale_change,
        color=color_list[i],           # 每个点不同颜色
        edgecolors=color_list[i]           # 边框颜色
        # label=f'Circle solid {i+1}' if i == 0 else None  # 只添加一次图例
    )

# 3. 绘制 positions_sub_2 (空心圆，不同颜色)
for i, (x, y) in enumerate(positions_sub_2):
    ax1.scatter(
        x, y,
        marker="o",                    # 圆形标记
        s=30/scale_change_2,
        linewidths=1/scale_change,
        color='white',                 # 内部填充白色
        edgecolors=color_list[i]     # 边框用不同颜色
        # label=f'Circle hollow {i+1}' if i == 0 else None  # 只添加一次图例
    )
# plt.tight_layout()
plt.savefig("fig13.png", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig13.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig13.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig13.eps", dpi=600,format= 'eps', bbox_inches='tight',pad_inches=0.05)
plt.show()

