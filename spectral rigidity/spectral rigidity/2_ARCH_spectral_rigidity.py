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
def load_spectrum(csv_path):
    data = pd.read_csv(csv_path, header=None).values
    L = data[:, 0]
    n_exp = data.shape[1] // 2

    all_delta = np.zeros((len(L), n_exp))
    for j in range(n_exp):
        all_delta[:, j] = data[:, 2*j+1]

    delta_mean = all_delta.mean(axis=1)
    delta_std  = all_delta.std(axis=1)
    
    return L, delta_mean, delta_std, all_delta
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
def delta3_poisson(L):
    return L / 15.0

def delta3_goe(L):
    gamma = 0.5772156649
    return (1.0 / (pi**2)) * (np.log(2 * pi * L) + gamma - 5/4 - pi*pi/8)
def delta3_gue(L):
    gamma = 0.5772156649
    return (1.0 / (2*(pi**2))) * (np.log(2 * pi * L) + gamma - 5/4)
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
def get_eigs(H):
    eigs_out = []
    
    for i,h in enumerate(H):
        file_name = f"D:data\\pu\\toe_xieduichen_{h:.2f}_2000_100_p.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values
        matrix = np.sort(matrix, axis=0)
        list = matrix[:,20] 
        list = list[int(len(list)*0.2):int(len(list)*0.8)]
        eigs  = Eigenvalues.get_eig_from_list(list)
        eigs_out.append(eigs)
    return eigs_out
H = [0.60,0.70,0.80,0.90]
# eigs_out = get_eigs(H)
H = [0.1,0.5,0.7,0.9]

def getunfoled_form_matrix(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"D:data\\pu\\toe_xieduichen_{h:.2f}_2000_100_p.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        matrix = matrix[:,41:42]  # 41: 42
        # matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=12))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings

def getunfoled_form_matrix_quanpu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"data\\ARCH_toe_{h:.2f}_2000_100_p.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=12))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_form_matrix_duichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"Data\\ARCH_toe_duichen_{h:.1f}_2000_100_p.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=12))

        combined_unfolded = combine_unfolded_instances(h_unfolded)
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
        # print(f"{i+1}/{len(eigs_out)}")
        unfoldings[f"H = {h:.2f}"] = combined_unfolded

    return unfoldings
def getunfoled_form_matrix_xieduichenzipu(H):
    unfoldings = {}
    for i,h in enumerate(H):
        print(f"processing H = {h:.2f}")
        file_name = f"data\\ARCH_toe_xieduichen_{h:.1f}_2000_100_p.csv"
        df = pd.read_csv(file_name, header=None)
        matrix = df.values

        matrix = matrix[:,41:42]  # 41: 42
        matrix = np.sort(matrix, axis=0)
        L_matrix = matrix.shape[1]

        h_unfolded = []
        for j in range((L_matrix)):
            list = matrix[:,j] 
            # list = list[int(len(list)*0.2):int(len(list)*0.8)]
            eigs  = Eigenvalues.get_eig_from_list(list)
            h_unfolded.append(eigs.unfold(smoother=SmoothMethod.Polynomial, degree=12))

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




# H = [0.1,0.5,0.7,0.9]
H = [0.5,0.9,0.7,0.1]
H_title = [0.1,0.5,0.7,0.9]
unfoldings_quanpu = getunfoled_form_matrix_quanpu(H)
unfoldings_duichenzipu = getunfoled_form_matrix_duichenzipu(H)
unfoldings_xieduichenzipu = getunfoled_form_matrix_xieduichenzipu(H)
color_1 = 'b'
color_2 = 'b'
N = len(unfoldings_quanpu)
# fig, axes = plt.subplots(nrows=3, ncols=N, figsize=(18, 16))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10.8))
# fig, ax1 = plt.subplots(figsize=(6, 6))
#尺度放缩
width_fig = 10 /  2.54
height_fig = 10/8*6 / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig, ax1 = plt.subplots(figsize=(width_fig, height_fig))
#*******设置字体Arial无衬线*******
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 展开为四个子图
ax2 ,ax3,ax4 = ax1,ax1,ax1

# 组为一个列表
ax_list = [ax1, ax1, ax1, ax1]
# H_list = ["H=0.60", "H=0.70", "H=0.80", "H=0.90"]
# fig, axes = plt.subplots(nrows=3, ncols=N, figsize=(18, 10.8))
# L = np.arange(1,25,0.2)
L = np.arange(1,30,0.5)
all_means = {"full":{}, "sym":{}, "skew":{}}
color_a = "#2A60D9"
color_b = "#559BD4"
color_c = "#73C2CD"
colors = plt.cm.get_cmap('coolwarm', (4))  # 'coolwarm' 是 matplotlib 内置的渐变色
# # 生成 20 个渐变色
color_list = [colors(i) for i in range(4)]
color_list = color_list[-4:]
base_dir="D:data\\delta\\"

# for idx, h in enumerate(H):

#     # 文件路径
#     # file_full = f"{base_dir}fbm_quanpu_{h:.2f}.csv"
#     # file_sym  = f"{base_dir}fbm_duichenzipu_{h:.2f}.csv"
#     # file_skew = f"{base_dir}fbm_xieduichenzipu_{h:.2f}.csv"

#     print(f"\nProcessing H = {h:.2f}")

    # ---------- 读取三类谱 ----------
    # L, m_full,  s_full,  _ = load_spectrum(file_full)
    # _, m_sym,   s_sym,   _ = load_spectrum(file_sym)
    # _, m_skew,  s_skew,  _ = load_spectrum(file_skew)

    # # 保存用于 H 对比图
    # all_means["full"][h] = (L, m_full)
    # all_means["sym"][h]  = (L, m_sym)
    # all_means["skew"][h] = (L, m_skew)

    # 理论曲线
p = delta3_poisson(L)
g = delta3_goe(L)

# ax = ax_list[idx]

# ====================================================
# 全谱（红色） + 误差带
# ====================================================
# ax.plot(L, m_full, color=color_a, linewidth=2.2, label="Full spectrum")
ax1.plot(L, p, color="green", linestyle="-", linewidth=2, label="Poisson")
ax1.plot(L, g, color="orange", linestyle="-", linewidth=2, label="GOE")
    # ax.scatter(L, m_full, color='k', s=20,label="Full" if idx==0 else None)
    # ci_up = m_full + 1.96 * s_full / np.sqrt(len(m_full))
    # ci_dn = m_full - 1.96 * s_full / np.sqrt(len(m_full))
    

# 第一行：全谱
# for j, (label, unfolded) in enumerate(unfoldings_quanpu.items()):
#     title = f"{label}"
#     print(title)
#     ax = ax_list[j]
#     unfolded.plot_spectral_rigidity(L=L,title=title, ensembles=["poisson","goe"], fig=fig, axes=ax)
    # ax.set_xlabel("")
    
    # Add (a1), (a2), ...
    # ax.text(0.02, 0.1, f"(a{j+1})", transform=axes[0, j].transAxes,
    #                 fontsize=16, va='bottom', ha='left')

    # ax.set_title(title, fontsize=14)
    # if j in (0,2) :
    #     ax.set_ylabel(r"$\Delta_3(L)$", fontsize=12, va='center')
    # else:
    #     ax.set_ylabel("")
    # if j != 0:
    # ax.legend().set_visible(False)
    # L = L
H = [0.9,0.5,0.7,0.1]
H_title = [0.1,0.5,0.7,0.9]
for j, (label, unfolded) in enumerate(unfoldings_quanpu.items()):

    title = f"{label}"
    title = rf"$\alpha_1$={H[j]:.1f}"
    print(title)
    ax = ax_list[j]
    color_1 = color_list[j]
    # unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig, axes=ax,marker="D",color_a=color_1,label_add=None)
    unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig,
                                     axes=ax,marker="D",color_a=color_1,
                                     s=30,linewidths=1,label_add="No label")
    # axes[1, j].set_xlabel("")
    
    # Add (b1), (b2), ...
    # axes[1, j].text(0.02, 0.1, f"(b{j+1})", transform=axes[1, j].transAxes,
    #                 fontsize=16, va='bottom', ha='left')

    # if j == 0:
    #     axes[1, j].set_ylabel(r"$\Delta_3(L)$ of Symmetric Subspectrum", fontsize=12, va='center')
    # else:
    #     axes[1, j].set_ylabel("")
    # if j != 0:
    # ax.legend().set_visible(False)
# 第二行：对称子谱
for j, (label, unfolded) in enumerate(unfoldings_duichenzipu.items()):
    # title = f"{label}"
    title = rf"$\alpha_1$={H_title[j]:.1f}"
    print(title)
    ax = ax_list[j]
    # f"H = {h:.2f}"
    color_1 = color_list[j]
    unfolded = unfoldings_duichenzipu[f"H = {H[j]:.2f}"]
    # unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig, axes=ax,marker="o",color_a=color_1,label_add=title)
    unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig,
                                     axes=ax,marker="o",color_a=color_1,
                                     s=30,linewidths=1,label_add="No label")
    # axes[1, j].set_xlabel("")
    
    # Add (b1), (b2), ...
    # axes[1, j].text(0.02, 0.1, f"(b{j+1})", transform=axes[1, j].transAxes,
    #                 fontsize=16, va='bottom', ha='left')

    # if j == 0:
    #     axes[1, j].set_ylabel(r"$\Delta_3(L)$ of Symmetric Subspectrum", fontsize=12, va='center')
    # else:
    #     axes[1, j].set_ylabel("")
    # if j != 0:
    # ax.legend().set_visible(False)

# 第三行：斜对称子谱
for j, (label, unfolded) in enumerate(unfoldings_xieduichenzipu.items()):
    # title = f"{label}"
    title = rf"$\alpha_1$={H_title[j]:.1f}"
    print(title)
    ax = ax_list[j]
    color_2 = color_list[j]
    unfolded = unfoldings_xieduichenzipu[f"H = {H[j]:.2f}"]
    # unfolded.plot_spectral_rigidity(L=L,title="",ensembles=[], fig=fig, axes=ax,marker="o",color_a = color_2,kongxin=True)
    unfolded.plot_spectral_rigidity(L=L, title="",ensembles=[], fig=fig, axes=ax, 
                                    marker="o", color_a=color_2,
                                    s=30,linewidths=1,
                                    kongxin=True)
    # ax.set_title(title, fontsize=16)
    # if j in (0,2) :
    #     ax.set_ylabel(r"$\Delta_3(L)$", fontsize=14, va='center')
    # else:
        # ax.set_ylabel("")
    # if j in (2,3) :
    #     ax.set_xlabel(r"$L$", fontsize=14, va='center')
    # else:
    #     ax.set_xlabel("")
    # if j != 0:
    #     ax.legend().set_visible(False)
# for ax in ax_list:
    # ax.set_ylim(0, 2)
# ax1.set_yticks(np.arange(0, 2.01, 0.5))
# ax1.set_ylabel(r"$\Delta_3(L)$", fontsize=14, va='center')
# ax1.set_xlabel(r"$L$", fontsize=14, va='center')
# # ax1.legend()
# ax1.legend(fontsize=12,frameon=False)
#刻度尺与标签
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax1.xaxis.labelpad = 2
ax1.yaxis.labelpad = 2
ax1.set_yticks(np.arange(0, 2.01, 0.5))
ylabel = r'$\overline{\Delta_3(L)}$'
ax1.set_ylabel(ylabel, fontsize=12)

ax1.set_xlabel(r"$L$", fontsize=12)
# ax1.legend(fontsize=8,frameon=False)
ax1.legend(
    fontsize=8,
    frameon=False,
    ncol=2,                # 一行显示两个图例
    loc='upper left',      # 以左上角为锚点
    bbox_to_anchor=(0.01, 0.98)  # 图例左上角放在数据坐标(0.1, 0.9)附近
)
# 要添加的文字和坐标
texts = [r"$\alpha_1$=0.1", r"$\alpha_1$=0.5", r"$\alpha_1$=0.7", r"$\alpha_1$=0.9"]
positions = [(2, 1.7), (2, 1.5), (2, 1.3), (2, 1.1)]
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
positions_all = [(6, 1.7), (6, 1.5), (6, 1.3), (6, 1.1)]
positions_sub_1 = [(12.5, 1.7), (12.5, 1.5), (12.5, 1.3), (12.5, 1.1)]
positions_sub_2 = [(13.5, 1.7), (13.5, 1.5), (13.5, 1.3), (13.5, 1.1)]
# 1. 绘制 positions_all (菱形标记，不同颜色)
for i, (x, y) in enumerate(positions_all):
    ax1.scatter(
        x, y,
        marker="D",                    # 菱形标记
        s=30/scale_change_2,
        linewidths=1.5/scale_change,
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
        linewidths=1.5/scale_change,
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
        linewidths=1.5/scale_change,
        color='white',                 # 内部填充白色
        edgecolors=color_list[i]     # 边框用不同颜色
        # label=f'Circle hollow {i+1}' if i == 0 else None  # 只添加一次图例
    )

# plt.tight_layout()
plt.savefig("fig09.png", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig09.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig09.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig09.eps", dpi=600,format= 'eps', bbox_inches='tight',pad_inches=0.05)
plt.show()
