import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter

H = np.arange(0.50,0.901,0.05)
width_fig = 18 /  2.54
height_fig = (18/8*5) / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(width_fig ,height_fig))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['patch.force_edgecolor'] = True
# plt.rcParams['savefig.transparent'] = False

# plt.show()

#准波数 q = 2*pi n /N
colors = plt.cm.get_cmap('coolwarm', (9))  # 'coolwarm' 是 matplotlib 内置的渐变色
    # # 生成 20 个渐变色
color_list = [colors(i) for i in range(9)]
colors = plt.cm.get_cmap('coolwarm', 9)
color_list = [colors(i)[:3] for i in range(9)]   # 去掉 alpha

# color_list = color_list[-:]
H = np.arange(0.50,0.901,0.05)
W = {}
L_NUM = np.arange(1,200.1,1)
q = 2*np.pi*L_NUM/2000
for i ,h in enumerate(H):
    # print(i, h)
    
    data_name = f"D:\\Data\\pu\\toe_{h:.2f}_200_100_p_10.csv"
    data = pd.read_csv(data_name, header=None)
    data = data.values
    data = np.sort(data,axis = 0)
    data = np.sqrt(data)
    W_h = np.mean(data,axis=1)
    # data = [np.sqrt(i) for i in data]
    L = data.shape[1]
    wc_h = 0
    W[f"H={h:.2f}"] = W_h

from scipy.stats import linregress
# 将 q 和 W 变成数组确保索引操作稳定
q = np.array(q)
half_idx = len(q) // 2
fit_idx = slice(0, half_idx)
log_q = np.log10(q[fit_idx])

beta_list = []


#(0,0) plot######################################################################################
# print(WC)
color_a = "#E38D8C"
color_b = "#9AB8D4"
# plt.figure()
for i,h in enumerate(H):
    color = color_list[i]
    omega = np.array(W[f"H={h:.2f}"])
    axes[0, 0].plot(q, W[f"H={h:.2f}"], color = color_list[i],linewidth=1.5/scale_change,label=fr"$H={h:.2f}$")
    # 拟合区域：虚线叠加
    # axes[0,0].plot(q[fit_idx], omega[fit_idx], '--', color=color, linewidth=1.5/scale_change)
    # 拟合区域：虚线叠加
    # axes[0,0].plot(q[fit_idx], omega[fit_idx], '--', color=color, linewidth=1.5/scale_change, alpha=0.7)

    # 计算 log-log 拟合斜率（β）
    log_w = np.log10(omega[fit_idx])
    slope, intercept, *_ = linregress(log_q, log_w)
    beta_list.append(slope)
# plt.plot(H, WC, color=color_a)
# plt.scatter(H, WC, c=color_a)
# 创建图例时去掉边框
# plt.legend(frameon=False)
import matplotlib.patches as patches

rect = patches.Rectangle(
    (10**(-2.5), 0.4),   
    10**(-1) - 10**(-2.5),  
    1 - 0.4,  
    linewidth=2/scale_change,
    linestyle='--',
    fill=False,
    edgecolor='black')

axes[0,0].add_patch(rect)
# original rectangle
xmin = 10**(-2.5)
xmax = 10**(-1)
ymin = 0.4
ymax = 1.0

# visual center on log–linear axes
x_start = (xmin * xmax) ** 0.5   # geometric mean for log x
y_start = (ymax + ymax) / 2      # arithmetic mean for linear y

# arrow target (右上)
# 终点：归一化坐标系中心点 (0.5, 0.5)
x_end = 0.5
y_end = 0.45

axes[0,0].annotate(
    '',
    xy=(x_end, y_end),          # 终点
    xycoords='axes fraction',   # 终点：归一化坐标 (0~1)
    xytext=(x_start, y_start),  # 起点
    textcoords='data',          # 起点：数据坐标
    arrowprops=dict(arrowstyle='->', linewidth=2/scale_change)
)

# inset_00 = inset_axes(axes[0, 0], width="45%", height="45%", loc='upper right', bbox_to_anchor=(0.95, 1.00))   # 右方向往里移)
inset_00 = inset_axes(
    axes[0, 0],
    width="45%",
    height="45%",
    loc='upper right',
    bbox_to_anchor=(-0.2, -0.01, 1.0, 1.0),   # ✔ 满足 4 元组要求
    bbox_transform=axes[0, 0].transAxes,
    borderpad=0
)

color_00 = "#2A60D9"
inset_00.plot(H,beta_list,'-',color = 'k',linewidth=1.5/scale_change)
inset_00.scatter(H,beta_list,color = 'k',s=30/scale_change_2)
for i, h in enumerate(H):
    # 找最大峰值位置
    # 在同一个 inset_ax 上绘图
    text = fr'$H={h:.2f},\ q^{{{beta_list[i]:.4f}}}$'

ax1 = axes[0, 0]
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax1.xaxis.labelpad = -9
ax1.yaxis.labelpad = 1
axes[0, 0].legend(fontsize=6,frameon=False)
axes[0, 0].set_yscale('log')
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel(r'$q$',fontsize=12)
axes[0, 0].set_ylabel(r'$\omega$',fontsize=12)



inset_00.text(0.45, 0.25, r"$\omega_H \sim q^{\gamma_H}$",transform=inset_00.transAxes,fontsize = 10)
inset_00.set_xlabel(r'$H$', fontsize=10)
# inset_00.yaxis.set_label_position("right")
inset_00.yaxis.set_label_position("right")
inset_00.yaxis.tick_right()  # 这是关键！
inset_00.set_ylabel(r'$\gamma_H$', fontsize=10, rotation=0)
inset_00.tick_params(axis='both', labelsize=6)   # 同时设置 x 和 y
inset_00.tick_params(axis='x', pad=0.5)   # x 轴刻度标签与轴的距离
inset_00.tick_params(axis='y', pad=2)   # y 轴刻度标签与轴的距离
inset_00.xaxis.labelpad = 0.5
inset_00.yaxis.labelpad = 5

# inset_00.tick_params(labelsize=6)
inset_00.tick_params(axis='both', which='minor')  # 隐藏次刻度标签
from scipy.stats import linregress

# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

#(0,1) plot######################################################################################
from mpl_toolkits.mplot3d import Axes3D
import glob

# 设定 ω 轴范围
omega_bins = np.arange(0, 2.0001, 0.02)  # 包括右端点
omega_centers = (omega_bins[:-1] + omega_bins[1:]) / 2

# 离散 P 值
P_values = np.arange(0.50, 0.95, 0.05)

# 假设每组 P 对应一个文件，如 data_P0.50.txt, data_P0.55.txt ...
g_list = []

for P in P_values:
    filename = f"D:\\Data\\pu\\toe_{P:.2f}_2000_100_p.csv"  # 修改为你自己的实际文件路径
    # omega_data = np.loadtxt(filename)  # 每组数据是一维数组
    df = pd.read_csv(filename, header=None)
    omega_data = df.values.flatten()
    omega_data = np.sqrt(omega_data)
    # omega_centers = [np.sqrt(i) for i in omega_data]
    # 用直方图估计 g(w)，密度=True 自动归一化
    hist, _ = np.histogram(omega_data, bins=omega_bins, density=False) 
    g_list.append(hist/100)

# 转为 2D 数组（P 行 × ω 列）
g_array = np.array(g_list)

# ---------- 关键修改：将 axes[1,0] 替换为 3D 子图 ----------
# --- 子图2: 替换为3D图 ---
fig.delaxes(axes[0,1])  # 移除默认2D子图
ax_3d = fig.add_subplot(2, 2, 2, projection='3d')  # 添加3D子图
axes[0,1] = ax_3d

# 绘制3D线框
Omega, P_grid = np.meshgrid(omega_centers, P_values)
wire = ax_3d.plot_wireframe(Omega, P_grid, g_array, color='black', linewidth=0.6, zorder=1)  # 确保3D图形在底层
# 在绘制3D图之后添加
wire = ax_3d.plot_wireframe(Omega, P_grid, g_array, 
                           color='black', 
                           linewidth=0.6/scale_change, 
                           zorder=1,
                           rasterized=True)  # 关键：栅格化
# 设置标签和视角
ax_3d.set_xlabel(r'$\omega$', fontsize=12)
ax_3d.set_ylabel(r'$H$', fontsize=12)
ax_3d.set_zlabel(r'$g(\omega)$', fontsize=12)
ax_3d.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax_3d.tick_params(axis='x', pad=-5)   # x 轴刻度标签与轴的距离
ax_3d.tick_params(axis='y', pad=-5)   # y 轴刻度标签与轴的距离
ax_3d.tick_params(axis='z', pad=-2)   # y 轴刻度标签与轴的距离
ax_3d.xaxis.labelpad = -9
ax_3d.yaxis.labelpad = -9
ax_3d.zaxis.labelpad = -3
ax_3d.view_init(elev=30, azim=-60)
# 反转 x 轴方向（ω 从大到小）
ax_3d.set_xlim(ax_3d.get_xlim()[::-1])
# 移除灰色背景
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False


# 关键修改：使用Line3DCollection绘制3D边框

# 设定 ω 轴范围
omega_bins = np.arange(0, 4.0001, 0.01)  # 包括右端点
omega_centers = (omega_bins[:-1] + omega_bins[1:]) / 2

# 离散 P 值
P_values = np.arange(0.50, 0.95, 0.05)

# 假设每组 P 对应一个文件，如 data_P0.50.txt, data_P0.55.txt ...
g_list = []

for P in P_values:
    filename = f"D:\\Data\\pu\\toe_{P:.2f}_2000_100_p.csv"  # 修改为你自己的实际文件路径
    # omega_data = np.loadtxt(filename)  # 每组数据是一维数组
    df = pd.read_csv(filename, header=None)
    omega_data = df.values.flatten()
    omega_data = np.sqrt(omega_data)
    # omega_centers = [np.sqrt(i) for i in omega_data]
    # 用直方图估计 g(w)，密度=True 自动归一化
    hist, _ = np.histogram(omega_data, bins=omega_bins, density=False) 
    g_list.append(hist/100)
for i,h in enumerate(P_values):
    axes[1, 0].plot(omega_centers,g_list[i],c=color_list[i],linewidth=1.5/scale_change)
    axes[1, 0].scatter(omega_centers,g_list[i],c=color_list[i],s=30/scale_change_2)
    # plt.plot(omega_centers, W[f"H={h:.2f}"], color = color_list[i],label=f"H={h:.2f}")
    # axes[1, 1].legend()
axes[1, 0].set_yscale('log')
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel(r'$\omega$', fontsize=12)
axes[1, 0].set_ylabel(r'$g(\omega)$',fontsize=12)
ax3 = axes[1, 0]
ax3.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax3.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax3.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax3.xaxis.labelpad = 0.5
ax3.yaxis.labelpad = 0.5

# 只创建一个 inset 子图
slope_g = []
# inset_ax = inset_axes(axes[1, 0], width="50%", height="50%", loc='center left', borderpad=2)
inset_ax = inset_axes(
    axes[1, 0],
    width="50%",
    height="50%",
    loc='upper right',
    bbox_to_anchor=(-0.40, -0.05,1.0, 1.0),
    bbox_transform=axes[1, 0].transAxes,
    borderpad=0.02
)

for i, h in enumerate(P_values):
    # 找最大峰值位置
    max_idx = np.argmax(g_list[i])

    # 拟合峰值右侧部分
    fit_omega = omega_centers[max_idx+1:]
    fit_values = g_list[i][max_idx+1:]
    # 去掉尾部10%

    cutoff = int(len(fit_omega) * 0.9)
    fit_omega = fit_omega[10:cutoff]
    fit_values = fit_values[10:cutoff]

    # 去除零值（避免 log 出现 -inf）
    valid = (fit_omega > 0) & (fit_values > 0)
    log_x = np.log10(fit_omega[valid])
    log_y = np.log10(fit_values[valid])

    # 线性拟合（log-log）
    slope, intercept, *_ = linregress(log_x, log_y)
    fit_line = 10 ** (slope * log_x + intercept)
    slope_g.append(slope)
    # 在同一个 inset_ax 上绘图
    inset_ax.plot(fit_omega[valid], fit_values[valid], 'o', color=color_list[i], markersize=2/scale_change_2)
    inset_ax.plot(fit_omega[valid], fit_line, '-', color=color_list[i], linewidth=1.0/scale_change,
                  )
    text = fr'$H={h:.2f},\ \omega^{{{slope:.1f}}}$'

# 设置坐标、图例等
inset_ax.set_xlabel(r'$\omega$', fontsize=10)
# inset_ax.set_ylabel(r'$\rho(\omega)$', fontsize=8)
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
# ax3 = axes[1, 0]

inset_ax.tick_params(axis='both', labelsize=6)   # 同时设置 x 和 y
inset_ax.tick_params(axis='x', pad=0.2)          # x 轴刻度标签与轴的距离
inset_ax.tick_params(axis='y', pad=0.2)          # y 轴刻度标签与轴的距离

# 只显示 x 轴刻度 1 和 10
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
inset_ax.set_xticks([1, 2,3])
inset_ax.set_xticklabels(['1', '2','3'])
inset_ax.xaxis.set_minor_formatter(NullFormatter())
inset_ax.xaxis.labelpad = 0
inset_ax.yaxis.labelpad = 0


inset_ax.tick_params(labelsize=8)
inset_ax.tick_params(axis='both', which='minor', labelsize=8)  # 隐藏次刻度标签

#(1,1) plot######################################################################################
WC = []
for i ,h in enumerate(H):
    print(i, h)
    data_name = f"D:\\Data\\pu\\toe_{h:.2f}_2000_100_p.csv"
    data = pd.read_csv(data_name, header=None)
    data = data.values
    data = np.sqrt(data)
    # data = [np.sqrt(i) for i in data]
    L = data.shape[1]
    wc_h = 0
    for j in range(L):
        wc_h += min(data[:,j])
    # print(wc_h / L)
    WC.append(wc_h/L)
# print(WC)
color_a = "#D5A19C"
color_b = "#DA635C"

axes[1, 1].plot(H, WC, color='k',linewidth=1.5/scale_change)
axes[1, 1].scatter(H, WC, c='k',s=40/scale_change_2)
# 线性拟合
slope, intercept, r_value, p_value, std_err = stats.linregress(H[1:], WC[1:])

# 拟合直线的y值
fit_line = slope * H + intercept

# 画拟合直线
# fr'$\omega_c \sim {slope:.2f} H$'
axes[1, 1].plot(H, fit_line, color=color_b,linewidth=1.5/scale_change, label=fr'$\omega_c \sim {slope:.3f} H$')



axes[1, 1].legend(fontsize=10,frameon=False)
axes[1, 1].set_xlabel(r'$H$',fontsize=12)
axes[1, 1].set_ylabel(r'$\omega_c$',fontsize=12)
ax4 = axes[1, 1]
ax4.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax4.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax4.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
ax4.xaxis.labelpad = 1
ax4.yaxis.labelpad = 1


#######################################################################################
# -------------------- 添加子图标注 --------------------
labels = ['(a)', '(c)', '(d)']
positions = [(0,0) ,(1,0), (1,1)]

for label, pos in zip(labels, positions):
    ax = axes[pos]
    if pos == (0,1):
        ax.text2D(0.02, 0.02, label, transform=ax.transAxes, fontsize=14)  # 3D子图用 text2D
    else:
        ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=14)
# 获取原始位置


# -------------------- 去除子图之间的空白 --------------------
# fig.tight_layout(pad=0.5)  # 自动调整布局
# fig.subplots_adjust(wspace=0.15, hspace=0.15)  # 控制子图间距（按需调）
plt.subplots_adjust(
    left=0.06,     # 整体左边距
    bottom=0.06,   # 整体下边距
    right=0.94,    # 右边距
    top=0.94,      # 上边距
    wspace=0.18,   # 子图之间水平间距
    hspace=0.18    # 子图之间垂直间距
)
# plt.tight_layout()
# -------------------- 3D子图添加一个四边形边框 --------------------
#3D 图上移一部分
pos = ax_3d.get_position()  # 获取当前 bbox，返回 Bbox对象
x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

new_y0 = y0 + 0.01  # 向上移动0.05（视需要调整）

ax_3d.set_position([x0, new_y0, w, h])

import matplotlib.patches as patches

# 获取子图位置 (figure坐标系，0~1)
pos_00 = axes[0,0].get_position()
pos_11 = axes[1,1].get_position()

# 计算矩形位置和大小
x_start = pos_11.x0
x_end = pos_11.x1
y_start = pos_00.y0
y_end = pos_00.y1

width = x_end - x_start
height = y_end - y_start

# 创建矩形patch (在figure坐标系)
rect = patches.Rectangle(
    (x_start, y_start),  # 左下角
    width,
    height,
    fill=False,
    edgecolor='black',
    linewidth=0.75,
    transform=fig.transFigure,  # 关键：坐标系是figure
    zorder=10
)

# 添加矩形到figure
fig.patches.append(rect)
# 添加 (b) 标签
fig.text(
    x_start + 0.008,         # x 位置，略偏右
    y_start + 0.006,         # y 位置，略偏上
    '(b)',                  # 显示的文本
    fontsize=14,
    # fontweight='bold',
    ha='left',
    va='bottom'
)
# -------------------- 显示图像 --------------------

plt.savefig("fig03.eps", dpi=600,format= 'eps',pad_inches=0.05)
plt.savefig("fig03.tiff", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig03.png", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig03.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig03.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
# import matplotlib

plt.show()