import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from def_plot_NE_SSEC_HSI_zuhe_2 import spectrum_density_plot_HSI,spectrum_density_plot_SSEC
from plot_Hurst_2_plot import plot_Hurst_HSI,plot_Hurst_SSEC
from plot_R_ave_fit_Hurst import plot_R_HSI,plot_R_SSEC
#拟合beta分布

# 数据
x = np.linspace(0, 10, 100)
y1, y2, y3, y4, y5, y6 = np.sin(x), np.cos(x), np.exp(-x/3)*np.sin(2*x), np.tan(x/5), np.sqrt(x), np.exp(-x/2)

# 创建 4行2列 的布局
#尺度放缩
width_fig = 18 /  2.54
height_fig = (12) / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig = plt.figure(figsize=(width_fig, height_fig))#constrained_layout=True
gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3)

# 第一行：合并两个子图

ax1 = fig.add_subplot(gs[0, :])   # 占据整行
# ax1.plot(x, y1, label='Plot 1')
# ax1.set_title('Merged Plot (Row 1)')
spectrum_density_plot_SSEC(ax=ax1)
ax1.text(0.02, 0.05, '(a)', transform=ax1.transAxes, fontsize=14)

# 第二行：两个独立子图
ax2 = fig.add_subplot(gs[1, 0])
# ax2.plot(x, y2, color='orange')
plot_Hurst_SSEC(ax2)
# ax2.set_title('Subplot 2')
ax2.text(0.02, 0.05, '(b)', transform=ax2.transAxes, fontsize=14)

ax3 = fig.add_subplot(gs[1, 1])
# ax3.plot(x, y3, color='green'
plot_R_SSEC(ax3)
# ax3.set_title('Subplot 3')
ax3.text(0.02, 0.05, '(c)', transform=ax3.transAxes, fontsize=14)


# plt.tight_layout()
# plt.savefig("fig15.svg",dpi=300)
plt.savefig("fig11.png", dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig11.pdf", dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig11.svg", dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig11.eps", dpi=600,format= 'eps',bbox_inches='tight', pad_inches=0.05)
plt.show()

