import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Brody分布的概率密度函数
def brody_distribution(s, nu):
    """
    Brody分布的概率密度函数
    """
    alpha = gamma((nu + 2) / (nu + 1)) ** (nu + 1)
    pdf = alpha * (nu + 1) * (s ** nu) * np.exp(-alpha * (s ** (nu + 1)))
    return pdf

# 生成s值范围
s = np.linspace(0, 4, 500)
def get_brody_data(s=np.linspace(0, 4, 500),nu=0.5,n_points=80,seed_set = None,noise_level=0.05):
    pdf_curve = brody_distribution(s, nu)
    # 创建散点数据 - x轴均匀分布
    if seed_set is not None:
        np.random.seed(seed_set)
    else:
        np.random.seed(42)  # 设置随机种子以便重现
    s_min, s_max = 0.1, 4  # 从0.1开始避免0值问题
    s_samples = np.linspace(s_min, s_max, n_points)
    # 计算理论PDF值
    pdf_theoretical = brody_distribution(s_samples, nu)
    # 添加相对噪声（与理论值成比例）
    relative_noise = pdf_theoretical * noise_level * np.random.randn(n_points)

    # 添加绝对噪声（固定幅度）
    absolute_noise = 0.05 * np.random.randn(n_points)
    absolute_noise =0 
    # 最终的散点数据（带波动）
    pdf_samples = pdf_theoretical + relative_noise + absolute_noise

    # 确保没有负值和异常值
    pdf_samples = np.maximum(pdf_samples, 0.01)
    pdf_samples = np.minimum(pdf_samples, pdf_theoretical.max() * 1.5)
    return pdf_curve, s_samples, pdf_samples

# 选择特定的nu值
nu = 0.5
pdf_curve = brody_distribution(s, nu)

# 创建散点数据 - x轴均匀分布
np.random.seed(42)  # 设置随机种子以便重现
n_points = 80  # 散点数量

# 在s轴上均匀分布80个点
# s的范围从0到2.5，覆盖主要区域
s_min, s_max = 0.1, 4  # 从0.1开始避免0值问题
s_samples = np.linspace(s_min, s_max, n_points)

# 计算理论PDF值
pdf_theoretical = brody_distribution(s_samples, nu)

# 为散点添加波动（模拟实验误差）
# 设置噪声参数
noise_level = 0.05  # 增加噪声水平，因为均匀分布的x值可能更分散

# 添加相对噪声（与理论值成比例）
relative_noise = pdf_theoretical * noise_level * np.random.randn(n_points)

# 添加绝对噪声（固定幅度）
absolute_noise = 0.05 * np.random.randn(n_points)
absolute_noise =0 
# 最终的散点数据（带波动）
pdf_samples = pdf_theoretical + relative_noise + absolute_noise

# 确保没有负值和异常值
pdf_samples = np.maximum(pdf_samples, 0.01)
pdf_samples = np.minimum(pdf_samples, pdf_theoretical.max() * 1.5)

# 创建图形
plt.figure(figsize=(14, 8))

# 绘制理论曲线
plt.plot(s, pdf_curve, 'royalblue', linewidth=3.5, 
         label=f'Brody分布理论曲线 (ν={nu})', zorder=5)

# 绘制散点数据
scatter = plt.scatter(s_samples, pdf_samples, 
                     c=pdf_samples,  # 使用颜色映射
                     cmap='viridis',
                     s=100, alpha=0.8,
                     edgecolors='white', linewidth=1.5,
                     label=f'模拟数据 (n={n_points}个点, x轴均匀分布)', 
                     zorder=6)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('P(s)值', fontsize=12)

# 添加从散点到理论曲线的垂直线（显示偏差）
for i in range(0, n_points, 2):  # 每隔一个点画一条线，避免太密集
    plt.plot([s_samples[i], s_samples[i]], 
             [pdf_theoretical[i], pdf_samples[i]],
             color='gray', alpha=0.2, linewidth=0.8, zorder=3)

# 添加理论曲线下的阴影区域
plt.fill_between(s, 0, pdf_curve, color='lightblue', alpha=0.2, zorder=1)

# 设置图形属性
plt.title(f'Brody分布与均匀x轴分布的模拟数据 (ν={nu})', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('归一化能级间隔 s', fontsize=14, fontweight='bold')
plt.ylabel('概率密度 P(s)', fontsize=14, fontweight='bold')

# 添加图例
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)

# 设置网格
plt.grid(True, alpha=0.3, linestyle='--', zorder=2)

# 设置坐标轴范围
plt.xlim(0, 2.6)
plt.ylim(-0.05, 1.3)

# 在右上角添加统计信息框
stats_text = f'统计信息:\n'
stats_text += f'ν = {nu}\n'
stats_text += f'数据点数: {n_points}\n'
stats_text += f's分布: [{s_min:.2f}, {s_max:.2f}]均匀分布\n'
stats_text += f's间隔: {(s_max-s_min)/(n_points-1):.3f}\n'
stats_text += f'理论-数据均方误差: {np.mean((pdf_samples - pdf_theoretical)**2):.5f}'

plt.text(1.75, 1.1, stats_text, 
         fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='top')

# 在左侧添加说明
info_text = f'散点分布特点:\n'
info_text += f'1. x轴均匀分布\n'
info_text += f'2. 80个等间距点\n'
info_text += f'3. y轴添加随机波动\n'
info_text += f'4. 颜色表示P(s)值'

plt.text(0.05, 0.95, info_text, 
         fontsize=11, 
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
         verticalalignment='top')

plt.tight_layout()
plt.show()

# 创建第二个图形：详细分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Brody分布均匀x轴散点数据分析 (ν={nu}, n={n_points})', fontsize=16, fontweight='bold')

# 子图1: 散点与理论曲线对比
ax1 = axes[0, 0]
ax1.plot(s, pdf_curve, 'royalblue', linewidth=2.5, label='理论曲线')
ax1.scatter(s_samples, pdf_samples, c='crimson', s=50, alpha=0.7, edgecolors='white', label='均匀分布散点')
ax1.set_xlabel('s')
ax1.set_ylabel('P(s)')
ax1.set_title('散点分布图')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: x轴分布直方图（验证均匀性）
ax2 = axes[0, 1]
ax2.hist(s_samples, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='navy')
ax2.axhline(y=1/(s_max-s_min), color='red', linestyle='--', linewidth=2, label='理论均匀分布')
ax2.set_xlabel('s')
ax2.set_ylabel('密度')
ax2.set_title('x轴分布直方图（验证均匀性）')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: 理论值与实际值的偏差
ax3 = axes[1, 0]
deviations = pdf_samples - pdf_theoretical
ax3.bar(range(n_points), deviations, alpha=0.7, color='orange', edgecolor='darkorange')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('数据点索引')
ax3.set_ylabel('偏差 (实际值 - 理论值)')
ax3.set_title(f'每个点的偏差 (平均偏差: {np.mean(deviations):.4f})')
ax3.grid(True, alpha=0.3)

# 子图4: 偏差的正态分布检验
ax4 = axes[1, 1]
ax4.hist(deviations, bins=15, density=True, alpha=0.7, color='lightgreen', edgecolor='darkgreen')

# 添加正态分布拟合曲线
from scipy.stats import norm
mu, std = norm.fit(deviations)
x = np.linspace(min(deviations), max(deviations), 100)
p = norm.pdf(x, mu, std)
ax4.plot(x, p, 'k', linewidth=2, label=f'正态拟合\nμ={mu:.3f}, σ={std:.3f}')

ax4.set_xlabel('偏差值')
ax4.set_ylabel('密度')
ax4.set_title('偏差分布的正态性检验')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 创建第三个图形：不同噪声水平的对比
plt.figure(figsize=(14, 6))

# 三个不同的噪声水平
noise_levels = [0.05, 0.15, 0.25]
colors = ['green', 'orange', 'red']

for i, noise in enumerate(noise_levels):
    # 生成带不同噪声的数据
    pdf_noisy = pdf_theoretical + pdf_theoretical * noise * np.random.randn(n_points) + 0.03 * np.random.randn(n_points)
    pdf_noisy = np.maximum(pdf_noisy, 0.01)
    
    plt.subplot(1, 3, i+1)
    plt.plot(s, pdf_curve, 'blue', linewidth=2, label='理论曲线')
    plt.scatter(s_samples, pdf_noisy, c=colors[i], s=50, alpha=0.7, edgecolors='white', 
                label=f'噪声={noise*100:.0f}%')
    
    plt.title(f'噪声水平: {noise*100:.0f}%')
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.3)

plt.suptitle('不同噪声水平下的均匀x轴散点分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 输出详细统计信息
print("=" * 60)
print(f"Brody分布均匀x轴散点数据详细统计 (ν={nu}, n={n_points})")
print("=" * 60)
print(f"x轴(s值)分布:")
print(f"  范围: [{s_min:.3f}, {s_max:.3f}]")
print(f"  间隔: {s_samples[1]-s_samples[0]:.4f}")
print(f"  标准差: {np.std(s_samples):.4f} (理论均匀分布: {(s_max-s_min)/np.sqrt(12):.4f})")
print()

print(f"y轴(P(s)值)统计:")
print(f"  理论值范围: [{pdf_theoretical.min():.4f}, {pdf_theoretical.max():.4f}]")
print(f"  实际值范围: [{pdf_samples.min():.4f}, {pdf_samples.max():.4f}]")
print(f"  平均偏差: {np.mean(deviations):.6f}")
print(f"  偏差标准差: {np.std(deviations):.6f}")
print(f"  均方误差(MSE): {np.mean(deviations**2):.6f}")
print(f"  平均绝对误差(MAE): {np.mean(np.abs(deviations)):.6f}")
print()

print(f"数据质量指标:")
print(f"  信噪比(SNR): {np.std(pdf_theoretical)/np.std(deviations):.2f}")
print(f"  相关系数: {np.corrcoef(pdf_theoretical, pdf_samples)[0,1]:.4f}")
print("=" * 60)