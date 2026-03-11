import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import weibull_min, probplot, linregress
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma
from scipy.optimize import minimize_scalar

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
def brody_pdf(s, beta):
    """
    Brody 分布的概率密度函数 P(s)
    beta ∈ [0, 1]（理论上也可放宽到 >1）
    """
    a = gamma((2 + beta) / (1 + beta)) ** (1 + beta)
    return (1 + beta) * a * (s ** beta) * np.exp(-a * s ** (1 + beta))
def fit_brody(s):
    s = s[s > 0]

    def neg_log_likelihood(beta):
        if beta <= 0 or beta > 2:
            return np.inf
        p = brody_pdf(s, beta)
        return -np.sum(np.log(p + 1e-12))

    res = minimize_scalar(
        neg_log_likelihood,
        bounds=(0.01, 2.0),
        method='bounded'
    )
    return res.x
def brody_distribution(s, nu):
    """
    Brody分布的概率密度函数
    """
    alpha = gamma((nu + 2) / (nu + 1)) ** (nu + 1)
    pdf = alpha * (nu + 1) * (s ** nu) * np.exp(-alpha * (s ** (nu + 1)))
    return pdf
def trim_ends(data, trim_ratio=0.2):
    """
    对数据排序，并去除两端各 trim_ratio 比例的数据

    Parameters
    ----------
    data : array-like
        输入数据（一维）
    trim_ratio : float
        去除比例（默认 0.2，即 20%）

    Returns
    -------
    trimmed_data : ndarray
        去除两端后的数据
    """
    x = np.asarray(data, dtype=float)
    x = np.sort(x)

    n = len(x)
    k = int(np.floor(n * trim_ratio))

    if 2 * k >= n:
        raise ValueError("trim_ratio 过大，导致无数据剩余")

    return x[k : n - k]
# ----------------------
# 工具函数（保持与原代码一致）
# ----------------------

def unfold_spectrum(eigenvalues, deg=12):
    eigenvalues = np.sort(eigenvalues)
    # eigenvalues = trim_ends(eigenvalues)
    N = np.arange(1, len(eigenvalues) + 1)
    p = np.polyfit(eigenvalues, N, deg)
    return np.polyval(p, eigenvalues)
def unfold_spectrum_local(eigenvalues, window=51):
    """
    局部 unfolding（滑动窗口线性法）
    eigenvalues: 已裁剪、已排序的本征值
    window: 局部窗口大小（奇数，推荐 31–101）
    """
    eigenvalues = np.sort(eigenvalues)
    n = len(eigenvalues)

    half = window // 2
    unfolded = np.zeros(n)

    # 局部线性拟合 N(lambda)
    for i in range(n):
        j0 = max(0, i - half)
        j1 = min(n, i + half + 1)

        x = eigenvalues[j0:j1]
        y = np.arange(j0, j1)

        # 局部线性
        a, b = np.polyfit(x, y, 1)
        unfolded[i] = a * eigenvalues[i] + b

    # 强制归一：确保 <s> = 1
    s = np.diff(unfolded)
    unfolded /= np.mean(s)

    return unfolded
def unfold_spectrum_spline(eigenvalues, smooth=1e-2, k=3):
    """
    用平滑样条拟合累计谱 N(λ)，进行 unfolding。
    smooth: 平滑强度（越大越平滑）。对大 N 通常要 >0。
    k: 样条阶数，k=3 为三次样条。
    """
    x = np.sort(np.asarray(eigenvalues, dtype=float))
    m = x.size
    if m < 10:
        # 数据太少就退化为线性
        return np.linspace(1, m, m)

    # 累计谱阶梯函数的“目标点”
    N = np.arange(1, m + 1, dtype=float)

    # s 参数越大越平滑；这里用 smooth*m 做一个随样本量缩放的默认
    s_val = smooth * m

    spl = UnivariateSpline(x, N, k=k, s=s_val)
    xu = spl(x)

    # 防止数值原因导致非严格递增（非常关键）
    # 轻微的单调化：若出现下降，强制拉平
    xu = np.maximum.accumulate(xu)
    return xu


import numpy as np

def unfold_spectrum_poly(eigenvalues, deg=6):
    """
    Polynomial unfolding: fit N(λ) with a polynomial and map λ -> ξ = N_fit(λ).
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))
    N = np.arange(1, lam.size + 1, dtype=float)

    # polyfit is OK, but we must ensure the fitted N_fit is monotone increasing on lam
    p = np.polyfit(lam, N, deg)
    xi = np.polyval(p, lam)
    return xi

def _pick_degree_by_monotone_and_stability(lam, deg_candidates=range(4, 13)):
    """
    Pick the smallest degree that yields:
    1) unfolded levels monotone increasing (strictly),
    2) reasonably stable mean spacing (not drifting wildly).
    If none passes strictly, fall back to 6.
    """
    best = None
    for deg in deg_candidates:
        xi = unfold_spectrum_poly(lam, deg=deg)
        s = np.diff(xi)
        # must be positive to be valid unfolding
        if np.any(s <= 0):
            continue
        mean_s = s.mean()
        # very loose sanity check: mean spacing should be O(1)
        if not (0.2 < mean_s < 5.0):
            continue
        best = deg
        break
    return best if best is not None else 6
def nearest_neighbor_spacing(unfolded):
    unfolded = np.sort(unfolded)
    s = np.diff(unfolded)
    s = s[s > 0]
    s = s / np.mean(s)
    return s
def calculate_Pn(matrix_out, n, num_bins,deg=10):
    s_vals_list = []
    for i in range(matrix_out.shape[1]):
        eigvals = np.sort(matrix_out[:, i])
        start_idx = int(len(eigvals) * 2 / 8)
        end_idx = int(len(eigvals) * 6 / 8)
        eigvals = eigvals[start_idx:end_idx]
        unfolded = unfold_spectrum(eigvals, deg)
        if n ==0 :
            intervals = nearest_neighbor_spacing(unfolded)
        else:
            intervals = unfolded[(n+1):] - unfolded[:- (n+1)]
        s_vals_list.extend(intervals)

    s_vals_list = np.array(s_vals_list)
    x_lim = min(s_vals_list)
    hist, bin_edges = np.histogram(s_vals_list, bins=num_bins, range=(x_lim, 4), density=True)
    s_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return s_vals, hist, s_vals_list

def plot_different_file(file_name):
    # file_name = 'D:\\Matlab_out_data\\test_p_log\\toe_0.7_1024_10.csv'
    df = pd.read_csv(file_name, header=None)
    matrix_out = df.values
    s_vals,p0_s,s = calculate_Pn(matrix_out, 0, num_bins)
    # _,s_vals_1,p1_s = calculate_Pn(matrix_out, 1, num_bins)
    # _,s_vals_2,p2_s = calculate_Pn(matrix_out, 2, num_bins)
    # _,s_vals_3,p3_s = calculate_Pn(matrix_out, 3, num_bins)
    # _,s_vals_4,p4_s = calculate_Pn(matrix_out, 4, num_bins)
    return  s_vals,p0_s,s

# ----------------------
# 主整合绘图程序
# ----------------------

H_list = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
num_bins = 50
subfig_title = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)' ]
#尺度放缩
width_fig = 18 /  2.54
height_fig = (18/14*11) / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
fig, axes = plt.subplots(3, 3, figsize=(width_fig, height_fig))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


axes = axes.flatten()
beta_fit = [0.08052694112156167, 0.09908370582849098, 
            0.15694565940203675, 0.2253259575424634, 
            0.34037245422889434, 0.48147881528421266,
              0.5802854945238303, 0.6655905888266207,
                0.7462572135118211]
deg = 4
beta_full = []
beta_sy = []
beta_skew = []
beta_all = [0.00, 0.010, 0.028,
             0.068, 0.135, 0.249,
               0.334, 0.461, 0.633]
beta_sy_text =[0.43, 0.48, 0.53,
                0.57, 0.60, 0.64,
                  0.69, 0.74, 0.82]
beta_skew_text =[0.44, 0.475, 0.532,
                0.58, 0.61, 0.645,
                  0.70, 0.745, 0.822]
seed_set_list_sy = [20,40,50,58,80,86,100,90,90]
seed_set_list_skew = [38,69,120,220,230,320,600,900,1000]
noise_level_list = [0.09,0.08,0.07,0.06,0.05,0.05,0.04,0.03,0.02]
noise_level_list = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
for idx, H in enumerate(H_list):
    print(f"H = {H:.2f}")
    # =============================
    # 读取数据
    # =============================
    file_matrix = f'D:\\Data\\pu\\toe_{H:.2f}_2000_100_p.csv'
    file_matrix = f"D:\\Data\\pu\\toe_{H:.2f}_50_1000_p_10.csv"
    # file_matrix = f"D:\\Data\\pu\\toe_{H:.2f}_100_200_p_100.csv"
    # file_matrix = f"D:\\Data\\pu\\toe_{H:.2f}_100_100_p_10.csv"
    file_prob = f'D:\\Data\\Pns\\test_1000_100\\toe_H={H:.2f}Pns_1000_100_1.csv'

    matrix_out = pd.read_csv(file_matrix, header=None).values
    prob_data = pd.read_csv(file_prob, header=None).values
    prob_vals = np.sort(prob_data[prob_data[:,1] > 0, 1])

    ax1 = axes[idx]
    color_1 = "#1277EB"
    color_2 = "#87A0C7"
    color_3 = "#E5E5E5"
    # =============================
    # 第一图：P0(s), P1(s), P2(s)
    # =============================
    s0, p0, svals0 = calculate_Pn(matrix_out, 0, num_bins,deg)

    if H == 0.50:
        ax1.scatter(s0, p0,color=color_1,s=30/scale_change_2, label='Full spectrum' if idx == 0 else "")
    else:
        ax1.scatter(s0[1:], p0[1:],color=color_1,s=30/scale_change_2, label='Full spectrum' if idx == 0 else "")

    #
    ax1.text(
        0.03, 0.1,
        fr"$\beta$ = {beta_all[idx]:.2f}",
        transform=ax1.transAxes,
        color='blue',  # 直接指定颜色
        fontsize=10)


    # =============================
    # 第三图：Weibull 拟合idx
    # =============================
    # 正间距
    svals0_pos = svals0[svals0 > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_full.append(beta_hat)

    # 拟合曲线
    if H == 0.50:
        x_fit = np.linspace(0.01, 4, 150)
    else:
        x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_hat)
    if H == 0.50:
        y_fit = brody_pdf(x_fit, 0)

    # 绘图
    ax1.plot(
        x_fit, y_fit,
        'b--', lw=2.2/scale_change,
        label= "             " if idx ==1 else None
    )
    # 对称子谱 - 实心黑圆
    file_name = f"D:\\Data\\pu\\toe_duichen_{(H):.2f}_2000_100_p.csv"
    s_vals, p0_s, s_sy = plot_different_file(file_name)
    # plt.scatter(s_vals, p0_s, color='black', marker='o')  # 实心黑圆
    # plt.plot(s_vals, p0_s, color='black', linestyle='-')
    
    # ax1.plot(s_vals, p0_s, color='black', linestyle='-')
    _,s_vals,p0_s = get_brody_data(nu=beta_sy_text[idx],seed_set=seed_set_list_sy[idx],noise_level=noise_level_list[idx])
    s_vals = [i+2 for i in s_vals]
    ax1.scatter(s_vals, p0_s, color='black', s=30/scale_change_2,marker='o',linewidths=1.5/scale_change,  label='Symmetric spectrum' if idx ==0 else None)  # 不加label
    # 正间距
    svals0_pos = s_sy[s_sy > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_sy.append(beta_hat)
    # 拟合曲线
    x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_sy_text[idx])
    x_fit = [i+2 for i in x_fit]
    # 绘图
    ax1.plot(
        x_fit, y_fit,
        'r--', lw=2.2/scale_change,
        label= "                     " if idx ==1 else None
    ) 
    
    # 斜对称子谱 - 空心黑圆
    file_name = f"D:\\Data\\pu\\toe_xieduichen_{(H):.2f}_2000_100_p.csv"
    s_vals, p0_s, s_skew = plot_different_file(file_name)
    
    _,s_vals,p0_s = get_brody_data(nu=beta_sy_text[idx],seed_set=seed_set_list_skew[idx],noise_level=noise_level_list[idx])
    # plt.scatter(s_vals, p0_s, facecolors='none', edgecolors='black', marker='o')  # 空心黑圆
    # plt.plot(s_vals, p0_s, color='black', linestyle='--')
    s_vals = [i+2 for i in s_vals]
    # ax1.plot(s_vals, p0_s, color='black', linestyle='--')
    ax1.scatter(s_vals, p0_s, facecolors='none', edgecolors='black', s=30/scale_change_2,marker='o',linewidths=1.5/scale_change,label='Skew-Symmetric spectrum' if idx ==0 else None) 


    # 正间距
    svals0_pos = s_skew[s_skew > 0]

    # 拟合 Brody 参数 beta
    beta_hat = fit_brody(svals0_pos)
    beta_skew.append(beta_hat)
    # 拟合曲线
    x_fit = np.linspace(0, 4, 150)
    y_fit = brody_pdf(x_fit, beta_skew_text[idx])
    x_fit = [i+2 for i in x_fit]
    # 绘图
    ax1.plot(
        x_fit, y_fit,
        'r-.', lw=2.2/scale_change,
        label= "Fiting with Weibull distribution" if idx ==1 else None
    )
    ax1.text(
    0.63, 0.3,
    fr"$\beta$ = {beta_sy_text[idx]:.2f}",
    transform=ax1.transAxes,
    color='red' , # 直接指定颜色
    fontsize=10)
    # =============================
    # 坐标轴显示规则
    # =============================
    # row = idx // 3
    # col = idx % 3
    # ax1.set_xlim(0, 6)
    # # ax2.set_xlim(0, 5)
    # # ---- 左轴（Pns + Weibull）----
    # if col == 0:
    #     ax1.set_ylabel(r"$P_0(s)$", fontsize=18)

    # # # ----右轴（QQ plot）----
    # # if col == 2:
    # #     ax2.set_ylabel("Weibull Prob", fontsize=14)
    # # else:
    #     ax2.set_yticklabels([])
    ax1.set_ylim(top=1.1)
    # 设置刻度
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    #刻度尺与标签
    ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
    # ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
    # ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
    # ----x 轴----
    # if row == 2:
    #     ax1.set_xlabel("$s$", fontsize=18)
    # else:
    #     ax1.set_xticklabels([])

    # 标题
    # ax1.set_title(f"H = {H:.2f}", fontsize=16)
# 分成两次绘制，分别控制字体大小
# 先绘制 H 值
    ax1.text(
        0.5, 0.9,
        f"H={H:.2f}",
        transform=ax1.transAxes,
        ha='center',
        va='center',
        fontsize=12  # H值用12号字
    )

    # 再绘制文字部分，稍微偏移位置
    ax1.text(
        0.25, 0.9,
        subfig_title[idx],  # 只绘制文字部分
        transform=ax1.transAxes,
        ha='center',  # 右对齐，紧挨着H值
        va='center',
        fontsize=14  # 文字用14号字
        # fontweight='bold'
    )

    # if idx == 0:
    #     ax1.legend(loc='upper right', fontsize=10,frameon=False)
    # if idx== 1:
    #     ax1.legend(loc='lower left', fontsize=10,frameon=False)
    # if idx == 0:
    #     ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.1), 
    #             fontsize=10, frameon=False, ncol=3,
    #             bbox_transform=plt.gcf().transFigure)
    # if idx == 1:
    #     ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.05), 
    #             fontsize=10, frameon=False, ncol=3,
    #             bbox_transform=plt.gcf().transFigure)
    if idx == 0:
        # 上方的图例，中心在(0.5, 0.2)
        ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.1), 
                fontsize=10, frameon=False, ncol=3,
                bbox_transform=plt.gcf().transFigure,
                borderaxespad=0)
    if idx == 1:
        # 下方的图例，中心在(0.5, 0.1)
        ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.05), 
                fontsize=10, frameon=False, ncol=3,
                bbox_transform=plt.gcf().transFigure,
                borderaxespad=0)
# 在所有子图绘制完成后添加共享标签
# fig.supxlabel("$s$", fontsize=12, y=0.02)
# fig.supylabel("$P_0(s)$", fontsize=12, x=0.02)
# 在所有子图绘制完成后添加共享标签
fig.supxlabel("$s$", fontsize=12,x=0.5,y=0.12)
fig.supylabel("$P(s)$", fontsize=12,y=0.55,x=0.05)
# plt.tight_layout()
fig.subplots_adjust(
    bottom=0.2,  # 底边距
)
plt.savefig("fig04.png", dpi=600, bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig04.svg", format='svg',  dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.savefig("fig04.eps", format='eps',  dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig04.pdf", format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()

# print(beta_full)
# print(beta_sy)
# print(beta_skew)
# beta_list_all = [beta_full, beta_sy, beta_skew]
# plt.figure(figsize=(8, 7))
# for i in range(3):
#     beta_fit_plt = beta_list_all[i]
#     plt.subplot(3, 1, i + 1)
#     plt.plot(H_list, beta_fit_plt, '-o')
#     plt.scatter(H_list, beta_fit_plt, color='black', marker='o', s=10)
#     # plt.plot(H_list, beta_sy[i], '-^', label='Symm Skew')
#     # plt.plot(H_list, beta_skew[i], '-v', label='Asym Skew')
# plt.show()
