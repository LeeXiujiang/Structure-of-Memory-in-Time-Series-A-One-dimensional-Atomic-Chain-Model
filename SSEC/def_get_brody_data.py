import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, eigvalsh
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import pandas as pd
import statistics
from def_unfolding_menthod import unflod_poly
from def_weibull_use import *
from def_H_beat_fit import beta_Hurst_SSEC,beta_Hurst_HSI
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def brody_distribution(s, nu):
    """
    Brody分布的概率密度函数
    """
    alpha = gamma((nu + 2) / (nu + 1)) ** (nu + 1)
    pdf = alpha * (nu + 1) * (s ** nu) * np.exp(-alpha * (s ** (nu + 1)))
    return pdf
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