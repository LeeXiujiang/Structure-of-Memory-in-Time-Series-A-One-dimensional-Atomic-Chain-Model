import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
def weibull_pdf(x, k, lam):
    """Weibull分布概率密度函数"""
    return (k/lam) * (x/lam)**(k-1) * np.exp(-(x/lam)**k)

def fit_weibull_distribution(spacings):
    """拟合Weibull分布并返回参数"""
    try:
        # 归一化间距
        normalized_spacings = spacings / np.mean(spacings)
        
        # 使用scipy的weibull_min分布进行拟合
        params = stats.weibull_min.fit(normalized_spacings, floc=0)
        k, loc, scale = params
        
        # 计算拟合优度
        hist, bin_edges = np.histogram(normalized_spacings, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算拟合的PDF
        fitted_pdf = stats.weibull_min.pdf(bin_centers, k, loc, scale)
        
        # 计算R²
        ss_res = np.sum((hist - fitted_pdf) ** 2)
        ss_tot = np.sum((hist - np.mean(hist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return k, scale, r_squared, normalized_spacings
    
    except Exception as e:
        print(f"拟合Weibull分布时出错: {e}")
        return None, None, None, None
