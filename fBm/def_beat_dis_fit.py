import numpy as np
from scipy import stats, optimize
from typing import Tuple, Dict

def fit_beta_from_density(
    x_data: np.ndarray,
    y_data: np.ndarray,
    method: str = 'mle'
) -> Dict[str, float]:
    """
    从概率密度数据拟合Beta分布
    
    参数:
    ----------
    x_data : np.ndarray
        概率密度对应的x值序列
    y_data : np.ndarray
        概率密度值序列（归一化的概率密度）
    method : str, 可选
        拟合方法: 'mle' (最大似然估计), 'curve_fit' (曲线拟合), 'lsq' (最小二乘)
        默认为 'mle'
    
    返回:
    ----------
    dict: 包含以下键的字典:
        - 'alpha': Beta分布参数α
        - 'beta': Beta分布参数β
        - 'a': 同alpha
        - 'b': 同beta
        - 'loc': 位置参数（最小值）
        - 'scale': 尺度参数（区间宽度）
        - 'x_fit': 拟合Beta分布的x坐标列表
        - 'y_fit': 拟合Beta分布的概率密度值列表
        - 'params': 完整参数元组 (alpha, beta, loc, scale)
        - 'r_squared': 拟合优度R²
        - 'rmse': 均方根误差
    """
    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)
    
    # 移除无效数据
    valid_mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        raise ValueError("所有概率密度值都无效")
    
    x = x[valid_mask]
    y = y[valid_mask]
    
    # 估计数据边界
    # 使用概率密度大于最大值的5%的位置作为边界
    y_threshold = 0.05 * np.max(y)
    valid_idx = np.where(y > y_threshold)[0]
    
    if len(valid_idx) > 0:
        lower_bound = np.min(x[valid_idx])
        upper_bound = np.max(x[valid_idx])
    else:
        lower_bound = np.min(x)
        upper_bound = np.max(x)
    
    # 稍微扩展边界
    data_range = upper_bound - lower_bound
    lower_bound -= 0.1 * data_range
    upper_bound += 0.1 * data_range
    
    # 计算位置和尺度参数
    loc = lower_bound
    scale = upper_bound - lower_bound
    
    # 标准化到[0,1]区间
    x_std = (x - loc) / scale
    y_std = y * scale  # PDF缩放
    
    # 裁剪到有效区间
    mask = (x_std >= 0) & (x_std <= 1)
    x_std = x_std[mask]
    y_std = y_std[mask]
    
    if len(x_std) < 3:
        # 如果有效点太少，使用所有点
        x_std = np.clip((x - loc) / scale, 0, 1)
    
    def beta_pdf(x_vals, a, b):
        """标准Beta分布PDF"""
        return stats.beta.pdf(x_vals, a, b)
    
    # 初始参数估计（加权矩估计）
    weights = y_std / np.sum(y_std)
    mean_val = np.sum(x_std * weights)
    variance = np.sum((x_std - mean_val)**2 * weights)
    
    if variance > 0 and 0 < mean_val < 1:
        alpha_init = mean_val * (mean_val * (1 - mean_val) / variance - 1)
        beta_init = (1 - mean_val) * (mean_val * (1 - mean_val) / variance - 1)
    else:
        alpha_init, beta_init = 2.0, 5.0
    
    alpha_init = max(alpha_init, 0.1)
    beta_init = max(beta_init, 0.1)
    
    # 根据选择的方法拟合
    if method == 'curve_fit':
        try:
            popt, _ = optimize.curve_fit(
                beta_pdf, x_std, y_std,
                p0=[alpha_init, beta_init],
                bounds=([0.1, 0.1], [100, 100])
            )
            alpha, beta = popt
        except:
            alpha, beta = alpha_init, beta_init
    
    elif method == 'lsq':
        def residuals(params):
            a, b = params
            return beta_pdf(x_std, a, b) - y_std
        
        result = optimize.least_squares(
            residuals, [alpha_init, beta_init],
            bounds=([0.1, 0.1], [100, 100])
        )
        alpha, beta = result.x
    
    else:  # 'mle' 或默认
        def neg_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            pdf_vals = beta_pdf(x_std, a, b)
            pdf_vals = np.clip(pdf_vals, 1e-10, None)
            log_likelihood = np.sum(weights * np.log(pdf_vals))
            return -log_likelihood
        
        result = optimize.minimize(
            neg_log_likelihood,
            [alpha_init, beta_init],
            bounds=[(0.1, 100), (0.1, 100)],
            method='L-BFGS-B'
        )
        alpha, beta = result.x if result.success else (alpha_init, beta_init)
    
    # 确保参数有效
    alpha = max(float(alpha), 0.1)
    beta = max(float(beta), 0.1)
    
    # 生成拟合曲线数据
    n_points = 500
    x_fit_std = np.linspace(0, 1, n_points)
    y_fit_std = beta_pdf(x_fit_std, alpha, beta)
    
    # 转换回原始坐标
    x_fit = loc + x_fit_std * scale
    y_fit = y_fit_std / scale
    
    # 计算拟合优度
    y_pred = beta_pdf(x_std, alpha, beta)
    ss_res = np.sum((y_std - y_pred)**2)
    ss_tot = np.sum((y_std - np.mean(y_std))**2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_std - y_pred)**2))
    
    return {
        'alpha': alpha,
        'beta': beta,
        'a': alpha,
        'b': beta,
        'loc': float(loc),
        'scale': float(scale),
        'x_fit': x_fit.tolist(),
        'y_fit': y_fit.tolist(),
        'params': (alpha, beta, float(loc), float(scale)),
        'r_squared': float(r_squared),
        'rmse': float(rmse)
    }


# 最小化版本（如果只需要最基本的功能）
def fit_beta_simple(x_data, y_data):
    """
    简化版Beta分布拟合函数
    
    返回:
    - alpha: α参数
    - beta: β参数
    - x_fit: 拟合曲线的x坐标
    - y_fit: 拟合曲线的y坐标
    """
    result = fit_beta_from_density(np.array(x_data), np.array(y_data))
    a = result['alpha']
    b = result['beta']
    skew = ( (2*(b-a)*np.sqrt(a+b+1)) / ((a+b+2)*np.sqrt(b*a)) )
    return {
        'alpha': result['alpha'],
        'beta': result['beta'],
        'x_fit': result['x_fit'],
        'y_fit': result['y_fit'],
        'skew' : skew
    }