import numpy as np
from scipy import stats, optimize
from scipy.special import beta as beta_func
from typing import Tuple, Dict, Optional

def fit_generalized_beta_from_density(
    x_data: np.ndarray,
    y_data: np.ndarray,
    method: str = 'mle',
    bounds_estimate: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    从概率密度数据拟合广义Beta分布（四参数Beta分布）
    
    广义Beta分布公式：f(x|a,b,α,β) = (x-a)^(α-1)(b-x)^(β-1) / [(b-a)^(α+β-1) * B(α,β)]
    
    参数:
    ----------
    x_data : np.ndarray
        概率密度对应的x值序列
    y_data : np.ndarray
        概率密度值序列（归一化的概率密度）
    method : str, 可选
        拟合方法: 'mle' (最大似然估计), 'curve_fit' (曲线拟合), 'lsq' (最小二乘)
        默认为 'mle'
    bounds_estimate : tuple, 可选
        边界估计 (lower_bound, upper_bound)，如果为None则自动估计
    
    返回:
    ----------
    dict: 包含以下键的字典:
        - 'alpha': Beta分布形状参数α
        - 'beta': Beta分布形状参数β
        - 'lower_bound': 分布下界a
        - 'upper_bound': 分布上界b
        - 'a': 同lower_bound
        - 'b': 同upper_bound
        - 'x_fit': 拟合Beta分布的x坐标列表
        - 'y_fit': 拟合Beta分布的概率密度值列表
        - 'params': 完整参数元组 (lower_bound, upper_bound, alpha, beta)
        - 'r_squared': 拟合优度R²
        - 'rmse': 均方根误差
    """
    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)
    mask = y > 0
    x = x[mask]
    y = y[mask]
    # 移除无效数据
    valid_mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        raise ValueError("所有概率密度值都无效")
    
    x = x[valid_mask]
    y = y[valid_mask]
    
    # 估计边界
    if bounds_estimate is None:
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
        lower_bound -= 0.05 * data_range
        upper_bound += 0.05 * data_range
    else:
        lower_bound, upper_bound = bounds_estimate
    
    # 确保边界有效
    if lower_bound >= upper_bound:
        raise ValueError(f"无效的边界: lower_bound={lower_bound}, upper_bound={upper_bound}")
    
    # 定义广义Beta分布PDF
    def generalized_beta_pdf(x_vals, a, b, alpha, beta):
        """广义Beta分布概率密度函数"""
        # 确保在区间内
        mask = (x_vals >= a) & (x_vals <= b)
        result = np.zeros_like(x_vals)
        
        if np.any(mask):
            # 计算标准化变量
            z = (x_vals[mask] - a) / (b - a)
            # 广义Beta分布公式
            result[mask] = (z**(alpha-1) * (1-z)**(beta-1) / 
                           ((b - a) * beta_func(alpha, beta)))
        return result
    
    # 初始参数估计
    # 首先估计形状参数（假设边界已知）
    x_std = (x - lower_bound) / (upper_bound - lower_bound)
    x_std = np.clip(x_std, 1e-10, 1-1e-10)  # 避免边界问题
    
    # 加权矩估计形状参数
    weights = y / np.sum(y)
    mean_val = np.sum(x_std * weights)
    variance = np.sum((x_std - mean_val)**2 * weights)
    
    if variance > 0 and 0 < mean_val < 1:
        alpha_init = mean_val * (mean_val * (1 - mean_val) / variance - 1)
        beta_init = (1 - mean_val) * (mean_val * (1 - mean_val) / variance - 1)
    else:
        alpha_init, beta_init = 2.0, 5.0
    
    alpha_init = max(alpha_init, 0.1)
    beta_init = max(beta_init, 0.1)
    
    # 初始参数向量
    init_params = [lower_bound, upper_bound, alpha_init, beta_init]
    
    # 根据选择的方法拟合
    if method == 'curve_fit':
        try:
            # 设置边界约束
            bounds = ([np.min(x)-10, np.max(x)+0.1, 0.1, 0.1],
                     [np.min(x)-0.1, np.max(x)+10, 100, 100])
            
            popt, _ = optimize.curve_fit(
                generalized_beta_pdf, x, y,
                p0=init_params,
                bounds=bounds,
                maxfev=5000
            )
            a_fit, b_fit, alpha_fit, beta_fit = popt
        except Exception as e:
            print(f"curve_fit失败: {e}, 使用初始估计")
            a_fit, b_fit, alpha_fit, beta_fit = init_params
    
    elif method == 'lsq':
        def residuals(params):
            a, b, alpha, beta = params
            if a >= b or alpha <= 0 or beta <= 0:
                return np.inf * np.ones_like(y)
            return generalized_beta_pdf(x, a, b, alpha, beta) - y
        
        bounds = ([np.min(x)-10, np.max(x)+0.1, 0.1, 0.1],
                 [np.min(x)-0.1, np.max(x)+10, 100, 100])
        
        result = optimize.least_squares(
            residuals, init_params,
            bounds=bounds,
            max_nfev=5000
        )
        a_fit, b_fit, alpha_fit, beta_fit = result.x
    
    else:  # 'mle' 或默认
        def neg_log_likelihood(params):
            a, b, alpha, beta = params
            
            # 参数约束
            if a >= b or alpha <= 0 or beta <= 0:
                return 1e10
            
            # 计算PDF
            pdf_vals = generalized_beta_pdf(x, a, b, alpha, beta)
            pdf_vals = np.clip(pdf_vals, 1e-10, None)
            
            # 使用y作为权重
            weights_norm = y / np.sum(y)
            log_likelihood = np.sum(weights_norm * np.log(pdf_vals))
            
            return -log_likelihood
        
        # 设置优化边界
        bounds = [(np.min(x)-10, np.min(x)-0.1),  # a的范围
                 (np.max(x)+0.1, np.max(x)+10),   # b的范围
                 (0.1, 100),                      # alpha的范围
                 (0.1, 100)]                      # beta的范围
        
        result = optimize.minimize(
            neg_log_likelihood,
            init_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if result.success:
            a_fit, b_fit, alpha_fit, beta_fit = result.x
        else:
            print("MLE优化失败，使用初始估计")
            a_fit, b_fit, alpha_fit, beta_fit = init_params
    
    # 确保参数有效
    a_fit = float(a_fit)
    b_fit = float(b_fit)
    alpha_fit = max(float(alpha_fit), 0.1)
    beta_fit = max(float(beta_fit), 0.1)
    
    # 确保边界顺序正确
    if a_fit > b_fit:
        a_fit, b_fit = b_fit, a_fit
    
    # 生成拟合曲线数据
    n_points = 500
    x_fit = np.linspace(a_fit + 1e-6, b_fit - 1e-6, n_points)
    y_fit = generalized_beta_pdf(x_fit, a_fit, b_fit, alpha_fit, beta_fit)
    
    # 计算拟合优度
    y_pred = generalized_beta_pdf(x, a_fit, b_fit, alpha_fit, beta_fit)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    return {
        'alpha': alpha_fit,
        'beta': beta_fit,
        'lower_bound': a_fit,
        'upper_bound': b_fit,
        'a': a_fit,
        'b': b_fit,
        'x_fit': x_fit.tolist(),
        'y_fit': y_fit.tolist(),
        'params': (a_fit, b_fit, alpha_fit, beta_fit),
        'r_squared': float(r_squared),
        'rmse': float(rmse)
    }


def fit_scipy_beta_from_density(
    x_data: np.ndarray,
    y_data: np.ndarray
) -> Dict[str, float]:
    """
    使用scipy的beta.fit方法拟合Beta分布
    
    这是标准四参数Beta分布的简化接口
    返回参数: (alpha, beta, loc, scale)
    """
    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)
    
    # 移除无效数据
    valid_mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    # 从概率密度数据生成伪样本
    # 基于概率密度值生成加权样本
    n_samples = 1000
    weights = y / np.sum(y)
    
    # 使用逆变换采样生成样本
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    cdf = np.cumsum(weights[sorted_indices])
    cdf = cdf / cdf[-1]  # 归一化
    
    # 生成均匀随机数
    u = np.random.rand(n_samples)
    # 线性插值生成样本
    samples = np.interp(u, cdf, x_sorted)
    
    # 使用scipy的beta.fit拟合
    try:
        alpha, beta, loc, scale = stats.beta.fit(samples, floc=np.min(samples)-1e-10)
        
        # 生成拟合曲线
        n_points = 500
        x_fit = np.linspace(loc + 1e-6, loc + scale - 1e-6, n_points)
        y_fit = stats.beta.pdf((x_fit - loc) / scale, alpha, beta) / scale
        
        # 计算拟合优度
        y_pred = stats.beta.pdf((x - loc) / scale, alpha, beta) / scale
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'loc': float(loc),
            'scale': float(scale),
            'lower_bound': float(loc),
            'upper_bound': float(loc + scale),
            'x_fit': x_fit.tolist(),
            'y_fit': y_fit.tolist(),
            'params': (float(alpha), float(beta), float(loc), float(scale)),
            'r_squared': float(r_squared),
            'rmse': float(rmse)
        }
    except Exception as e:
        raise ValueError(f"scipy beta.fit失败: {e}")


# 简化版广义Beta分布拟合
def fit_generalized_beta_simple(x_data, y_data):
    """
    简化版广义Beta分布拟合函数
    
    返回:
    - alpha: α参数
    - beta: β参数
    - lower_bound: 下界
    - upper_bound: 上界
    - x_fit: 拟合曲线的x坐标
    - y_fit: 拟合曲线的y坐标
    """
    result = fit_generalized_beta_from_density(
        np.array(x_data), 
        np.array(y_data),
        method='mle'
    )
    a = result['alpha']
    b = result['beta']
    skew = ( (2*(b-a)*np.sqrt(a+b+1)) / ((a+b+2)*np.sqrt(b*a)) )
    return {
        'alpha': result['alpha'],
        'beta': result['beta'],
        'lower_bound': result['lower_bound'],
        'upper_bound': result['upper_bound'],
        'x_fit': result['x_fit'],
        'y_fit': result['y_fit'],
        'skew' : skew
    }