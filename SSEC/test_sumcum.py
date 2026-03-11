import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_fitted_matrix_with_noise(hurst_year_matrix, noise_level=0.01):
    """
    根据线性拟合公式计算矩阵中每个元素的结果，并添加高斯噪声
    
    Parameters:
    hurst_year_matrix: numpy array, 包含 Hurst 指数的矩阵
    noise_level: float, 噪声水平（标准差）
    
    Returns:
    result_matrix: numpy array, 根据 y = slope*x + intercept 计算的结果矩阵
    result_matrix_with_noise: numpy array, 加入噪声后的结果矩阵
    noise_matrix: numpy array, 添加的噪声矩阵
    """
    # 线性拟合的参考数据点
    X_fit = [0.55, 0.60, 0.65, 0.70]  # h值
    Y_fit = [0.46, 0.425, 0.39, 0.35]  # R值
    
    # 进行线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_fit, Y_fit)
    
    print(f"使用的拟合公式: y = {slope:.4f} * x + {intercept:.4f}")
    print(f"拟合质量 R² = {r_value**2:.4f}")
    
    # 根据公式计算矩阵中每个元素的结果
    result_matrix = slope * hurst_year_matrix + intercept
    
    # 生成高斯噪声（均值为0，标准差为noise_level）
    noise_matrix = np.random.normal(0, noise_level, hurst_year_matrix.shape)
    
    # 加入噪声后的结果
    result_matrix_with_noise = result_matrix + noise_matrix
    
    return result_matrix, result_matrix_with_noise, noise_matrix, slope, intercept

def add_adaptive_noise(hurst_year_matrix, result_matrix, relative_noise_level=0.05):
    """
    添加与数值大小成比例的适应性噪声
    
    Parameters:
    hurst_year_matrix: numpy array, 原始 Hurst 矩阵
    result_matrix: numpy array, 拟合结果矩阵
    relative_noise_level: float, 相对噪声水平
    
    Returns:
    result_matrix_with_noise: numpy array, 加入噪声后的结果矩阵
    """
    # 计算每个位置的噪声标准差（与拟合值成比例）
    noise_std = np.abs(result_matrix) * relative_noise_level
    
    # 生成适应性噪声
    adaptive_noise = np.random.normal(0, noise_std)
    
    # 加入噪声
    result_matrix_with_noise = result_matrix + adaptive_noise
    
    return result_matrix_with_noise, adaptive_noise

# 完整的示例
def comprehensive_analysis_with_noise(hurst_year_matrix, noise_level=0.01):
    """
    综合分析：计算拟合结果并添加不影响整体的噪声
    
    Parameters:
    hurst_year_matrix: numpy array, Hurst 指数矩阵
    noise_level: float, 噪声水平
    """
    # 线性拟合参数
    X_fit = [0.55, 0.60, 0.65, 0.70]
    Y_fit = [0.46, 0.425, 0.39, 0.35]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_fit, Y_fit)
    
    print("=" * 60)
    print("带噪声的线性拟合分析")
    print("=" * 60)
    print(f"拟合方程: y = {slope:.6f}x + {intercept:.6f}")
    print(f"决定系数 R²: {r_value**2:.6f}")
    print(f"噪声水平: {noise_level}")
    
    # 计算拟合值
    fitted_matrix = slope * hurst_year_matrix + intercept
    
    # 添加噪声
    fitted_matrix_with_noise, noise_matrix, _, _ = calculate_fitted_matrix_with_noise(
        hurst_year_matrix, noise_level
    )
    
    # 统计比较
    original_mean = np.mean(fitted_matrix)
    noisy_mean = np.mean(fitted_matrix_with_noise)
    mean_difference = np.abs(original_mean - noisy_mean)
    
    original_std = np.std(fitted_matrix)
    noisy_std = np.std(fitted_matrix_with_noise)
    std_difference = np.abs(original_std - noisy_std)
    
    print(f"\n统计比较:")
    print(f"原始均值: {original_mean:.6f}")
    print(f"加噪均值: {noisy_mean:.6f}")
    print(f"均值差异: {mean_difference:.6f} ({mean_difference/original_mean*100:.2f}%)")
    print(f"原始标准差: {original_std:.6f}")
    print(f"加噪标准差: {noisy_std:.6f}")
    print(f"标准差差异: {std_difference:.6f} ({std_difference/original_std*100:.2f}%)")
    
    return fitted_matrix, fitted_matrix_with_noise, noise_matrix

# 可视化函数
def plot_comparison(original_matrix, noisy_matrix, noise_matrix, years=None):
    """
    可视化原始结果和加噪结果的比较
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始拟合结果
    im1 = axes[0, 0].imshow(original_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('原始拟合结果')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 加噪后的结果
    im2 = axes[0, 1].imshow(noisy_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('加噪后的结果')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 噪声分布
    axes[1, 0].hist(noise_matrix.flatten(), bins=30, alpha=0.7, color='red')
    axes[1, 0].axvline(0, color='black', linestyle='--')
    axes[1, 0].set_xlabel('噪声值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('噪声分布')
    
    # 差异比较
    difference = noisy_matrix - original_matrix
    im3 = axes[1, 1].imshow(difference, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('差异 (加噪 - 原始)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建示例 Hurst 矩阵
    np.random.seed(42)  # 为了可重复性
    hurst_year_matrix = np.random.uniform(0.55, 0.70, (6, 8))
    
    print("原始 Hurst 矩阵:")
    print(hurst_year_matrix)
    
    # 进行带噪声的分析
    original_results, noisy_results, noise = comprehensive_analysis_with_noise(
        hurst_year_matrix, noise_level=0.005
    )
    
    print("\n原始拟合结果 (前5个元素):")
    print(original_results.flatten()[:5])
    
    print("\n加噪后结果 (前5个元素):")
    print(noisy_results.flatten()[:5])
    
    print("\n添加的噪声 (前5个元素):")
    print(noise.flatten()[:5])
    
    # 可视化比较
    plot_comparison(original_results, noisy_results, noise)

# 批量测试不同噪声水平
def test_noise_levels(hurst_year_matrix, noise_levels=[0.001, 0.005, 0.01, 0.02]):
    """
    测试不同噪声水平的影响
    """
    results = {}
    
    for noise_level in noise_levels:
        print(f"\n测试噪声水平: {noise_level}")
        original, noisy, noise = comprehensive_analysis_with_noise(
            hurst_year_matrix, noise_level
        )
        
        # 计算相关系数
        correlation = np.corrcoef(original.flatten(), noisy.flatten())[0, 1]
        results[noise_level] = {
            'correlation': correlation,
            'mean_difference': np.abs(np.mean(original) - np.mean(noisy)),
            'std_difference': np.abs(np.std(original) - np.std(noisy))
        }
        
        print(f"与原始结果的相关系数: {correlation:.6f}")
    
    return results