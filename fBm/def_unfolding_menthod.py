import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
import matplotlib.pyplot as plt

def select_best_poly_degree(eigs, degree_range=range(1, 11), criterion='aic', plot=False):
    """
    自动选择最佳多项式拟合阶数
    参数：
        eigs: ndarray，本征值数组，建议为已排序
        degree_range: 可选多项式阶数的范围，如 range(1,11)
        criterion: 'aic' 或 'rss'，选择准则
        plot: 是否绘图展示残差

    返回：
        best_degree: 最佳拟合阶数
        best_func: 对应的拟合函数（lambda）
        unfolded_best: 最佳展开结果
    """
    eigs = np.sort(eigs)
    steps = np.arange(1, len(eigs) + 1)

    best_score = np.inf
    best_degree = None
    best_func = None
    unfolded_best = None

    scores = []  # 存储每个 degree 对应的 AIC/RSS 值

    for deg in degree_range:
        coef = polyfit(eigs, steps, deg)
        unfolded = polyval(eigs, coef)
        residual = steps - unfolded
        rss = np.sum(residual**2)
        n = len(eigs)
        k = deg + 1  # 多项式参数个数
        
        if criterion == 'aic':
            score = n * np.log(rss / n) + 2 * k
        elif criterion == 'rss':
            score = rss
        else:
            raise ValueError("criterion 只能为 'aic' 或 'rss'")

        scores.append(score)

        if score < best_score:
            best_score = score
            best_degree = deg
            best_func = lambda x, c=coef: polyval(x, c)
            unfolded_best = unfolded

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(list(degree_range), scores, marker='o')
        plt.xlabel("Polynomial Degree")
        plt.ylabel("AIC" if criterion == 'aic' else "RSS")
        plt.title("Optimal Degree Selection")
        plt.grid(True)
        plt.show()

    return best_degree, best_func, unfolded_best

# ========================================================================
# 1. 多项式拟合（Polynomial Fitting）
# ========================================================================
from numpy.polynomial.polynomial import polyfit, polyval

def unflod_poly(eigs, degree,best_find = True):
    """
    使用多项式拟合累计态密度 N(λ)，实现谱展开。

    参数：
        eigs : ndarray
            原始本征值（需为一维数组）。
        degree : int
            拟合的多项式阶数。

    返回：
        unfolded : ndarray
            展开后的本征值（平滑累计态密度函数对应值）。
        steps : ndarray
            原始阶数（1 到 len(eigs)）。
        func : callable
            拟合的累计态密度函数 N_fit(x)。
    """
    eigs = np.sort(eigs)
    #choose of degree
    if best_find:
        best_degree, best_func, unfolded_best = select_best_poly_degree(eigs)
        print("Best polynomial fitting degree is {}".format(best_degree))

        degree = best_degree
    # 构造阶数序列 N(λ)：1, 2, ..., len(eigs)
    steps = np.arange(0, len(eigs)) + 1

    # 多项式拟合：用 eigs 拟合 steps，返回多项式系数（幂从低到高）
    poly_coef = polyfit(eigs, steps, degree)

    # 对 eigs 套用拟合的多项式，得到展开后的本征值
    unfolded = polyval(eigs, poly_coef)

    # 返回一个可调用函数 func(x) 作为 N_fit(x)
    func = lambda x: polyval(x, poly_coef)

    return unfolded, steps, func


# ========================================================================
# 2. 样条插值（Spline Interpolation）
# ========================================================================
from scipy.interpolate import UnivariateSpline

def unflod_spline(eigs, s=0.5):
    """
    使用一元样条插值方法拟合累计态密度，实现谱展开。

    参数：
        eigs : ndarray
            原始本征值。
        s : float
            平滑因子，值越小越接近插值，越大越平滑。

    返回：
        unfolded : ndarray
            展开后的本征值。
        steps : ndarray
            原始阶数。
        spline : callable
            样条插值函数对象，可用于后续计算。
    """
    eigs = np.sort(eigs)  # 保证本征值有序
    steps = np.arange(1, len(eigs) + 1)  # N(λ)：1, 2, ..., n

    spline = UnivariateSpline(eigs, steps, s=s)  # 构造样条插值函数

    unfolded = spline(eigs)  # 对原始本征值进行插值，得到展开谱

    return unfolded, steps, spline


# ========================================================================
# 3. 高斯卷积平滑（Gaussian Kernel Smoothing）
# ========================================================================
from scipy.ndimage import gaussian_filter1d

def unflod_gaussian(eigs, sigma=2):
    """
    使用高斯滤波器对 N(λ) 进行平滑处理。

    参数：
        eigs : ndarray
            原始本征值。
        sigma : float
            高斯核的标准差，决定平滑程度。

    返回：
        smoothed : ndarray
            平滑后的累计态密度。
        steps : ndarray
            原始阶数。
    """
    eigs = np.sort(eigs)
    steps = np.arange(1, len(eigs) + 1)

    smoothed = gaussian_filter1d(steps, sigma=sigma)  # 卷积平滑

    return smoothed, steps


# ========================================================================
# 4. 局部加权回归（LOWESS / LOESS）
# ========================================================================
from statsmodels.nonparametric.smoothers_lowess import lowess

def unflod_lowess(eigs, frac=0.1):
    """
    使用局部加权回归（LOWESS）对 N(λ) 进行平滑。

    参数：
        eigs : ndarray
            原始本征值。
        frac : float
            每个拟合点使用的数据比例（范围 0~1）。

    返回：
        smoothed : ndarray
            平滑后的累计态密度。
        steps : ndarray
            原始阶数。
    """
    eigs = np.sort(eigs)
    steps = np.arange(1, len(eigs) + 1)

    # lowess 对 (x=eigs, y=steps) 进行局部回归拟合
    smoothed = lowess(steps, eigs, frac=frac, return_sorted=False)

    return smoothed, steps


# ========================================================================
# 5. 指数平滑（Exponential Smoothing）
# ========================================================================
def unflod_exp_smooth(eigs, alpha=0.2):
    """
    使用简单指数平滑方法对 N(λ) 进行平滑。

    参数：
        eigs : ndarray
            原始本征值。
        alpha : float
            平滑系数，范围 0~1，越大越强调最新值。

    返回：
        smoothed : ndarray
            平滑后的累计态密度。
        steps : ndarray
            原始阶数。
    """
    eigs = np.sort(eigs)
    steps = np.arange(1, len(eigs) + 1)

    # 初始化 smoothed 数组
    smoothed = np.zeros_like(steps, dtype=float)
    smoothed[0] = steps[0]  # 初始值直接等于第一个 step

    # 迭代进行指数平滑
    for i in range(1, len(steps)):
        smoothed[i] = alpha * steps[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed, steps
from PyEMD import EMD

def emd_detrend(unfolded):
    """'Detrend' the unfolded eigenvalues via Empirical Mode Decomposition.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.

    Returns
    -------
    detrended: ndarray
        The 'detrended' values (i.e. with last IMF residue removed).

    References
    ----------
    Morales, I. O., Landa, E., Stránský, P., & Frank, A. (2011). Improved
    unfolding by detrending of statistical fluctuations in quantum spectra.
    Physical Review E, 84(1). doi:10.1103/physreve.84.016203
    """
    spacings = np.diff(unfolded)
    s_av = np.average(spacings)
    s_i = spacings - s_av

    ns = np.zeros([len(unfolded)], dtype=int)
    delta_n = np.zeros([len(unfolded)])
    for n in range(len(unfolded)):
        delta_n[n] = np.sum(s_i[0:n])
        ns[n] = n

    # last member of IMF basis is the trend
    trend = EMD().emd(delta_n)[-1]
    detrended_delta = delta_n - trend

    # see Morales (2011) DOI: 10.1103/PhysRevE.84.016203, Eq. 15
    unfolded_detrend = np.empty([len(unfolded)])
    for i in range(len(unfolded_detrend)):
        if i == 0:
            unfolded_detrend[i] = unfolded[i]
            continue
        unfolded_detrend[i] = detrended_delta[i - 1] + unfolded[0] + (i - 1) * s_av

    return unfolded_detrend
DEFAULT_SPLINE_DEGREE = 3
DEFAULT_POLY_DEGREE = 7
DEFAULT_SPLINE_SMOOTH = 1.4
from scipy.optimize import curve_fit

def fit(eigs, method='Poly', degree=4,
    spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
    detrend: bool = False,
    return_callable: bool = False
):
    """Alias for Li_fit function"""
    return Li_fit(eigs, method, degree, spline_smooth, detrend, return_callable)

def Li_fit(eigs,method='Poly',degree = 4,
    spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
    detrend: bool = False,
    return_callable: bool = False
):
    """Computer the specified smoothing function values for a set of eigenvalues.

    Parameters
    ----------
    eigs: ndarray
        The sorted eigenvalues

    smoother: "poly" | "spline" | "gompertz" | lambda
        The type of smoothing function used to fit the step function

    degree: int
        The degree of the polynomial or spline

    spline_smooth: float
        The smoothing factors passed into scipy.interpolate.UnivariateSpline

    detrend: bool
        Whether or not to perform EMD detrending before returning the
        unfolded eigenvalues.

    return_callable: bool
        If true, return a function that closes over the fit parameters so
        that, e.g., additional values can be fit later.


    Returns
    -------
    unfolded: ndarray
        the unfolded eigenvalues

    steps: ndarray
        the step
    """
    steps = np.arange(0, len(eigs)) + 1
    if method == 'Poly' :
        poly_coef = polyfit(eigs, steps, degree)
        unfolded = polyval(eigs, poly_coef)
        func = lambda x: polyval(x, poly_coef) if return_callable else None
        if detrend:
            unfolded = emd_detrend(unfolded)
        return unfolded, steps, func  # type: ignore


    if method == "Exponential":
        func = lambda t, a, b: a * np.exp(-b * t)  # type: ignore
        [a, b], cov = curve_fit(
            func,
            eigs,
            steps,
            p0=(len(eigs) / 2, 0.5),
        )
        unfolded = func(eigs, a, b)  # type: ignore
        if detrend:
            unfolded = emd_detrend(unfolded)
        return unfolded, steps, func  # type: ignore

# raise RuntimeError("Unreachable!")
def get_unfolded_from_eig_matrix_fit(matrix_val,n=0,menthod = 'Ploy',degree = 7,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    # unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    # spacings = []
    matrix_val_unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded,_,_ = unflod_poly(eig_val,degree=degree)
        # if percent:
        #     eig_unfloded = eig_unfloded[int(L*lower): int(L*upper)]
        L = len(eig_unfloded)
        eig_unfloded = np.sort(eig_unfloded)
        matrix_val_unfloded[:,i] = eig_unfloded

        

        # unfloded.extend(eig_unfloded)

    

    return matrix_val_unfloded