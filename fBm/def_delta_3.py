import  numpy as np
from numpy import float64 as f64
class prange(object):
    """ Provides a 1D parallel iterator that generates a sequence of integers.
    In non-parallel contexts, prange is identical to range.
    """
    def __new__(cls, *args):
        return range(*args)
RIGIDITY_GRID = 100 
class ConvergenceError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args) 
#
def _step_function_fast(eigs,x):
    """optimized version that does not repeatedly call np.sum(eigs <= x), since
    this function needed to be called extensively in rigidity calculation."""
    ret = np.zeros((len(x)), dtype=np.int64)
    if x[-1] <= eigs[0]:  # early return if all values are just zero
        return ret
    if x[0] > eigs[-1]:  # early return if all x values above eigs
        n = len(eigs)
        for i in range(len(ret)):
            ret[i] = n
        return ret

    # find the first index of x where we hit values of x that are actually
    # within the range of eigs[0], eigs[-1]
    j = 0  # index into x
    while j < len(x) and x[j] < eigs[0]:
        j += 1
    # now j is the index of the first x value with a nonzero step function value

    i = 0  # index into eigs
    count = 0
    while j < len(x) and i < len(eigs):
        if x[j] >= eigs[i]:  # x could start in middle of eigs
            i += 1
            count += 1
            continue
        while j < len(x) and x[j] < eigs[i]:  # look ahead
            ret[j] = count
            j += 1

    while j < len(x):  # keep going for any remaining values of x
        ret[j] = count
        j += 1

    return ret
def _slope(x, y):
    """Perform linear regression to compute the slope."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev)
    var = np.sum(x_dev * x_dev)
    if var == 0.0:
        return 0.0  # type: ignore
    return cov / var  # type: ignore
def _intercept(x, y, slope):
    return np.mean(y) - slope * np.mean(x)  # type: ignore
def _sq_lin_deviation(eigs, steps, K, w, grid):
    """计算最佳拟合直线的阶梯函数在 grid 区域内平方偏差。
    Compute the sqaured deviation of the staircase function of the best fitting
    line, over the region in `grid`.
    参数
    ----------
    eigs: ndarray
        已排序的原始特征值

    steps: ndarray
        在 `grid` 上计算的阶梯函数值。

    K: float
        最佳拟合直线的斜率。

    w: float
        最佳拟合直线的截距。

    grid: ndarray
        阶梯函数计算的取值网格。

    返回
    -------
    sq_deviations: ndarray
        平方偏差值。
    """
    ret = np.empty(len(grid))
    for i in range(len(grid)):
        n = steps[i]
        deviation = n - K * grid[i] - w
        ret[i] = deviation * deviation
    return ret
def _int_simps_nonunif(grid, vals):
    """
    Simpson rule for irregularly spaced data. Copied shamelessly from
    https://en.wikipedia.org/w/index.php?title=Simpson%27s_rule&oldid=938527913#Composite_Simpson's_rule_for_irregularly_spaced_data
    for compilation here with Numba, and to overcome the extremely slow performance
    problems with scipy.integrate.simps.
    参数
    ----------
    grid: 浮点数列表或np.array
            函数值的采样点
    vals: 浮点数列表或np.array
            采样点处的函数值
    返回
    -------
    float: 积分的近似值
    """
    N = len(grid) - 1
    h = np.diff(grid)

    result = f64(0.0)
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += vals[i] * (h[i]**3 + h[i-1]**3 + 3.0*h[i]*h[i-1]*hph) / (6*h[i]*h[i-1])
        result += vals[i-1] * (2.0*h[i-1]**3 - h[i]**3 + 3.0*h[i]*h[i-1]**2) / (6*h[i-1] * hph)
        result += vals[i+1] * (2.0*h[i]**3 - h[i-1]**3 + 3.0*h[i-1]*h[i]**2) / (6*h[i] * hph)

    if (N + 1) % 2 == 0:
        result += vals[N] * (2*h[N-1]**2 + 3.0*h[N-2]*h[N-1]) / (6*(h[N-2] + h[N-1]))
        result += vals[N-1] * (h[N-1]**2 + 3*h[N-1]*h[N-2]) / (6*h[N-2])
        result -= vals[N-2] * h[N-1]**3 / (6*h[N-2]*(h[N-2] + h[N-1]))
    return result
def _integrate_fast(grid, values) :
    """scipy.integrate.trapz 的运行速度极其缓慢，无法满足我们的需求。
        这个微小的重写似乎能带来近 20 倍的加速。不过，由于是梯形积分法，其精度较差。
    """
    integral = 0
    for i in range(len(grid) - 1):
        w = grid[i + 1] - grid[i]
        h = values[i] + values[i + 1]
        integral += w * h / 2
    return integral  # type: ignore
def kahan_add(current_sum, update, carry_over):
    """
    Returns
    -------
    updated_sum: float
        Updated sum.

    carry_over: float
        Carried-over value (often named "c") in pseudo-code.
    """
    remainder = update - carry_over
    lossy = current_sum + remainder
    c = (lossy - current_sum) - remainder
    updated_sum = lossy
    return updated_sum, c
def delta_L(unfolded,L,gridsize= 100,max_iters= int(1e6),min_iters = 100, tol = 0.01,use_simpson= True,show_progress= False):
    buf = 1000
    # prog_interval = CONVERG_PROG_INTERVAL
    # if L > 100:
    #     prog_interval *= 10

    delta_running = np.zeros((buf,))
    k = np.uint64(0)
    c = np.float64(0.0)  # compensation (carry-over) term for Kahan summation
    d3_mean = np.float64(0.0)
    # if show_progress:
    #     print(CONVERG_PROG, 0, ITER_COUNT)
    while True:
        if k != 0:  # awkward, want uint64 k
            k += 1
        start = np.random.uniform(unfolded[0], unfolded[-1])
        grid = np.linspace(start - L / 2, start + L / 2, gridsize)
        steps = _step_function_fast(unfolded, grid)  # performance bottleneck
        K = _slope(grid, steps)
        w = _intercept(grid, steps, K)
        y_vals = _sq_lin_deviation(unfolded, steps, K, w, grid)
        if use_simpson:
            delta3_c = _int_simps_nonunif(grid, y_vals)  # O(len(grid))
        else:
            delta3_c = _integrate_fast(grid, y_vals)  # O(len(grid))
        d3 = delta3_c / L
        if k == 0:  # initial value
            d3_mean = d3
            delta_running[0] = d3_mean
            k += 1
            continue
        else:
            # Regular sum
            # d3_mean = (k * d3_mean + d3) / (k + 1)
            # d3_mean += (d3 - d3_mean) / k

            # Kahan sum - but can we be sure Numba isn't optimizing away?
            update = (d3 - d3_mean) / k  # mean + update is new mean
            d3_mean, c = kahan_add(current_sum=d3_mean, update=update, carry_over=c)
            # remainder = update - c
            # lossy = d3_mean + remainder
            # c = (lossy - d3_mean) - remainder
            # d3_mean = lossy
            delta_running[int(k) % buf] = d3_mean

        # if show_progress and k % prog_interval == 0:
        #     print(CONVERG_PROG, int(k * 2), ITER_COUNT)  # x2 for safety factor
        if k >= max_iters:
            break
        if (
            (k > min_iters)
            and (k % buf == 0)  # all buffer values must have changed
            and (np.abs(np.max(delta_running) - np.min(delta_running)) < tol)
        ):
            break

    # if show_progress:
    #     print(CONVERG_PROG, int(k), ITER_COUNT)
    converged = np.abs(np.max(delta_running) - np.min(delta_running)) < tol
    # I don't think it matters much if we use median, mean, max, or min for the
    # final returned single value, given our convergence criterion is so strict
    # return d3_mean, converged, k
    return np.mean(delta_running), converged, k  # type: ignore

RIGIDITY_PROG = "\033[2K Spectral-rigidity progress:"
PERCENT = "%"
def delta_parallel(unfolded,L_vals,tol=0.01,max_iters=int(1e6),gridsize=1000,min_iters=100,use_simpson=True,show_progress = False):
    prog_interval = len(L_vals) // 50
    if prog_interval == 0:
        prog_interval = 1
    delta3 = np.zeros(L_vals.shape)
    iters = np.zeros(L_vals.shape, dtype=np.int64)
    converged = np.zeros_like(L_vals, dtype=np.bool_)
    for i in prange(len(L_vals)):
        L = L_vals[i]
        delta3[i], converged[i], iters[i] = delta_L(
            unfolded=unfolded,
            L=L,
            gridsize=gridsize,
            max_iters=max_iters,
            min_iters=min_iters,
            tol=tol,
            use_simpson=use_simpson,
        )

        if show_progress and i % prog_interval == 0:
            prog = int(100 * np.sum(delta3 != 0) / len(delta3))
            print(RIGIDITY_PROG, prog, PERCENT)
    return delta3, converged, iters

def spectral_rigidity(unfolded,L= np.arange(2, 20, 0.5),tol= f64(0.01),max_iters = 0,gridsize=100,integration= "simps",show_progress = True):
    """计算特定展开方式下的谱刚性（spectral rigidity）。
    该方法通过随机采样计算给定特征值及其展开后的谱刚性(delta_3, ∆₃)[1]。采用辛普森法计算阶梯函数与线性拟合偏差的内部积分，并对每个L值迭代随机采样区间中心点c，直至满足收敛条件。

    参数
    ----------
    unfolded: ndarray
        展开后的特征值数组

    L: ndarray 
        需要计算谱刚性的L值数组

    max_iters: int = -1
        对每个L值随机选择区间中心点c的迭代次数上限（区间[c-L/2,c+L/2]）
        
        对于N×N的GOE矩阵（N∈[5000,10000,20000]）：
        - 当使用高次多项式展开或解析展开时
        - 默认tol=0.01和max_iters=int(1e4)时
        算法在L≈70-90范围内可收敛。更大的L值需要将max_iters增至约1e5。
        较小矩阵（如N=2000）在默认参数下L>60时难以收敛。

    tol: float = 0.01
        收敛判据。当最近1000次计算值的范围小于tol时判定收敛

    gridsize: int = 100
        每个内部积分在[c-L/2,c+L/2]区间上的网格点数。
        较小值会增加delta_3采样值的方差，建议保持默认值

    integration: "simps"|"trapz"
        积分方法："simps"(默认，辛普森法)或"trapz"(梯形法，较快但精度略低)

    返回值
    -------
    L : ndarray
        基于输入生成的L值数组

    delta3 : ndarray
        各L值对应的谱刚性计算结果

    注意事项
    ---------
    1. 该算法经过深度优化（Python层面），即使在高网格密度和大迭代次数下仍能快速执行
    2. 计算效率大致为O(L_grid_size)，增加网格密度对性能影响最大
    3. 建议对每个L值采用较大迭代次数（如1e5次），特别是L>50时
    4. 计算结果与理论值的匹配程度取决于：
    - 展开函数的选择
    - 矩阵尺寸
    - 大L值时收敛较慢（示例说明见原文）

    参考文献
    ---------
    [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier
    """
    # delta3 = _spectral_iter_grid(
    #     unfolded=unfolded,
    #     L_vals=L.copy().ravel(),
    #     gridsize=gridsize,
    #     use_simpson=True,
    # )
    if max_iters <= 0:
        success, max_iters = delta_L(
            unfolded=unfolded,
            L=float(np.max(L)),
            gridsize=RIGIDITY_GRID,
            max_iters = int(1e9),
            min_iters=10 * 100,  # also safety
            tol=tol,
            use_simpson=integration != "trapz",
            show_progress=show_progress,
        )[1:]
        max_iters = int(max_iters * 2)  # precaution
        print(f"The max_iters was automatically determined as {max_iters}")

        if not success:
            raise ConvergenceError(
                f"For the largest L-value in your provided Ls, {np.max(L)}, the "
                f"spectral rigidity at L did not converge in {100} "
                "iterations. Either reduce the range of L values, reduce the "
                "`tol` tolerance value, or manually set `max_iters` to be some "
                "value other than the default of 0 to disable this check. Note "
                "the convergence criterion involves the range on the last 1000 "
                "values, which are themselves iteratively-computed means, so is "
                "a somewhat strict convergence criterion. However, setting "
                "`max_iters` too low then provides NO guarantee on the error for "
                "non-converging L values."
            )

    delta3, converged, iters = delta_parallel(
        unfolded=unfolded,
        L_vals=L.copy().ravel(),
        gridsize=gridsize,
        max_iters=max_iters,
        min_iters=100,
        tol=tol,
        use_simpson=integration != "trapz",
        show_progress=show_progress,
    )
    return L, delta3, converged, iters
def spectral_rigidity_possion(Ls):
    delta_3_possion = Ls / 15
    return Ls,delta_3_possion

p = np.pi
y = np.euler_gamma
def spectral_rigidity_goe(unfolded,L):
    s = L / np.mean(unfolded[1:] - unfolded[:-1]) if unfolded is not None else L
    goe = (1 / (p**2)) * (np.log(2 * p * s) + y - 5 / 4 - (p**2) / 8)
    # delta_3_possion = Ls / 15
    return L, goe

def spectral_rigidity_gue(unfolded,L):
    s = L / np.mean(unfolded[1:] - unfolded[:-1]) if unfolded is not None else L
    gue = (1 / (2 * (p**2))) * (np.log(2 * p * s) + y - 5 / 4)
    # delta_3_possion = Ls / 15
    return L, gue

def spectral_rigidity_goe(unfolded,L):
    s = L / np.mean(unfolded[1:] - unfolded[:-1]) if unfolded is not None else L
    gse = (1 / (4 * (p**2))) * (np.log(4 * p * s) + y - 5 / 4 + (p**2) / 8)
    # delta_3_possion = Ls / 15
    return L, gse

# Removed circular import
from def_unfolding_menthod import *
def unfolding_method_choose(eig_vals,menthod= 'Poly',degree = 5):
    #特征值展开 方法选择
    if menthod == 'Poly':
        eig_unfloded,stps,func = unflod_poly(eig_vals,degree=5)

    if menthod == 'Spline':
        eig_unfloded,stps,func = unflod_spline(eig_vals)
   
    if menthod == 'Gaussian':
        eig_unfloded,stps = unflod_gaussian(eig_vals)
    
    if menthod == 'Exp':
        eig_unfloded,stps = unflod_exp_smooth(eig_vals)
    return eig_unfloded
    
def get_unfloded_from_eig_list(list_val,menthod= 'Poly',degree = 5,percent = True,lower = 0.2,upper = 0.8,average_spacing = True,print_curvce = False): 
    """

    # 返回值
    list_vals: 一维数组，包含所有特征值
    x: 一维数组，包含bin_mindle vals
    y: 一维数组，包含特征值 x 对应的 的密度
    """
    list_val = np.sort(list_val)

    # sum_c = np.sum(list_val)
    # print (f"sum of eig_vals = {sum_c:.0f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    #缩放
    L = len(list_val)
    if L == 0:
        print("list is empty")
    
    
    if percent:
        list_val = list_val[int(L*lower): int(L*upper)]
    #特征值 展开
    eig_unfloded = unfolding_method_choose(list_val,menthod=menthod,degree=degree)
    eig_unfloded = np.sort(eig_unfloded)

    #缩放
    # if scale:
    #     list_val = list_val / np.sqrt(L)
    # if bin_size <= 0:
    #     raise ValueError("bin size must be positive")



    return eig_unfloded


def get_unfolded_from_eig_matrix(matrix_val,menthod = 'Ploy',degree = 5,percent = True,lower = 0.0,upper = 1,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    unfloded = []
    # if percent:
    #     matrix_val = matrix_val[int(L*lower): int(L*upper),:]
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded = unfolding_method_choose(eig_val,menthod=menthod,degree=degree)
        eig_unfloded = np.sort(eig_unfloded)
        if percent:
            eig_val = eig_val[int(L*lower): int(L*upper)]
        
        

        unfloded.extend(eig_unfloded)

    

    return np.sort(unfloded)

def get_unfolded_from_eig_matrix_2(matrix_val,menthod = 'Ploy',degree = 5,percent = True,lower = 0.0,upper = 1,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded = unfolding_method_choose(eig_val,menthod=menthod,degree=degree)
        eig_unfloded = np.sort(eig_unfloded)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        unfloded[:,i] = eig_unfloded
        

        # unfloded.extend(eig_unfloded)

    

    return unfloded
def delta_3_of_possion(L = np.arange(2,20.01,0.05)):
    delta_3 = L / 15
    return L,delta_3

def delta_3_of_GOE(L = np.arange(2,20.01,0.05)):
    delta_3 = np.log(L) / ((np.pi)**2)
    return L,delta_3
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
DEFAULT_SPLINE_SMOOTH = 1.4
DEFAULT_SPLINE_DEGREE = 3


#特征值展开

def get_unfolded_from_eig_matrix_3(matrix_val,menthod = 'Ploy',degree = 7,percent = True,lower = 0.0,upper = 1,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded,_,_ = unflod_poly(eig_val,degree=degree)
        eig_unfloded = np.sort(eig_unfloded)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        unfloded[:,i] = eig_unfloded
        

        # unfloded.extend(eig_unfloded)

    

    return unfloded

from def_unfolding_menthod import Li_fit
def get_unfolded_from_eig_matrix_fit_fit(matrix_val,method = 'Ploy',degree = 4,print_curvce=False): 
    
    

    # L = matrix_val.shape[0]
    # sum_c =  np.sum(eig_vals) / matrix_val.shape[1]
    # print (f"sum of eig_vals = {sum_c:.4f}" if sum_c > 0 else f" sum of eig_vals = {sum_c:.4f}")
    num_of_system = matrix_val.shape[1]
    matrix_val = np.sort(matrix_val,axis=0)
    # unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    # spacings = []
    # lower = 0.2
    # upper = 0.8
    # percent = True
    # L_1 = matrix_val.shape[0]
    # L = len(range(L_1)[int(L_1*lower): int(L_1*upper)])
    matrix_val_unfloded = np.zeros((matrix_val.shape[0],num_of_system))
    for i in range(num_of_system):
        eig_val = matrix_val[:,i]
        eig_val = np.sort(eig_val)
        
        L = len(eig_val)
        # if percent:
        #     eig_val = eig_val[int(L*lower): int(L*upper)]
        eig_unfloded,_,_ =Li_fit(eig_val,method,degree=degree)
        # if percent:
        #     eig_unfloded = eig_unfloded[int(L*lower): int(L*upper)]
        L = len(eig_unfloded)
        eig_unfloded = np.sort(eig_unfloded)
        matrix_val_unfloded[:,i] = eig_unfloded
    
    return matrix_val_unfloded