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
def _step_function_fast(eigenvalues, grid_points):
    """Optimized version that does not repeatedly call np.sum(eigenvalues <= grid_points)."""
    result = np.zeros(len(grid_points), dtype=np.int64)
    n_eigs = len(eigenvalues)
    n_points = len(grid_points)
    
    if grid_points[-1] <= eigenvalues[0]:  # all values zero
        return result
    if grid_points[0] > eigenvalues[-1]:  # all values max
        result.fill(n_eigs)
        return result

    # Find first index where grid_points >= eigenvalues[0]
    j = 0
    while j < n_points and grid_points[j] < eigenvalues[0]:
        j += 1

    i = count = 0
    while j < n_points and i < n_eigs:
        if grid_points[j] >= eigenvalues[i]:
            i += 1
            count += 1
        else:
            while j < n_points and grid_points[j] < eigenvalues[i]:
                result[j] = count
                j += 1

    result[j:] = count  # fill remaining values
    return result
def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Perform linear regression to compute the slope.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable values
    y : np.ndarray
        Dependent variable values
        
    Returns
    -------
    float
        Slope of the linear regression line
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
        
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.sum((x - x_mean) * (y - y_mean))
    var = np.sum((x - x_mean) ** 2)
    
    return 0.0 if var == 0.0 else cov / var  # type: ignore
def _intercept(x, y, slope):
    return np.mean(y) - slope * np.mean(x)  # type: ignore
def _sq_lin_deviation(eigs, steps, K, w, grid):
    """Compute the sqaured deviation of the staircase function of the best fitting
    line, over the region in `grid`.

    Parameters
    ----------
    eigs: ndarray
        The raw, sorted eigenvalues

    steps: ndarray
        The step function values which were computed on `grid`.

    K: float
        The calculated slope of the line of best fit.

    w: float
        The calculated intercept.

    grid: ndarray
        The grid of values for which the step function was evaluated.

    Returns
    -------
    sq_deviations: ndarray
        The squared deviations.
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

    Parameters
    ----------
    grid: list or np.array of floats
            Sampling points for the function values

    vals: list or np.array of floats
            Function values at the sampling points

    Returns
    -------
    float: approximation for the integral
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
    """scipy.integrate.trapz is excruciatingly slow and unusable for our purposes.
    This tiny rewrite seems to result in a near 20x speedup. However, being trapezoidal
    integration, it is quite inaccurate."""
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