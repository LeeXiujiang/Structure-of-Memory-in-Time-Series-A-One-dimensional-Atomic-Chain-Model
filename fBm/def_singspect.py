# 基于以下文献和代码实现：
# Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S., Bunde, A., Stanley, H. E.,
# 《非平稳时间序列的多重分形去趋势波动分析》，Physica A, 316(1-4), 87-114, 2002
# 同时参考了 nolds 项目 (https://github.com/CSchoel/nolds) 以及
# Espen A. F. Ihlen 的工作，《Matlab 中的多重分形去趋势波动分析简介》，Front. Physiol., 2012
import numpy as np
from numpy import polyfit,poly1d
# 提供的函数名称
__all__ = [
    'singularity_spectrum',
    'scaling_exponents',
    'hurst_exponents',
    'singularity_spectrum_plot',
    'scaling_exponents_plot',
    'hurst_exponents_plot'
]

def singularity_spectrum(lag, mfdfa, q, lim=[False, False], interpolate=False):
    """
    提取波动函数的斜率，以获得奇异强度 α 和奇异谱 f(α)。
    注意：由于MFDFA过程中的规律性增强，α 通常以 >1 为中心。

    参数
    ----------
    lag: np.array
        用于 MFDFA 的窗口大小数组。

    mfdfa: np.ndarray
        来自 MFDFA 的波动函数矩阵。

    q: np.array
        分形指数，必须包含多个点。

    lim: list
        指定计算斜率时的最小与最大 lag 范围，None 表示使用完整范围。

    interpolate: int
        是否插值平滑 q 空间（尚未实现）。

    返回
    -------
    alpha: np.array
        奇异强度 α，反映多重分形的强度。α 的宽度越大，多重分形越明显。

    f: np.array
        奇异谱 f(α)，最大值通常为 1，表示最主要的分形尺度。
    """
    if lim[0] is False:
        lim[0] = int(lag.size // 8)
    if lim[1] is False:
        lim[1] = int(lag.size // 1.5)
    q = _clean_q(q)
    _, tau = scaling_exponents(lag, mfdfa, q, lim, interpolate)
    alpha = np.gradient(tau) / np.gradient(q)
    f = _falpha(tau, alpha, q)
    return alpha, f


def scaling_exponents(lag, mfdfa, q, lim=[False, False], interpolate=False):
    """
    计算多重分形的缩放指数 τ(q)，其定义为：
    τ(q) = qh(q) - 1

    若 τ(q) 与 q 的关系为线性，则数据是单重分形；若为非线性，则为多重分形。

    参数和返回值说明与上函数一致。
    """
    if lim[0] is False:
        lim[0] = int(lag.size // 8)
    if lim[1] is False:
        lim[1] = int(lag.size // 1.5)
    q = _clean_q(q)
    slopes = _slopes(lag, mfdfa, q, lim, interpolate)
    return q, (q * slopes) - 1


def hurst_exponents(lag, mfdfa, q, lim=[False, False], interpolate=False):
    """
    计算广义 Hurst 指数 h(q)，即 MFDFA 中各 q 值对应 DFA 的斜率。

    参数和返回值说明与上函数一致。
    """
    if lim[0] is False:
        lim[0] = int(lag.size // 8)
    if lim[1] is False:
        lim[1] = int(lag.size // 1.5)
    q = _clean_q(q)
    hq = _slopes(lag, mfdfa, q, lim, interpolate)
    return q, hq


def _slopes(lag, mfdfa, q, lim=[None, None], modified=True, interpolate=False):
    """
    提取每个 q 对应的斜率，用于后续计算 α、f(α) 或 τ(q)。
    """
    if lim[0] is False:
        lim[0] = int(lag.size // 8)
    if lim[1] is False:
        lim[1] = int(lag.size // 1.5)
    q = _clean_q(q)
    q = np.asarray_chkfinite(q, dtype=float)
    if mfdfa.shape[1] != q.shape[0]:
        raise ValueError("波动函数与 q 的维度不一致。")
    slopes = np.zeros(len(q))
    for i in range(len(q)):
        slopes[i] = polyfit(np.log(lag[lim[0]:lim[1]]),
                            np.log(mfdfa[lim[0]:lim[1], i]), 1)[1]
    return slopes


def _falpha(tau, alpha, q):
    """
    计算奇异谱 f(α)
    """
    return q * alpha - tau


# 以下是绘图函数

def singularity_spectrum_plot(alpha, f):
    """
    绘制奇异谱 f(α)
    """
    fig, ax = _plotter(alpha, f)
    ax.set_ylabel(r'f(α)')
    ax.set_xlabel(r'α')
    return fig, ax


def scaling_exponents_plot(q, tau):
    """
    绘制缩放指数 τ(q)
    """
    fig, ax = _plotter(q, tau)
    ax.set_ylabel(r'tau')
    ax.set_xlabel(r'q')
    return fig, ax


def hurst_exponents_plot(q, hq):
    """
    绘制广义 Hurst 指数 h(q)
    """
    fig, ax = _plotter(q, hq)
    ax.set_ylabel(r'h(q)')
    ax.set_xlabel(r'q')
    return fig, ax


def _clean_q(q):
    """
    清理 q 数组，剔除 |q| < 0.1 的值（q=0 时不收敛）
    """
    q = np.asarray_chkfinite(q, dtype=float)
    q = q[(q < -.1) + (q > .1)]
    return q.flatten()


def _plotter(x, y):
    """
    辅助绘图函数
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k')
    return fig, ax
def _missing_library() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            ("'matplotlib' is required to output the singularity "
             "spectrum plots. Please install 'matplotlib'."
             )
        )

    return
