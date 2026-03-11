# 基于以下文献：
# Kantelhardt, J. W. 等人，Multifractal detrended fluctuation analysis of nonstationary time series.
# Physica A, 316(1-4), 87-114, 2002
# 以及 GitHub 上的 nolds 工具包：https://github.com/CSchoel/nolds
# 和 Espen A. F. Ihlen 的 Matlab 教程：https://doi.org/10.3389/fphys.2012.00141

# 多重分形去趋势波动分析（MFDFA）
import numpy as np
from numpy import polyfit,polyval
from emddetrender import *
def MFDFA(timeseries, lag, order=1, q=2, stat=False, modified=False,
          extensions={'EMD': False, 'eDFA': False, 'window': False}):
    """
    对时间序列进行多重分形去趋势波动分析（MFDFA）。
    MFDFA 生成波动函数 F²(q,s)，其中 s 为分段长度，q 为分形阶数。

    处理步骤：
    1. 累加时间序列 Xₜ 得到 Yₜ = cumsum(Xₜ)
    2. 将 Yₜ 分成 Ns 个长度为 s 的段
    3. 每段使用阶数为 m 的多项式 y_{v,i} 拟合，计算方差 F²(v,s)
    4. 得到总体波动函数 F_q²(s) 并分析其随 s 的变化规律

    如果数据是单分形的且 Hurst 指数接近 0，建议使用 modified=True 进行二次积分处理。

    参数说明
    ----------
    timeseries : np.ndarray
        一维时间序列数组（N, 1）

    lag : np.ndarray
        用于计算的窗口大小数组。注意 lag 的最小值应大于 order+1。
        对于长度较短的序列不应取较大的 lag 值。

    order : int
        拟合多项式的阶数，默认 1。order=0 表示不去趋势，即普通波动分析。

    q : np.ndarray
        分形阶数数组，默认 2。q=2 表示标准 DFA，代码中会去除 q=0 的情况。

    stat : bool
        若为 True，返回波动函数的标准差。

    modified : bool
        若为 True，会对时间序列进行再次积分，适用于 H≈0 的强反相关数据。

    extensions : dict
        额外功能选项：
        - 'EMD'：若非 False，表示传入了 EMD 分解后选中的 IMF 索引列表，将使用外部 EMD 去趋势；
        - 'eDFA'：若为 True，将调用 eDFA 方法；
        - 'window'：若为正整数，表示启用滑动窗口方式处理短时间序列。

    返回
    -------
    lag : np.ndarray
        实际用于计算的窗口大小数组，已筛除不合规的值。

    f : np.ndarray
        形状为 (len(lag), len(q)) 的波动函数矩阵。

    其他返回值（根据参数 stat 和 eDFA 决定）：
    - f_std：每个 q 的标准差；
    - f_eDFA：每个窗口长度的 DFA 极值差。

    """

    # 处理 lag 数组，只保留大于 order+1 的值，并转换为整数
    lag = lag[lag > order + 1]
    lag = np.round(lag).astype(int)

    # 确保输入是 1 维时间序列
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "输入时间序列必须是一维数组"

    timeseries = timeseries.reshape(-1, 1)

    # 数据长度
    N = timeseries.shape[0]

    # 若使用滑动窗口，检查其合法性
    window = False
    if 'window' in extensions:
        if extensions['window'] is not False:
            window = extensions['window']
            assert isinstance(window, int), "window 必须是整数"
            assert window > 0, "window 必须大于 0"

    # 转换分形阶数 q 为浮点数，并剔除 |q| < 0.1 的值（无法收敛）
    q = np.asarray_chkfinite(q, dtype=float)
    q = q[(q < -.1) + (q > .1)]
    q = q.reshape(-1, 1)

    # 多项式拟合用的横轴
    X = np.linspace(1, lag.max(), lag.max())

    # 一阶积分
    Y = np.cumsum(timeseries - np.mean(timeseries))

    # 若启用二阶积分
    if modified is True:
        Y = np.cumsum(Y - np.mean(Y))

    # 初始化结果数组
    f = np.empty((0, q.size))
    if stat is True:
        f_std = np.empty((0, q.size))
    if ('eDFA', True) in extensions.items():
        f_eDFA = np.empty((0, q.size))

    # 若使用 EMD 去趋势
    if 'EMD' in extensions:
        if extensions['EMD'] is not False:
            assert isinstance(extensions['EMD'], list), "EMD 需为 IMF 索引列表"
            Y = detrendedtimeseries(Y, extensions['EMD'])
            order = 0  # EMD 已完成去趋势，跳过多项式拟合

    # 遍历所有 lag
    for i in lag:

        # 标准方式（不使用滑动窗口）
        if window is False:
            Y_ = Y[:N - N % i].reshape((N - N % i) // i, i)
            Y_r = Y[N % i:].reshape((N - N % i) // i, i)

            if order == 0:
                F = np.append(np.var(Y_, axis=1), np.var(Y_r, axis=1))
            else:
                p = polyfit(X[:i], Y_.T, order)
                p_r = polyfit(X[:i], Y_r.T, order)
                F = np.append(np.var(Y_ - polyval(X[:i], p), axis=1),
                              np.var(Y_r - polyval(X[:i], p_r), axis=1))

        # 使用滑动窗口的方式
        if window is not False:
            F = np.empty(0)
            for j in range(0, i - 1, window):
                N_0 = N - j
                Y_ = Y[j:N - N_0 % i].reshape((N - N_0 % i) // i, i)

                if order == 0:
                    F = np.append(F, np.var(Y_, axis=1))
                else:
                    p = polyfit(X[:i], Y_.T, order)
                    F = np.append(F, np.var(Y_ - polyval(X[:i], p), axis=1))

        # 计算 q 阶波动函数
        f = np.append(f,
                      np.float_power(
                          np.mean(np.float_power(F, q / 2), axis=1),
                          1 / q.T
                      ),
                      axis=0)

        # 计算标准差（若启用）
        if stat is True:
            f_std = np.append(f_std,
                              np.float_power(
                                  np.std(np.float_power(F, q / 2), axis=1),
                                  1 / q.T
                              ),
                              axis=0)

        # 计算 DFA 极差指标
        if ('eDFA', True) in extensions.items():
            f_eDFA = np.append(f_eDFA, eDFA(F))

    # 返回结果
    if stat is False:
        if ('eDFA', True) in extensions.items():
            return lag, f, np.vstack(f_eDFA)
        else:
            return lag, f
    if stat is True:
        if ('eDFA', True) in extensions.items():
            return lag, f, f_std, np.vstack(f_eDFA)
        else:
            return lag, f, f_std


def eDFA(F):
    """
    eDFA 方法是基于每个窗口的最大值和最小值之差衡量非平稳性的指标：

        dF_q^2(s) = max(F_q^2(s)) - min(F_q^2(s))

    参数
    ----------
    F : np.ndarray
        由 MFDFA 生成的波动函数

    返回
    -------
    res : np.ndarray
        每个窗口对应的极值差

    参考文献
    ----------
    Pavlov 等人，"Detrended fluctuation analysis of cerebrovascular responses...",
    CNSNS 85, 105232, 2020
    """
    return np.max(F) - np.min(F)
