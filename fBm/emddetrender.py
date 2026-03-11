# 本函数基于 Dawid Laszuk 的 PyEMD 函数“Python implementation of
# Empirical Mode Decomposition algorithm” https://github.com/laszukdawid/PyEMD，
# 使用 Apache 2.0 许可证。

# from PyEMD import EMD
import numpy as np

# PyEMD 的导入在函数内部进行

__all__ = [
    'detrendedtimeseries',
    'IMFs'
]


def detrendedtimeseries(timeseries: np.ndarray, modes: list) -> np.ndarray:
    """
    该函数计算给定时间序列的本征模函数（IMFs），并减去用户选择的 IMF 模式，
    从而获得去趋势后的时间序列。此方法基于 Dawid Laszuk 的 PyEMD，
    可在 https://github.com/laszukdawid/PyEMD 获取。

    参数
    ----------
    timeseries: np.ndarray
        一维时间序列，长度为 `N`。

    modes: list
        整数列表，表示要从 `timeseries` 中减去/去趋势的 IMF 索引。

    返回值
    -------
    detrendedTimeseries: np.ndarray
        去趋势后的一维时间序列。

    警告
    --------
    要使用经验模态分解（EMD）进行去趋势分析，必须安装 ``pyEMD`` 库。

    .. 代码::

        pip install EMD-signal

    备注
    -----
    .. 版本新增:: 0.3

    参考文献
    ----------
    .. [Huang1998] N. E. Huang 等, “The empirical mode decomposition and the Hilbert spectrum for non-linear and non-stationary time series analysis”, Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998.
    .. [Rilling2003] G. Rilling 等, “On Empirical Mode Decomposition and its algorithms”, IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing NSIP-03, Grado (意大利), 2003年6月.
    """

    # 使用 pyEMD 获取本征模函数（IMFs）
    IMF = IMFs(timeseries)

    # 从原始时间序列中减去选定的 IMF 模式
    detrendedTimeseries = timeseries - np.sum(IMF[modes, :], axis=0)

    return detrendedTimeseries


def IMFs(timeseries: np.ndarray) -> np.ndarray:
    """
    提取给定时间序列的本征模函数（IMFs）。

    参数
    ----------
    timeseries: np.ndarray
        一维时间序列，长度为 `N`。

    备注
    -----
    .. 版本新增:: 0.3

    返回值
    -------
    IMFs: np.ndarray
        本征模函数（IMFs），即经验模态分解得到的分量。其形状为 `(..., timeseries.size)`，
        第一维表示分量数，最后一个分量为残差项。
    """

    # 检查是否已安装 EMD-signal 库
    _missing_library()
    from PyEMD import EMD

    # 初始化 pyEMD 的 EMD 实例
    emd = EMD()

    # 获取本征模函数（IMFs）
    IMFs = emd(timeseries)

    # 返回形状为 (..., timeseries.size) 的 IMF 数组
    return IMFs


def _missing_library() -> None:
    try:
        import PyEMD.EMD as _EMD
    except ImportError:
        raise ImportError(
            ("需要安装 PyEMD 才能进行经验模态分解。请使用 'pip install EMD-signal' 安装该库。")
        )
    return
