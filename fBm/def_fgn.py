# 本代码基于 Christopher Flynn 的 fbm 项目：
# https://github.com/crflynn/fbm，
# 以及 Robert B. Davies 和 D. S. Harte 的论文：
# “Tests for Hurst effect.” Biometrika 74, no. 1 (1987): 95-101.



import numpy as np

__all__ = [
    'fgn',
]


def fgn(N: int, H: float) -> np.ndarray:
    """
    生成具有 Hurst 指数 H ∈ (0,1) 的分数高斯噪声（fGn）。
    当 H = 1/2 时，该过程退化为普通高斯白噪声。
    当前实现方法为 Davies–Harte 方法，该方法在 H 接近 0 时失效。
    后续版本中将实现 Cholesky 分解法与 Hosking 方法。

    参数
    ----------
    N: int
        要生成的分数高斯噪声序列长度。

    H: float
        Hurst 指数，取值范围在 (0,1)。

    返回值
    -------
    f: np.ndarray
        长度为 N 的分数高斯噪声数组，具有指定的 Hurst 指数 H。
    """

    # 参数断言
    assert isinstance(N, int), "序列长度必须是整数"
    assert isinstance(H, float), "Hurst 指数必须是浮点数，取值范围为 (0,1)"

    # 生成索引序列
    k = np.linspace(0, N - 1, N)

    # 相关函数
    cor = 0.5 * (abs(k - 1) ** (2 * H)
                 - 2 * abs(k) ** (2 * H)
                 + abs(k + 1) ** (2 * H)
                 )

    # 相关函数的特征值（通过快速傅里叶变换）
    eigenvals = \
        np.sqrt(
            np.fft.fft(
                np.real(
                    np.concatenate(
                        [cor[:], 0, cor[1:][::-1]], axis=None
                    )
                )
            )
        )

    # 生成两个独立的标准正态分布随机序列
    gn = np.random.normal(0.0, 1.0, N)
    gn2 = np.random.normal(0.0, 1.0, N)

    # Davies–Harte 方法的实现
    w = np.concatenate(
        [
            (eigenvals[0] / np.sqrt(2 * N)) * gn[0],
            (eigenvals[1:N] / np.sqrt(4 * N)) * (gn[1:] + 1j * gn2[1:]),
            (eigenvals[N] / np.sqrt(2 * N)) * gn2[0],
            (eigenvals[N + 1:] / np.sqrt(4 * N))
            * (gn[1:][::-1] - 1j * gn2[1:][::-1])
        ], axis=None)

    # 快速傅里叶变换，保留前 N 项作为结果
    f = np.fft.fft(w).real[:N] * ((1.0 / N) ** H)

    # TODO: 优化速度
    # TODO: 支持生成多个样本，理论上较容易实现
    # TODO: 实现 Cholesky 分解法
    # TODO: 实现 Hosking 方法
    return f
