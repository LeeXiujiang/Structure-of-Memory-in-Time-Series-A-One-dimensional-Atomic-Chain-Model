import numpy as np

def compute_eigenvalues(N):
    """
    输入矩阵大小 N，返回 GOE, GUE, GSE 的特征值。
    返回字典格式：
    {
        "GOE": 特征值数组,
        "GUE": 特征值数组,
        "GSE": 特征值数组 (长度为 2N)
    }
    """

    def generate_goe(N):
        A = np.random.randn(N, N)
        return (A + A.T) / 2

    def generate_gue(N):
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        return (A + A.conj().T) / 2

    def generate_gse(N):
        A = np.random.randn(N, N)
        B = np.random.randn(N, N)
        H = np.block([
            [A, B],
            [-B.T, A.T]
        ])
        return (H + H.T) / 2  # 保证实对称

    # 构造矩阵并求特征值
    goe_eigs = np.linalg.eigvalsh(generate_goe(N))
    gue_eigs = np.linalg.eigvalsh(generate_gue(N))
    gse_eigs = np.linalg.eigvalsh(generate_gse(N))  # 2N 维

    return goe_eigs, gue_eigs, gse_eigs
