import numpy as np

def Overlap_DFA(ts, s, degree=5):
    N = len(ts)
    windows = np.array([ts[i:i+s] for i in range(N - s + 1)]).T
    t = np.arange(s)

    windows_fit = np.zeros_like(windows)
    for i in range(windows.shape[1]):
        coeffs = np.polyfit(t, windows[:, i], degree)
        windows_fit[:, i] = np.polyval(coeffs, t)

    return windows - windows_fit


def Doran_unbiased_self_corr_var(Z_D, W_A, W_B):
    S_len, cols = Z_D.shape
    total = 0
    for k in range(cols):
        w_a, w_b = W_A[:, k], W_B[:, k]
        p_a = np.dot(w_a[:-1], w_a[1:]) / ((2 * S_len - 1) * np.std(w_a))
        p_b = np.dot(w_b[:-1], w_b[1:]) / ((2 * S_len - 1) * np.std(w_b))
        p_a += ((1 + p_a) + 2 * p_a) / S_len
        p_b += ((1 + p_b) + 2 * p_b) / S_len
        v = (p_a + p_b) * (1 - 1 / (2 * S_len)) * np.std(w_a)
        total += v
    return np.sqrt(total) / np.sqrt(S_len * cols)

def cal_UDFA(ts, s, degree=5):
    print(ts[:6])
    ts = np.cumsum(ts) 
    print(ts[:5])
    Z_D = Overlap_DFA(ts, s, degree)
    S_len, cols = Z_D.shape

    # 构造 W_A
    W_A = np.repeat(Z_D, 2, axis=0)

    # 构造 W_B = W_A * 交替符号矩阵
    signs = np.tile(np.repeat(((-1) ** (np.arange(S_len) + 1)), 2).reshape(-1, 1), (1, cols))
    W_B = W_A * signs

    return Doran_unbiased_self_corr_var(Z_D, W_A, W_B)
