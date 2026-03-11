import numpy as np

def Overlap_DFA(ts,s,degree=1):
    N= len(ts)
    windows = np.array([ts[i:i+s] for i in range(len(ts) - s + 1)]).T
    # sliding_window = np.zeros(s,N-s+1)
    # for i in range(N-s+1):
    #     sliding_window[:,i] = ts[i:i+s]

    #趋势拟合
    windows_fit = np.zeros_like(windows)
    for i in range(windows.shape[1]):    # 遍历每一列（每个窗口）
        # 1. 对当前窗口数据拟合三次多项式
        coeffs = np.polyfit(range(1,s+1), windows[:, i], deg=degree)
        # 2. 用多项式生成拟合曲线
        fitted_values = np.polyval(coeffs, range(1,s+1))
        # 3. 将拟合值存入新数组
        windows_fit[:, i] = fitted_values
    
    Z_DK = windows - windows_fit

    return Z_DK
def Overlap_DFA_OLS(ts, s, degree=1):
    """
    使用最小二乘法进行去趋势的重叠DFA分析
    
    参数:
        ts: 输入时间序列 (1D数组)
        s: 窗口大小(分析尺度)
        degree: 多项式拟合阶数(默认为1-线性去趋势)
        
    返回:
        Z_DK: 去趋势后的窗口数据矩阵
    """
    N = len(ts)
    # 生成重叠窗口矩阵 (s × (N-s+1))
    windows = np.array([ts[i:i+s] for i in range(N - s + 1)]).T
    
    # 初始化去趋势后的矩阵
    Z_DK = np.zeros_like(windows)
    
    # 准备最小二乘设计矩阵 (范德蒙矩阵)
    x = np.arange(1, s+1)  # 窗口内坐标
    X = np.vander(x, degree+1, increasing=True)  # 设计矩阵
    
    for i in range(windows.shape[1]):
        # 当前窗口数据
        y = windows[:, i]
        
        # 最小二乘求解
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 计算拟合值
        fitted_values = X @ coeffs
        
        # 去趋势
        Z_DK[:, i] = y - fitted_values
    
    return Z_DK
    #Wak
    S_len, cols = Z_DK.shape
    W_A = np.repeat(Z_DK, 2, axis=0)  # 沿行方向重复每个元素两次
    if S_len / W_A.shape[0] != 2:
        print("Error: S_len / WA.shape[0] != 2")

    #Wbk
    # 生成符号交替因子 (-1)^{m+1}（m从1开始）
    m_indices = np.arange(1, S_len+1)          # m = 1, 2, ..., S
    signs = (-1) ** (m_indices + 1)        # [1, -1, 1, -1, ...] for S=3: [1, -1, 1]
    signs_expanded = np.repeat(signs, 2)   # 每个符号重复两次 [1, 1, -1, -1, 1, 1]

    # 扩展符号矩阵到与 W_A 相同形状
    signs_matrix = np.tile(signs_expanded.reshape(-1, 1), (1, cols))

    # 生成 W_B^k：W_A * 符号矩阵
    W_B = Z_DK * signs_matrix3qw
def Doran_unbiased_self_corr_var(Z_D,W_A,W_B):
    S_len, cols = Z_D.shape   #s n-S+1
    # selfcorr_A = []
    # selfcorr_B = []
    V_s_k = 0
    for k in range(cols):
        # z_d_k = Z_D[:,k]
        w_a_k = W_A[:,k]
        w_b_k = W_B[:,k]
        # p_a = (sum(e*e for e in z_d_k) + sum(z_d_k[i]*z_d_k[i+1] for i in range(S_len-1) )) / ( (2*S_len -1)*np.std(w_a_k) )
        p_a = np.dot(w_a_k[:-1], w_a_k[1:]) / ( (2*S_len -1)*(np.var(w_a_k)) )
        p_b = np.dot(w_b_k[:-1], w_b_k[1:]) / ( (2*S_len -1)* (np.var(w_b_k)) )
        m = 0 
        p_a_ = p_a + ( (1+p_a)*(m+1) + 2*p_a )/ (S_len)
        p_b_ = p_b + ( (1+p_b)*(m+1) + 2*p_b )/ (S_len)
        # selfcorr_A.append(p_a_)
        # selfcorr_B.append(p_b_)
        v_s_k = (p_a_+p_b_)* (1- 1/(2*S_len))*(np.var(w_a_k))
        V_s_k += v_s_k
    V_s_k = np.sqrt(V_s_k) / np.sqrt(S_len * (cols))
    return V_s_k
def cal_UDFA(ts,s,degree=1):
    # print(ts[:5])
    ts_mean = np.mean(ts)
    # print(f"mean {ts_mean}")
    ts = np.cumsum(ts) 
    ts = ts - ts_mean
    # print(ts[:5])
    Z_D = Overlap_DFA_OLS(ts,s,degree)
    # print(Z_D.shape)
    #Wak
    S_len, cols = Z_D.shape
    W_A = np.repeat(Z_D, 2, axis=0)  # 沿行方向重复每个元素两次
    if W_A.shape[0] / S_len != 2:
        print("Error: S_len / WA.shape[0] != 2")

    #Wbk
    # 生成符号交替因子 (-1)^{m+1}（m从1开始）
    m_indices = np.arange(1, S_len+1)          # m = 1, 2, ..., S
    signs = (-1) ** (m_indices + 1)        # [1, -1, 1, -1, ...] for S=3: [1, -1, 1]
    signs_expanded = np.repeat(signs, 2)   # 每个符号重复两次 [1, 1, -1, -1, 1, 1]

    # 扩展符号矩阵到与 W_A 相同形状
    signs_matrix = np.tile(signs_expanded.reshape(-1, 1), (1, cols))

    # 生成 W_B^k：W_A * 符号矩阵
    W_B = W_A * signs_matrix
 
    ##计算UDFA
    #方差V_s_k 
    V_s_k = Doran_unbiased_self_corr_var(Z_D,W_A,W_B)
    
    return V_s_k