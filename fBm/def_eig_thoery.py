import numpy as np
from scipy.linalg import toeplitz, circulant, eigvalsh
from scipy.optimize import curve_fit
def fit_power_decay(t, y, constant_a=False):
    t = np.asarray(t)
    y = np.asarray(y)

    # 定义拟合函数（两种情况）
    if constant_a:
        def func(t, alpha):
            return 1 / (t + 1) ** alpha
        p0 = [1.0]  # 初始猜测 alpha
        bounds = (0.01, 5)  # 给 alpha 设置合理范围
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)
        return popt[0]
    else:
        def func(t, a, alpha):
            return a / (t + 1) ** alpha
        p0 = [1.0, 1.0]  # 初始猜测 a, alpha
        bounds = ([0.0001, 0.01], [10, 5])  # 设置 a 和 alpha 的范围
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)
        return popt[1], popt[0]  # 返回顺序为 alpha, a
#
def circulant_eigenvalues(c):
    """计算循环矩阵的特征值"""
    return np.fft.fft(c)

def circulant_eigenvectors(n):
    """计算循环矩阵的特征向量基"""
    return np.fft.ifft(np.eye(n))
def fbm_theory(h,L,return_toe_matrix=True,delta_range=(-0.03, 0.03)):
    # 在delta_range范围内随机生成delta
    delta = np.random.uniform(low=delta_range[0], high=delta_range[1])

    slope_h = 2-2*h + delta

    tau = range(L)
    Corr = [1/((i+1)**slope_h) for i in tau]
    if return_toe_matrix:
        toeplitz_matrix = toeplitz(Corr)
        return toeplitz_matrix
    else:
        return Corr
def toe_cir(Tk):
    N = len(Tk)
    Ck = [None] * N
    Ck[0] = Tk[0]
    
    for k in range(1,N):
        Ck[k] = Tk[k] +Tk[N-k]
    # circulant_matrix = circulant(Ck)
    return Ck

def fbm_Cir_menthod(h,L):
    Corr = fbm_theory(h,L,False)
    Ck = toe_cir(Corr)

    return Ck


def create_fbm_theory_eigenvalue(H=np.arange(0.50,0.901,0.05),L=2000,T=100,eigenvector = False,method = "toe",t_k_theory  = False):
    eigs_dict_H = {}
    eigenvctors_dict_H = {}

    if method == 'toe':
        for i,h in enumerate(H):
            print(F"Processing H = {h:.2f}")

            matrix_eigenvalue_h = np.zeros((L,T))
            dict_eigenvectors_h = {}

            for t in range(T):
                toe_matrix_corr = fbm_theory(h,L)
                if eigenvector:
                    toe_eigenvalues,toe_eigenvectors =  np.linalg.eigh(toe_matrix_corr)
                    matrix_eigenvalue_h[:,t] = toe_eigenvalues
                    dict_eigenvectors_h[t] = toe_eigenvectors
                else:
                    toe_eigenvalues = np.linalg.eigvalsh(toe_matrix_corr)
                    matrix_eigenvalue_h[:,t] = toe_eigenvalues
            
            eigs_dict_H[f"H={h:.2f}"] = matrix_eigenvalue_h
            if eigenvector:
                eigenvctors_dict_H[f"H={h:.2f}"] = dict_eigenvectors_h
        
        if eigenvector:
            return eigs_dict_H,eigenvctors_dict_H
        else:
            return eigs_dict_H
    
    if method == "cir":
        for i,h in enumerate(H):
            print(F"Processing H = {h:.2f}")

            matrix_eigenvalue_h = np.zeros((L,T))
            dict_eigenvectors_h = {}

            for t in range(T):
                Ck = fbm_Cir_menthod(h,L)
                if eigenvector:

                    # cir_eigenvalues,cir_eigenvectors =  np.linalg.eig(cir_matrix_corr)
                    cir_eigenvalues = np.fft.fft(Ck)  
                    cir_eigenvectors = np.fft.ifft(np.eye(L))

                    matrix_eigenvalue_h[:,t] = cir_eigenvalues
                    dict_eigenvectors_h[t] = cir_eigenvectors
                else:
                    # cir_eigenvalues = np.linalg.eigvals(cir_matrix_corr)
                    cir_eigenvalues = np.fft.fft(Ck) 
                    matrix_eigenvalue_h[:,t] = cir_eigenvalues
            
            eigs_dict_H[f"H={h:.2f}"] = matrix_eigenvalue_h
            if eigenvector:
                eigenvctors_dict_H[f"H={h:.2f}"] = dict_eigenvectors_h
        
        if eigenvector:
            return eigs_dict_H,eigenvctors_dict_H
        else:
            return eigs_dict_H        


