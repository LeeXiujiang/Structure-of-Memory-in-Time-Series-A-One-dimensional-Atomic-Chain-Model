import numpy as np
from scipy.linalg import toeplitz, circulant, eigvalsh,eigvals
from scipy.optimize import curve_fit


# def toe_cir_eigs(Tk,fft=True):
#     N = len(Tk)
#     Ck = [None] * N
#     Ck[0] = Tk[0]
    
#     for k in range(1,N):
#         Ck[k] = Tk[k] +Tk[N-k]
#     circulant_matrix = circulant(Ck)
#     if fft:
#         eigvals_ = np.fft.fft(Ck)  # 特征值是第一行的DFT
#     else:
#         circulant_matrix = circulant(Ck)
#         eigvals_ = eigvalsh(circulant_matrix)
#     return eigvals_ 
def toe_cir_eigs(Toe_matrix,ftt = True):
    N = Toe_matrix.shape[0]
    T_k = Toe_matrix[:,0]
    Ck = [None] * N
    Ck[0] = T_k[0]
    
    for k in range(1,N):
        Ck[k] = T_k[k] +T_k[N-k]
    # circulant_matrix = circulant(Ck)
    if ftt:
        eigvals_ = np.fft.fft(Ck)  # 特征值是第一行的DFT
    else:
        circulant_matrix = circulant(Ck)
        eigvals_ = eigvalsh(circulant_matrix)
    return eigvals_
def toe_cir_asymmetric(Toe_matrix,ftt=True):
    N = Toe_matrix.shape[0]
    c = Toe_matrix[:,0]   #toeplitz matrix first第一列 
    r = Toe_matrix[0,:]   #toeplitz matrix first 第一行

    # Tk = np.concatenate((c, r[1:][::-1]))  # 第一列 + 第一行（去掉第一个并倒序）
    # if len(Tk) != (2*N - 1):
    #     print("error: please check the input and stop run the code!")
    
    C_k = [None]*N
    C_k[0] = c[0]
    
    for i in range(1,N):
        C_k[i] = r[i] + c[N-i]
    # circulant_matrix = circulant(Ck)
    if ftt:
        eigvals_ = np.fft.fft(C_k)  # 特征值是第一行的DFT
    else:
        circulant_matrix = circulant(C_k)
        eigvals_ = eigvals(circulant_matrix)
    return eigvals_
    
    # circulant_matrix = circulant(Ck)
    return 
