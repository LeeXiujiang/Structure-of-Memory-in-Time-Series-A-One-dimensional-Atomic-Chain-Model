import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde, norm
import math

def find_subplot_layout(T):
    if T == 1:
        return 1, 1
    m = math.ceil(math.sqrt(T))
    n = math.ceil(T / m)
    # 尝试使 m 和 n 更接近
    while abs(m - n) > 1 and (m - 1) * n >= T:
        m -= 1
    return m, n
def nnsd_matplotlib(data_dict, H, x_min=0, x_max=4):


    x_vals = np.linspace(x_min, x_max, 1000)
    
    for h in H:
        h_label = f'H={h:.2f}'
        data = data_dict[h]
        data = data[data >= 0]
        # 计算KDE，限制在数据范围内
        kde = gaussian_kde(data)
        # 只绘制指定范围内的数据
        mask = (x_vals >= min(data)) & (x_vals <= max(data))

    
    return x_vals[mask], kde(x_vals[mask])

def get_density(vals,bin_size=80,lower = 0,upper = 8,print_curvce = False):
    vals = np.array(vals)
    L1 = len(vals)
    vals = vals[vals >= 0]
    vals = vals[vals <= upper]
    print(f"Percent of spacings = : {len(vals)/L1*100}")

    N = bin_size
    lower = min(vals)
    dx = (upper - lower) / float(N)  #带宽
    unit = 1.0 / (float(len(vals)) * dx)

    density = dict()   # 存储每个区间内的密度
    for i in range(N):
        lower_bound = lower + i * dx
        upper_bound = lower + (i + 1) * dx
        # 统计当前区间内的数据个数
        count = sum(unit for val in vals if lower_bound <= val < upper_bound)
        
        # 将结果存入字典，key为区间列表
        density[tuple([lower_bound, upper_bound])] = count    
    x,y=[],[]
    s = 0
    for k in range(len(density)):
        key,value = list(density.items())[k]
        # 计算 bim = (left + right) / 2
        left, right = key
        bim = (left + right) / 2
        s += value * (right - left)
        x.append(bim)
        y.append(value)
    if print_curvce:
        print(f"Area under curve = {s}")
    
    #返回密度值
    return x,y

def plot_nnsd(spaings_dict,H,color_gradient=True,plot_one = False): 
    # if color_gradient:
    #     cmap = plt.cm.get_cmap('viridis')
    plot_eig = dict()
    Ne = dict()
    for i,h in enumerate(H):
        eigs = spaings_dict[f"H={h:.2f}"]
        x,y = get_density(eigs,bin_size=40,upper=4,print_curvce=False)
        plot_eig[f"H={h:.2f}"] = x
        Ne[f"H={h:.2f}"] = y

    if plot_one:

        plt.figure()
        for h in H:
            
            plt.title("Spectral Density")
            plt.plot(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])
            plt.scatter(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],label = f"H={h:.2f}")
            # plt.loglog(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],"o",label=h)
            plt.legend()

        plt.show()   
    else:
        m,n = find_subplot_layout(len(H))
        #后续可添加只最后一行显示横坐标
        #    第一列显示纵坐标
        plt.figure()
        for i,h in enumerate(H):
            plt.subplot(m,n,i+1)
            plt.title(f"H={h:.2f}")
            plt.plot(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])
            plt.scatter(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"])
            plt.legend()
        
        plt.show()