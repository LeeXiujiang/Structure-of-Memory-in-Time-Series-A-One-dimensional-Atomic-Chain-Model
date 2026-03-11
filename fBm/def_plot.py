import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import List  

def find_subplot_layout(T):
    if T == 1:
        return 1, 1
    m = math.ceil(math.sqrt(T))
    n = math.ceil(T / m)
    # 尝试使 m 和 n 更接近
    while abs(m - n) > 1 and (m - 1) * n >= T:
        m -= 1
    return m, n
def generate_gradient_two_colors(colorA, colorB, N=256):
    """
    生成 colorA 到 colorB 的渐变颜色列表（仅支持 2 色）
    :param colorA: 起始颜色（名称或十六进制，如 'blue' 或 '#0000FF'）
    :param colorB: 结束颜色
    :param N: 返回的颜色数量
    :return: 渐变颜色列表，格式为 [(R1,G1,B1), (R2,G2,B2), ...]
    """
    # 将颜色名称转换为 RGB（范围 0-1）
    rgbA = mcolors.to_rgb(colorA)
    rgbB = mcolors.to_rgb(colorB)
    
    # 线性插值
    gradient = np.linspace(0, 1, N)
    colors = []
    for t in gradient:
        r = rgbA[0] + (rgbB[0] - rgbA[0]) * t
        g = rgbA[1] + (rgbB[1] - rgbA[1]) * t
        b = rgbA[2] + (rgbB[2] - rgbA[2]) * t
        colors.append((r, g, b))
    return colors

import seaborn as sns

def generate_gradient_seaborn(color_list, N=256):
    """
    使用 Seaborn 生成渐变颜色
    :param color_list: 颜色列表（支持名称或十六进制）
    :param N: 颜色数量
    :return: RGB 元组列表
    """
    palette = sns.color_palette(color_list, n_colors=N)
    return palette
def plot_Ne(plot_eig,Ne,H,color_gradient=True,plot_one = True): 
    # if color_gradient:
    #     cmap = plt.cm.get_cmap('viridis')
        # # 定义颜色渐变范围（从蓝色到红色）
    len_H = len(H)
    colors = plt.cm.get_cmap('coolwarm', (2*len_H-1))  # 'coolwarm' 是 matplotlib 内置的渐变色
    # # 生成 20 个渐变色
    color_list = [colors(i) for i in range(len_H)]
    color_list = color_list[-len_H:]
    if plot_one:

        plt.figure()
        for i,h in enumerate(H):
            color = color_list[i]  # 统一颜色
            # plt.title("Spectral Density")
            plt.plot(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],c=color)
            plt.scatter(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],c = color,label = f"H={h:.2f}")
            # plt.loglog(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],"o",label=h)
            plt.legend()
            plt.ylim(top=3)
            plt.xlim(left=-1)

        plt.show()   
    else:
        m,n = find_subplot_layout(len(H))
        #后续可添加只最后一行显示横坐标
        #    第一列显示纵坐标
        plt.figure()
        for i,h in enumerate(H):
            plt.subplot(m,n,i+1)
            plt.title(f"H={h:.2f}")
            plt.plot(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],color_list[i])
            plt.scatter(plot_eig[f"H={h:.2f}"],Ne[f"H={h:.2f}"],c= color_list[i])
            plt.ylim(top=3)
            plt.legend()
        
        plt.show()





    