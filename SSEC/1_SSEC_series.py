import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 文件路径
# ===============================
file_name_SSEC = "D:\\Data\\SSEC\\SSEC_1.csv"
file_name_HSI  = "D:\\Data\\HSI\\HSI_1.csv"

year_list = [2019, 2020, 2021, 2022, 2023, 2024]

# ===============================
# 2. 读取数据
# ===============================
# --- SSEC ---
df_ssec = pd.read_csv(
    file_name_SSEC,
    parse_dates=["trade_time"]
).rename(columns={"trade_time": "time", "close": "close"})

df_ssec = df_ssec.set_index("time").sort_index()

# --- HSI ---
df_hsi = pd.read_csv(
    file_name_HSI,
    parse_dates=["date"]
).rename(columns={"date": "time", "close": "close"})

df_hsi = df_hsi.set_index("time").sort_index()

# ===============================
# 3. 颜色设置（统一）
# ===============================
colors = plt.cm.viridis(np.linspace(0, 1, len(year_list)))

# ===============================
# 4. 创建画布
# ===============================

#尺度放缩
width_fig = 8 /  2.54
height_fig = 5.5 / 2.54
scale_change = 2.54
scale_change_2 = 2.54*2.54
plt.figure(figsize=(width_fig, height_fig))
ax_ssec = plt.gca()
#*******设置字体Arial无衬线*******
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

# ax_ssec, ax_hsi = axes

# ===============================
# 5. (a) SSEC 原始价格序列
# ===============================
for i, year in enumerate(year_list):
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    price = df_ssec.loc[start:end, "close"].dropna()
    if len(price) == 0:
        continue

    ax_ssec.plot(
        price.index,
        price.values,
        color=colors[i],
        lw=0.8/scale_change,

    )

    # 年份标注
    ax_ssec.text(
        price.index[0],
        price.iloc[0],
        f"{year}",
        fontsize=8,
        color=colors[i],
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=colors[i],
            linewidth=0.8/scale_change,
            alpha=0.8
        )
    )

#刻度尺与标签
ax1= ax_ssec
ax1.tick_params(axis='both', labelsize=8)   # 同时设置 x 和 y
ax1.tick_params(axis='x', pad=1)   # x 轴刻度标签与轴的距离
ax1.tick_params(axis='y', pad=1)   # y 轴刻度标签与轴的距离
# ax_ssec.set_title("(a) SSEC price series (2019–2024)", fontsize=13)
# ax_ssec.text(
#     0.05,
#     0.02,
#     "(a)",
#     fontsize=14,
#     color='k',
#     transform=ax_ssec.transAxes
# )
ax_ssec.set_ylabel("Price",fontsize=12)
ax_ssec.set_xlabel("Time",fontsize=12)
ax_ssec.grid(alpha=0.3)

plt.savefig("fig10.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig10.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig10.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.savefig("fig10.eps", dpi=600,format= 'eps', bbox_inches='tight', pad_inches=0.05)
# ===============================
# 7. 排版与显示
# ===============================
plt.tight_layout()
plt.show()
