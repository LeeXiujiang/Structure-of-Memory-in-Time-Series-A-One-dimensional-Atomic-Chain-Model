from datetime import datetime, timedelta

# from datetime import datetime
from dateutil.relativedelta import relativedelta

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_month_days(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 29 if is_leap_year(year) else 28

def add_months(dt, months):
    year = dt.year + (dt.month + months - 1) // 12
    month = (dt.month + months - 1) % 12 + 1
    max_days = get_month_days(year, month)
    new_day = min(dt.day, max_days)
    return datetime(year, month, new_day)
def get_start_end_daily(start_date, end_date, months_per_window=9):
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt   = datetime.strptime(end_date, '%Y-%m-%d')

    start_list_daily = []
    end_list_daily = []

    current = start_dt

    while current <= end_dt:
        period_end = add_months(current, months_per_window)
        
        if period_end > end_dt:
            period_end = end_dt  # 超过就截断
            start_list_daily.append(current.strftime('%Y-%m-%d'))
            end_list_daily.append(period_end.strftime('%Y-%m-%d'))
            break  # 最后一个窗口生成后退出循环

        start_list_daily.append(current.strftime('%Y-%m-%d'))
        end_list_daily.append(period_end.strftime('%Y-%m-%d'))
        current += timedelta(days=1)  # 每次向后滑一天

    print(f"总共 {len(start_list_daily)} 个区间")
    return len(start_list_daily), start_list_daily, end_list_daily
def add_months_2(dt, months):
    return dt + relativedelta(months=months)

def get_start_end_monthly(start_date, end_date, months_per_window=9):
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt   = datetime.strptime(end_date, '%Y-%m-%d')

    start_list, end_list = [], []
    current = start_dt

    while current <= end_dt:
        period_end = add_months_2(current, months_per_window)
        if period_end > end_dt:
            period_end = end_dt  # ⚡ 超过就取 end_date
            start_list.append(current.strftime('%Y-%m-%d'))
            end_list.append(period_end.strftime('%Y-%m-%d'))
            break  # 最后一个窗口生成后退出循环

        start_list.append(current.strftime('%Y-%m-%d'))
        end_list.append(period_end.strftime('%Y-%m-%d'))
        current = add_months(current, 1)

    print(f"总共 {len(start_list)} 个区间")
    return len(start_list), start_list, end_list

# get_start_end_dail
"""
start_date = '2023-01-01'
end_date = '2024-12-31'

start_dt = datetime.strptime(start_date, '%Y-%m-%d')
end_dt = datetime.strptime(end_date, '%Y-%m-%d')
# 间隔1天
print("=== 间隔1天 ===")
current = start_dt
start_list_daily = []
end_list_daily = []

while current <= end_dt:
    period_end = add_months(current, 15)
    if period_end <= end_dt:
        start_list_daily.append(current.strftime('%Y-%m-%d'))
        end_list_daily.append(period_end.strftime('%Y-%m-%d'))
    current += timedelta(days=1)

print(f"start_list_daily (前10个): {start_list_daily[:10]}")
print(f"end_list_daily (前10个): {end_list_daily[:10]}")
print(f"start_list_daily (后5个): {start_list_daily[-5:]}")
print(f"end_list_daily (后5个): {end_list_daily[-5:]}")
print(f"总共 {len(start_list_daily)} 个区间")

# 间隔1个月
print("\n=== 间隔1个月 ===")
current = start_dt
start_list_monthly = []
end_list_monthly = []

while current <= end_dt:
    period_end = add_months(current, 15)
    if period_end > end_dt:
        break
    start_list_monthly.append(current.strftime('%Y-%m-%d'))
    end_list_monthly.append(period_end.strftime('%Y-%m-%d'))
    current = add_months(current, 1)

print(f"start_list_monthly: {start_list_monthly}")
print(f"end_list_monthly: {end_list_monthly}")
print(f"总共 {len(start_list_monthly)} 个区间")

# 如果需要完整的列表输出，可以取消下面的注释：
# print("\n=== 完整列表 ===")
# print("间隔1天的所有start:", start_list_daily)
# print("间隔1天的所有end:", end_list_daily)
# print("间隔1个月的所有start:", start_list_monthly)
# print("间隔1个月的所有end:", end_list_monthly)
"""