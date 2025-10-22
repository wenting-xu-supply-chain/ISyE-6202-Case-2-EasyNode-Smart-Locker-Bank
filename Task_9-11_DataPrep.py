import pandas as pd
import numpy as np
from scipy.stats import norm
import pulp
import math
import os

# Peak Daily Demand by Size
data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"
Packages_arrival_file = os.path.join(data_dir, "task3_demand_instance_steady_mode.csv")
Packages_arrival = pd.read_csv(Packages_arrival_file)

def assign_size(ft3):
    if 1 <= ft3 <= 3:
        return 'S'
    elif 4 <= ft3 <= 9:
        return 'M'
    else:
        return 'L'

Packages_arrival['real_deposit_dt'] = pd.to_datetime(Packages_arrival['real_deposit_dt'], errors='coerce')
Packages_arrival['real_pickup_dt'] = pd.to_datetime(Packages_arrival['real_pickup_dt'], errors='coerce')

Packages_arrival['size'] = Packages_arrival['min_size_ft3'].apply(assign_size)

# 创建“存入”事件：时间, 尺寸, 变化量 (+1)
deposits = Packages_arrival[['real_deposit_dt', 'size']].copy()
deposits = deposits.rename(columns={'real_deposit_dt': 'time'})
deposits['change'] = 1

# 创建“取出”事件：时间, 尺寸, 变化量 (-1)
pickups = Packages_arrival[['real_pickup_dt', 'size']].copy()
pickups = pickups.rename(columns={'real_pickup_dt': 'time'})
pickups['change'] = -1

events_df = pd.concat([deposits, pickups], ignore_index=True)

# 丢弃时间缺失的事件（NaT）
events_df = events_df[events_df['time'].notna()].copy()

# 将时间排序（如果存在相同时间点，先处理 deposit 再 pickup 可以影响峰值；这里按 change 降序确保 +1 先于 -1）
events_df.sort_values(by=['time', 'change'], ascending=[True, False], inplace=True)
events_df.reset_index(drop=True, inplace=True)

# 初始化
current_counts = {'S': 0, 'M': 0, 'L': 0}
daily_peak_storage = {}
current_date = None
current_day_peak = None

for row in events_df.itertuples(index=False):
    event_time = row.time
    event_date = event_time.date()
    size = row.size
    change = row.change

    # 初次设置 current_date 和 current_day_peak（取当前 counts 的快照作为当天起始值）
    if current_date is None:
        current_date = event_date
        current_day_peak = {
            'S': current_counts['S'],
            'M': current_counts['M'],
            'L': current_counts['L'],
            'Total': current_counts['S'] + current_counts['M'] + current_counts['L']
        }

    # 如果进入新的一天：保存前一天的峰值，并为新的一天初始化峰值为当前 counts（事件尚未被处理）
    if event_date != current_date:
        daily_peak_storage[current_date] = current_day_peak.copy()
        current_date = event_date
        current_day_peak = {
            'S': current_counts['S'],
            'M': current_counts['M'],
            'L': current_counts['L'],
            'Total': current_counts['S'] + current_counts['M'] + current_counts['L']
        }

    # 处理事件（先更新 counts，再比较是否超过当天峰值）
    current_counts[size] += change
    current_total = current_counts['S'] + current_counts['M'] + current_counts['L']

    if current_total > current_day_peak['Total']:
        current_day_peak['S'] = current_counts['S']
        current_day_peak['M'] = current_counts['M']
        current_day_peak['L'] = current_counts['L']
        current_day_peak['Total'] = current_total

# 循环结束后，保存最后一天的峰值（如果有）
if current_date is not None and current_day_peak is not None:
    daily_peak_storage[current_date] = current_day_peak.copy()

# 把结果转换为 DataFrame 并保存（在循环外）
results_df = pd.DataFrame.from_dict(daily_peak_storage, orient='index')
results_df.index.name = 'Date'
results_df.reset_index(inplace=True)
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df.sort_values(by='Date', inplace=True)
Peak_daily_demand_by_size = results_df[['Date', 'S', 'M', 'L']]

data_dir_1 = "D:\Fall 2025\ISYE6335\Case\Case 2"
out_file = os.path.join(data_dir_1, "Peak_daily_demand_by_size.csv")
Peak_daily_demand_by_size.to_csv(out_file, index=False)

# Date
Packages_arrival['desired_deposit_dt'] = pd.to_datetime(Packages_arrival['desired_deposit_dt'])
Packages_arrival['date'] = Packages_arrival['desired_deposit_dt'].dt.date
# Size
bins = [0, 3, 9, float('inf')]
labels = ['S', 'M', 'L']
Packages_arrival['size_group'] = pd.cut(Packages_arrival['min_size_ft3'], bins=bins, labels=labels, right=True, include_lowest=True)
# Output
demand_counts = Packages_arrival.groupby(['date', 'size_group']).size().reset_index(name='count')
Daily_demand_by_size = demand_counts.pivot_table(
    index='date', 
    columns='size_group', 
    values='count', 
    fill_value=0 
).reset_index()
column_order = ['date', 'S', 'M', 'L']
Daily_demand_by_size = Daily_demand_by_size.reindex(columns=column_order, fill_value=0)
output_file = "Daily_demand_by_size.csv"
Daily_demand_by_size.to_csv(output_file, index=False)
