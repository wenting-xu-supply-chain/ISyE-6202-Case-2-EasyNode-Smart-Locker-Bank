import pandas as pd
import numpy as np
import os

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"
Periodic_optimization_results_df = os.path.join(data_dir, "dynamic_periodic_optimization_results.csv")
Periodic_optimization_results = pd.read_csv(Periodic_optimization_results_df)
Daily_demand_by_size_df = os.path.join(data_dir, "Daily_demand_by_size.csv")
Daily_demand_by_size = pd.read_csv(Daily_demand_by_size_df)

# Demand
df_demand = Daily_demand_by_size.rename(columns={
    'S': 'S_demand',
    'M': 'M_demand',
    'L': 'L_demand'
})

df_demand['date'] = pd.to_datetime(df_demand['date'])
df_demand.sort_values('date', inplace=True)

start_date = df_demand['date'].min()
df_demand['Period'] = (df_demand['date'] - start_date).dt.days // 28 + 1

# Supply
df_capacity = Periodic_optimization_results[['Period', 'S', 'M', 'L']]
df_capacity = df_capacity.rename(columns={
    'S': 'S_capacity',
    'M': 'M_capacity',
    'L': 'L_capacity'
})

df_merged = pd.merge(df_demand, df_capacity, on='Period', how='left')

# Utilization = Daily Demand / Periodic Capacity
for size in ['S', 'M', 'L']:
    demand_col = f"{size}_demand"
    capacity_col = f"{size}_capacity"
    util_col = f"{size}_utilization"
    
    df_merged[util_col] = df_merged[demand_col] / df_merged[capacity_col]

output_columns = [
    'date', 
    'Period', 
    'S_demand', 'S_capacity', 'S_utilization',
    'M_demand', 'M_capacity', 'M_utilization',
    'L_demand', 'L_capacity', 'L_utilization'
]

output_file = 'daily_utilization.csv'
df_utilization = df_merged[output_columns]
df_utilization.to_csv(output_file, index=False)

# Line chart
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(df_utilization['date'], df_utilization['S_utilization'], label='S')
plt.plot(df_utilization['date'], df_utilization['M_utilization'], label='M')
plt.plot(df_utilization['date'], df_utilization['L_utilization'], label='L')

avg_S = np.percentile(df_utilization['S_utilization'], 80)
avg_M = np.percentile(df_utilization['M_utilization'], 80)
avg_L = np.percentile(df_utilization['L_utilization'], 80)

plt.axhline(y=avg_S, color='blue', linestyle='--', alpha=0.7, label=f'S avg ({avg_S:.2f})')
plt.axhline(y=avg_M, color='orange', linestyle='--', alpha=0.7, label=f'M avg ({avg_M:.2f})')
plt.axhline(y=avg_L, color='green', linestyle='--', alpha=0.7, label=f'L avg ({avg_L:.2f})')

plt.title('Daily Utilization by Size')
plt.xlabel('Date')
plt.ylabel('Utilization (Demand / Capacity)')
plt.legend(title='Size', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()