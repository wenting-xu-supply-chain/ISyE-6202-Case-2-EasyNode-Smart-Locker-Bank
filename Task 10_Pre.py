import pandas as pd
import numpy as np
from scipy.stats import norm
import pulp
import math
import os

# Daily Demand by Size
data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2\Task_3_Outcome"
Packages_arrival_file = os.path.join(data_dir, "task3_demand_instance_random_mode.csv")
Packages_arrival = pd.read_csv(Packages_arrival_file)
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

# Max Demand in period
Max_packages_periods = []
num_periods = int(math.ceil(len(Daily_demand_by_size) / 28))
Safety_factor = 1 + norm.ppf(0.995) * 0.25
for p in range(num_periods):
    start_idx = p * 28
    end_idx = min((p + 1) * 28, len(Daily_demand_by_size))
    
    period_data = Daily_demand_by_size.iloc[start_idx:end_idx]
    
    if not period_data.empty:
        max_s = np.max(period_data["S"]) * Safety_factor
        max_m = np.max(period_data["M"]) * Safety_factor
        max_l = np.max(period_data["L"]) * Safety_factor
        
        Max_packages_periods.append({
            "period": p + 1,
            "S": math.ceil(max_s),
            "M": math.ceil(max_m),
            "L": math.ceil(max_l)
        })

Max_packages_df = pd.DataFrame(Max_packages_periods)
Max_packages_df.to_csv("periodic_max_demand.csv", index=False)

Max_packages_periods_df = pd.read_csv("periodic_max_demand.csv")
Max_packages_periods = Max_packages_periods_df.to_dict('records')