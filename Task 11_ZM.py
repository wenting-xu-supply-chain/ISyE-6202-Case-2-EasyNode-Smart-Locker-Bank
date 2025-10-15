import pandas as pd
import os
import numpy as np
from scipy.stats import norm

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"

Locker_prob_file = os.path.join(data_dir, "Locker Demand Distribution.csv")
Module_design_file = os.path.join(data_dir, "Module design.csv")
Daily_demand_file = os.path.join(data_dir, "simulated_daily_demand.csv")
Periodic_module_file = os.path.join(data_dir, "periodic_module_assignment.csv")

Locker_prob = pd.read_csv(Locker_prob_file)
Module_design = pd.read_csv(Module_design_file)
Daily_demand = pd.read_csv(Daily_demand_file)
Periodic_module = pd.read_csv(Periodic_module_file)

# From Periodic to Daily
col_add = ['S', 'M', 'L']
col_key = ['year', 'period']

Periodic_supply = Periodic_module[col_key + col_add].copy()
Periodic_supply = Periodic_supply.drop_duplicates(subset = col_key)

Daily_demand_supply = pd.merge(
    Daily_demand,
    Periodic_supply,
    on = col_key,
    how = 'left'
)

#Calculate daily service level
Safety_factor = 1 + norm.ppf(0.995)*0.25
Daily_demand_value = Daily_demand_supply["final_daily_demand"]*Safety_factor

Demand_distribution = Locker_prob.set_index('Locker')['Demand_probability'].to_dict()
Daily_target = {k: np.ceil(v * Daily_demand_value) for k, v in Demand_distribution.items()}
Total_target = sum(Daily_target.values())


Service_level=[]
Shortage_S = np.maximum(Daily_target["S"] - Daily_demand_supply["S"], 0)
Shortage_M = np.maximum(Daily_target["M"] - Daily_demand_supply["M"], 0)
Shortage_L = np.maximum(Daily_target["L"] - Daily_demand_supply["L"], 0)
Total_shortage = Shortage_S + Shortage_M + Shortage_L

Daily_service_level = 1 - (Total_shortage / Total_target)
Daily_demand_supply["Service level"] = Daily_service_level

Daily_demand_supply_df = pd.DataFrame(Daily_demand_supply)
Daily_demand_supply_df.to_csv(os.path.join(data_dir, "Daily_demand_supply.csv"), index=False)