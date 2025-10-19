import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import pulp

Cost = {"M1": 1941.8, "M2": 1941.8, "M3": 1911.0, "M4": 1911.0, "M5": 1911.0}

# Revenue & module composition
Revenue = {"S": 6.1, "M": 11.95, "L": 16.35}
Module_type= {
    'M1': {'S': 6,  'M': 6,  'L': 2},
    'M2': {'S': 10, 'M': 10, 'L': 0},
    'M3': {'S': 6,  'M': 6,  'L': 4},
    'M4': {'S': 12, 'M': 6,  'L': 2},
    'M5': {'S': 4,  'M': 4,  'L': 6}
}

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"
Daily_demand_file = os.path.join(data_dir, "daily_package_counts_by_size_full_range.csv")

Daily_demand = pd.read_csv(Daily_demand_file)
Safety_factor = 1 + norm.ppf(0.995) * 0.25
Max_packages = {
    "S": float(np.max(Daily_demand["S"] * Safety_factor)),
    "M": float(np.max(Daily_demand["M"] * Safety_factor)),
    "L": float(np.max(Daily_demand["L"] * Safety_factor))
}


model = pulp.LpProblem("Locker_Module_Design", pulp.LpMinimize)

# Decision variables
module_names = ["M1","M2","M3","M4","M5"]
M = pulp.LpVariable.dicts("M", module_names, lowBound=0, cat="Integer")

# Related variables
locker_counts = {}
for size in ["S","M","L"]:
    locker_counts[size] = pulp.lpSum([Module_type[m][size] * M[m] for m in module_names])

total_modules_width = pulp.lpSum([M[m] for m in module_names])

# Objective
Total_Revenue = pulp.lpSum([locker_counts[s] * Revenue[s] for s in ["S","M","L"]])

Lambda = 1/(5*13*28)
Total_Module_Cost = pulp.lpSum([M[m] * Cost[m] for m in module_names])

model += Total_Module_Cost, "Total_Cost"

# Constraints
# 1) 每种尺寸的 locker 数量 >= Max_packages（你的“Total_modules >= Max_packages”）
for s in ["S","M","L"]:
    model += locker_counts[s] >= Max_packages[s], f"Demand_cover_{s}"
# 2) 交互模块距离线性化：6 * M1 >= total_modules_width  （线性替代 ceil）
model += 6 * M["M1"] >= total_modules_width, "Interactive_module_spacing"
# 3) 非负性（通过变量域已保证）

# 可选：你可能想把总模块数设上限，避免解无限大（例如预算或场地限制）
# model += total_modules_width <= 1000, "Capacity_upper_bound"

# 求解
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
result_status = model.solve(solver)

print("Status:", pulp.LpStatus[result_status])

# 输出解
solution = {m: int(pulp.value(M[m])) for m in module_names}
print("Module counts (solution):", solution)

# 计算并报告各项指标的数值
S_count = sum(Module_type[m]['S'] * solution[m] for m in module_names)
M_count = sum(Module_type[m]['M'] * solution[m] for m in module_names)
L_count = sum(Module_type[m]['L'] * solution[m] for m in module_names)
module_cost_val = sum(solution[m] * Cost[m] for m in module_names)

Slab_width = sum(solution[m] for m in module_names) *2

print(f"Lockers S/M/L = {S_count}/{M_count}/{L_count}")
print("Slab width = ", Slab_width)
print(f"Module cost = {module_cost_val:.2f}")