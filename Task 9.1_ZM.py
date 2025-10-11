import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import math

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"

Locker_prob_file = os.path.join(data_dir, "Locker Demand Distribution.csv")
Module_design_file = os.path.join(data_dir, "Module design.csv")
Locker_prob = pd.read_csv(Locker_prob_file)
Module_design = pd.read_csv(Module_design_file)

Annual_demand_max = 150000
Periodic_share_max = 0.13
Daily_share_max = 0.2
Safety_factor = 1 + norm.ppf(0.995)*0.25 # 1 + z * CV
Daily_demand_max = Annual_demand_max * Periodic_share_max * Daily_share_max * Safety_factor

Locker_key = list(Locker_prob["Locker"])
Module_key = list(Module_design["Module"])

Demand_distribution = Locker_prob.set_index('Locker')['Demand_probability'].to_dict()
Module_counts_matrix = Module_design.set_index('Module')[Locker_key]

Target_counts = {k: v * Daily_demand_max for k, v in Demand_distribution.items()}
Target_N_locker = np.array([Target_counts[k] for k in Locker_key], dtype=float)

N_module = {m: 0 for m in Module_key}
Current_N_locker = np.zeros_like(Target_N_locker)
Total_modules = 0

# Step 1: 尽量使用 M4 填到 80%
Target_80 = Target_N_locker * 0.8
M4_number = Module_counts_matrix.loc["M4", Locker_key].values

# 如果 M4 在某些 locker 上为 0，说明用 M4 无法覆盖这些 locker 的 80% -> 不应仅依赖 M4
if np.any((M4_number == 0) & (Target_80 > 0)):
    print("注意：M4 在某些 locker 上数量为 0，不能单靠 M4 达到这些 locker 的 80%。将先用 M4 能增加的部分，再用别的 module 补齐。")

# 为了避免无限循环，计算一个合理的上限（例如 10000 个 module）
max_iter = 10000
iter_count = 0

needed_m4 = 0
positive_mask = M4_number > 0
if positive_mask.any():
    # 对每个 locker（M4能提供的）计算ceil((Target80 - current)/M4_number)
    req = np.ceil(np.maximum(0, Target_80[positive_mask] - Current_N_locker[positive_mask]) / M4_number[positive_mask])
    needed_m4 = int(req.max()) if req.size>0 else 0

if needed_m4 > 0:
    Current_N_locker += M4_number * needed_m4
    N_module['M4'] = needed_m4
    Total_modules += needed_m4

# Step 2: 用最有效的 module 补齐到 target
# 设定安全上限防止无限循环
max_iter = 10000
iter_count = 0

while np.any(Target_N_locker - Current_N_locker > 0) and iter_count < max_iter:
    Difference = Target_N_locker - Current_N_locker
    largest_difference_index = np.argmax(Difference)  # index of max remaining deficit
    if Difference[largest_difference_index] <= 0:
        break  
    
    locker_needed = Locker_key[largest_difference_index]
    # 找出对该 locker 提供最多数量的 module（若多个相同，取第一个）
    best_module = None
    max_number = -1
    for module in Module_key:
        number = Module_counts_matrix.loc[module, locker_needed]
        if number > max_number:
            max_number = number
            best_module = module

    # 如果所有模块对该 locker 都是 0，说明无法满足 -> 警告并跳出
    if max_number <= 0:
        print(f"无法通过任何 module 补充 locker '{locker_needed}'（所有 module 对应数量均为 0）。请检查 Module_design 或目标是否可达。")
        break

    # 计算一次需要多少个该模块才能把该 locker 的差补到 <=0（采用 ceil）
    needed = math.ceil(Difference[largest_difference_index] / max_number)
    N_module[best_module] += needed
    Total_modules += needed
    Module_number = Module_counts_matrix.loc[best_module, Locker_key].values.astype(float)
    Current_N_locker += Module_number * needed

    iter_count += 1

if iter_count >= max_iter:
    print("达到最大迭代次数，可能存在不可达的目标或逻辑问题。请检查模块组合与目标。")

final_solution = {k: int(v) for k, v in N_module.items() if v > 0}
print("Total modules:", Total_modules)
print("Modules used:", final_solution)
print("Current lockers achieved (by type):")
for lk, val in zip(Locker_key, Current_N_locker):
    print(f"  {lk}: {val:.0f} / target {Target_N_locker[Locker_key.index(lk)]:.0f}")
