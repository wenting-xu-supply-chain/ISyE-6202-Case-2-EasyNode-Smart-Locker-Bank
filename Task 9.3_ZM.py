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

M4_proportion = np.arange(0.0, 1.01, step=0.01)
records=[]

for p in M4_proportion:
    N_module = {m: 0 for m in Module_key}
    Current_N_locker = np.zeros_like(Target_N_locker)
    Total_modules = 0
    unreachable = False

    # Step 1: 用 M4 达到 p * target
    M4_vec = Module_counts_matrix.loc["M4", Locker_key].values.astype(float)
    target_partial = Target_N_locker * p
    positive_mask = M4_vec > 0
    needed_m4 = 0
    if positive_mask.any():
        req = np.ceil(np.maximum(0, target_partial[positive_mask] - Current_N_locker[positive_mask]) / M4_vec[positive_mask])
        needed_m4 = int(req.max()) if req.size>0 else 0

    if needed_m4 > 0:
        Current_N_locker += M4_vec * needed_m4
        N_module['M4'] = needed_m4
        Total_modules += needed_m4

    # Step 2: 补齐到 full target
    max_iter = 10000
    iter_count = 0
    while np.any(Target_N_locker - Current_N_locker > 0) and iter_count < max_iter:
        Difference = Target_N_locker - Current_N_locker
        largest_idx = int(np.argmax(Difference))
        locker_needed = Locker_key[largest_idx]

        best_module = None
        max_number = -1
        for module in Module_key:
            number = Module_counts_matrix.loc[module, locker_needed]
            if number > max_number:
                max_number = number
                best_module = module

        if max_number <= 0:
            unreachable = True
            break

        needed = int(np.ceil(Difference[largest_idx] / max_number))
        N_module[best_module] += needed
        Total_modules += needed
        Module_vec = Module_counts_matrix.loc[best_module, Locker_key].values.astype(float)
        Current_N_locker += Module_vec * needed

        iter_count += 1

    # 判断是否每类都 >= target
    no_shortage = (not unreachable) and np.all(Current_N_locker >= Target_N_locker - 1e-9)
    overfill_vec = np.maximum(0, Current_N_locker - Target_N_locker)
    sum_overfill = float(np.sum(overfill_vec)) if no_shortage else np.nan

    records.append({
        "M4_share": p,
        "Total_modules": (Total_modules if not unreachable else np.nan),
        "final_lockers": Current_N_locker.copy(),
        "overfill_vec": overfill_vec,
        "sum_overfill": sum_overfill,
        "no_shortage": no_shortage,
        "modules_used": {k:int(v) for k,v in N_module.items() if v>0}
    })

df = pd.DataFrame(records)

# 只在 no_shortage == True 的行里找最小 sum_overfill
df_valid = df[df['no_shortage'] == True].copy()
if df_valid.empty:
    print("No solution")
else:
    best = df_valid.loc[df_valid['sum_overfill'].idxmin()]
    print(f"M4 Proportion: {best['M4_share']*100:.0f}%")
    print(f"Total modules: {best['Total_modules']}")
    print("Modules used:", best['modules_used'])
    for lk, final_val, targ, of in zip(Locker_key, best['final_lockers'], Target_N_locker, best['overfill_vec']):
        print(f"  {lk}: {final_val:.0f} / {targ:.0f}  overfill={of:.0f}")
    print(f"sum_overfill = {best['sum_overfill']:.0f}")