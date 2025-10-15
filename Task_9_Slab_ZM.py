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

Daily_demand_file = os.path.join(data_dir, "simulated_daily_demand.csv")
Daily_demand = pd.read_csv(Daily_demand_file)

Daily_demand_max = Daily_demand["final_daily_demand"].max()

Locker_key = list(Locker_prob["Locker"])
Module_key = list(Module_design["Module"])

Demand_distribution = Locker_prob.set_index('Locker')['Demand_probability'].to_dict()
Module_counts_matrix = Module_design.set_index('Module')[Locker_key]

Target_counts = {k: v * Daily_demand_max for k, v in Demand_distribution.items()}
Target_N_locker = np.array([Target_counts[k] for k in Locker_key], dtype=float)

M2_proportion = np.arange(0.0, 1.01, step=0.01)
records=[]

for p in M2_proportion:
    N_module = {m: 0 for m in Module_key}
    Current_N_locker = np.zeros_like(Target_N_locker)
    Total_modules = 0
    unreachable = False

    # Step 1: use M2 to reach p * target
    if "M2" not in Module_counts_matrix.index:
        records.append({
            "M2_share": p,
            "Total_modules": np.nan,
            "final_lockers": Current_N_locker.copy(),
            "overfill_vec": np.maximum(0, Current_N_locker - Target_N_locker),
            "sum_overfill": np.nan,
            "no_shortage": False,
            "modules_used": {}
        })
        continue

    M2_vec = Module_counts_matrix.loc["M2", Locker_key].values.astype(float)
    target_partial = Target_N_locker * p
    positive_mask = M2_vec > 0
    needed_M2 = 0
    if positive_mask.any():
        req = np.ceil(np.maximum(0, target_partial[positive_mask] - Current_N_locker[positive_mask]) / M2_vec[positive_mask])
        needed_M2 = int(req.max()) if req.size>0 else 0

    if needed_M2 > 0:
        Current_N_locker += M2_vec * needed_M2
        N_module['M2'] = needed_M2
        Total_modules += needed_M2

    if "M1" not in Module_counts_matrix.index:
        records.append({
            "M2_share": p,
            "Total_modules": np.nan,
            "final_lockers": Current_N_locker.copy(),
            "overfill_vec": np.maximum(0, Current_N_locker - Target_N_locker),
            "sum_overfill": np.nan,
            "no_shortage": False,
            "modules_used": {k:int(v) for k,v in N_module.items() if v>0},
            "note": "M1_missing"
        })
        continue
    else:
        if N_module.get('M1', 0) < 1:
            N_module['M1'] = N_module.get('M1', 0) + 1
            M1_vec = Module_counts_matrix.loc["M1", Locker_key].values.astype(float)
            Current_N_locker += M1_vec
            Total_modules += 1

    # Step 2: find the max difference locker, find the module with that max locker to reach full target
    # For every 6 non-M1, add 1 M1
    non_m1_counter = 0

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

        if best_module != "M1":
            N_module[best_module] += needed
            Total_modules += needed
            Module_vec = Module_counts_matrix.loc[best_module, Locker_key].values.astype(float)
            Current_N_locker += Module_vec * needed

            non_m1_counter += needed

            if non_m1_counter >= 6:
                to_add_m1 = non_m1_counter // 6
                N_module['M1'] = N_module.get('M1', 0) + to_add_m1
                M1_vec = Module_counts_matrix.loc["M1", Locker_key].values.astype(float)
                Current_N_locker += M1_vec * to_add_m1
                Total_modules += to_add_m1
                non_m1_counter -= to_add_m1 * 6
        else:
            N_module['M1'] = N_module.get('M1', 0) + needed
            Module_vec = Module_counts_matrix.loc['M1', Locker_key].values.astype(float)
            Current_N_locker += Module_vec * needed
            Total_modules += needed

        iter_count += 1

    # 判断是否每类都 >= target
    no_shortage = (not unreachable) and np.all(Current_N_locker >= Target_N_locker - 1e-9)
    overfill_vec = np.maximum(0, Current_N_locker - Target_N_locker)
    sum_overfill = float(np.sum(overfill_vec)) if no_shortage else np.nan

    records.append({
        "M2_share": p,
        "Total_modules": (Total_modules if not unreachable else np.nan),
        "final_lockers": Current_N_locker.copy(),
        "overfill_vec": overfill_vec,
        "sum_overfill": sum_overfill,
        "no_shortage": no_shortage,
        "modules_used": {k:int(v) for k,v in N_module.items() if v>0}
    })

df = pd.DataFrame(records)
df_valid = df[df['no_shortage'] == True].copy()
if df_valid.empty:
    print("No solution (no p satisfies no-shortage with M1 rule).")
else:
    best = df_valid.loc[df_valid['sum_overfill'].idxmin()]
    print(f"M2 Proportion: {best['M2_share']*100:.0f}%")
    print(f"Total modules: {best['Total_modules']}")
    print("Modules used:", best['modules_used'])
    for lk, final_val, targ, of in zip(Locker_key, best['final_lockers'], Target_N_locker, best['overfill_vec']):
        print(f"  {lk}: {final_val:.0f} / {targ:.0f}  overfill={of:.0f}")
    print(f"sum_overfill = {best['sum_overfill']:.0f}")

Slab_Height = 15
Slab_Width = best['Total_modules']*2
print(f"Slab: Height = {Slab_Height}, Width = {Slab_Width}")
