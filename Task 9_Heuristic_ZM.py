import pandas as pd
import os
import numpy as np
from scipy.stats import norm

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"

# 读取数据
Locker_prob_file = os.path.join(data_dir, "Locker Demand Distribution.csv")
Module_design_file = os.path.join(data_dir, "Module design.csv")
Daily_demand_file = os.path.join(data_dir, "simulated_daily_demand.csv")

Locker_prob = pd.read_csv(Locker_prob_file)
Module_design = pd.read_csv(Module_design_file)
Daily_demand = pd.read_csv(Daily_demand_file)

Locker_key = list(Locker_prob["Locker"])
Module_key = list(Module_design["Module"])

Demand_distribution = Locker_prob.set_index('Locker')['Demand_probability'].to_dict()
Module_counts_matrix = Module_design.set_index('Module')[Locker_key]

# Heuristic_module 返回 DataFrame（所有比例的计算结果）
def Heuristic_module(Daily_demand_max_in_period):
    Slab_limit = 70
    Target_counts = {k: v * Daily_demand_max_in_period for k, v in Demand_distribution.items()}
    Target_N_locker = np.array([Target_counts[k] for k in Locker_key], dtype=float)

    M2_proportion = np.arange(0.0, 1.01, step=0.01)
    records=[]

    for p in M2_proportion:
        N_module = {m: 0 for m in Module_key}
        Current_N_locker = np.zeros_like(Target_N_locker)
        Total_modules = 0
        unreachable = False
        note = None

        # Step 1: use M2 to reach p * target
        if "M2" in Module_counts_matrix.index:
            M2_vec = Module_counts_matrix.loc["M2", Locker_key].values.astype(float)
            target_partial = Target_N_locker * p
            positive_mask = M2_vec > 0
            needed_m2 = 0
            if positive_mask.any():
                req = np.ceil(np.maximum(0, target_partial[positive_mask] - Current_N_locker[positive_mask]) / M2_vec[positive_mask])
                needed_m2 = int(req.max()) if req.size>0 else 0

            if needed_m2 > 0:
                if Total_modules + needed_m2 > Slab_limit:
                    unreachable = True
                    note = "slab_exceeded_on_M2"
                else:
                    Current_N_locker += M2_vec * needed_m2
                    N_module['M2'] = needed_m2
                    Total_modules += needed_m2
        else:
            note = "M2_missing"

        # Step 1b: ensure at least one M1
        if "M1" in Module_counts_matrix.index and N_module.get('M1', 0) < 1:
            if Total_modules + 1 <= Slab_limit:
                N_module['M1'] += 1
                M1_vec = Module_counts_matrix.loc["M1", Locker_key].values.astype(float)
                Current_N_locker += M1_vec
                Total_modules += 1
            else:
                unreachable = True
                note = "slab_exceeded_on_initial_M1"
        elif "M1" not in Module_counts_matrix.index:
            note = "M1_missing"

        # Step 2: greedy fill largest shortage
        non_m1_counter = 0
        max_iter = 10000
        iter_count = 0

        while np.any(Target_N_locker - Current_N_locker > 0) and iter_count < max_iter and not unreachable:
            Difference = Target_N_locker - Current_N_locker
            largest_idx = int(np.argmax(Difference))
            locker_needed = Locker_key[largest_idx]

            # 找最合适 module
            best_module = None
            max_number = -1
            for module in Module_key:
                number = Module_counts_matrix.loc[module, locker_needed]
                if number > max_number:
                    max_number = number
                    best_module = module

            if max_number <= 0:
                unreachable = True
                note = "no_module_covers_locker"
                break

            needed = int(np.ceil(Difference[largest_idx] / max_number))

            # slab check
            if Total_modules + needed > Slab_limit:
                unreachable = True
                note = f"slab_exceeded_on_{best_module}_add"
                break

            N_module[best_module] += needed
            Total_modules += needed
            Module_vec = Module_counts_matrix.loc[best_module, Locker_key].values.astype(float)
            Current_N_locker += Module_vec * needed

            if best_module != "M1":
                non_m1_counter += needed
                if non_m1_counter >= 6 and "M1" in Module_counts_matrix.index:
                    to_add_m1 = non_m1_counter // 6
                    if Total_modules + to_add_m1 <= Slab_limit:
                        N_module['M1'] += to_add_m1
                        M1_vec = Module_counts_matrix.loc["M1", Locker_key].values.astype(float)
                        Current_N_locker += M1_vec * to_add_m1
                        Total_modules += to_add_m1
                        non_m1_counter -= to_add_m1 * 6

            iter_count += 1

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
            "modules_used": {k:int(v) for k,v in N_module.items() if v>0},
            "note": note
        })

    df = pd.DataFrame(records)
    df_valid = df[df['no_shortage'] == True].copy()
    return df_valid


# 整个 year-period 循环，生成最终表格
final_results = []

Safety_factor = 1 + norm.ppf(0.995)*0.25
max_demand_per_period = Daily_demand.groupby(["year","period"])['final_daily_demand'].max() * Safety_factor

for (year, period), max_demand in max_demand_per_period.items():
    df_valid = Heuristic_module(max_demand)
    if df_valid.empty:
        continue
    best = df_valid.loc[df_valid['sum_overfill'].idxmin()]
    modules_used = best['modules_used']
    final_lockers = best['final_lockers']

    # locker 分组
    S_lockers = [lk for lk in Locker_key if lk.startswith("S")]
    M_lockers = [lk for lk in Locker_key if lk.startswith("M")]
    L_lockers = [lk for lk in Locker_key if lk.startswith("L")]

    Target_counts = {k: v * max_demand for k, v in Demand_distribution.items()}

    row = {
        "year": year,
        "period": period,
        "Total Module": best['Total_modules'],
        "M1": modules_used.get("M1", 0),
        "M2": modules_used.get("M2", 0),
        "M3": modules_used.get("M3", 0),
        "M4": modules_used.get("M4", 0),
        "M5": modules_used.get("M5", 0),
        "S": final_lockers[[Locker_key.index(lk) for lk in S_lockers]].sum(),
        "S Target": np.array([Target_counts[lk] for lk in S_lockers]).sum(),
        "M": final_lockers[[Locker_key.index(lk) for lk in M_lockers]].sum(),
        "M Target": np.array([Target_counts[lk] for lk in M_lockers]).sum(),
        "L": final_lockers[[Locker_key.index(lk) for lk in L_lockers]].sum(),
        "L Target": np.array([Target_counts[lk] for lk in L_lockers]).sum(),
    }
    final_results.append(row)

periodic_module_assignment_df = pd.DataFrame(final_results)
periodic_module_assignment_df.to_csv(os.path.join(data_dir, "periodic_module_assignment.csv"), index=False)