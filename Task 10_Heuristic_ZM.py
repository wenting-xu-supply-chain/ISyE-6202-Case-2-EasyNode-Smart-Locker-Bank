import pandas as pd
import numpy as np
from scipy.stats import norm
import pulp
import math
import os

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"
Max_packages_periods_df = os.path.join(data_dir, "periodic_mean_demand.csv")
Max_packages_periods = pd.read_csv(Max_packages_periods_df)
Max_packages_periods = Max_packages_periods.to_dict('records')

Daily_demand_by_size_df = os.path.join(data_dir, "Daily_demand_by_size.csv")
Daily_demand_by_size = pd.read_csv(Daily_demand_by_size_df)

# Parameters
Cost = {"M1": 1941.8, "M2": 1941.8, "M3": 1911.0, "M4": 1911.0, "M5": 1911.0}
Revenue = {"S": 6.1, "M": 11.95, "L": 16.35}
Module_type= {
    'M1': {'S': 6, 'M': 6, 'L': 2}, 'M2': {'S': 10, 'M': 10, 'L': 0},
    'M3': {'S': 6, 'M': 6, 'L': 4}, 'M4': {'S': 12, 'M': 6, 'L': 2},
    'M5': {'S': 4, 'M': 4, 'L': 6}
}
Ergonomic_cost = {'S': 0.67, 'M': 0.425, 'L': 0.55} 
module_names = ["M1","M2","M3","M4","M5"]
locker_sizes = ["S","M","L"]
PERIOD_DAYS = 28 

results = []
# 用于存储上一个周期最优解的变量，初始化为0
previous_solution = {m: 0 for m in module_names} 

for p_data in Max_packages_periods:
    period = p_data['period']
    Max_packages = {s: p_data[s] for s in locker_sizes}

    start_row = (period - 1) * PERIOD_DAYS
    end_row = period * PERIOD_DAYS
    period_daily_demand_df = Daily_demand_by_size.iloc[start_row:end_row]
    period_daily_demand_list = period_daily_demand_df.to_dict('records')
    actual_days_in_period = len(period_daily_demand_list)

    model = pulp.LpProblem(f"Locker_Design_P{period}", pulp.LpMaximize)

    # Decision variables: # of modules
    M = pulp.LpVariable.dicts("M", module_names, lowBound=0, cat="Integer")

    if period > 1:
        # Y_add[m]: # of modules added
        Y_add = pulp.LpVariable.dicts("Y_add", module_names, lowBound=0, cat="Integer")
        # Y_remove[m]: # of modules removed
        Y_remove = pulp.LpVariable.dicts("Y_remove", module_names, lowBound=0, cat="Integer")
        # M[m] - M[m]_prev = Y_add[m] - Y_remove[m]
        for m in module_names:
            model += M[m] - previous_solution[m] == Y_add[m] - Y_remove[m], f"Change_def_{m}_P{period}"

    # Objetive function
    # # of lockers
    locker_counts = {}
    for size in locker_sizes:
        locker_counts[size] = pulp.lpSum([Module_type[m][size] * M[m] for m in module_names])
    
    actual_rentals = {}
    for s in locker_sizes:
        actual_rentals[s] = pulp.LpVariable.dicts(
            f"ActualRentals_{s}_P{period}",
            range(actual_days_in_period),
            lowBound=0,
            cat="Continuous"
        )

    # Total_Revenue:
    Total_Revenue = pulp.lpSum(
        [actual_rentals[s][d] * Revenue[s]
         for s in locker_sizes
         for d in range(actual_days_in_period)]
    )
    
    # Total_Ergonomic_Cost:
    Total_Ergonomic_Cost = pulp.lpSum(
        [actual_rentals[s][d] * Ergonomic_cost[s]
         for s in locker_sizes
         for d in range(actual_days_in_period)]
    )

    # Total_Module_Cost: one-time
    if period == 1:
        # Setup
        Total_Module_Cost = pulp.lpSum([M[m] * Cost[m] for m in module_names])
    else:
        # Change: C * (Y_add[m] + Y_remove[m])
        Total_Module_Cost = pulp.lpSum([Cost[m] * (Y_add[m] + Y_remove[m]) for m in module_names])

    # Max: Total_Profit
    model += Total_Revenue - (Total_Module_Cost + Total_Ergonomic_Cost), f"Total_Profit_P{period}"

    # Constraint
    # C1: Locker Capacity >= Max Demand
    for s in locker_sizes:
        model += locker_counts[s] >= Max_packages[s], f"Demand_cover_{s}_P{period}"

    # C2: Linearization of M1 = ceil(Total_width / 6)
    Total_width = pulp.lpSum([M[m] for m in module_names])
    # 6 * M1 >= Total_width
    model += 6 * M["M1"] >= Total_width, f"Interactive_spacing_c1_P{period}"
    # 6 * M1 <= Total_width + 5 (5 = 6 - 1)
    model += 6 * M["M1"] <= Total_width + 5, f"Interactive_spacing_c2_P{period}"

    # C3: Slab
    slab = 202
    model += pulp.lpSum([M[m] for m in module_names]) <= slab/2, f"Cumulative_Slab_limit_P{period}"
    
    # C4:
    for d in range(actual_days_in_period):
        demand_today = period_daily_demand_list[d]
        for s in locker_sizes:
            # 约束a: 每日实际出租量 <= 总容量
            model += actual_rentals[s][d] <= locker_counts[s], f"Capacity_limit_{s}_{d}_P{period}"
            # 约束b: 每日实际出租量 <= 当日需求
            model += actual_rentals[s][d] <= demand_today[s], f"Demand_limit_{s}_{d}_P{period}"

    # Solution
    # 使用 CBC 求解器，设置求解时间限制
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=100)
    result_status = model.solve(solver)
    
    # 提取结果
    if pulp.LpStatus[result_status] == "Optimal":
        solution = {m: int(pulp.value(M[m])) for m in module_names}
        
        s_count = int(pulp.value(locker_counts['S']))
        m_count = int(pulp.value(locker_counts['M']))
        l_count = int(pulp.value(locker_counts['L']))
        
        m_change = {m: solution[m] - previous_solution[m] for m in module_names}
        
        revenue_val = pulp.value(Total_Revenue)
        module_cost_val = pulp.value(Total_Module_Cost)
        erg_cost_val = pulp.value(Total_Ergonomic_Cost)
        profit_val = pulp.value(model.objective)
      
        previous_solution = solution
    else:
        print(f"Warning: Period {period} did not find an optimal solution. Status: {pulp.LpStatus[result_status]}.")
        # 使用上一个周期的解
        solution = previous_solution 

        m_change = {m: 0 for m in module_names}

        s_count = int(sum(Module_type[m]['S'] * solution[m] for m in module_names))
        m_count = int(sum(Module_type[m]['M'] * solution[m] for m in module_names))
        l_count = int(sum(Module_type[m]['L'] * solution[m] for m in module_names))
        current_capacity = {'S': s_count, 'M': m_count, 'L': l_count}
        
        # Calculate profit
        revenue_val = 0
        erg_cost_val = 0
        for d in range(actual_days_in_period):
            demand_today = period_daily_demand_list[d]
            for s in locker_sizes:
                # 实际出租量 = min(容量, 当日需求)
                rented = min(current_capacity[s], demand_today[s])
                revenue_val += rented * Revenue[s]
                erg_cost_val += rented * Ergonomic_cost[s]
        module_cost_val = 0
        
        profit_val = revenue_val - (module_cost_val + erg_cost_val)

    results.append({
        'Period': period,
        'M1': solution['M1'], 'M2': solution['M2'], 'M3': solution['M3'], 'M4': solution['M4'], 'M5': solution['M5'],
        'M1_change': m_change['M1'], 'M2_change': m_change['M2'], 'M3_change': m_change['M3'], 'M4_change': m_change['M4'], 'M5_change': m_change['M5'],
        'S': s_count, 'M': m_count, 'L': l_count,
        'Total_Revenue': revenue_val, 
        'Total_Module_Cost': module_cost_val, 
        'Total_Ergonomic_Cost': erg_cost_val, 
        'Total_Profit': profit_val
    })

# Output
results_df = pd.DataFrame(results)
results_df['Cumulative_Profit'] = results_df['Total_Profit'].cumsum()
output_file = "dynamic_periodic_optimization_results.csv"
results_df.to_csv(output_file, index=False)