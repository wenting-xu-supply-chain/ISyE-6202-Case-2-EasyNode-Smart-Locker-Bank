import pandas as pd
import numpy as np
from scipy.stats import norm
import pulp
import math
import os

# 1. Loading
data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2" 

# 文件 1: 用于计算 目标函数 (利润)
Daily_demand_by_size_df = os.path.join(data_dir, "Daily_demand_by_size.csv")
Daily_demand_by_size = pd.read_csv(Daily_demand_by_size_df)

# 文件 2: 用于计算 约束 C1 (服务水平)
Peak_daily_demand_by_size_df = os.path.join(data_dir, "Peak_daily_demand_by_size.csv")
Peak_daily_demand_by_size = pd.read_csv(Peak_daily_demand_by_size_df)

# 2. Parameters
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

SLAB_WIDTH = 48
MODULE_LIMIT = SLAB_WIDTH / 2 

# 3. Optimization with loop
results = []
previous_solution = {m: 0 for m in module_names} 

infeasible_periods = []

# 根据 Daily 文件的长度，自动计算总周期数
total_periods = math.ceil(len(Daily_demand_by_size) / PERIOD_DAYS)
print(f"数据加载完毕。将开始计算 {total_periods} 个周期...")

for period in range(1, total_periods + 1):
    
    start_row = (period - 1) * PERIOD_DAYS
    end_row = period * PERIOD_DAYS

    # 1. 为约束 C1 加载 PEAK 数据 (来自 Peak_daily_demand_by_size.csv)
    peak_period_daily_demand_df = Peak_daily_demand_by_size.iloc[start_row:end_row]
    
    # 2. 为利润计算 (C4 和 else) 加载 DAILY 数据 (来自 Daily_demand_by_size.csv)
    period_daily_demand_df = Daily_demand_by_size.iloc[start_row:end_row]
    period_daily_demand_list = period_daily_demand_df.to_dict('records')
    actual_days_in_period = len(period_daily_demand_list)
    
    # 如果数据不完整 (例如最后一个周期)，则跳过
    if actual_days_in_period == 0:
        continue

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

    # Total_Revenue: (基于 C4, C4 基于 Daily 数据)
    Total_Revenue = pulp.lpSum(
        [actual_rentals[s][d] * Revenue[s]
         for s in locker_sizes
         for d in range(actual_days_in_period)]
    )
    
    # Total_Ergonomic_Cost: (基于 C4, C4 基于 Daily 数据)
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
        # Change: C * Y_add[m]
        Total_Module_Cost = pulp.lpSum([Cost[m] * Y_add[m] for m in module_names])

    # Max: Total_Profit
    model += Total_Revenue - (Total_Module_Cost + Total_Ergonomic_Cost), f"Total_Profit_P{period}"

    # Constraint
    
    # C1: Daily Service Level >= 99.5% (基于 PEAK 数据)
    Z_score = norm.ppf(0.995)
    CV = 0.25
    Safety_Factor = 1 + Z_score * CV
    max_daily = peak_period_daily_demand_df[locker_sizes].max()
    for s in locker_sizes:
        required_capacity = math.ceil(max_daily[s] * Safety_Factor)
        model += locker_counts[s] >= required_capacity, f"Service_Level_cover_{s}_P{period}"

    # C2: Linearization of M1
    model += M["M1"] >= 1, "At_least_one_M1"

    # C3: Slab
    model += pulp.lpSum([M[m] for m in module_names]) <= MODULE_LIMIT, f"Cumulative_Slab_limit_P{period}"
    
    # C4: 利润约束 (基于 DAILY 数据)
    for d in range(actual_days_in_period):
        demand_today = period_daily_demand_list[d] # <-- 来自 DAILY 数据
        for s in locker_sizes:
            # 约束a: 每日实际出租量 <= 总容量
            model += actual_rentals[s][d] <= locker_counts[s], f"Capacity_limit_{s}_{d}_P{period}"
            # 约束b: 每日实际出租量 <= 当日需求
            model += actual_rentals[s][d] <= demand_today[s], f"Demand_limit_{s}_{d}_P{period}"

    # Solution
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=100)
    result_status = model.solve(solver)
    
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
        # 发生 Infeasible (C1 和 C3 冲突)
        # 记录下这个周期的编号 ---
        infeasible_periods.append(period)

        print(f"Warning: Period {period} 发生约束冲突 (Infeasible). C1 (服务水平) 需求超出了 C3 (Slab限制)。")
        print(f"         将沿用 Period {period-1} 的容量。")
        
        # 使用上一个周期的解
        solution = previous_solution 
        m_change = {m: 0 for m in module_names} # 没有任何变动

        # 重新计算这一期的 S, M, L 容量 (因为它们不是 'Optimal' 解的一部分)
        s_count = int(sum(Module_type[m]['S'] * solution[m] for m in module_names))
        m_count = int(sum(Module_type[m]['M'] * solution[m] for m in module_names))
        l_count = int(sum(Module_type[m]['L'] * solution[m] for m in module_names))
        current_capacity = {'S': s_count, 'M': m_count, 'L': l_count}
        
        # 关键: 利润计算 (仍然基于 DAILY 数据)
        revenue_val = 0
        erg_cost_val = 0
        for d in range(actual_days_in_period):
            demand_today = period_daily_demand_list[d] # 来自 DAILY 数据
            for s in locker_sizes:
                # 实际出租量 = min(容量, 当日需求)
                rented = min(current_capacity[s], demand_today[s])
                revenue_val += rented * Revenue[s]
                erg_cost_val += rented * Ergonomic_cost[s]
        
        # 模块成本为 0, m_change = 0
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

print("\n" + "="*50)
print("代码运行完毕，逻辑已按要求修正。")
print(f"结果已保存到 {output_file}")

if infeasible_periods:
    print("\n以下周期 (Periods) 未能找到最优解 (由于C1服务水平与C3场地限制冲突):")
    print(infeasible_periods)
else:
    print("\n所有周期均已成功找到最优解。")
