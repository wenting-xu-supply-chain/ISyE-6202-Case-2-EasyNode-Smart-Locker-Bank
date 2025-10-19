import pandas as pd
import numpy as np
from scipy.stats import norm
import pulp
import math

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

file_name = "daily_package_counts_by_size_full_range.csv"
Daily_demand = pd.read_csv(file_name)

Safety_factor = 1 + norm.ppf(0.995) * 0.25
period_length = 28
num_periods = int(math.ceil(len(Daily_demand) / period_length))

Max_packages_periods = []
for p in range(num_periods):
    start_idx = p * period_length
    end_idx = min((p + 1) * period_length, len(Daily_demand))
    
    period_data = Daily_demand.iloc[start_idx:end_idx]
    
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


results = []
# 用于存储上一个周期最优解的变量，初始化为0
previous_solution = {m: 0 for m in module_names} 

for p_data in Max_packages_periods:
    period = p_data['period']
    Max_packages = {s: p_data[s] for s in locker_sizes}
    
    model = pulp.LpProblem(f"Locker_Design_P{period}", pulp.LpMaximize)

    # 决策变量: M[m] - 模块m的数量 (整数)
    M = pulp.LpVariable.dicts("M", module_names, lowBound=0, cat="Integer")

    # --- 辅助变量和成本线性化 (P2+) ---
    if period > 1:
        # Y_add[m]: 增加的模块数 (整数)
        Y_add = pulp.LpVariable.dicts("Y_add", module_names, lowBound=0, cat="Integer")
        # Y_remove[m]: 移除的模块数 (整数)
        Y_remove = pulp.LpVariable.dicts("Y_remove", module_names, lowBound=0, cat="Integer")
        
        # 线性化约束: M[m] - M[m]_prev = Y_add[m] - Y_remove[m]
        for m in module_names:
            model += M[m] - previous_solution[m] == Y_add[m] - Y_remove[m], f"Change_def_{m}_P{period}"

    # --- 目标函数组件 ---
    
    # Locker counts (Linear expression)
    locker_counts = {}
    for size in locker_sizes:
        locker_counts[size] = pulp.lpSum([Module_type[m][size] * M[m] for m in module_names])
    
    # 1. 总收入 (Total_Revenue): 收入是每天的，需要乘以周期天数
    Total_Revenue = pulp.lpSum([locker_counts[s] * Revenue[s] * PERIOD_DAYS for s in locker_sizes])
    
    # 2. 总人体工程学成本 (Total_Ergonomic_Cost): 成本是每天的，需要乘以周期天数
    Total_Ergonomic_Cost = pulp.lpSum([Ergonomic_cost[s] * locker_counts[s] * PERIOD_DAYS for s in locker_sizes])

    # 3. 总模块成本 (Total_Module_Cost): 这项成本只发生一次（P1）或一次变动（P2+）
    if period == 1:
        # P1 成本: 全额安装成本
        Total_Module_Cost = pulp.lpSum([M[m] * Cost[m] for m in module_names])
    else:
        # P2+ 成本: 模块变化成本 C * (|M[m] - prev_M[m]|) = C * (Y_add[m] + Y_remove[m])
        Total_Module_Cost = pulp.lpSum([Cost[m] * (Y_add[m] + Y_remove[m]) for m in module_names])

    # 目标: 最大化 Total_Profit (周期内总利润)
    model += Total_Revenue - (Total_Module_Cost + Total_Ergonomic_Cost), f"Total_Profit_P{period}"

    # --- 约束 ---
    
    # C1: 需求覆盖 (Locker Capacity >= Max Demand)
    for s in locker_sizes:
        model += locker_counts[s] >= Max_packages[s], f"Demand_cover_{s}_P{period}"

    # C2: 交互模块约束 (Linearization of M1 = ceil(Total_width / 6))
    Total_width = pulp.lpSum([M[m] for m in module_names])
    # 6 * M1 >= Total_width
    model += 6 * M["M1"] >= Total_width, f"Interactive_spacing_c1_P{period}"
    # 6 * M1 <= Total_width + 5 (5 = 6 - 1)
    model += 6 * M["M1"] <= Total_width + 5, f"Interactive_spacing_c2_P{period}"

    # C3: 地基上限
    slab = 172
    model += pulp.lpSum([M[m] for m in module_names]) <= slab/2, f"Cumulative_Slab_limit_P{period}"
    
    # 求解
    # 使用 CBC 求解器，设置求解时间限制
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    result_status = model.solve(solver)
    
    # 提取结果
    if pulp.LpStatus[result_status] == "Optimal":
        solution = {m: int(pulp.value(M[m])) for m in module_names}
        
        # 计算和存储各项指标
        s_count = int(pulp.value(locker_counts['S']))
        m_count = int(pulp.value(locker_counts['M']))
        l_count = int(pulp.value(locker_counts['L']))
        
        # 计算变动情况
        m_change = {m: solution[m] - previous_solution[m] for m in module_names}
        
        results.append({
            'Period': period,
            'S_Demand': Max_packages['S'], 'M_Demand': Max_packages['M'], 'L_Demand': Max_packages['L'],
            'M1_prev': previous_solution['M1'], 'M2_prev': previous_solution['M2'], 'M3_prev': previous_solution['M3'], 'M4_prev': previous_solution['M4'], 'M5_prev': previous_solution['M5'],
            'M1': solution['M1'], 'M2': solution['M2'], 'M3': solution['M3'], 'M4': solution['M4'], 'M5': solution['M5'],
            'M1_change': m_change['M1'], 'M2_change': m_change['M2'], 'M3_change': m_change['M3'], 'M4_change': m_change['M4'], 'M5_change': m_change['M5'],
            'S_lockers': s_count, 'M_lockers': m_count, 'L_lockers': l_count,
            'Total_Revenue': pulp.value(Total_Revenue), 'Total_Module_Cost': pulp.value(Total_Module_Cost), 
            'Total_Ergonomic_Cost': pulp.value(Total_Ergonomic_Cost), 'Total_Profit': pulp.value(model.objective)
        })
        
        # 更新 previous_solution
        previous_solution = solution
    else:
        print(f"Warning: Period {period} did not find an optimal solution. Status: {pulp.LpStatus[result_status]}")
        # 如果模型未找到解，为避免后续周期错误，可以选择保留上一个周期的配置
        previous_solution = previous_solution # 保持不变

# --- 4. 最终输出 ---
results_df = pd.DataFrame(results)
results_df['Cumulative_Profit'] = results_df['Total_Profit'].cumsum()
output_file = "dynamic_periodic_optimization_results.csv"
results_df.to_csv(output_file, index=False)

print(f"\n--- 动态优化模型已构建 ---")
print(f"该模型已在 {len(Max_packages_periods)} 个周期上迭代，并考虑到每个周期（P2+）模块变动的成本。")
print(f"结果已保存到 {output_file} 文件。")