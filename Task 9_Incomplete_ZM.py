import pandas as pd
import os
import numpy as np
from scipy.stats import norm

#Load
data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"
Locker_prob_file = os.path.join(data_dir, "Locker Demand Distribution.csv")
Locker_prob = pd.read_csv(Locker_prob_file)
Module_design_file = os.path.join(data_dir, "Module design.csv")
Module_design = pd.read_csv(Module_design_file)

#Slab size
Annual_demand_max = 150000
Periodic_share_max = 0.13
Daily_share_max = 0.2
Safety_factor = 1 + norm.ppf(0.995)*0.25 # 1 + z * CV
Daily_demand_max = Annual_demand_max * Periodic_share_max * Daily_share_max * Safety_factor

Average_demand_size = (Locker_prob['Locker_size'] * Locker_prob['Demand_probability']).sum()
Total_size_required = Daily_demand_max*Average_demand_size

Average_module_size = Module_design['Total Size (ft3)'].mean()
Total_module_number = np.ceil(Total_size_required/Average_module_size) # round up??
print(Total_module_number)

#Heuristic
Slab_limit = Total_module_number

Module_key_col = Module_design["Module"]
Module_key = list(Module_key_col)

Locker_key_col = Locker_prob["Locker"]
Locker_key = list(Locker_key_col)

Demand_distribution = Locker_prob.set_index('Locker')['Demand_probability'].to_dict()

#缺一个demand的表！
Periodic_demand (Year,Period) = Annual_average_demand(Year)*Period_share(Period)*Safety_factor

Module_counts_matrix = Module_design.set_index('Module')[Locker_key]

def Heuristic_modular_periodic_assignment(Periodic_demand) -> dict:
    Target_counts = {k: v * Periodic_demand 
                    for k, v in Demand_distribution.items()} # {S:100, M:200, L:300, XL:400}
    Target_N_locker = np.array([Target_counts[k] for k in Locker_key]) #[100, 200, 300, 400]
    N_module = {m: 0 for m in Module_key} #{"M1":0, "M2":0, "M3":0, "M4":0, "M5":0, "M6":0}
    Current_N_locker = np.zeros_like(Target_N_locker)
    Total_modules = 0

    #先尝试用尽可能多的M6填充到满足80%的target_number
    Target_80 = Target_N_locker * 0.8
    N_M6 = 0
    M6_number = Module_counts_matrix.loc["M6", Locker_key].values
    while Total_modules < Slab_limit:
        if np.all(Current_N_locker >= Target_80):
            break
        else:
            Current_N_locker += M6_number
            N_M6 += 1
            Total_modules += 1
    N_module['M6'] = N_M6

    while Total_modules < Slab_limit:
        Difference = Target_N_locker - Current_N_locker
        if np.all(Difference == 0): #无法处理negative!??
            break
        else: #找到差的最多的那一种locker
            largest_difference_index = np.argmax(Difference)
            Locker_needed = Locker_key[largest_difference_index] 
        #找到拥有最多这种locker的module
        Best_module = None
        max_number = -1
        for module in Module_key: 
            number = Module_counts_matrix.loc[module, Locker_needed]
            if number > max_number:
                max_number = number
                Best_module = module
        Module_needed = Best_module
        N_module[Module_needed] += 1
        Total_modules += 1
        Module_number = Module_counts_matrix.loc[Module_needed, Locker_key].values
        Current_N_locker += Module_number
    final_solution = {k: int(v) for k, v in N_module.items() if v > 0}
    return final_solution