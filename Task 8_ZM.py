import pandas as pd
import os

data_dir = "D:\Fall 2025\ISYE6335\Case\Case 2"

#1*1,1*2,1*3

#Design
Locker_size = {
    'S': {'space': 3, 'range': (1, 3)},
    'M': {'space': 9, 'range': (4, 9)},
    'L': {'space': 18, 'range': (10, 18)}}

Module_type= {
    'M1': {'S': 6, 'M': 6, 'L': 2},
    'M2': {'S': 6, 'M': 12, 'L': 0},
    'M3': {'S': 8, 'M': 8, 'L': 2},
    'M4': {'S': 12, 'M': 6, 'L': 2},
    'M5': {'S': 0, 'M': 6, 'L': 6},
    'M6': {'S': 2, 'M': 8, 'L': 4}}

#Calculate total probability for each locker_size
Space_prob = {1: 0.08, 2: 0.12, 3: 0.15, 
              4: 0.12, 5: 0.08, 6: 0.06, 7: 0.05, 8: 0.05, 9: 0.05, 
              10: 0.04, 11: 0.04, 12: 0.03, 13: 0.03, 14: 0.03, 15: 0.02, 16: 0.02, 17: 0.02, 18: 0.01}

Locker_prob = []
for category, data in Locker_size.items():
    min_space, max_space = data['range']
    total_prob = sum(
        prob for space, prob in Space_prob.items() 
        if min_space <= space <= max_space)
    Locker_prob.append({
        'Locker': category,
        'Locker_size': data['space'],
        'Demand_probability': total_prob})
    
df_locker_prob = pd.DataFrame(Locker_prob)
#print(df_locker_prob)
df_locker_prob.to_csv(os.path.join(data_dir, "Locker Demand Distribution.csv"), index=False)

#Calculate module size
def calculate_module_size(mtype: dict) -> float: 
    Module_size = 0
    for key, count in mtype.items():
        if key in Locker_size:
            Module_size += count * Locker_size[key]['space']
    return Module_size

module_data = []
for key, count in Module_type.items():
    data = {'Module': key, **count}
    data['Total Lockers'] = sum(count.values())
    data['Total Size (ft3)'] = calculate_module_size(count)
    module_data.append(data)
df_modules = pd.DataFrame(module_data)

columns_order = [
    'Module',
    'S', 'M', 'L',
    'Total Lockers',
    'Total Size (ft3)']
df_modules = pd.DataFrame(module_data, columns=columns_order)
#print(df_modules)
df_modules.to_csv(os.path.join(data_dir, "Module design.csv"), index=False)