import os, math, warnings, bisect
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Configuration & Parameters (Ensure consistency with T5/T7/T8/T9)
# Input Files
DEMAND_INSTANCE_FILE = os.path.join("task3_demand_instance_steady_mode.csv") # Use the demand instance you want to test
CONFIG_SCHEDULE_FILE = os.path.join("dynamic_periodic_optimization_results.csv")
# --- 注意：请确保这个路径是正确的 ---
MODULE_DEFINITIONS_FILE = os.path.join("D:\Fall 2025\ISYE6335\Case\Case 2\Task_8_Outcome", "Module_design.csv")

OUT_DIR = "Task_10_11_Outcome_2"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
GRID_HEIGHT = 15
PERIOD_DAYS = 28
R_D = {'S': 6.10, 'M': 11.95, 'L': 16.35}
SIZE_UNITS = {'S': 1, 'M': 2, 'L': 3}
LOCKER_CAPACITY_FT3 = {'S': 3.0, 'M': 9.0, 'L': 18.0}
SLA_DAILY_TARGET = 0.995
WARMUP_DAYS = 2.0

# Ergonomic Parameters (Required for Utility Functions)
ROW_HEIGHT_CM = 10
BASE_FLOOR_TO_ROW0_CM = 30
USER_HEIGHT_DIST_CM = {'p50': 170}
ARM_REACH_OFFSET_CM = 70
REACH_SAFETY_MARGIN_CM = 10

# Heuristic Parameters (Required for AssignmentAgent)
LOOKAHEAD_HOURS = 48
SIZE_PROTECTION_LARGE_THRESHOLD = 0.15

RANDOM_SEED = 20251020
np.random.seed(RANDOM_SEED)

# Utility Functions (From Task 7)
def get_ergonomic_cost(req_size_cat, assigned_size_cat, y_pos):
    y = y_pos + 1; cost = 0.0; size_map = {'S': 1, 'M': 2, 'L': 3}
    d_req, d_assigned = size_map.get(req_size_cat), size_map.get(assigned_size_cat)
    if d_req is None or d_assigned is None or d_assigned < d_req: return float('inf')
    if d_req == 1 and d_assigned == 1:   cost = 0.01 * (115 - 15*y) if y <= 7 else 0.01 * (-95 + 15*y)
    elif d_req == 1 and d_assigned == 2: cost = 0.01 * ((75 + (65/6)) - (65/6)*y) if y <= 7 else 0.01 * (10 - 7*(65/6) + (65/6)*y)
    elif d_req == 1 and d_assigned == 3: cost = 0.01 * ((50 + (40/6)) - (40/6)*y) if y <= 7 else 0.01 * (10 - 7*(40/6) + (40/6)*y)
    elif d_req == 2 and d_assigned == 2: cost = 0.01 * ((10 - (65/14)) + (65/14)*y)
    elif (d_req == 2 and d_assigned == 3) or (d_req == 3 and d_assigned == 3):
        cost = 0.01 * ((10 - (90/14)) + (90/14)*y)
    return max(0, cost)

def reachable_row_range_for_user():
    h = np.random.normal(USER_HEIGHT_DIST_CM['p50'], 3.0)
    reach_max_cm = h + ARM_REACH_OFFSET_CM - REACH_SAFETY_MARGIN_CM
    reach_min_cm = max(0, h * 0.35)
    max_row = min(GRID_HEIGHT - 1, int(round((reach_max_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    min_row = max(0, int(round((reach_min_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    return (min_row, max_row) if min_row <= max_row else (0, GRID_HEIGHT - 1)

# Data Structures (Modified for Modularity - Task 10)
class Locker:
    def __init__(self, locker_id, size_category, position_rc):
        self.id = locker_id; self.size = size_category; self.position = position_rc
        self.schedule = []
    def is_available(self, start, end):
        if not self.schedule: return True
        idx = bisect.bisect_right(self.schedule, (start, start))
        if idx > 0 and self.schedule[idx - 1][1] > start: return False
        if idx < len(self.schedule) and self.schedule[idx][0] < end: return False
        return True
    def book(self, start, end):
        bisect.insort(self.schedule, (start, end))

class ModularLockerBank:
    """
    Task 10 Implementation: Supports dynamic reconfiguration based on module counts.
    """
    def __init__(self, module_definitions, height=GRID_HEIGHT):
        self.height = height
        self.module_definitions = module_definitions
        self.lockers = []
        self.num_config_counts = {'S': 0, 'M': 0, 'L': 0}

    def reconfigure(self, new_module_counts):
        """
        Reconfigures the bank. CRITICAL ASSUMPTION: Reconfiguration happens instantaneously at the start of the period.
        """
        self.lockers = []
        self.num_config_counts = {'S': 0, 'M': 0, 'L': 0}

        # Generate the layout sequence
        layout_sequence = []
        # Iterate in a consistent order
        sorted_module_names = sorted(self.module_definitions.keys())
        
        for module_name in sorted_module_names:
            count = new_module_counts.get(module_name, 0)
            if count > 0:
                module_composition = self.module_definitions[module_name]
                # Simplified layout strategy: stack S, then M, then L for each module instance.
                for _ in range(int(count)):
                    layout_sequence.extend(['S'] * int(module_composition.get('S', 0)))
                    layout_sequence.extend(['M'] * int(module_composition.get('M', 0)))
                    layout_sequence.extend(['L'] * int(module_composition.get('L', 0)))

        # Build the physical layout (stacks bottom-up, left-right)
        current_row, current_col, lid = 0, 0, 0
        for size in layout_sequence:
            h_units = SIZE_UNITS[size]
            # Move to next column if it doesn't fit
            if current_row + h_units > self.height:
                current_row, current_col = 0, current_col + 1

            center_row = current_row + (h_units - 1) / 2.0
            self.lockers.append(Locker(lid, size, (int(round(center_row)), current_col)))
            current_row += h_units
            lid += 1
            self.num_config_counts[size] += 1

class AssignmentAgent:
    # Uses the Smarter heuristic
    def __init__(self, locker_bank: ModularLockerBank):
        self.lb = locker_bank

    @staticmethod
    def _size_value(cat):
        return {'S': 1, 'M': 2, 'L': 3}.get(cat, 0)

    def _get_candidates(self, req_cat, start, end, rr):
        req_v = self._size_value(req_cat)
        return [
            L for L in self.lb.lockers
            if self._size_value(L.size) >= req_v
            and (rr[0] <= L.position[0] <= rr[1])
            and L.is_available(start, end)
        ]

    def assign_smarter(self, req_cat, start, end, rr):
        cands = self._get_candidates(req_cat, start, end, rr)
        if not cands: return None

        # Dynamically calculate large utilization
        total_large = self.lb.num_config_counts['L']
        # Check current utilization state
        num_large_used = sum(1 for L in self.lb.lockers if L.size == 'L' and L.schedule and L.schedule[-1][1] > start)

        large_remain_ratio = (total_large - num_large_used) / total_large if total_large > 0 else 0
        return max(cands, key=lambda L: self._smarter_score(L, req_cat, start, end, large_remain_ratio))

    def _smarter_score(self, locker: Locker, req_cat, start, end, large_remain_ratio):
        # (Scoring logic identical to T7)
        req_v, size_diff = self._size_value(req_cat), self._size_value(locker.size) - self._size_value(req_cat)
        size_score = 1.0 / (1 + 2.0 * max(0, size_diff))
        ergo_soft = 1.0 - min(1.0, abs(locker.position[0] - GRID_HEIGHT/2) / (GRID_HEIGHT/2)) * 0.6
        next_start = float('inf')
        idx = bisect.bisect_right(locker.schedule, (end, end))
        if idx < len(locker.schedule):
            next_start = locker.schedule[idx][0]
        future = 1.0 if next_start == float('inf') else max(0.0, min(1.0, (next_start - end).total_seconds()/3600 / LOOKAHEAD_HOURS))
        protect_penalty = 0.0
        if locker.size == 'L' and size_diff > 0:
            if large_remain_ratio < SIZE_PROTECTION_LARGE_THRESHOLD:
                protect_penalty = (SIZE_PROTECTION_LARGE_THRESHOLD - large_remain_ratio) * 0.8
        return (0.40 * size_score) + (0.30 * ergo_soft) + (0.30 * future) - protect_penalty

# Modular Rolling Horizon Simulator (Task 10/11 Implementation)
# --- 这是更新后的版本 ---
class ModularRollingHorizonSimulator:
    def __init__(self, demand_df, config_schedule_df, module_definitions, module_costs, warmup_days=WARMUP_DAYS):
        self.demand = demand_df.copy()
        self.config_schedule = config_schedule_df
        self.module_definitions = module_definitions
        self.module_costs = module_costs
        self.warmup_days = warmup_days
        
        # --- MODIFICATIONS ---
        self.results = [] # 将存储所有结果字典
        self.periodic_summaries = [] # 将存储每个周期的摘要文本
        self.periodic_kpis = [] # 将存储每个周期的 kpi 字典
        # --- END MODIFICATIONS ---
        
        self.total_module_cost = 0.0 # 这仍然是总成本

    def _req_category(self, v_ft3):
        v = int(math.ceil(float(v_ft3))) if pd.notna(v_ft3) else 1
        if v <= 3: return 'S'
        if v <= 9: return 'M'
        return 'L'

    def run(self):
        start_ts = self.demand['real_deposit_dt'].min()
        warmup_cut = start_ts + pd.Timedelta(days=self.warmup_days)

        # 按周期分组需求
        self.demand['Period'] = (self.demand['real_deposit_dt'] - start_ts).dt.days // PERIOD_DAYS + 1
        grouped_demand = self.demand.groupby('Period')

        bank = ModularLockerBank(self.module_definitions)
        agent = AssignmentAgent(bank)

        print("Starting Modular Rolling Horizon Simulation (Task 10)...")

        module_names = list(self.module_definitions.keys())
        
        for index, config_row in tqdm(self.config_schedule.iterrows(), total=len(self.config_schedule), desc="Simulating Periods"):
            period_num = int(config_row['Period']) # 确保周期是整数

            # 1. 重新配置 (Task 10)
            new_counts = {m: config_row[m] for m in module_names if m in config_row.index}
            
            # 关键：在重新配置前清除时间表 (模拟瞬时变化)
            for locker in bank.lockers:
                locker.schedule = []
                
            bank.reconfigure(new_counts)
            agent.lb = bank # 更新代理的引用

            # 2. 跟踪模块成本 (变更成本)
            period_cost = 0
            for m in module_names:
                change_col = f'{m}_change'
                if change_col in config_row.index:
                    change = config_row[change_col]
                    if change > 0: # 仅在添加模块时产生-成本
                        period_cost += change * self.module_costs.get(m, 0)
            self.total_module_cost += period_cost # 添加到总成本

            # 3. 处理该周期的需求
            # --- MODIFICATION: 使用一个临时列表来存放该周期的结果 ---
            period_results_list = []
            
            if period_num not in grouped_demand.groups:
                # 即使没有需求，仍需要创建一个摘要条目
                pass # 下面的摘要逻辑将处理空的数据帧
            else:
                period_demand_df = grouped_demand.get_group(period_num)
                period_demand_df = period_demand_df.sort_values(by='real_deposit_dt')
                demand_array = period_demand_df[['package_id', 'min_size_ft3', 'real_deposit_dt', 'real_pickup_dt']].to_numpy()

                for r_data in demand_array:
                    pkg, min_size_ft3, start_dt, end_dt = r_data
                    start, end = pd.to_datetime(start_dt), pd.to_datetime(end_dt)
                    req_cat = self._req_category(min_size_ft3)
                    rr = reachable_row_range_for_user()

                    chosen = agent.assign_smarter(req_cat, start, end, rr)
                    
                    row = {
                        'package_id': pkg, 'Period': period_num, 'required_size': req_cat,
                        'package_volume_ft3': float(min_size_ft3), 'status': 'rejected',
                        'date': start.date(), 'keep_for_eval': start >= warmup_cut, 'ergonomic_cost': 0.0,
                        'assigned_locker_size': None
                    }
                    if chosen:
                        chosen.book(start, end)
                        row.update({
                            'status': 'accepted', 'assigned_locker_size': chosen.size,
                            'ergonomic_cost': get_ergonomic_cost(req_cat, chosen.size, chosen.position[0])
                        })
                    
                    # --- MODIFICATION: 附加到周期性列表 ---
                    period_results_list.append(row)

            # --- NEW SECTION: 4. 总结该周期的结果 ---
            period_df = pd.DataFrame(period_results_list)
            period_eval_df = pd.DataFrame() # 在没有结果的情况下创建空 df
            
            if not period_df.empty:
                # 筛选以进行评估
                period_eval_df = period_df[period_df['keep_for_eval']]

            # 为该周期定义一个标题
            title = f"Period {period_num} Performance Summary"
            
            # 调用辅助函数获取摘要文本和 KPI 字典
            summary_text, kpi_dict = self._summarize_dataframe(period_eval_df, period_cost, title)
            
            # 存储结果
            kpi_dict['Period'] = period_num # 添加周期编号
            self.periodic_summaries.append(summary_text)
            self.periodic_kpis.append(kpi_dict)

            # --- NEW SECTION: 5. 将周期性结果添加到总结果列表 ---
            self.results.extend(period_results_list)

        # --- MODIFICATION: 在最后将总结果列表转换为 DataFrame ---
        self.results = pd.DataFrame(self.results)

    # --- NEW HELPER FUNCTION ---
    def _summarize_dataframe(self, eval_df, module_cost, title="Performance Summary"):
        """
        用于计算给定 dataframe 和模块成本的 KPI 的辅助函数。
        返回: (summary_text, kpi_dict)
        """
        if eval_df.empty:
            summary_text = (
                f"\n--- {title} ---\n"
                f"   --- NO DATA FOR EVALUATION (This period might be all warmup) ---"
            )
            kpis = {'Period': title.split(' ')[1]} # 默认的空 KPIs
            return summary_text, kpis

        # --- 服务水平 KPIs ---
        daily_sl = eval_df.groupby('date')['status'].apply(lambda s: (s == 'accepted').mean()).reset_index(name='service_level')
        overall_sl = (eval_df['status']=='accepted').mean()
        min_daily_sl = daily_sl['service_level'].min() if not daily_sl.empty else 0
        breach_days = (daily_sl['service_level'] < SLA_DAILY_TARGET).sum() if not daily_sl.empty else 0

        # --- 财务 KPIs ---
        served = eval_df[eval_df['status'] == 'accepted']
        S_d = served.groupby('required_size').size().reindex(['S','M','L'], fill_value=0)
        revenue = float(sum(R_D[d] * S_d[d] for d in ['S','M','L']))
        ergo_cost = float(served['ergonomic_cost'].sum())
        
        # 使用传入的 module_cost
        total_cost = module_cost + ergo_cost
        profit = revenue - total_cost

        # --- 利用率 ---
        util_overall = 0.0
        oversize_rate = 0.0
        if not served.empty:
            cap = served['assigned_locker_size'].map(LOCKER_CAPACITY_FT3)
            util_overall = float(served['package_volume_ft3'].sum() / cap.sum()) if cap.sum() > 0 else 0.0
            oversize_rate = float((served['assigned_locker_size'] != served['required_size']).mean())

        # --- 报告文本 ---
        summary_text = (
            f"\n--- {title} ---\n"
            f"  [Service Level]\n"
            f"   Total Packages: {len(eval_df)}\n"
            f"  Accepted Packages: {len(served)}\n"
            f"  Overall Service Level: {overall_sl:.3%}\n"
            f"  Minimum Daily SL: {min_daily_sl:.3%}\n"
            f"  Days Breaching SLA Target ({SLA_DAILY_TARGET:.1%}): {breach_days}\n"
            f"  ------------------------------------------\n"
            f"  [Financial Performance]\n"
            f"  Total Revenue: ${revenue:,.2f}\n"
            f"  Total Module Cost (Setup+Changes): ${module_cost:,.2f}\n"
            f"  Total Ergonomic Cost: ${ergo_cost:,.2f}\n"
            f"  TOTAL COST: ${total_cost:,.2f}\n"
            f"  TOTAL PROFIT: ${profit:,.2f}\n"
            f"  ------------------------------------------\n"
            f"  [Utilization]\n"
            f"  Volume Utilization (Overall): {util_overall:.2%}\n"
            f"  Oversize Rate: {oversize_rate:.2%}\n"
        )
        
        # --- KPI 字典 ---
        kpis = {
            'Total_Packages': len(eval_df),
            'Accepted_Packages': len(served),
            'Overall_SL': overall_sl,
            'Min_Daily_SL': min_daily_sl,
            'Breach_Days': breach_days,
            'Revenue': revenue,
            'Module_Cost': module_cost,
            'Ergo_Cost': ergo_cost,
            'Total_Cost': total_cost,
            'Total_Profit': profit,
            'Util_Volume': util_overall,
            'Oversize_Rate': oversize_rate
        }
        
        return summary_text, kpis

    # --- MODIFIED summarize() FUNCTION ---
    def summarize(self):
        """
        Task 11 Implementation: KPI calculation and reporting.
        Prints periodic summaries first, then the overall summary.
        """
        print("\n" + "=" * 50)
        print("--- Task 11: Periodic Performance Assessment ---")
        print("=" * 50)
        
        if not self.periodic_kpis:
            print("No periodic data to report.")
        else:
            # 打印所有存储的文本摘要
            for summary_text in self.periodic_summaries:
                print(summary_text)
                print("-" * 50)
            
            # 将周期性 KPI 保存到新的 CSV 文件
            df_periodic = pd.DataFrame(self.periodic_kpis)
            df_periodic = df_periodic.set_index('Period')
            periodic_csv_path = os.path.join(OUT_DIR, "modular_summary_periodic.csv")
            
            try:
                df_periodic.to_csv(periodic_csv_path)
                print(f"\nPeriodic KPI summary saved to: {periodic_csv_path}")
            except Exception as e:
                print(f"\nError saving periodic CSV: {e}")

        print("\n" + "=" * 50)
        print("--- Task 11: Overall Performance Assessment ---")
        print("=" * 50)
        
        # 筛选总结果以进行评估
        eval_df = self.results[self.results['keep_for_eval']]
        
        if eval_df.empty:
            print("--- NO DATA FOR OVERALL EVALUATION ---")
            return

        # 调用辅助函数获取总计结果
        overall_title = "Overall Modular Rolling Horizon Performance"
        overall_summary_text, _ = self._summarize_dataframe(eval_df, self.total_module_cost, overall_title)
        
        print(overall_summary_text)
        
        # 保存详细日志和总摘要
        self.results.to_csv(os.path.join(OUT_DIR, "modular_simulation_details.csv"), index=False)
        overall_summary_path = os.path.join(OUT_DIR, "modular_summary_overall.txt")
        with open(overall_summary_path, 'w', encoding='utf-8') as f:
            f.write(overall_summary_text)
        print(f"\nOverall summary saved to: {overall_summary_path}")

# Main Execution
def main():
    # 1. 加载数据
    print("Loading data...")
    try:
        demand_df = pd.read_csv(DEMAND_INSTANCE_FILE, parse_dates=['real_deposit_dt', 'real_pickup_dt'])
        config_schedule_df = pd.read_csv(CONFIG_SCHEDULE_FILE)
        df_modules = pd.read_csv(MODULE_DEFINITIONS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading input files: {e}. Ensure Tasks 3, 8, and 9/10 (optimizer) are complete.")
        return
    except Exception as e:
        print(f"An error occurred loading files: {e}")
        print(f"Check path for MODULE_DEFINITIONS_FILE: {MODULE_DEFINITIONS_FILE}")
        return

    # 2. 解析模块定义
    # 假设 'Cost' 列在 Module_design.csv 中存在
    if 'Cost' not in df_modules.columns:
        print("Warning: 'Cost' column not found in Module_design.csv. Module costs will be 0.")
        module_costs = {row['Module']: 0 for index, row in df_modules.iterrows()}
    else:
        module_costs = df_modules.set_index('Module')['Cost'].to_dict()
        
    module_definitions = {}
    for index, row in df_modules.iterrows():
        # 我们只需要组合 (S, M, L) 来生成模拟器布局
        module_definitions[row['Module']] = {'S': row['S'], 'M': row['M'], 'L': row['L']}

    # 3. 初始化并运行模拟器
    simulator = ModularRollingHorizonSimulator(
        demand_df=demand_df,
        config_schedule_df=config_schedule_df,
        module_definitions=module_definitions,
        module_costs=module_costs
    )
    
    simulator.run()
    
    # 4. 总结 (Task 11 评估)
    simulator.summarize()
    print(f"\n Task 10/11 complete. Results saved to {OUT_DIR}")

if __name__ == '__main__':
    main()