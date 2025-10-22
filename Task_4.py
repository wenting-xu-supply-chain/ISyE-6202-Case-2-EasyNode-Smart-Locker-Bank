# -*- coding: utf-8 -*-
"""
Task 4 - Dynamic Assignment Heuristics
Author: (your name)
Last Update: 2025-10-15

功能概述：
1) 实现了两种截然不同的动态分配启发式算法，完全满足 Task 4 要求。
2) Strawman Heuristic: 一个简单的“首次适应”(First-Fit)算法，选择ID最小的可用储物柜。
3) Smarter Heuristic: 一个基于“成本最小化”的智能算法，综合考虑人体工学成本和尺寸浪费的机会成本。
4) 人体工学模型: 直接使用了案例中提供的人体工学成本函数。
5) 结果输出: 将两种算法的详细分配结果分别保存到 Task_4_Outcome 文件夹中。
"""

import os, json, math, warnings, bisect
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# =========================
# 全局参数
# =========================
INPUT_DEMAND_CSV = "Task_3_Outcome/task3_demand_instance_random_mode.csv"
OUTPUT_DIR = "Task_4_Outcome"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_HEIGHT = 15
ROW_HEIGHT_CM = 10
BASE_FLOOR_TO_ROW0_CM = 30
SIZE_UNITS = {'small': 1, 'medium': 2, 'large': 3}
USER_HEIGHT_DIST_CM = {'p05': 155, 'p50': 170, 'p95': 185}
ARM_REACH_OFFSET_CM = 70
REACH_SAFETY_MARGIN_CM = 10
BASE_FEE = 5.0 # Although not used for decision logic here, kept for context
OVERSIZE_OPPORTUNITY_COST_PER_LEVEL = 4.0 # Penalty for using an oversized locker
RANDOM_SEED = 20251010
np.random.seed(RANDOM_SEED)

# Baseline configuration
NUM_SMALL = 44
NUM_MEDIUM = 51
NUM_LARGE = 30

# =========================
# 工具函数与数据结构
# =========================
def get_ergonomic_cost(req_size_cat, assigned_size_cat, y_pos):
    """Calculates the ergonomic cost based on the provided formula."""
    y = y_pos + 1
    cost = 0.0
    size_map = {'small': 1, 'medium': 2, 'large': 3}
    d_req = size_map.get(req_size_cat)
    d_assigned = size_map.get(assigned_size_cat)
    if d_req is None or d_assigned is None or d_assigned < d_req:
        return float('inf')

    if d_req == 1 and d_assigned == 1:   cost = 0.01 * (115 - 15*y) if y <= 7 else 0.01 * (-95 + 15*y)
    elif d_req == 1 and d_assigned == 2: cost = 0.01 * ((75 + (65/6)) - (65/6)*y) if y <= 7 else 0.01 * (10 - 7*(65/6) + (65/6)*y)
    elif d_req == 1 and d_assigned == 3: cost = 0.01 * ((50 + (40/6)) - (40/6)*y) if y <= 7 else 0.01 * (10 - 7*(40/6) + (40/6)*y)
    elif d_req == 2 and d_assigned == 2: cost = 0.01 * ((10 - (65/14)) + (65/14)*y)
    elif d_req == 2 and d_assigned == 3: cost = 0.01 * ((10 - (90/14)) + (90/14)*y)
    elif d_req == 3 and d_assigned == 3: cost = 0.01 * ((10 - (90/14)) + (90/14)*y)
    return max(0, cost)

def reachable_row_range_for_user():
    """Simulates a user's height and calculates their reachable locker row range."""
    h = np.random.normal(USER_HEIGHT_DIST_CM['p50'], 3.0)
    reach_max_cm = h + ARM_REACH_OFFSET_CM - REACH_SAFETY_MARGIN_CM
    reach_min_cm = max(0, h * 0.35)
    max_row = min(GRID_HEIGHT - 1, int(round((reach_max_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    min_row = max(0, int(round((reach_min_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    return (min_row, max_row) if min_row <= max_row else (0, GRID_HEIGHT - 1)

class Locker:
    """Represents a single locker with an efficient schedule management system."""
    def __init__(self, locker_id, size_category, position_rc):
        self.id, self.size, self.position, self.schedule = locker_id, size_category, position_rc, []
    def is_available(self, start, end):
        if not self.schedule: return True
        # Find insertion point for the start of the new interval
        idx = bisect.bisect_right(self.schedule, (start, start))
        # Check for overlap with the previous interval
        if idx > 0 and self.schedule[idx - 1][1] > start: return False
        # Check for overlap with the next interval
        if idx < len(self.schedule) and self.schedule[idx][0] < end: return False
        return True
    def book(self, start, end):
        if start < end:
            bisect.insort(self.schedule, (start, end))

class LockerBank:
    """Manages the creation and layout of all lockers."""
    def __init__(self, num_s, num_m, num_l, height=GRID_HEIGHT):
        self.height = height
        self.lockers = []
        self._create_layout(num_s, num_m, num_l)
    def _create_layout(self, num_s, num_m, num_l):
        specs = [('small', num_s), ('medium', num_m), ('large', num_l)]
        current_row, current_col, lid = 0, 0, 0
        for size, cnt in specs:
            h_units = SIZE_UNITS[size]
            for _ in range(cnt):
                if current_row + h_units > self.height:
                    current_row, current_col = 0, current_col + 1
                # Use the geometric center for ergonomic calculations
                center_row_float = current_row + (h_units - 1) / 2.0
                self.lockers.append(Locker(lid, size, (int(round(center_row_float)), current_col)))
                current_row += h_units
                lid += 1
    def total_lockers(self): return len(self.lockers)

class AssignmentAgent:
    """Contains the logic for different assignment heuristics."""
    def __init__(self, locker_bank: LockerBank):
        self.lb = locker_bank
        self._build_req_size_map()
    def _build_req_size_map(self):
        m = {}; [m.update({s: 'small'}) for s in range(1, 4)]; [m.update({s: 'medium'}) for s in range(4, 10)]; [m.update({s: 'large'}) for s in range(10, 19)]
        self.req_size_map = m
    @staticmethod
    def _size_value(cat): return {'small': 1, 'medium': 2, 'large': 3}.get(cat, 0)

    def _get_available_candidates(self, req_cat, start, end, rr):
        """Finds all lockers that are large enough, reachable, and available."""
        req_v = self._size_value(req_cat)
        return [L for L in self.lb.lockers if self._size_value(L.size) >= req_v and (rr[0] <= L.position[0] <= rr[1]) and L.is_available(start, end)]

    # ==================================
    # a. Strawman Heuristic
    # ==================================
    def assign_strawman(self, req_cat, start, end, rr):
        """
        Simple, myopic, first-fit heuristic.
        Finds all available lockers and picks the one with the lowest ID.
        """
        candidates = self._get_available_candidates(req_cat, start, end, rr)
        if not candidates:
            return None
        # Sort by locker ID to ensure a deterministic "first-fit"
        candidates.sort(key=lambda l: l.id)
        return candidates[0]

    # ==================================
    # b. Smarter Heuristic
    # ==================================
    def assign_smarter_cost_min(self, req_cat, start, end, rr):
        """
        Smarter heuristic based on cost minimization.
        It selects the locker that minimizes the sum of ergonomic cost and an oversize opportunity cost.
        """
        candidates = self._get_available_candidates(req_cat, start, end, rr)
        if not candidates:
            return None

        best_locker, min_total_cost = None, float('inf')
        req_v = self._size_value(req_cat)

        for locker in candidates:
            # 1. Calculate Ergonomic Cost using the provided function
            ergo_cost = get_ergonomic_cost(req_cat, locker.size, locker.position[0])

            # 2. Calculate Oversize Opportunity Cost
            # This penalizes using a locker that is larger than necessary.
            size_difference = self._size_value(locker.size) - req_v
            oversize_cost = size_difference * OVERSIZE_OPPORTUNITY_COST_PER_LEVEL
            
            # 3. Total Cost
            total_cost = ergo_cost + oversize_cost

            if total_cost < min_total_cost:
                best_locker, min_total_cost = locker, total_cost
        
        return best_locker

# =========================
# 主程序
# =========================
def run_simulation(heuristic_name, demand_df, locker_bank):
    """A simple simulation runner to process demand with a given heuristic."""
    agent = AssignmentAgent(locker_bank)
    results = []
    
    demand_array = demand_df[['package_id', 'min_size_ft3', 'real_deposit_dt', 'real_pickup_dt']].to_numpy()

    for r_data in tqdm(demand_array, desc=f"Simulating ({heuristic_name})"):
        pkg, min_size_ft3, start_dt, end_dt = r_data
        start, end = pd.to_datetime(start_dt), pd.to_datetime(end_dt)
        
        req_cat = agent.req_size_map.get(int(math.ceil(float(min_size_ft3))) if pd.notna(min_size_ft3) else 1, 'large')
        rr = reachable_row_range_for_user()
        
        chosen = None
        if heuristic_name == 'strawman':
            chosen = agent.assign_strawman(req_cat, start, end, rr)
        elif heuristic_name == 'smarter':
            chosen = agent.assign_smarter_cost_min(req_cat, start, end, rr)

        result_row = {'package_id': pkg, 'required_size': req_cat, 'status': 'rejected', 'assigned_locker_id': None, 'assigned_locker_size': None}
        if chosen:
            chosen.book(start, end)
            result_row.update({'status': 'accepted', 'assigned_locker_id': chosen.id, 'assigned_locker_size': chosen.size})
        results.append(result_row)
        
    return pd.DataFrame(results)

def main():
    """Main function to run and compare the two heuristics."""
    try:
        demand_df = pd.read_csv(INPUT_DEMAND_CSV, parse_dates=['real_deposit_dt','real_pickup_dt'])
    except FileNotFoundError:
        print(f"Error: Demand file not found at '{INPUT_DEMAND_CSV}'. Please run Task 3 first.")
        return
        
    print(f"Loaded {len(demand_df):,} records for simulation.")

    # --- Run Strawman Simulation ---
    print("\n--- Running Simulation with Strawman Heuristic ---")
    bank_a = LockerBank(NUM_SMALL, NUM_MEDIUM, NUM_LARGE)
    results_a = run_simulation('strawman', demand_df, bank_a)
    results_a.to_csv(os.path.join(OUTPUT_DIR, 'strawman_assignment_results.csv'), index=False)
    
    # --- Run Smarter Simulation ---
    print("\n--- Running Simulation with Smarter (Cost Minimization) Heuristic ---")
    bank_b = LockerBank(NUM_SMALL, NUM_MEDIUM, NUM_LARGE)
    results_b = run_simulation('smarter', demand_df, bank_b)
    results_b.to_csv(os.path.join(OUTPUT_DIR, 'smarter_assignment_results.csv'), index=False)

    # --- Final Comparison ---
    print("\n--- Simulation Complete: Performance Comparison ---")
    sl_a = results_a['status'].value_counts(normalize=True).get('accepted', 0)
    sl_b = results_b['status'].value_counts(normalize=True).get('accepted', 0)
    print(f"Strawman Heuristic Service Level: {sl_a:.3%}")
    print(f"Smarter Heuristic Service Level:  {sl_b:.3%}")
    
    # Generate README
    with open(os.path.join(OUTPUT_DIR, 'README_Task4.txt'), 'w', encoding='utf-8') as f:
        f.write("Task 4: Dynamic Assignment Heuristics Output\n\n")
        f.write("This directory contains the output of two different assignment strategies:\n\n")
        f.write("1. strawman_assignment_results.csv:\n")
        f.write("   - Heuristic: Simple 'First-Fit'. Assigns to the first available locker (by ID) that fits.\n")
        f.write("   - Characteristics: Myopic, easy to implement, not optimized.\n\n")
        f.write("2. smarter_assignment_results.csv:\n")
        f.write("   - Heuristic: 'Cost Minimization'. Selects the locker that minimizes a total cost function.\n")
        f.write("   - Cost Components: \n")
        f.write("     a) Ergonomic Cost: Based on the formula from Faugere & Montreuil (2020).\n")
        f.write("     b) Oversize Opportunity Cost: A penalty for using a locker larger than required.\n")
        f.write("   - Characteristics: Smarter, considers both user comfort and space efficiency.\n")

    print(f"\n✅ Detailed results and README saved in the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    main()
