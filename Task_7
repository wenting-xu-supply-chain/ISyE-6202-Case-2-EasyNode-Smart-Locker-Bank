# -*- coding: utf-8 -*-
"""
Task 7 - Optimized Fixed-Configuration Locker Bank Design (SA with Integrated Layout Optimization)
Last Update: 2025-10-20

Optimizations Applied:
- Two-Stage Sampling Strategy
- Core Simulation Optimizations (Numeric Time, Locker Indexing, Merged Availability Check, itertuples)
- Numba Acceleration for Ergonomics & Pre-calculated Reach Ranges
- Optimized Repair Loop (Avoid Redundant Simulations)
- Parallel Evaluation (for multiple instances)
"""

import os, math, bisect, warnings, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import time

# Strategy: Import Numba for JIT acceleration
try:
    from numba import jit
except ImportError:
    print("Warning: Numba not installed (`pip install numba`). Proceeding without JIT acceleration.")
    def jit(*args, **kwargs):
        return lambda f: f

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ==============================
# ==== Directories & Inputs ====
# ==============================
OUT_DIR = "Task_7_Outcome"
os.makedirs(OUT_DIR, exist_ok=True)

# NOTE: Update this path to your actual demand file location
INPUT_DEMAND_CSV = "Task_3_Outcome/task3_demand_instance_steady_mode.csv"

# ==============================
# ==== Grid & Ergonomics
# ==============================
GRID_HEIGHT = 15
ROW_HEIGHT_CM = 10
BASE_FLOOR_TO_ROW0_CM = 30

SIZE_UNITS = {'small': 1, 'medium': 2, 'large': 3}
LOCKER_CAPACITY_FT3 = {'small': 3.0, 'medium': 9.0, 'large': 18.0}
SIZE_MAP_NUM = {'small': 1, 'medium': 2, 'large': 3}

USER_HEIGHT_DIST_CM = {'p50': 170}
ARM_REACH_OFFSET_CM = 70
REACH_SAFETY_MARGIN_CM = 10

# Economics and Targets
SLA_DAILY_TARGET = 0.995
WARMUP_DAYS = 2.0
RANDOM_SEED = 20251020
RNG = np.random.default_rng(RANDOM_SEED) # Use modern NumPy RNG
random.seed(RANDOM_SEED)

R_D = {'small': 6.10, 'medium': 11.95, 'large': 16.35}
C_L = {'small': 91.0, 'medium': 182.0, 'large': 273.0}
C_M = 50.0; C_W = 500.0; C_S = 50.0

AMORT_LAMBDA = 1.0
N_M_COVER = 14

# Baseline locker counts
BASELINE_COUNTS = {'small': 60, 'medium': 71, 'large': 41}

# Heuristic Parameters
LOOKAHEAD_HOURS = 48
SIZE_PROTECTION_LARGE_THRESHOLD = 0.15
# Optimization: Pre-calculate seconds for numeric comparison
LOOKAHEAD_SECONDS = LOOKAHEAD_HOURS * 3600

# ==============================
# ==== Simulated Annealing  ====
# ==============================
MAX_ITERS = 50 # Increased significantly; adjust based on time constraints
T_START = 50000.0
ALPHA = 0.98 # Slower cooling rate for better exploration

# Strategy: Two-Stage Sampling
T7_SAMPLE_SIZE_FULL = 423723 # Full size for final evaluation
T7_SAMPLE_SIZE_OPT = 50000   # Reduced size for SA optimization phase

# Strategy: Configure parallel workers
MAX_WORKERS = multiprocessing.cpu_count()

# Objective shaping
MU_OVERSIZE = 1e6; XI_UNDERUTIL = 4e5; TARGET_VOL_UTIL = 0.72
MAX_REPAIR_ADDS = 15

# ==============================
# ==== Ergonomic Cost & Reach (Optimized)
# ==============================

# Strategy: Numba Accelerated Ergonomic Cost
@jit(nopython=True)
def get_ergonomic_cost_numba(d_req, d_assigned, y_pos):
    y = y_pos + 1
    cost = 0.0

    # Use a large number instead of float('inf') for Numba compatibility
    if d_assigned < d_req: return 1e9

    if d_req == 1 and d_assigned == 1:
        cost = 0.01 * (115 - 15*y) if y <= 7 else 0.01 * (-95 + 15*y)
    elif d_req == 1 and d_assigned == 2:
        val = 65/6
        cost = 0.01 * ((75 + val) - val*y) if y <= 7 else 0.01 * (10 - 7*val + val*y)
    elif d_req == 1 and d_assigned == 3:
        val = 40/6
        cost = 0.01 * ((50 + val) - val*y) if y <= 7 else 0.01 * (10 - 7*val + val*y)
    elif d_req == 2 and d_assigned == 2:
        cost = 0.01 * ((10 - (65/14)) + (65/14)*y)
    elif (d_req == 2 and d_assigned == 3) or (d_req == 3 and d_assigned == 3):
        cost = 0.01 * ((10 - (90/14)) + (90/14)*y)

    return max(0.0, cost)

def get_ergonomic_cost(req_size_cat, assigned_size_cat, y_pos):
    d_req = SIZE_MAP_NUM.get(req_size_cat)
    d_assigned = SIZE_MAP_NUM.get(assigned_size_cat)
    if d_req is None or d_assigned is None: return float('inf')
    # Ensure y_pos is a valid integer index for robustness
    y_pos_corrected = max(0, min(GRID_HEIGHT - 1, int(round(y_pos))))
    return get_ergonomic_cost_numba(d_req, d_assigned, y_pos_corrected)

# Optimization: Pre-calculated Reach Ranges (avoids np.random.normal in loop)
N_PRECALC_REACH = 10000

def _calculate_reach(h):
    reach_max_cm = h + ARM_REACH_OFFSET_CM - REACH_SAFETY_MARGIN_CM
    reach_min_cm = max(0, h * 0.35)
    max_row = min(GRID_HEIGHT - 1, int(round((reach_max_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    min_row = max(0, int(round((reach_min_cm - BASE_FLOOR_TO_ROW0_CM) / ROW_HEIGHT_CM)))
    return (min_row, max_row) if min_row <= max_row else (0, GRID_HEIGHT - 1)

# Generate the pool of reach ranges
_heights = RNG.normal(USER_HEIGHT_DIST_CM['p50'], 3.0, N_PRECALC_REACH)
PRECALCULATED_REACH_RANGES = [_calculate_reach(h) for h in _heights]

def reachable_row_range_for_user_fast():
    # Fast lookup from the pre-calculated pool
    return random.choice(PRECALCULATED_REACH_RANGES)

# ==============================
# ==== Locker Simulation =======
# ==============================
class Locker:
    def __init__(self, locker_id, size_category, position_rc):
        self.id = locker_id; self.size = size_category; self.position = position_rc
        # Strategy: Schedule stores tuples of (start_s, end_s) (numeric)
        self.schedule = []

    # Strategy: Combined availability check and future prediction
    def check_availability(self, start, end):
        """
        Checks availability and returns the start time of the next booking.
        Returns: (is_available: bool, next_booking_start: float)
        """
        if not self.schedule:
            return True, float('inf')

        # Find insertion point (Single bisect call)
        idx = bisect.bisect_right(self.schedule, (start, start))

        # 1. Check overlap with the previous booking
        if idx > 0 and self.schedule[idx - 1][1] > start:
            return False, 0.0

        # 2. Check overlap with the next booking
        if idx < len(self.schedule):
            next_start = self.schedule[idx][0]
            if next_start < end:
                return False, 0.0
            # Available, return the start of the next booking
            return True, next_start

        # Available, no future bookings
        return True, float('inf')

    def book(self, start, end):
        bisect.insort(self.schedule, (start, end))

class LockerBank:
    def __init__(self, layout_sequence, height=GRID_HEIGHT):
        self.height = height
        self.lockers = []
        self.layout_sequence = layout_sequence.copy()

        # Strategy: Indexing lockers by size
        self.lockers_by_size = {'small': [], 'medium': [], 'large': []}

        self._create_layout(layout_sequence)
        self.num_config = pd.Series(layout_sequence).value_counts().reindex(['small', 'medium', 'large'], fill_value=0).to_dict()

    def _create_layout(self, layout_sequence):
        current_row, current_col, lid = 0, 0, 0
        for size in layout_sequence:
            h_units = SIZE_UNITS[size]
            if current_row + h_units > self.height:
                current_row, current_col = 0, current_col + 1

            center_row = current_row + (h_units - 1) / 2.0
            L = Locker(lid, size, (int(round(center_row)), current_col))
            self.lockers.append(L)
            self.lockers_by_size[size].append(L) # Add to index

            current_row += h_units
            lid += 1
        self.width = (max((L.position[1] for L in self.lockers), default=-1) + 1)

class AssignmentAgent:
    def __init__(self, locker_bank: LockerBank):
        self.lb = locker_bank
        self.total_large = locker_bank.num_config['large']

    @staticmethod
    def _size_value(cat):
        return SIZE_MAP_NUM.get(cat, 0)

    # Strategy: Integrated optimized assignment logic
    def assign_t5_smarter(self, req_cat, start, end, rr):
        # start/end are numeric seconds

        # 1. Determine sizes to check (Strategy: Indexing)
        if req_cat == 'small':
            sizes_to_check = ['small', 'medium', 'large']
        elif req_cat == 'medium':
            sizes_to_check = ['medium', 'large']
        else:
            sizes_to_check = ['large']

        # 2. Calculate large utilization (Strategy: Indexing optimization)
        num_large_used = 0
        if self.total_large > 0:
            # Iterate only over large lockers
            for L in self.lb.lockers_by_size['large']:
                # Check if currently in use (last booking hasn't ended)
                if L.schedule and L.schedule[-1][1] > start:
                   num_large_used += 1
            large_remain_ratio = (self.total_large - num_large_used) / self.total_large
        else:
            large_remain_ratio = 0

        # 3. Iterate, check availability, and score simultaneously
        best_score = -float('inf')
        best_locker = None

        for size in sizes_to_check:
            for L in self.lb.lockers_by_size[size]:
                # Check ergonomics constraint
                if not (rr[0] <= L.position[0] <= rr[1]):
                    continue

                # Strategy: Combined availability check
                is_available, next_start = L.check_availability(start, end)

                if is_available:
                    # Calculate score using the obtained next_start
                    score = self._smarter_score_optimized(L, req_cat, end, large_remain_ratio, next_start)
                    if score > best_score:
                        best_score = score
                        best_locker = L

        return best_locker

    def _smarter_score_optimized(self, locker: Locker, req_cat, end, large_remain_ratio, next_start):
        req_v = self._size_value(req_cat)
        size_diff = self._size_value(locker.size) - req_v

        size_score = 1.0 / (1 + 2.0 * max(0, size_diff))
        ergo_soft = 1.0 - min(1.0, abs(locker.position[0] - GRID_HEIGHT/2) / (GRID_HEIGHT/2)) * 0.6

        # Strategy: Use pre-calculated next_start and LOOKAHEAD_SECONDS
        if next_start == float('inf'):
            future = 1.0
        else:
            # (next_start - end) is the gap in seconds
            future = max(0.0, min(1.0, (next_start - end) / LOOKAHEAD_SECONDS))

        protect_penalty = 0.0
        if locker.size == 'large' and size_diff > 0:
            if large_remain_ratio < SIZE_PROTECTION_LARGE_THRESHOLD:
                protect_penalty = (SIZE_PROTECTION_LARGE_THRESHOLD - large_remain_ratio) * 0.8

        return (0.40 * size_score) + (0.30 * ergo_soft) + (0.30 * future) - protect_penalty

class LockerBankSimulator:
    """
    MODIFIED: Expects pre-processed numeric demand data.
    """
    def __init__(self, locker_bank, demand_df_numeric, agent, warmup_days=WARMUP_DAYS):
        self.lb = locker_bank; self.demand = demand_df_numeric; self.agent = agent
        self.results = pd.DataFrame()
        # Calculate warmup cut in seconds
        self.warmup_cut_s = warmup_days * 24 * 3600

    def run(self):
        rows = []
        # Optimization: Use itertuples for faster iteration
        demand_tuples = self.demand.itertuples(index=False)

        for r in tqdm(demand_tuples, total=len(self.demand), desc=f"Sim (N={len(self.demand)})"):
            start, end = r.start_s, r.end_s
            req_cat = r.req_cat
            keep_for_eval = start >= self.warmup_cut_s

            # Use the fast reach range function
            rr = reachable_row_range_for_user_fast()

            # Uses the optimized agent and numeric time
            chosen = self.agent.assign_t5_smarter(req_cat, start, end, rr)

            row = {
                'package_id': r.package_id,
                'required_size': req_cat,
                'package_volume_ft3': float(r.min_size_ft3), 'status': 'rejected',
                'date': r.date,
                'keep_for_eval': keep_for_eval,
                'ergonomic_cost': 0.0
            }
            if chosen:
                chosen.book(start, end)
                row.update({
                    'status': 'accepted', 'assigned_locker_size': chosen.size,
                    # Strategy: Uses the Numba-accelerated cost function
                    'ergonomic_cost': get_ergonomic_cost(req_cat, chosen.size, chosen.position[0])
                })
            rows.append(row)
        self.results = pd.DataFrame(rows)

# ==============================
# ==== Evaluation Functions ====
# ==============================

# NOTE: This function must be defined at the top level for multiprocessing
def _evaluate_config(demand_df_numeric, cfg_sequence, warmup_days=WARMUP_DAYS):
    """
    Evaluates a configuration sequence by running the optimized simulation.
    """
    bank = LockerBank(cfg_sequence)
    agent = AssignmentAgent(bank)
    # Initialize simulator with numeric data
    sim = LockerBankSimulator(bank, demand_df_numeric, agent, warmup_days=warmup_days)
    sim.run()

    # (Rest of the evaluation logic remains the same, but executes faster)
    df = sim.results
    eval_df = df[df['keep_for_eval']]
    if eval_df.empty:
        return dict(feasible=False, overall_sl=0.0, min_daily_sl=0.0, revenue=0.0, ergonomic_cost=0.0,
                    vol_util_overall=0.0, oversize_rate=0.0, capex_amort=0.0, total_cost=0.0,
                    width=getattr(bank, 'width', 0), height=GRID_HEIGHT, worst_day=None,
                    worst_day_rejects_by_size={'small':0,'medium':0,'large':0})

    # SLA Calculation
    day_grp = eval_df.groupby('date')
    daily_sl = day_grp['status'].apply(lambda x: (x == 'accepted').mean())
    min_daily_sl = float(daily_sl.min()) if not daily_sl.empty else 0.0
    feasible = (min_daily_sl >= SLA_DAILY_TARGET)

    # Worst Day Analysis (for repair) - Strategy: Optimized Repair relies on this info
    worst_day = daily_sl.idxmin() if not daily_sl.empty else None
    if worst_day is not None:
        dday = eval_df[eval_df['date'] == worst_day]
        rejects = dday[dday['status'] != 'accepted']
        rej_by_size = rejects['required_size'].value_counts().reindex(
            ['small','medium','large'], fill_value=0).to_dict()
    else:
        rej_by_size = {'small':0,'medium':0,'large':0}

    overall_sl = (eval_df['status'] == 'accepted').mean()

    # Revenue and Cost
    served = eval_df[eval_df['status'] == 'accepted']
    S_d = served.groupby('required_size').size().reindex(['small','medium','large'], fill_value=0)
    revenue = float(sum(R_D[d] * S_d[d] for d in ['small','medium','large']))
    ergo_cost = float(served['ergonomic_cost'].sum())

    # Utilization
    if served.empty:
        util_overall, oversize_rate = 0.0, 0.0
    else:
        cap = served['assigned_locker_size'].map(LOCKER_CAPACITY_FT3)
        util_overall = float(served['package_volume_ft3'].sum() / cap.sum()) if cap.sum() > 0 else 0.0
        oversize_rate = float((served['assigned_locker_size'] != served['required_size']).mean())

    # CAPEX
    W = getattr(bank, 'width', 0)
    H = GRID_HEIGHT
    M = int(np.ceil(W / N_M_COVER)) if W > 0 else 0

    cfg_counts = bank.num_config
    capex = sum(C_L[d] * cfg_counts[d] for d in ['small','medium','large']) + C_M * M + C_W * W + 2.0 * C_S * (W + H)
    capex_amort = AMORT_LAMBDA * capex
    total_cost = ergo_cost + capex_amort

    return dict(
        feasible=feasible, overall_sl=float(overall_sl), min_daily_sl=float(min_daily_sl),
        revenue=float(revenue), ergonomic_cost=float(ergo_cost), vol_util_overall=float(util_overall),
        oversize_rate=float(oversize_rate), capex_amort=float(capex_amort), total_cost=float(total_cost),
        width=W, height=H, worst_day=worst_day, worst_day_rejects_by_size=rej_by_size
    )

# Strategy: Wrapper function for multiprocessing compatibility
def _evaluate_config_wrapper(args):
    return _evaluate_config(*args)

# Strategy: Parallelized multi-instance evaluation & Optimized Repair integration
def _evaluate_multi(instances_numeric, cfg_sequence, warmup_days=WARMUP_DAYS, parallel=True):
    if not instances_numeric:
         return {}

    agg = []
    tasks = [(df, cfg_sequence, warmup_days) for df in instances_numeric]

    # Determine execution mode
    run_parallel = parallel and len(instances_numeric) > 1 and MAX_WORKERS > 1

    if run_parallel:
        # Parallel execution
        print(f"Evaluating {len(instances_numeric)} instances in parallel using {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(_evaluate_config_wrapper, tasks))
        agg = results
    else:
        # Sequential execution (Tqdm already handled within _evaluate_config -> LockerBankSimulator)
        for task in tasks:
            agg.append(_evaluate_config_wrapper(task))

    # Aggregation logic
    feasible_all = True
    worst_min_sl = 1.0
    # Strategy: Store details of the worst performance for repair
    worst_info = {'instance_idx': None, 'day': None, 'rejects_by_size': None}

    for i, k in enumerate(agg):
        if not k['feasible']: feasible_all = False
        # Strategy: Track the worst performance details
        if k['min_daily_sl'] < worst_min_sl:
            worst_min_sl = k['min_daily_sl']
            worst_info['instance_idx'] = i
            worst_info['day'] = k['worst_day']
            # This relies on _evaluate_config returning 'worst_day_rejects_by_size'
            worst_info['rejects_by_size'] = k['worst_day_rejects_by_size']

    out = {}
    out['feasible'] = feasible_all
    out['min_daily_sl'] = float(min(k['min_daily_sl'] for k in agg))
    out['overall_sl']   = float(np.average([k['overall_sl'] for k in agg]))
    out['revenue']      = float(sum(k['revenue'] for k in agg))
    out['ergonomic_cost']=float(sum(k['ergonomic_cost'] for k in agg))
    out['width']        = agg[0]['width']
    out['height']       = agg[0]['height']
    out['capex_amort']  = agg[0]['capex_amort']
    out['total_cost']   = float(out['ergonomic_cost'] + out['capex_amort'])
    out['vol_util_overall'] = float(np.average([k['vol_util_overall'] for k in agg]))
    out['oversize_rate']    = float(np.average([k['oversize_rate'] for k in agg]))
    
    # Strategy: Pass the detailed info out
    out['worst_instance_idx'] = worst_info['instance_idx']
    out['worst_day'] = worst_info['day']
    out['worst_rejects_by_size'] = worst_info['rejects_by_size']
    return out

# ==============================
# ==== Repair & Objective  =====
# ==============================
def _objective_feasible_first(kpi):
    # (Unchanged logic)
    if not kpi or not kpi.get('feasible', False): return -float('inf')
    score = float(kpi['revenue'] - kpi['total_cost'])
    score -= MU_OVERSIZE * float(kpi.get('oversize_rate', 0.0))
    util_gap = max(0.0, TARGET_VOL_UTIL - float(kpi.get('vol_util_overall', 0.0)))
    score -= XI_UNDERUTIL * util_gap
    return score

# Strategy: Optimized repair loop
def _repair_to_feasible(instances_numeric, cfg_sequence):
    """
    MODIFIED: Optimized to avoid redundant simulations by using info from _evaluate_multi.
    """
    cfg_sequence = cfg_sequence.copy()
    
    # Track best infeasible solution in case repair fails (Robustness improvement)
    best_infeasible_kpi = None
    best_infeasible_cfg = cfg_sequence.copy()
    max_sl = -1.0

    for _ in range(MAX_REPAIR_ADDS):
        # Run simulation (optimized/parallel)
        kpi = _evaluate_multi(instances_numeric, cfg_sequence)
        if kpi['feasible']:
            return cfg_sequence, kpi, True

        # Track best infeasible
        if kpi['min_daily_sl'] > max_sl:
            max_sl = kpi['min_daily_sl']
            best_infeasible_kpi = kpi
            best_infeasible_cfg = cfg_sequence.copy()

        # Strategy: Use the rejection info directly from the KPI dictionary.
        # This avoids running a second simulation inside the loop.
        rej = kpi.get('worst_rejects_by_size')

        if rej is None:
            # Fallback if info is missing
            cfg_sequence.append('small')
            continue

        # Add the size most needed
        if sum(rej.values()) > 0:
            target_size = max(rej, key=lambda s: rej[s])
        else:
            target_size = 'small' # Fallback

        cfg_sequence.extend([target_size] * 10)

    # Exceeded repair cap. Return the best configuration found during repair.
    print(f"Warning: Repair failed after {MAX_REPAIR_ADDS} attempts. Returning best infeasible config (Min SL={max_sl:.4f}).")
    if best_infeasible_kpi:
         return best_infeasible_cfg, best_infeasible_kpi, False
    
    # Final fallback
    final_kpi = _evaluate_multi(instances_numeric, cfg_sequence)
    return cfg_sequence, final_kpi, final_kpi['feasible']

# ==============================
# ==== Neighborhood (SA)  ======
# ==============================
def _get_random_neighbor(cfg_sequence):
    # (Unchanged logic for generating neighbors)
    new_sequence = cfg_sequence.copy()
    move_type = random.choices(['swap_position', 'change_count'], weights=[0.4, 0.6], k=1)[0]

    if move_type == 'swap_position' and len(new_sequence) >= 2:
        idx1, idx2 = random.sample(range(len(new_sequence)), 2)
        new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    else:
        mod_type = random.choice(['add', 'remove', 'swap_type'])
        if mod_type == 'add':
            size_to_add = random.choice(['small', 'medium', 'large'])
            insert_idx = random.randint(0, len(new_sequence))
            new_sequence.insert(insert_idx, size_to_add)
        elif mod_type == 'remove' and len(new_sequence) > 50: # Keep a reasonable minimum bank size
            remove_idx = random.randint(0, len(new_sequence) - 1)
            new_sequence.pop(remove_idx)
        elif mod_type == 'swap_type' and len(new_sequence) > 0:
            idx = random.randint(0, len(new_sequence) - 1)
            current_type = new_sequence[idx]
            possible_types = [t for t in ['small', 'medium', 'large'] if t != current_type]
            if possible_types:
                new_sequence[idx] = random.choice(possible_types)
    return new_sequence

# ==============================
# ==== SA Search (Multi)   ====
# ==============================
def _search_sa_multi(instances_numeric, baseline_counts):
    hist = []
    T = T_START

    # Initial sequence generation
    initial_sequence = (['small'] * baseline_counts['small'] +
                        ['medium'] * baseline_counts['medium'] +
                        ['large'] * baseline_counts['large'])
    random.shuffle(initial_sequence)

    # Start from baseline repaired to feasibility (Uses optimized repair)
    print("Repairing initial configuration...")
    cur_cfg, cur_kpi, success = _repair_to_feasible(instances_numeric, initial_sequence)
    if not success:
        print("WARNING: Initial configuration could not be repaired to feasibility.")
        
    cur_obj = _objective_feasible_first(cur_kpi)
    best_cfg, best_kpi, best_obj = cur_cfg.copy(), cur_kpi.copy(), cur_obj

    def get_counts(seq):
        return pd.Series(seq).value_counts().reindex(['small', 'medium', 'large'], fill_value=0).to_dict()

    hist.append({**get_counts(cur_cfg), **cur_kpi, 'objective': cur_obj, 'phase': 'init', 'temp': T, 'accepted': True})
    print(f"Init: Obj={cur_obj:.2f}, Feasible={cur_kpi['feasible']}, Cfg={get_counts(cur_cfg)}")

    for it in range(MAX_ITERS):
        print(f"\n--- Iteration {it+1}/{MAX_ITERS} (T={T:.4f}) ---")
        cand_cfg = _get_random_neighbor(cur_cfg)

        # Evaluate candidate (Uses parallel/optimized evaluation)
        kpi0 = _evaluate_multi(instances_numeric, cand_cfg)
        
        if not kpi0['feasible']:
            # Try to repair (Uses optimized repair)
            print(f"Candidate infeasible (SL={kpi0['min_daily_sl']:.4f}). Attempting repair...")
            cand_cfg_rep, kpi_r, ok = _repair_to_feasible(instances_numeric, cand_cfg)
            if not ok:
                print("Repair failed.")
                hist.append({**get_counts(cand_cfg), **kpi0, 'objective': -float('inf'),
                             'phase': f'iter{it+1}', 'temp': T, 'accepted': False})
                T *= ALPHA
                continue
            cand_cfg, new_kpi = cand_cfg_rep, kpi_r
        else:
            new_kpi = kpi0

        new_obj = _objective_feasible_first(new_kpi)
        delta = new_obj - cur_obj

        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        elif T > 1e-6:
            try:
                if np.exp(delta / T) > random.random():
                    accept = True
            except OverflowError:
                 accept = True # Handle overflow if delta/T is very large (means acceptance)

        if accept:
            print(f"Accepted. Obj: {new_obj:.2f} (Delta: {delta:.2f})")
            cur_cfg, cur_kpi, cur_obj = cand_cfg, new_kpi, new_obj
        else:
            print(f"Rejected. Obj: {new_obj:.2f}")

        if cur_obj > best_obj:
            best_obj = cur_obj
            best_cfg, best_kpi = cur_cfg.copy(), cur_kpi.copy()
            print(f"*** New Best Found! Obj={best_obj:.2f}, W={best_kpi['width']} ***")

        hist.append({**get_counts(cand_cfg), **new_kpi, 'objective': new_obj,
                     'phase': f'iter{it+1}', 'temp': T, 'accepted': accept})
        T *= ALPHA

        if (it + 1) % 25 == 0:
            print(f"\nCheckpoint Iter {it+1}: Best={best_obj:.0f}, BestCfg={get_counts(best_cfg)}")

    print(f"\nSA finished. BestObj={best_obj:.2f}, BestCfg={get_counts(best_cfg)}, Feasible={best_kpi['feasible']}")
    return best_cfg, best_kpi, pd.DataFrame(hist)

# ==============================
# ==== Layout Export ===========
# ==============================
def generate_layout_report(cfg_sequence, filepath):
    # (Unchanged logic)
    bank = LockerBank(cfg_sequence)
    layout_data = [
        {'locker_id': L.id, 'size': L.size, 'center_row_index': L.position[0],
         'column_index': L.position[1], 'sequence_index': i}
        for i, L in enumerate(bank.lockers)
    ]
    df_layout = pd.DataFrame(layout_data)
    df_layout = df_layout.sort_values(by=['column_index', 'center_row_index'])
    df_layout.to_csv(filepath, index=False)
    print(f"✅ Layout saved: {filepath}")

# ==============================
# ==== Preprocessing (New) =====
# ==============================
# Optimization: Helper function for Numeric Time Conversion and Pre-calculation
def preprocess_demand(df_raw):
    """Converts datetime to numeric (seconds) and pre-calculates required category."""
    print("Preprocessing demand data (Numeric conversion and categorization)...")
    df = df_raw.copy()
    for col in ['real_deposit_dt', 'real_pickup_dt']:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    df = df.sort_values('real_deposit_dt')
    start_ts = df['real_deposit_dt'].min()

    # Convert to seconds from the start
    df['start_s'] = (df['real_deposit_dt'] - start_ts).dt.total_seconds()
    df['end_s'] = (df['real_pickup_dt'] - start_ts).dt.total_seconds()
    df['date'] = df['real_deposit_dt'].dt.date # Ensure date is available for SLA grouping

    # Pre-calculate required category
    def _req_category(v_ft3):
        v = int(math.ceil(float(v_ft3))) if pd.notna(v_ft3) else 1
        if v <= 3: return 'small'
        if v <= 9: return 'medium'
        return 'large'

    df['req_cat'] = df['min_size_ft3'].apply(_req_category)
    return df

# ==============================
# ==== Main Run (Optimized) ====
# ==============================
def run_task7():
    start_time = time.time()

    if not os.path.exists(INPUT_DEMAND_CSV):
        print(f"ERROR: Missing demand file: {INPUT_DEMAND_CSV}")
        print("Please ensure the file exists before running the optimization.")
        return

    # Load the full dataset
    print("Loading demand data...")
    df_load = pd.read_csv(INPUT_DEMAND_CSV, parse_dates=['real_deposit_dt', 'real_pickup_dt', 'desired_deposit_dt', 'desired_pickup_dt'])

    # Strategy: Two-Stage Sampling
    # 1. Prepare Full Instance (Raw)
    if len(df_load) > T7_SAMPLE_SIZE_FULL:
        print(f"Sampling {T7_SAMPLE_SIZE_FULL} demands (Full) from {len(df_load)}...")
        df_full_raw = df_load.sample(n=T7_SAMPLE_SIZE_FULL, random_state=RANDOM_SEED)
    else:
        df_full_raw = df_load

    # 2. Prepare Optimization Instance (Raw subset)
    if len(df_full_raw) > T7_SAMPLE_SIZE_OPT:
        print(f"Creating reduced sample ({T7_SAMPLE_SIZE_OPT}) for optimization phase.")
        df_opt_raw = df_full_raw.sample(n=T7_SAMPLE_SIZE_OPT, random_state=RANDOM_SEED)
    else:
        df_opt_raw = df_full_raw
    
    # 3. Preprocess (Numeric conversion)
    df_full_numeric = preprocess_demand(df_full_raw)
    df_opt_numeric = preprocess_demand(df_opt_raw)

    instances_full = [df_full_numeric]
    instances_opt = [df_opt_numeric]
    
    # Note: To leverage parallelization, instances_opt/instances_full should contain multiple distinct scenarios.

    # --- SA Optimization Phase (using instances_opt) ---
    print("\n=== Starting SA Optimization (Reduced Sample) ===")
    best_cfg_sequence, best_kpi_opt, traj = _search_sa_multi(instances_opt, BASELINE_COUNTS)


    # --- Validation Phase (using instances_full) ---
    print("\n=== Starting Validation Phase (Full Sample) ===")
    print("Re-evaluating best configuration on the full sample...")
    # Force sequential evaluation if only one instance, otherwise use parallel if configured
    use_parallel = len(instances_full) > 1
    best_kpi_full = _evaluate_multi(instances_full, best_cfg_sequence, parallel=use_parallel)

    # Check feasibility on the full set. If sampling caused an infeasible solution, repair it now.
    if not best_kpi_full['feasible']:
        print(f"WARNING: Best configuration infeasible on full sample (Min SL={best_kpi_full['min_daily_sl']:.4f}). Repairing...")
        best_cfg_sequence, best_kpi_full, success = _repair_to_feasible(instances_full, best_cfg_sequence)
        if not success:
            print("CRITICAL: Failed to repair the final configuration on the full sample.")
        else:
            print("Successfully repaired configuration.")
    else:
        print("Configuration is feasible on the full sample.")

    final_cfg_sequence = best_cfg_sequence
    final_kpi = best_kpi_full
    final_counts = pd.Series(final_cfg_sequence).value_counts().to_dict()

    # --- Baseline Evaluation (using instances_full) ---
    print("\n=== Evaluating Baseline (Full Sample) ===")
    baseline_sequence = (['small'] * BASELINE_COUNTS['small'] +
                         ['medium'] * BASELINE_COUNTS['medium'] +
                         ['large'] * BASELINE_COUNTS['large'])
    random.shuffle(baseline_sequence)
    base_cfg_repaired, base_kpi, _ = _repair_to_feasible(instances_full, baseline_sequence)
    base_counts = pd.Series(base_cfg_repaired).value_counts().to_dict()

    # --- Reporting ---
    base_profit = base_kpi['revenue'] - base_kpi['total_cost']
    final_profit = final_kpi['revenue'] - final_kpi['total_cost']
    base_obj = _objective_feasible_first(base_kpi)
    final_obj = _objective_feasible_first(final_kpi)

    def format_counts(counts):
        return {'S': counts.get('small',0), 'M': counts.get('medium',0), 'L': counts.get('large',0)}

    # Comparison DataFrame generation
    comp = pd.DataFrame([
        ['Baseline(Repair)', format_counts(base_counts)['S'], format_counts(base_counts)['M'], format_counts(base_counts)['L'],
         base_kpi['overall_sl'], base_kpi['min_daily_sl'], base_kpi['feasible'],
         base_kpi['revenue'], base_kpi['ergonomic_cost'], base_kpi['capex_amort'], base_kpi['total_cost'], base_profit,
         base_kpi['width'], base_obj],
        ['Recommended', format_counts(final_counts)['S'], format_counts(final_counts)['M'], format_counts(final_counts)['L'],
         final_kpi['overall_sl'], final_kpi['min_daily_sl'], final_kpi['feasible'],
         final_kpi['revenue'], final_kpi['ergonomic_cost'], final_kpi['capex_amort'], final_kpi['total_cost'], final_profit,
         final_kpi['width'], final_obj]
    ], columns=[
        'Design','Small','Medium','Large',
        'Avg_SL','Min_Daily_SL','Feasible',
        'Revenue','ErgoCost','Capex_Amort','TotalCost','Profit','W','Objective'
    ])
    comp_path = os.path.join(OUT_DIR, "task7_comparison.csv")
    comp.to_csv(comp_path, index=False)

    traj_path = os.path.join(OUT_DIR, "task7_search_traj.csv")
    traj.to_csv(traj_path, index=False)

    layout_path = os.path.join(OUT_DIR, "task7_recommended_layout.csv")
    generate_layout_report(final_cfg_sequence, layout_path)

    end_time = time.time()
    print(f"\n=== ✅ Task 7 Completed (Optimized SA) ===")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Baseline(Repair):  Feasible={base_kpi['feasible']} | Profit={base_profit:.0f} | Config={format_counts(base_counts)}")
    print(f"Recommended:       Feasible={final_kpi['feasible']} | Profit={final_profit:.0f} | Config={format_counts(final_counts)}")
    print(f"Files saved to: {OUT_DIR}")

# Required for multiprocessing to work correctly across platforms
if __name__ == "__main__":
    try:
        # 'spawn' is generally safer than 'fork' across different OSes
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Start method already set
    
    # Uncomment the following line to run the task
    run_task7()
