# --- 1. Import Required Libraries ---
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# --- 2. Configuration and Constants ---

# File path for the Excel data source
EXCEL_FILE_PATH = 'ISyE 6202 + 6335 EasyNode Smart Locker Bank Casework  Fall 2025.xlsx'
SHEET_NAME = 'Deposit2Pickup'

# Key business parameters from the case study
MAX_ANNUAL_DEMAND = 150000      # Most optimistic demand projection (Year 5, Scenario 5.5)
PEAK_4_WEEK_SHARE = 0.13        # Share of demand occurring in the busiest 4-week period
PEAK_DAY_OF_WEEK_SHARE = 0.20   # Share of weekly demand occurring on the busiest day (Friday)
SERVICE_LEVEL_TARGET = 0.995    # The target probability of having a locker available for a new package

# Probabilities for different package sizes, used to determine the locker mix
SPACE_REQUIREMENT_PROB = {
    'small': 0.08 + 0.12 + 0.15,                           # For packages 1-3 ft³
    'medium': 0.12 + 0.08 + 0.06 + 0.05 + 0.05 + 0.05,       # For packages 4-9 ft³
    'large': 0.04 + 0.04 + 0.03 + 0.03 + 0.03 + 0.02 + 0.02 + 0.02 + 0.01 # For packages 10-18 ft³
}

# Parameters for the physical layout of the locker bank
LOCKER_GRID_UNITS = {'small': 1, 'medium': 2, 'large': 3} # Vertical space units each locker size occupies
GRID_HEIGHT = 15          # Total height of the locker bank in grid units
UI_MODULE_WIDTH = 2       # Width of the user interface/payment module in grid units

print("--- Starting Baseline Locker Bank Design Calculation ---\n")

# --- 3. Step 1: Calculate Peak Daily Demand ---
# This step estimates the number of packages arriving on the busiest day of the year.
print("Step 1: Calculating peak daily demand...")

# Calculate the number of parcels arriving during the peak 4-week period
parcels_in_peak_period = MAX_ANNUAL_DEMAND * PEAK_4_WEEK_SHARE
# Convert this to a weekly average during the peak period
avg_parcels_per_week_peak = parcels_in_peak_period / 4
# Calculate the number of parcels arriving on the single busiest day
peak_day_demand = avg_parcels_per_week_peak * PEAK_DAY_OF_WEEK_SHARE

print(f"Projected peak daily arrivals: {peak_day_demand:.2f}\n")

# --- 4. Step 2: Calculate Average Package Stay Duration ---
# This step determines how long, on average, a package stays in a locker.
print("Step 2: Calculating average package stay duration...")

# Read the deposit-to-pickup time probability matrix from the Excel file
df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, skiprows=9, nrows=24, usecols='D:AB')
# Convert the data into a numpy array and scale it, as original values were multiplied by 10,000
prob_matrix = df.iloc[:24, :24].values / 10000.0

# Calculate the weighted average duration
total_duration = 0
for deposit_hour in range(24):
    for pickup_hour in range(24):
        # Calculate the duration in hours, accounting for pickups that occur on the next day
        duration = (pickup_hour - deposit_hour) if pickup_hour >= deposit_hour else (pickup_hour + 24 - deposit_hour)
        # Add the duration weighted by its probability
        total_duration += prob_matrix[deposit_hour, pickup_hour] * duration

# The average duration is the total weighted duration divided by the sum of all probabilities (which should be 1)
avg_duration_hours = total_duration / np.sum(prob_matrix)

print(f"Average package stay duration: {avg_duration_hours:.2f} hours\n")


# --- 5. Step 3: Estimate Total Number of Lockers Required ---
# This step uses queueing theory to find the number of lockers needed to meet the service level target.
print("Step 3: Estimating the total number of lockers required...")

# a) Calculate the average hourly arrival rate (lambda) on the peak day
arrival_rate_hourly = peak_day_demand / 24

# b) Use Little's Law (L = λ * W) to find the average number of occupied lockers at any given time.
#    L = average items in system, λ = arrival rate, W = average time in system.
avg_occupied_lockers = arrival_rate_hourly * avg_duration_hours
print(f"Average number of occupied lockers (L): {avg_occupied_lockers:.2f}")

# c) Use the Poisson distribution to find the capacity needed to achieve the target service level.
#    We need the number of lockers 'C' such that the probability of (occupied lockers <= C) is at least 99.5%.
#    This is found using the Percent Point Function (PPF), which is the inverse of the CDF.
mu = avg_occupied_lockers  # The average (L) is the mean of our Poisson distribution
total_lockers_needed = int(poisson.ppf(SERVICE_LEVEL_TARGET, mu))

print(f"To achieve a {SERVICE_LEVEL_TARGET*100}% service level, an estimated {total_lockers_needed} lockers are needed.\n")


# --- 6. Step 4: Determine Locker Mix and Layout ---
# This step allocates the total number of lockers into sizes and calculates the physical dimensions.
print("Step 4: Determining locker mix and physical layout...")

# Calculate the number of lockers for each size based on probability
num_s = round(total_lockers_needed * SPACE_REQUIREMENT_PROB['small'])
num_m = round(total_lockers_needed * SPACE_REQUIREMENT_PROB['medium'])
# The number of large lockers is calculated by subtraction to ensure the total is correct
num_l = total_lockers_needed - num_s - num_m

# Calculate the total vertical grid space required by all lockers
total_grid_units_needed = (num_s * LOCKER_GRID_UNITS['small']) + \
                          (num_m * LOCKER_GRID_UNITS['medium']) + \
                          (num_l * LOCKER_GRID_UNITS['large'])

# Calculate the required width in grid units. This is the total grid space divided by the height,
# plus the space for the UI module.
total_width_units = (total_grid_units_needed / GRID_HEIGHT) + UI_MODULE_WIDTH
# Since the width must be a whole number of units, we round up to the nearest integer.
required_width = int(np.ceil(total_width_units))
print("Locker mix and layout determined successfully.\n")


# --- 7. Step 5: Display the Final Design Proposal ---
print("===========================================================")
print("--- Task 1: Baseline Fixed-Configuration Locker Bank Design ---")
print("===========================================================")
print(f"\nTotal Number of Lockers: {total_lockers_needed}")
print("\nLocker Size Distribution:")
print(f"  - Small (S):  {num_s} units")
print(f"  - Medium (M): {num_m} units")
print(f"  - Large (L):  {num_l} units")
print("\nPhysical Layout Proposal:")
print(f"  - Height: {GRID_HEIGHT} grid units")
print(f"  - Width:  {required_width} grid units (includes UI module)")
print("\n===========================================================\n")


# --- 8. Step 6: Visualize the Design ---
# This step creates a simple bar chart to show the distribution of locker sizes.
print("Generating a visualization of the locker distribution...")

# Data for the bar chart
sizes = ['Small', 'Medium', 'Large']
counts = [num_s, num_m, num_l]
colors = ['#4CAF50', '#FFC107', '#F44336']

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(sizes, counts, color=colors)

# Add titles and labels for clarity
plt.title('Proposed Locker Size Distribution', fontsize=16)
plt.xlabel('Locker Size', fontsize=12)
plt.ylabel('Number of Lockers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the chart
plt.show()

print("Process complete.")
