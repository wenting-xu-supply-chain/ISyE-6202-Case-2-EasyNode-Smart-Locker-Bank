import numpy as np
import matplotlib.pyplot as plt
import os ### MODIFICATION: Import the os module ###

def generate_mixed_arrival_times(N, lsp_proportion=0.8, lsp_mean=15, lsp_std=5):
    """
    Generates N package arrival times using a mixed model for LSPs and customers.

    Args:
        N (int): The total number of packages arriving in the hour.
        lsp_proportion (float): The assumed proportion of arrivals from LSPs.
        lsp_mean (float): The mean arrival time for LSPs (e.g., 15 minutes).
        lsp_std (float): The standard deviation for LSP arrival times.

    Returns:
        list: A sorted list of N arrival times for all packages.
    """
    if N <= 0:
        return []

    # 1. Calculate sub-totals for each group
    n_lsp = int(round(N * lsp_proportion))
    n_cust = N - n_lsp

    # 2. Generate timestamps for each group
    # LSP arrivals (Normally distributed and clipped to be within the hour)
    lsp_times = np.random.normal(loc=lsp_mean, scale=lsp_std, size=n_lsp)
    lsp_times = np.clip(lsp_times, 0, 60) # Ensure times are within [0, 60]

    # Customer arrivals (Uniformly distributed)
    cust_times = np.random.uniform(0, 60, n_cust)

    # 3. Combine and sort
    all_times = np.concatenate((lsp_times, cust_times))
    all_times.sort()

    return list(all_times)

# --- Demonstration ---
num_arrivals = 1000 # Increased for a better visualization
example_times = generate_mixed_arrival_times(num_arrivals, lsp_proportion=0.8)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.hist(example_times, bins=60, color='skyblue', edgecolor='black')
plt.title('Distribution of Package Arrival Times')
plt.xlabel('Time (minutes past the hour)')
plt.ylabel('Number of Packages')
plt.grid(axis='y', alpha=0.75)

### MODIFICATION START ###
# Create the output folder and define the full file path
output_folder = "Task_2_Outcome"
os.makedirs(output_folder, exist_ok=True) # Creates the folder if it doesn't exist

filename = 'arrival_times_histogram.png'
output_filepath = os.path.join(output_folder, filename)

# Save the plot to the new path
plt.savefig(output_filepath)
### MODIFICATION END ###

plt.show()

print(f"Generated and visualized mixed-model arrival times for {num_arrivals} packages.")
### MODIFICATION: Update the print statement to show the new path ###
print(f"The histogram has been saved as '{output_filepath}'")
