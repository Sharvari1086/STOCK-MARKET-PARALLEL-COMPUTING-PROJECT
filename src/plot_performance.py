
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

file_path = "results/performance.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Run correlation.exe first to generate results/performance.csv"
    )

df = pd.read_csv(file_path)
result = df.iloc[0]

sequential_time = float(result["SequentialTime"])
parallel_time = float(result["ParallelTime"])
threads = int(result["Threads"])
speedup = float(result["Speedup"])
efficiency = float(result["EfficiencyPercent"])
max_difference = float(result["MaxDifference"])
diagonal_error = float(result["MaxDiagonalError"])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Execution time comparison
time_ax = axes[0, 0]

bars = time_ax.bar(
    ["Sequential", "OpenMP Parallel"],
    [sequential_time, parallel_time]
)

time_ax.set_title("Actual Execution Time")
time_ax.set_ylabel("Time (seconds)")
time_ax.bar_label(bars, fmt="%.6f s", padding=4)

# Speedup
speed_ax = axes[0, 1]

bars = speed_ax.bar(
    ["OpenMP"],
    [speedup]
)

speed_ax.set_title("Speedup")
speed_ax.set_ylabel("Times Faster")
speed_ax.bar_label(bars, fmt="%.2fx", padding=4)

# Parallel efficiency
eff_ax = axes[1, 0]

bars = eff_ax.bar(
    ["OpenMP"],
    [efficiency]
)

eff_ax.set_title(f"Parallel Efficiency ({threads} Threads)")
eff_ax.set_ylabel("Efficiency (%)")
eff_ax.set_ylim(0, max(100, efficiency * 1.15))
eff_ax.bar_label(bars, fmt="%.2f%%", padding=4)

# Correctness check
check_ax = axes[1, 1]
check_ax.axis("off")

status = (
    "PASSED"
    if max_difference < 1e-9 and diagonal_error < 1e-6
    else "CHECK REQUIRED"
)

check_text = (
    "CORRECTNESS CHECK\n\n"
    f"Status: {status}\n\n"
    f"Maximum sequential-parallel difference:\n"
    f"{max_difference:.3e}\n\n"
    f"Maximum diagonal error from 1.0:\n"
    f"{diagonal_error:.3e}\n\n"
    "Diagonal correlation values should be near 1."
)

check_ax.text(
    0.05,
    0.85,
    check_text,
    transform=check_ax.transAxes,
    fontsize=11,
    verticalalignment="top"
)

# Final layout and save
fig.suptitle(
    "Stock Correlation: Sequential vs OpenMP",
    fontsize=15
)

plt.tight_layout()

output_path = "results/performance_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Print actual measured results
print("\nACTUAL BENCHMARK RESULTS")
print("-" * 35)
print(f"Sequential time:      {sequential_time:.6f} seconds")
print(f"Parallel time:        {parallel_time:.6f} seconds")
print(f"Speedup:              {speedup:.3f}x")
print(f"Parallel efficiency:  {efficiency:.2f}%")
print(f"Maximum difference:   {max_difference:.3e}")
print(f"Correctness:          {status}")
print(f"Graph saved to:       {output_path}")
