
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/stock_metrics_performance.csv")
row = df.iloc[0]

sequential_time = row["SequentialTime"]
parallel_time = row["ParallelTime"]
threads = int(row["Threads"])
speedup = row["Speedup"]
efficiency = row["EfficiencyPercent"]
max_difference = row["MaxDifference"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].bar(
    ["Sequential", "OpenMP Parallel"],
    [sequential_time, parallel_time]
)
axes[0, 0].set_title("Stock Metrics Execution Time")
axes[0, 0].set_ylabel("Time (seconds)")

axes[0, 1].bar(
    ["Measured Speedup"],
    [speedup]
)
axes[0, 1].axhline(1, linestyle="--", label="Baseline: 1x")
axes[0, 1].set_title("Speedup")
axes[0, 1].set_ylabel("Speedup (x)")
axes[0, 1].legend()

axes[1, 0].bar(
    ["Parallel Efficiency"],
    [efficiency]
)
axes[1, 0].set_title(f"Parallel Efficiency ({threads} threads)")
axes[1, 0].set_ylabel("Efficiency (%)")

axes[1, 1].bar(
    ["Maximum Difference"],
    [max_difference]
)
axes[1, 1].set_title("Sequential vs Parallel Correctness")
axes[1, 1].set_ylabel("Absolute difference")

for ax in axes.flat:
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(
    "results/stock_metrics_performance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Sequential time:", sequential_time, "seconds")
print("Parallel time:", parallel_time, "seconds")
print("Speedup:", speedup, "x")
print("Efficiency:", efficiency, "%")
print("Maximum result difference:", max_difference)
print("Saved results/stock_metrics_performance.png")
