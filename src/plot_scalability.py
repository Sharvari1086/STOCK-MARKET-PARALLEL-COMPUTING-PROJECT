
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/scalability.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    df["Threads"],
    df["ParallelTime"],
    marker="o"
)
axes[0].set_title("Scalability: Execution Time")
axes[0].set_xlabel("Number of Threads")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_xticks(df["Threads"])
axes[0].grid(True)

axes[1].plot(
    df["Threads"],
    df["Speedup"],
    marker="o",
    label="Measured speedup"
)
axes[1].plot(
    df["Threads"],
    df["Threads"],
    linestyle="--",
    label="Ideal speedup"
)
axes[1].set_title("Measured vs Ideal Speedup")
axes[1].set_xlabel("Number of Threads")
axes[1].set_ylabel("Speedup (times)")
axes[1].set_xticks(df["Threads"])
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

output_path = "results/scalability_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSCALABILITY RESULTS")
print(df.to_string(index=False))
print(f"\nGraph saved to {output_path}")
