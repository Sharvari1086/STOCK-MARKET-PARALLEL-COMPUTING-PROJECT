
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_FILE = os.path.join(RESULTS_DIR, "correlation_matrix.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "correlation_heatmap.png")

# Load correlation matrix
df = pd.read_csv(INPUT_FILE, header=None)
matrix = df.apply(pd.to_numeric, errors="coerce").to_numpy()

# Handle invalid values
matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

# Ensure the matrix is square
if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError("The correlation matrix must be square.")

# Keep correlation values within their valid range
matrix = np.clip(matrix, -1.0, 1.0)

n = matrix.shape[0]

# Professional plotting style
sns.set_theme(style="white", context="paper", font_scale=1.0)

fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

# Diverging colors:
# Blue = negative, white = zero, red = positive
heatmap = sns.heatmap(
    matrix,
    ax=ax,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    xticklabels=False,
    yticklabels=False,
    linewidths=0,
    rasterized=True,
    cbar_kws={
        "label": "Pearson Correlation Coefficient",
        "shrink": 0.82,
        "pad": 0.025,
        "ticks": [-1, -0.5, 0, 0.5, 1]
    }
)

# Labels and title
ax.set_title(
    f"Pairwise Stock Correlation Matrix ({n} Stocks)",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlabel("Stock Index", fontsize=11, labelpad=8)
ax.set_ylabel("Stock Index", fontsize=11, labelpad=8)

# Show a few index ticks without overcrowding the figure
step = max(1, n // 8)
positions = np.arange(0, n, step)

ax.set_xticks(positions + 0.5)
ax.set_yticks(positions + 0.5)

ax.set_xticklabels(positions, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(positions, rotation=0, fontsize=8)

# Keep the matrix oriented consistently
ax.tick_params(axis="both", length=0)

# Color bar formatting
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)
cbar.set_label(
    "Pearson Correlation Coefficient",
    fontsize=10,
    labelpad=10
)

# Save high-resolution figures
fig.savefig(
    OUTPUT_FILE,
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print(f"Heatmap saved to: {OUTPUT_FILE}")
print(f"Matrix dimensions: {n} x {n}")
print(f"Correlation range: [{matrix.min():.6f}, {matrix.max():.6f}]")
