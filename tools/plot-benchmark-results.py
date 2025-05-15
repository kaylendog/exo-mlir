import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) != 5:
    print("Usage: python plot-benchmark-results.py <csv_path> <level> <proc> <output>")
    sys.exit(1)

csv_path = sys.argv[1]
level = sys.argv[2]
proc = sys.argv[3]
output = sys.argv[4]

df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(8, 6))

compiler_colors = {"exocc": "#33a02c", "exomlir": "#1f78b4"}  # green, blue

for compiler, group in df.groupby("compiler"):
    ax.errorbar(
        group["size"],
        group["mean"],
        yerr=group["stddev"],
        fmt="o",
        label=compiler,
        color=compiler_colors[compiler],
        capsize=5,
        markersize=5,
        linestyle="--",
        linewidth=1.5,
    )

ax.set_xlabel("Problem Size")
ax.set_ylabel("CPU Time (ns)")
ax.set_title(f"{proc} {level}")
ax.set_xscale("log", base=2)
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)


os.makedirs(os.path.dirname(output), exist_ok=True)

plt.tight_layout()
plt.savefig(output)
plt.close()
