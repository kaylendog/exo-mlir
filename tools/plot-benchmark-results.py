import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) != 4:
    print("Usage: python plot-benchmark-results.py <csv_path> <level> <proc>")
    sys.exit(1)

csv_path = sys.argv[1]
level = sys.argv[2]
proc = sys.argv[3]

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

out_dir = os.path.join(os.getcwd(), f"build/plots/{level}")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, f"{proc}.png")
plt.tight_layout()
plt.savefig(out_path)
plt.close()
