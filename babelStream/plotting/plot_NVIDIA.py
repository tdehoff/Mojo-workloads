import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter

df = pd.read_csv("results_NVIDIA.csv", skipinitialspace=True)

df["backend"] = df["backend"].astype(str).str.strip()
df["routine"] = df["routine"].astype(str).str.strip()

routine_order = ["Copy", "Mul", "Add", "Triad", "Dot"]
df["routine"] = pd.Categorical(df["routine"], categories=routine_order, ordered=True)

palette = {"Mojo": "#0e2841", "CUDA": "#76b900"}

plt.figure(figsize=(12, 6))

ax = sns.stripplot(
    data=df,
    x="routine",
    y="BW_GBs",
    hue="backend",
    dodge=True,
    jitter=0.3,
    palette=palette,
    size=5,
    alpha=0.5,
)

peak_bw = 3900
ax.axhline(peak_bw, color='red', linestyle='--', linewidth=1)
ax.text(5 - 0.5, 3950, f'Theoretical Peak = {peak_bw} GB/s',
        color="#757575", fontsize=18, ha='right')

ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 5]))
ax.set_ylim(bottom=2000)

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)

ax.set_xlabel(None)

ax.text(0, 3650, "Higher is better",
        color="#757575", fontsize=17, ha='left')
ax.text(0, 3450, "Vector size = $2^{25}$ elements",
        color="#757575", fontsize=17, ha='left')

ax.set_ylabel("Bandwidth (GB/s)", fontsize=17)
ax.tick_params(axis='both', labelsize=17)
ax.yaxis.set_tick_params(which='minor', labelsize=17)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Mojo',
           markerfacecolor=palette["Mojo"], markersize=9, alpha=1),
    Line2D([0], [0], marker='o', color='w', label='CUDA',
           markerfacecolor=palette["CUDA"], markersize=9, alpha=1),
]

ax.legend(handles=legend_elements, loc="lower left", fontsize=14, frameon=True)

ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3, zorder=0)

plt.tight_layout()
plt.show()
