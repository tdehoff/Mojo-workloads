import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter

df = pd.read_csv("results_NVIDIA.csv", skipinitialspace=True)

df["precision"] = df["precision"].str.strip().replace({"double": "float64", "float": "float32"})
df["backend"] = df["backend"].str.strip().str.lower().replace({"mojo": "Mojo", "cuda": "CUDA"})
df["config"] = df.apply(lambda r: f"L={r['L']}\n({r['blk_x']}×{r['blk_y']}×{r['blk_z']})", axis=1)
config_order = df["config"].drop_duplicates().tolist()
df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)

df32 = df[df["precision"] == "float32"]
df64 = df[df["precision"] == "float64"]

palette = {"Mojo": "#0e2841", "CUDA": "#76b900"}
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for ax, d, title in zip(axes, [df32, df64], ["Single Precision", "Double Precision"]):
    sns.stripplot(
        data=d,
        x="config",
        y="BW_GBs",
        hue="backend",
        dodge=True,
        jitter=0.3,
        ax=ax,
        palette=palette,
        size=5,
        alpha=0.5,
        legend=True
    )
    ax.legend_.remove()

    peak_bw = 3900
    ax.axhline(peak_bw, color='red', linestyle='--', linewidth=1)
    ax.text(4 - 0.5, peak_bw + 150, f'Theoretical Peak = {peak_bw} GB/s',
            color="#757575", fontsize=15, ha='right')

    ax.set_yscale("log")

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))

    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 5]))

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)

    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.tick_params(axis='y', which='minor', labelsize=13)

    ax.set_ylabel("Bandwidth (GB/s)", fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(top=5000)

axes[0].text(0, 2800, "Higher is better",
        color="#757575", fontsize=17, ha='left')

for ax in axes:
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelbottom=True, labelsize=13)

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.2, zorder=0)  # optional x grid

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Mojo',
           markerfacecolor=palette["Mojo"], markersize=9, alpha=1),
    Line2D([0], [0], marker='o', color='w', label='CUDA',
           markerfacecolor=palette["CUDA"], markersize=9, alpha=1),
]

axes[1].legend(handles=legend_elements, loc="lower left", fontsize=14, frameon=True)

plt.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.16, hspace=0.35)
plt.tight_layout()
plt.show()
