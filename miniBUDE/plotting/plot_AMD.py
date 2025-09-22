import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("results_AMD.csv")

palette = {
    "Mojo": "#0e2841",
    "HIP-fastmath": "#ed1c24",
    "HIP-no-fastmath": "#772D27"
}

linestyles = {
    "Mojo": "solid",
    "HIP-fastmath": "solid",
    "HIP-no-fastmath": "dashed",
}

markers = {
    "HIP-no-fastmath": "^",
}

def plot_for_wgsize(wgsize, filename):
    fig, ax = plt.subplots(figsize=(11, 6))
    for backend in ["Mojo", "HIP-no-fastmath", "HIP-fastmath"]:
        subset = df[(df["backend"] == backend) & (df["wgsize"] == wgsize)]
        subset = subset.sort_values(by="ppwi")
        linestyle = linestyles.get(backend, "solid")
        color = palette.get(backend, "#000000")
        marker = markers.get(backend, "o")
        ax.plot(subset["ppwi"], subset["gflops/s"], marker=marker, label=backend, color=color, linestyle=linestyle)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("PPWI", fontsize=17)
    ax.set_ylabel("GFLOP/s", fontsize=17)
    ax.legend(fontsize=14)

    if wgsize == 64:
        ax.text(14, 5000, "Theoretical peak = 122.6 TFLOPs",
            color="#757575", fontsize=15, ha='left')
        ax.text(14, 4200, "Higher is better",
            color="#757575", fontsize=15, ha='left')
    else:
        ax.text(14, 2000, "Theoretical peak = 122.6 TFLOPs",
            color="#757575", fontsize=15, ha='left')
        ax.text(14, 1700, "Higher is better",
            color="#757575", fontsize=15, ha='left')

    plt.show()

plot_for_wgsize(8, "wgsize_8.pdf")
plot_for_wgsize(64, "wgsize_64.pdf")
