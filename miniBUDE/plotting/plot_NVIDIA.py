import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("results_NVIDIA.csv", on_bad_lines="skip")

palette = {
    "Mojo": "#0e2841",
    "CUDA-fastmath": "#76b900",
    "CUDA-no-fastmath": "#587c19"
}

linestyles = {
    "Mojo": "solid",
    "CUDA-fastmath": "solid",
    "CUDA-no-fastmath": "dashed",
}

markers = {
    "CUDA-no-fastmath": "^",
}

def plot_for_wgsize(wgsize, filename):
    fig, ax = plt.subplots(figsize=(11, 6))
    for backend in ["Mojo", "CUDA-no-fastmath", "CUDA-fastmath"]:
        subset = df[(df["backend"] == backend) & (df["wgsize"] == wgsize)]
        subset = subset.sort_values(by="ppwi")
        color = palette.get(backend, "#000000")
        linestyle = linestyles.get(backend, "solid")
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
        ax.set_ylim(top=10000)
        ax.text(14, 6000, "Theoretical peak = 60 TFLOPs",
            color="#757575", fontsize=15, ha='left')
        ax.text(14, 5300, "Higher is better",
            color="#757575", fontsize=15, ha='left')
    else:
        ax.set_ylim(top=5000)
        ax.text(14, 3000, "Theoretical peak = 60 TFLOPs",
            color="#757575", fontsize=15, ha='left')
        ax.text(14, 2700, "Higher is better",
            color="#757575", fontsize=15, ha='left')

    plt.show()

plot_for_wgsize(8, "wgsize_8.pdf")
plot_for_wgsize(64, "wgsize_64.pdf")
