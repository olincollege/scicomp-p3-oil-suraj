import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#Plot raw signal, all IMFs, and residue as stacked subplots.
def plot_decomposition(signal, imfs, residue, save_path="decomposition.png"):
    n_panels = len(imfs) + 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2 * n_panels), sharex=True)

    axes[0].plot(signal, color="black", linewidth=0.5)
    axes[0].set_ylabel("Raw ($)")
    axes[0].set_title("ICEEMDAN Decomposition of WTI Crude Oil Prices",
                       fontsize=13, fontweight="bold")

    for i, imf in enumerate(imfs):
        axes[i + 1].plot(imf, color="steelblue", linewidth=0.5)
        axes[i + 1].set_ylabel(f"IMF {i+1}")

    axes[-1].plot(residue, color="firebrick", linewidth=0.5)
    axes[-1].set_ylabel("Residue")
    axes[-1].set_xlabel("Trading day")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")

   
#Overlay actual vs predicted prices on the test set.
def plot_forecast(actual, predicted, title, save_path="forecast.png"):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(actual, color="black", linewidth=0.8, label="Actual")
    ax.plot(predicted, color="steelblue", linewidth=0.8, alpha=0.8, label="Predicted")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Test day")
    ax.set_ylabel("Price ($)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")

#Bar chart comparing MAPE and RMSE across models.
def plot_model_comparison(results, save_path="comparison.png"):
    names = [r[0] for r in results]
    mapes = [r[1] for r in results]
    rmses = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(names, mapes, color=["gray", "steelblue", "firebrick"])
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("Mean Absolute Percentage Error", fontweight="bold")
    for bar, val in zip(bars1, mapes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}%", ha="center", fontsize=10)

    bars2 = ax2.bar(names, rmses, color=["gray", "steelblue", "firebrick"])
    ax2.set_ylabel("RMSE ($)")
    ax2.set_title("Root Mean Square Error", fontweight="bold")
    for bar, val in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"${val:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")
