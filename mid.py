"""
mid.py — Mid-project code conversation demo.

  1. Data pipeline works (load, split, normalize)
  2. ICEEMDAN decomposes real oil prices into IMFs
  3. Reconstruction is perfect (IMFs + residue = original)
  4. Frequency separation is correct (high freq to low freq)
  5. Decomposition plot 

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_wti, split_data
from iceemdan import iceemdan


def main():
    print("Loadin WTI crude oil prices")

    prices, dates = load_wti()
    train, val, test = split_data(prices)

    print(f"  Total samples:    {len(prices)}")
    print(f"  Train:            {len(train)}")
    print(f"  Validation:       {len(val)}")
    print(f"  Test:             {len(test)}")
    print(f"  Price range:      ${prices.min():.2f} - ${prices.max():.2f}")
    print()


    N_REAL = 50


    print(f"ICEEMDAN decomposition (n_realizations={N_REAL})")
    print(f"  Signal length:    {len(train)} samples")
    print(f"  noise_std=0.08, max_imfs=11")

    imfs, residue = iceemdan(train, max_imfs=11, noise_std=0.08,
                              n_realizations=N_REAL, seed=42)

    print(f"  Extracted {len(imfs)} IMFs + 1 residue")


 
    print("Reconstruction check")

    reconstructed = np.sum(imfs, axis=0) + residue
    recon_error = np.max(np.abs(train - reconstructed))

    print(f"  Max error: {recon_error:.2e}")
    if recon_error < 1e-6:
        print("  PASS — IMFs + residue = original signal")
    else:
        print("  FAIL — reconstruction error too large")
    print()

    print("Frequency analysis")

    print(f"  {'Component':<12} {'Zero crossings':>15} {'Std dev':>10}")
    print(f"  {'-'*12} {'-'*15} {'-'*10}")

    for i, imf in enumerate(imfs):
        zc = np.sum(np.diff(np.sign(imf)) != 0)
        print(f"  IMF {i+1:<7} {zc:>15} {np.std(imf):>10.2f}")

    zc_res = np.sum(np.diff(np.sign(residue)) != 0)
    print(f"  {'Residue':<12} {zc_res:>15} {np.std(residue):>10.2f}")
    print()


    print("decomposition plot")


    n_panels = len(imfs) + 2  # raw + IMFs + residue
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2 * n_panels),
                              sharex=True)

    axes[0].plot(train, color="black", linewidth=0.5)
    axes[0].set_ylabel("Raw ($)")
    axes[0].set_title("ICEEMDAN Decomposition — WTI Training Data",
                       fontsize=13, fontweight="bold")

    for i, imf in enumerate(imfs):
        axes[i + 1].plot(imf, color="steelblue", linewidth=0.5)
        axes[i + 1].set_ylabel(f"IMF {i+1}")

    axes[-1].plot(residue, color="firebrick", linewidth=0.5)
    axes[-1].set_ylabel("Residue")
    axes[-1].set_xlabel("Trading day")

    plt.tight_layout()
    plt.savefig("decomposition.png", dpi=150, bbox_inches="tight")
    print("  Saved -> decomposition.png")
    plt.close()


    print(f"  Data:                {len(prices)} trading days loaded")
    print(f"  Decomposition:       {len(imfs)} IMFs + 1 residue")
    print(f"  Reconstruction:      {recon_error:.2e} max error")
    print(f"  Plot:                decomposition.png")



if __name__ == "__main__":
    main()
