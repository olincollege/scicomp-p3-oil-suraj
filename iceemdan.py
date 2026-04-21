"""
iceemdan.py  From-scratch ICEEMDAN signal decomposition.

Implements the Improved Complete Ensemble Empirical Mode Decomposition
with Adaptive Noise (ICEEMDAN) algorithm from Torres et al. (2011),
as applied in Li et al. (2019) for crude oil price forecasting.

Decomposes a nonlinear, nonstationary signal into a set of Intrinsic Mode
Functions (IMFs) ordered from highest to lowest frequency, plus a monotonic
residue capturing the long-term trend.
 
Algorithm hierarchy:
    find_extrema  make_envelopes   sift    emd   iceemdan
"""

import numpy as np
from scipy.interpolate import CubicSpline


def find_extrema(signal):
    """Locate local maxima and minima via vectorized first-difference sign changes.
 
    Uses np.diff to compute the derivative sign, then detects sign transitions:
    negative change = local maximum, positive change = local minimum.
    Flat segments are handled by forward-filling the last nonzero sign value.
 
    Endpoints (index 0 and N-1) are appended to both lists to anchor the
    cubic spline envelopes at the signal boundaries.
 
    Returns:
        maxima: int array of indices where local maxima occur
        minima: int array of indices where local minima occur
    """
    N = len(signal)
    d = np.diff(signal)
    sign_d = np.sign(d)
    nonzero = sign_d != 0
    if np.any(nonzero):
        idx = np.where(nonzero, np.arange(len(sign_d)), 0)
        np.maximum.accumulate(idx, out=idx)
        sign_d = sign_d[idx]
    sign_changes = np.diff(sign_d)
    maxima = np.where(sign_changes < 0)[0] + 1
    minima = np.where(sign_changes > 0)[0] + 1

    if len(maxima) > 0:
        maxima = np.concatenate(([0], maxima, [N - 1]))
    if len(minima) > 0:
        minima = np.concatenate(([0], minima, [N - 1]))

    return maxima.astype(int), minima.astype(int)


def make_envelopes(signal):
    """Construct upper and lower envelopes via cubic spline interpolation.
 
    Fits a cubic spline through local maxima (upper envelope) and through
    local minima (lower envelope). Their mean approximates the local trend
    that sifting removes.
 
    Raises ValueError if fewer than 2 extrema exist (monotonic signal).
    """
    t = np.arange(len(signal))
    maxima, minima = find_extrema(signal)

    if len(maxima) < 2 or len(minima) < 2:
        raise ValueError("Not enough extrema for envelope construction.")

    upper = CubicSpline(maxima, signal[maxima])(t)
    lower = CubicSpline(minima, signal[minima])(t)

    return upper, lower


def sift(signal, max_iterations=100, threshold=1e-9):
    """Extract one IMF by iterative mean-envelope subtraction.
 
    Repeatedly computes the mean of upper and lower envelopes and subtracts
    it from the signal until the Cauchy convergence criterion is satisfied:
        sum((h - h_next)^2) / sum(h^2) < threshold
 
    Most signals converge in 10-30 iterations. The max_iterations ceiling
    prevents infinite loops on pathological inputs.
 
    Returns the extracted IMF (a zero-mean oscillatory component).
    """
    h = signal.copy()

    for _ in range(max_iterations):
        try:
            upper, lower = make_envelopes(h)
        except ValueError:
            break

        mean_env = (upper + lower) / 2.0
        h_next = h - mean_env

        denom = np.sum(h ** 2)
        if denom > 0:
            change = np.sum((h - h_next) ** 2) / denom
            if change < threshold:
                h = h_next
                break

        h = h_next

    return h


def emd(signal, max_imfs=11, max_sift_iter=100, sift_threshold=1e-9):
    """Empirical Mode Decomposition — extract all IMFs from a signal.
 
    Repeatedly sifts the residue to peel off one IMF at a time, ordered from
    highest frequency to lowest. Terminates when the residue has fewer than
    2 extrema (monotonic) or max_imfs components have been extracted.
 
    Invariant: sum(imfs) + residue == signal (to machine precision).
    """
    imfs = []
    residue = signal.copy()

    for _ in range(max_imfs):
        maxima, minima = find_extrema(residue)
        if len(maxima) < 2 or len(minima) < 2:
            break

        imf = sift(residue, max_sift_iter, sift_threshold)
        imfs.append(imf)
        residue = residue - imf

    return np.array(imfs), residue


def iceemdan(signal, max_imfs=11, noise_std=0.08, n_realizations=500,
             max_sift_iter=100, sift_threshold=1e-9, seed=42):
    """Improved Complete Ensemble EMD with Adaptive Noise.
 
    Wraps EMD with noise averaging to eliminate mode mixing. At each
    decomposition stage k:
        1. Add the k-th pre-decomposed noise IMF to the current residue
        2. Sift the perturbed signal to extract a candidate IMF
        3. Repeat for all N noise realizations
        4. Average the N candidates to produce the final IMF k
 
    The averaging cancels noise artifacts while preserving real structure.
    More realizations = cleaner decomposition but longer runtime.
 
    Parameters (defaults match paper Table 1):
        signal:          1D price array (e.g. 5,338 training days)
        max_imfs:        cap on extracted components (paper: 11)
        noise_std:       noise level as fraction of signal std (paper: 0.05)
        n_realizations:  noise trials to average (paper: 500)
        max_sift_iter:   sifting iteration ceiling (paper: 5000)
        sift_threshold:  Cauchy convergence criterion (1e-9)
        seed:            random seed for reproducibility
 
    Returns:
        imfs:    2D array (n_imfs, signal_length), high-freq first
        residue: 1D array, the monotonic trend
    """
    rng = np.random.default_rng(seed)
    N = len(signal)
    sigma = noise_std * np.std(signal)

    #Pre-decompose all noise realizations
    noise_imfs_all = []
    for i in range(n_realizations):
        noise = rng.normal(0, 1, size=N)
        noise_imfs, _ = emd(noise, max_imfs, max_sift_iter, sift_threshold)
        noise_imfs_all.append(noise_imfs)

    #ICEEMDAN  at stage k, add k-th noise IMF to residue, sift, average
    imfs = []
    residue = signal.copy()

    for k in range(max_imfs):
        maxima, minima = find_extrema(residue)
        if len(maxima) < 2 or len(minima) < 2:
            break

        candidates = []
        for i in range(n_realizations):
            noise_imfs = noise_imfs_all[i]
            if k < len(noise_imfs):
                noise_component = sigma * noise_imfs[k]
            else:
                noise_component = np.zeros(N)

            perturbed = residue + noise_component
            candidate = sift(perturbed, max_sift_iter, sift_threshold)
            candidates.append(candidate)

        imf_k = np.mean(candidates, axis=0)
        imfs.append(imf_k)
        residue = residue - imf_k

    return np.array(imfs), residue


if __name__ == "__main__":
    # Synthetic test two sine waves at different frequencies and linear trend
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 2 * t

    print("EMD on synthetic signal (500 points, 2 sines + trend):")
    imfs, residue = emd(signal, max_imfs=5)
    recon_error = np.max(np.abs(signal - (np.sum(imfs, axis=0) + residue)))
    print(f"  IMFs extracted:        {len(imfs)}")
    print(f"  Reconstruction error:  {recon_error:.2e}")
    for i, imf in enumerate(imfs):
        zc = np.sum(np.diff(np.sign(imf)) != 0)
        print(f"  IMF {i+1}: {zc} zero crossings")
    print()

    print("ICEEMDAN (n_realizations=10):")
    imfs_ice, res_ice = iceemdan(signal, max_imfs=5, n_realizations=10)
    recon_error_ice = np.max(np.abs(signal - (np.sum(imfs_ice, axis=0) + res_ice)))
    print(f"  IMFs extracted:        {len(imfs_ice)}")
    print(f"  Reconstruction error:  {recon_error_ice:.2e}")
    for i, imf in enumerate(imfs_ice):
        zc = np.sum(np.diff(np.sign(imf)) != 0)
        print(f"  IMF {i+1}: {zc} zero crossings")
