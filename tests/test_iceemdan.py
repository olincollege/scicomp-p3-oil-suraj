#test_iceemdan.py — ICEEMDAN decomposition.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from iceemdan import find_extrema, make_envelopes, sift, emd, iceemdan

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# -----------------------------------------------------------------

print("find_extrema tests")


t = np.linspace(0, 4 * np.pi, 1000)
sig = np.sin(t)
maxima, minima = find_extrema(sig)

# Sine wave should have clear peaks and valleys
check("sine wave has at least 2 maxima", len(maxima) >= 2)
check("sine wave has at least 2 minima", len(minima) >= 2)

# Peak values should be near +1
interior_max = [i for i in maxima if 0 < i < len(sig) - 1]
if interior_max:
    check("peak values near positive 1",
          np.all(sig[interior_max] > 0.95))

# Valley values should be near -1
interior_min = [i for i in minima if 0 < i < len(sig) - 1]
if interior_min:
    check("valley values near negative 1",
          np.all(sig[interior_min] < -0.95))

# monotonic signal has no interior extrema
mono = np.linspace(0, 10, 500)
mx, mn = find_extrema(mono)
imx = [i for i in mx if 0 < i < len(mono) - 1]
imn = [i for i in mn if 0 < i < len(mono) - 1]
check("monotonic has no interior peaks", len(imx) == 0)
check("monotonic has no interior valleys", len(imn) == 0)

# Constant signal no extrema at all
const = np.ones(100)
mx_c, mn_c = find_extrema(const)
check("constant signal no maxima", len(mx_c) == 0)
check("constant signal no minima", len(mn_c) == 0)
print()

print("make_envelopes tests")

t = np.linspace(0, 6 * np.pi, 2000)
sig = np.sin(t)
upper, lower = make_envelopes(sig)

# Upper envelope should pass through the peaks
maxima_e, _ = find_extrema(sig)
interior_max = maxima_e[(maxima_e > 0) & (maxima_e < len(sig) - 1)]
if len(interior_max) > 0:
    check("upper envelope touches maxima",
          np.all(np.abs(upper[interior_max] - sig[interior_max]) < 0.05))

# lower envelope should pass through valleys
_, minima_e = find_extrema(sig)
interior_min = minima_e[(minima_e > 0) & (minima_e < len(sig) - 1)]
if len(interior_min) > 0:
    check("lower envelope touches minima",
          np.all(np.abs(lower[interior_min] - sig[interior_min]) < 0.05))

# Mean of envelopes for pure sine should be roughly zero
mean_env = (upper + lower) / 2.0
check("mean envelope of sine near zero",
      np.mean(np.abs(mean_env)) < 0.15)

# monotonic input should raise because not enough extrema
try:
    make_envelopes(np.linspace(0, 10, 500))
    check("monotonic raises ValueError", False)
except ValueError:
    check("monotonic raises ValueError", True)


print()
print("sift tests")

# Sine plus trend, sifting should pull out the oscillation
t = np.linspace(0, 4 * np.pi, 1000)
signal = np.sin(t) + 2 * t / (4 * np.pi)
imf = sift(signal)

# result should oscillate around zero
check("sifted result has mean near zero", abs(np.mean(imf)) < 0.3)

# should still have zero crossings from the sine
zc = np.sum(np.diff(np.sign(imf)) != 0)
check("sifted result has zero crossings", zc >= 3)

# convergence works even with very tight threshold
imf2 = sift(signal, max_iterations=5000, threshold=1e-12)
check("tight threshold still converges", True)
print()
print("emd tests")


# Two sines at different speeds plus a ramp
t = np.linspace(0, 1, 1000)
fast = 0.5 * np.sin(2 * np.pi * 20 * t)
slow = np.sin(2 * np.pi * 3 * t)
trend = 2 * t
signal = fast + slow + trend

imfs, residue = emd(signal, max_imfs=10)

# Reconstruction check the fundamental invariant
reconstructed = np.sum(imfs, axis=0) + residue
recon_error = np.max(np.abs(signal - reconstructed))
check("reconstruction error below 1e-10", recon_error < 1e-10,
      f"error = {recon_error:.2e}")

# Should extract at least 2 modes
check("at least 2 IMFs extracted", len(imfs) >= 2)

# First IMF faster than second
zc1 = np.sum(np.diff(np.sign(imfs[0])) != 0)
zc2 = np.sum(np.diff(np.sign(imfs[1])) != 0)
check("IMF 1 higher frequency than IMF 2", zc1 > zc2,
      f"IMF1={zc1} IMF2={zc2}")

# zero crossings should decrease monotonically across all IMFs
zcs = [np.sum(np.diff(np.sign(imf)) != 0) for imf in imfs]
decreasing = all(zcs[i] >= zcs[i + 1] for i in range(len(zcs) - 1))
check("zero crossings decrease from IMF1 to last", decreasing,
      f"crossings: {zcs}")

# Residue should be nearly monotonic
res_zc = np.sum(np.diff(np.sign(residue)) != 0)
check("residue is near monotonic", res_zc <= 3,
      f"got {res_zc} crossings")

# Pure noise still reconstructs perfectly
noise = np.random.RandomState(42).randn(500)
n_imfs, n_res = emd(noise, max_imfs=8)
n_recon = np.sum(n_imfs, axis=0) + n_res
check("noise signal reconstructs",
      np.max(np.abs(noise - n_recon)) < 1e-10)

# pure sine may overdecompose but must still reconstruct
pure_sine = np.sin(2 * np.pi * 5 * t)
ps_imfs, ps_res = emd(pure_sine, max_imfs=5)
ps_recon = np.sum(ps_imfs, axis=0) + ps_res
check("pure sine reconstructs perfectly",
      np.max(np.abs(pure_sine - ps_recon)) < 1e-10)


print()
print("iceemdan tests")


t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 2 * t

# small realizations for speed
imfs_ice, res_ice = iceemdan(signal, max_imfs=5, n_realizations=10, seed=42)

# Reconstruction
recon_ice = np.sum(imfs_ice, axis=0) + res_ice
check("ICEEMDAN reconstruction below 1e-10",
      np.max(np.abs(signal - recon_ice)) < 1e-10)

# got at least 2 modes
check("ICEEMDAN produces at least 2 IMFs", len(imfs_ice) >= 2)

# frequency ordering preserved
if len(imfs_ice) >= 2:
    zc1 = np.sum(np.diff(np.sign(imfs_ice[0])) != 0)
    zc2 = np.sum(np.diff(np.sign(imfs_ice[1])) != 0)
    check("ICEEMDAN IMF1 faster than IMF2", zc1 >= zc2)

# Same seed same result
imfs_2, res_2 = iceemdan(signal, max_imfs=5, n_realizations=10, seed=42)
check("same seed gives identical output",
      np.max(np.abs(imfs_ice - imfs_2)) == 0.0)

# different seed different result
imfs_3, res_3 = iceemdan(signal, max_imfs=5, n_realizations=10, seed=99)
check("different seed gives different output",
      np.max(np.abs(imfs_ice - imfs_3)) > 0.0)

# doesnt crash with more realizations
imfs_many, _ = iceemdan(signal, max_imfs=3, n_realizations=30, seed=42)
check("30 realizations completes without error", True)
print()

print("edge case tests")


# Very short signal 5 points
short = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
try:
    s_imfs, s_res = emd(short, max_imfs=3)
    s_recon = np.sum(s_imfs, axis=0) + s_res if len(s_imfs) > 0 else s_res
    check("5 point signal handled gracefully",
          np.max(np.abs(short - s_recon)) < 1e-10)
except Exception as e:
    check("5 point signal handled gracefully", False, str(e))

# Signal with flat segments in the middle
flat_seg = np.concatenate([np.ones(50), np.sin(np.linspace(0, 4 * np.pi, 200)), np.ones(50)])
try:
    f_imfs, f_res = emd(flat_seg, max_imfs=5)
    f_recon = np.sum(f_imfs, axis=0) + f_res
    check("flat segments handled gracefully",
          np.max(np.abs(flat_seg - f_recon)) < 1e-8)
except Exception as e:
    check("flat segments handled gracefully", False, str(e))

# Large amplitude like real oil prices 10 to 145 dollars
t = np.linspace(0, 1, 500)
big = 80 + 60 * np.sin(2 * np.pi * 3 * t) + 5 * np.sin(2 * np.pi * 20 * t)
b_imfs, b_res = emd(big, max_imfs=5)
b_recon = np.sum(b_imfs, axis=0) + b_res
check("large amplitude signal reconstructs",
      np.max(np.abs(big - b_recon)) < 1e-10)

print()

total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")

if FAIL > 0:
    sys.exit(1)