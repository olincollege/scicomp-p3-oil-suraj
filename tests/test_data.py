#test_data.py — tests for splitting, normalization, windowing

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data import split_data, MinMaxScaler, create_windows

PASS = 0
FAIL = 0
def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  PASS  {name}")
    else: FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("split_data")


prices = np.arange(1000, dtype=np.float64)
train, val, test = split_data(prices)

# everything accounted for
check("splits sum to total",
      len(train) + len(val) + len(test) == len(prices))

# 80/20 outer
check("train block is 80 pct",
      abs((len(train) + len(val)) / len(prices) - 0.8) < 0.01)
check("test is 20 pct",
      abs(len(test) / len(prices) - 0.2) < 0.01)

# 80/20 inner
check("val is 20 pct of train block",
      abs(len(val) / (len(train) + len(val)) - 0.2) < 0.02)

# sequential no gaps
check("starts at zero", train[0] == 0)
check("train to val continuous", train[-1] + 1 == val[0])
check("val to test continuous", val[-1] + 1 == test[0])
check("ends at last index", test[-1] == len(prices) - 1)

# tiny dataset
s_tr, s_v, s_te = split_data(np.arange(10, dtype=np.float64))
check("10 points splits", len(s_tr) + len(s_v) + len(s_te) == 10)
print()

print("MinMaxScaler")


scaler = MinMaxScaler()
data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
normed = scaler.fit_transform(data)

check("min maps to 0", normed[0] == 0.0)
check("max maps to 1", normed[-1] == 1.0)
check("all in 0 to 1", normed.min() >= 0.0 and normed.max() <= 1.0)
check("middle is 0.5", abs(normed[2] - 0.5) < 1e-10)

# roundtrip
check("inverse recovers original",
      np.max(np.abs(data - scaler.inverse_transform(normed))) < 1e-10)

# out of range
tn = scaler.transform(np.array([0.0, 60.0]))
check("below min goes negative", tn[0] < 0)
check("above max goes over 1", tn[1] > 1)

# constant
check("constant gives zeros",
      np.all(MinMaxScaler().fit_transform(np.array([5.0, 5.0, 5.0])) == 0.0))

check("stores x_min", scaler.x_min == 10.0)
check("stores x_max", scaler.x_max == 50.0)
print()

print("create_windows")


series = np.arange(20, dtype=np.float64)
X, y = create_windows(series, lag=6, horizon=1)

check("row count correct", X.shape[0] == len(series) - 6)
check("col count equals lag", X.shape[1] == 6)
check("y length matches", len(y) == X.shape[0])

# content
check("first X is 0-5", np.array_equal(X[0], [0, 1, 2, 3, 4, 5]))
check("first y is 6", y[0] == 6.0)
check("last X is 13-18", np.array_equal(X[-1], [13, 14, 15, 16, 17, 18]))
check("last y is 19", y[-1] == 19.0)

# horizon 3
X3, y3 = create_windows(series, lag=6, horizon=3)
check("horizon 3 target is index 8", y3[0] == 8.0)
check("horizon 3 fewer rows", len(y3) < len(y))

# lag 1
X1, y1 = create_windows(series, lag=1, horizon=1)
check("lag 1 single column", X1.shape[1] == 1)
check("lag 1 first is 0", X1[0, 0] == 0.0)
check("lag 1 target is 1", y1[0] == 1.0)

# short series
Xs, ys = create_windows(np.array([1.0, 2.0, 3.0]), lag=2, horizon=1)
check("3 points lag 2 gives 1 window", len(ys) == 1)
check("short X correct", np.array_equal(Xs[0], [1, 2]))
check("short y correct", ys[0] == 3.0)

print()
print("=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
print("=" * 60)
if FAIL > 0: sys.exit(1)
