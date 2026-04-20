#test_evaluate.py — tests for MAPE and RMSE
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evaluate import mape, rmse

PASS = 0
FAIL = 0
def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  PASS  {name}")
    else: FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("MAPE")


actual = np.array([100.0, 200.0, 300.0])

# perfect prediction
check("perfect prediction gives zero",
      mape(actual, actual) == 0.0)

# 10% high on everything
check("10 percent over gives 10",
      abs(mape(actual, actual * 1.10) - 10.0) < 1e-10)

# 10% low same thing because absolute
check("10 percent under gives 10",
      abs(mape(actual, actual * 0.90) - 10.0) < 1e-10)

#  mixed errors still 10
check("mixed plus minus 10 percent",
      abs(mape(np.array([100.0, 100.0]), np.array([110.0, 90.0])) - 10.0) < 1e-10)

# zero actual gets skipped
check("zero in actual doesnt blow up",
      np.isfinite(mape(np.array([0.0, 100.0]), np.array([5.0, 110.0]))))

# single value
check("single value",
      abs(mape(np.array([50.0]), np.array([55.0])) - 10.0) < 1e-10)
print()

print("RMSE")

# perfect
check("perfect prediction gives zero",
      rmse(actual, actual) == 0.0)

# constant 5 dollar miss
check("constant 5 dollar error",
      abs(rmse(actual, actual + 5.0) - 5.0) < 1e-10)

# outlier punishment
r = rmse(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 7.0]))
mae = np.mean([1.0, 1.0, 1.0, 7.0])
check("penalizes outliers more than MAE", r > mae)

# known exact value
check("errors 3 and 4 give sqrt 12.5",
      abs(rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) - np.sqrt(12.5)) < 1e-10)

#  sign doesnt matter
check("symmetric",
      abs(rmse(actual, actual + 5.0) - rmse(actual, actual - 5.0)) < 1e-10)

# single value
check("single value",
      abs(rmse(np.array([10.0]), np.array([13.0])) - 3.0) < 1e-10)

# scales with units
check("scales with data",
      abs(rmse(actual * 1000, actual * 1000 + 5000.0) - 5000.0) < 1e-6)
print()
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL > 0: sys.exit(1)
