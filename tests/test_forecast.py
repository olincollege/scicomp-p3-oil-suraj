
#test_forecast.py — tests for random walk, raw ridge, decomposed ridge

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from forecast import (random_walk, train_raw_ridge, predict_raw_ridge,
                       train_decomposed_ridge, predict_decomposed_ridge)
from iceemdan import emd
from evaluate import mape, rmse

PASS = 0
FAIL = 0
def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  PASS  {name}")
    else: FAIL += 1; print(f"  FAIL  {name}  {detail}")



print("random_walk")


prices = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
rw_actual, rw_pred = random_walk(prices)

# Prediction is yesterday
check("predicts yesterday as today",
      np.array_equal(rw_pred, prices[:-1]))

# actual is tomorrow
check("actual is next day",
      np.array_equal(rw_actual, prices[1:]))

check("lengths match", len(rw_actual) == len(rw_pred))

# Constant prices means perfect
ca, cp = random_walk(np.ones(100) * 50.0)
check("constant gives MAPE 0", mape(ca, cp) == 0.0)
check("constant gives RMSE 0", rmse(ca, cp) == 0.0)
print()

print("raw ridge")


np.random.seed(42)
t = np.linspace(0, 10, 500)
train = 50 + 10 * np.sin(t) + np.random.randn(500) * 0.1
test = 50 + 10 * np.sin(t + 10) + np.random.randn(500) * 0.1

model, scaler = train_raw_ridge(train, lag=6)

check("model returned", model is not None)
check("scaler returned", scaler is not None)
check("scaler fitted", scaler.x_min is not None)

# Predictions in range
actual, predicted = predict_raw_ridge(model, scaler, test, lag=6)
check("prediction length matches actual", len(predicted) == len(actual))
check("predictions in reasonable range",
      predicted.min() > 0 and predicted.max() < 200,
      f"[{predicted.min():.1f}, {predicted.max():.1f}]")
check("RMSE is finite", np.isfinite(rmse(actual, predicted)))
print()

print("decomposed ridge (with val tunin)")


# Build synthetic train, val, test signals
np.random.seed(42)
t = np.linspace(0, 4, 400)
train_sig = 30 + 5*np.sin(2*np.pi*t) + 2*np.sin(2*np.pi*5*t)
val_sig = 30 + 5*np.sin(2*np.pi*(t+4)) + 2*np.sin(2*np.pi*5*(t+4))
test_sig = 30 + 5*np.sin(2*np.pi*(t+8)) + 2*np.sin(2*np.pi*5*(t+8))

# Decompose all three
train_imfs, train_res = emd(train_sig, max_imfs=5)
val_imfs, val_res = emd(val_sig, max_imfs=5)
test_imfs, test_res = emd(test_sig, max_imfs=5)

# Train needs both train and val decompositions (new signature)
n_components = len(train_imfs) + 1

# Handle case where val has different IMF count than train
min_imfs = min(len(train_imfs), len(val_imfs))
fitted = train_decomposed_ridge(
    train_imfs[:min_imfs], train_res,
    val_imfs[:min_imfs], val_res,
    lag=6
)

check("one model per component", len(fitted) == min_imfs + 1,
      f"got {len(fitted)} for {min_imfs + 1} components")

check("each entry is model scaler tuple",
      all(len(f) == 2 for f in fitted))

# Predict
min_test_imfs = min(min_imfs, len(test_imfs))
actual, predicted = predict_decomposed_ridge(
    fitted,
    test_imfs[:min_test_imfs], test_res,
    lag=6
)

check("predictions not empty", len(predicted) > 0)
check("actual and predicted same length", len(actual) == len(predicted))
check("all predictions finite", np.all(np.isfinite(predicted)))
check("all actuals finite", np.all(np.isfinite(actual)))
check("MAPE computable", np.isfinite(mape(actual, predicted)))
print()

print("IMF count mismatch")


# test with fewer IMFs than train
short_test = test_sig[:100]
short_imfs, short_res = emd(short_test, max_imfs=5)

actual_m, pred_m = predict_decomposed_ridge(fitted, short_imfs, short_res, lag=6)
check("fewer test IMFs handled",
      len(pred_m) > 0 and np.all(np.isfinite(pred_m)))
print()

print("integration")


# Decomposed should differ from raw
raw_model, raw_scaler = train_raw_ridge(train_sig, lag=6)
raw_actual, raw_predicted = predict_raw_ridge(raw_model, raw_scaler, test_sig, lag=6)

min_len = min(len(predicted), len(raw_predicted))
check("decomposed differs from raw",
      not np.allclose(predicted[:min_len], raw_predicted[:min_len]))
print()

total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")

if FAIL > 0: sys.exit(1)