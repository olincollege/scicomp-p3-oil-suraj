"""
evaluate.py — Forecast evaluation metrics.
 
MAPE (Mean Absolute Percentage Error):
    How far off in percentage terms. Lower is better.
 
RMSE (Root Mean Square Error):
    How far off in dollars, penalizing large misses. Lower is better.
 
Both metrics are defined.
"""

import numpy as np


def mape(actual, predicted):
    """Mean Absolute Percentage Error Returns a percentage (e.g. 0.43)."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def rmse(actual, predicted):
    """Root Mean Square Error Returns value in same units as input."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return np.sqrt(np.mean((actual - predicted) ** 2))


if __name__ == "__main__":
    actual = np.array([100.0, 200.0, 300.0])

    assert mape(actual, actual) == 0.0
    assert rmse(actual, actual) == 0.0
    print("Perfect prediction:    MAPE=0, RMSE=0")

    m = mape(actual, actual * 1.10)
    print(f"10% overpredict:       MAPE={m:.2f}%")

    r = rmse(actual, actual + 5.0)
    print(f"Constant $5 error:     RMSE={r:.2f}")
