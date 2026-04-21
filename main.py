
import numpy as np
from data import load_wti, split_data
from iceemdan import iceemdan
from evaluate import mape, rmse
from forecast import (random_walk, train_raw_ridge, predict_raw_ridge,
                       train_decomposed_ridge, predict_decomposed_ridge)
from plots import plot_decomposition, plot_forecast, plot_model_comparison


def main():
    # N_REAL = 50 for fast runs, 500 for final
    N_REAL = 50
    print("Loading WTI data")
    prices, dates = load_wti()
    train, val, test = split_data(prices)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print()

    print(f"Decomposing training data (n_realizations={N_REAL})")
    train_imfs, train_residue = iceemdan(train, max_imfs=11, noise_std=0.05,
                                          n_realizations=N_REAL, seed=42)
    print(f"  {len(train_imfs)} IMFs + 1 residue")

    recon = np.sum(train_imfs, axis=0) + train_residue
    recon_err = np.max(np.abs(train - recon))
    print(f"  Reconstruction error: {recon_err:.2e}")
    print()

    print("Training Ridge models per component")
    print(f"Decomposing validation data (n_realizations={N_REAL})")
    val_imfs, val_residue = iceemdan(val, max_imfs=11, noise_std=0.05,
                                      n_realizations=N_REAL, seed=42)
    print(f"  {len(val_imfs)} IMFs + 1 residue")
    print()

    fitted = train_decomposed_ridge(train_imfs, train_residue,
                                     val_imfs, val_residue, lag=6)
    print(f"  {len(fitted)} models trained")

    print("Training raw Ridge (no decomposition)")
    raw_model, raw_scaler = train_raw_ridge(train, lag=6)
    print()

    print(f"Decomposing test data (n_realizations={N_REAL})")
    test_imfs, test_residue = iceemdan(test, max_imfs=11, noise_std=0.05,
                                        n_realizations=N_REAL, seed=42)
    print(f"  {len(test_imfs)} IMFs + 1 residue")
    print()

    print("Predicting Oil Prices")

    #Random Walk
    rw_actual, rw_pred = random_walk(test)

    #Raw Ridge (no decomposition)
    raw_actual, raw_pred = predict_raw_ridge(raw_model, raw_scaler, test)

    #iCEEMDAN and Ridge
    ice_actual, ice_pred = predict_decomposed_ridge(fitted, test_imfs,
                                                      test_residue, lag=6)
    print()


    rw_mape, rw_rmse = mape(rw_actual, rw_pred), rmse(rw_actual, rw_pred)
    raw_mape, raw_rmse = mape(raw_actual, raw_pred), rmse(raw_actual, raw_pred)
    ice_mape, ice_rmse = mape(ice_actual, ice_pred), rmse(ice_actual, ice_pred)

    print("BENCHMARK RESULTS")
    print(f"  {'Model':<28} {'MAPE':>8} {'RMSE':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}")
    print(f"  {'Random Walk':<28} {rw_mape:>7.4f}% {rw_rmse:>7.4f}")
    print(f"  {'Raw Ridge (no decomp)':<28} {raw_mape:>7.4f}% {raw_rmse:>7.4f}")
    print(f"  {'ICEEMDAN+RR (yours)':<28} {ice_mape:>7.4f}% {ice_rmse:>7.4f}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}")
    print(f"  {'ICEEMDAN and RR (paper)':<28} {'~0.43%':>8} {'~0.34':>8}")
    print()

    print("Saving plots")
    plot_decomposition(train, train_imfs, train_residue)
    plot_forecast(ice_actual, ice_pred,
                   f"ICEEMDAN + Ridge Forecast (MAPE={ice_mape:.2f}%)",
                   save_path="forecast_iceemdan.png")
    plot_forecast(raw_actual, raw_pred,
                   f"Raw Ridge Forecast (MAPE={raw_mape:.2f}%)",
                   save_path="forecast_raw.png")
    plot_model_comparison([
        ("Random Walk", rw_mape, rw_rmse),
        ("Raw Ridge", raw_mape, raw_rmse),
        ("ICEEMDAN+RR", ice_mape, ice_rmse),
    ])
    print()
    print("Completed Run, Check Plots.")


if __name__ == "__main__":
    main()
