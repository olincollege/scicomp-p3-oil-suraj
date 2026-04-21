"""
forecast.py — Forecasting models for ICEEMDAN-decomposed oil prices.
 
Three models:
    random_walk:              naive baseline (tomorrow = today)
    train/predict_raw_ridge:  Ridge on raw prices (no decomposition)
    train/predict_decomposed: one Ridge per ICEEMDAN component with
                              per-component alpha tuning on validation set
 
The decomposed approach follows the paper's Stage 2 (individual forecasting)
and Stage 3 (ensemble by addition).
"""
import numpy as np
from sklearn.linear_model import Ridge
from data import MinMaxScaler, create_windows


def random_walk(prices):
    #Tomorrow = today. Random
    return prices[1:], prices[:-1]


def train_raw_ridge(train, lag=6):
    #Train Ridge on only on raw prices (no decomposition). Returns model + scaler.
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(train)
    X, y = create_windows(norm, lag=lag)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, scaler


def predict_raw_ridge(model, scaler, test, lag=6):
    #Predict with only using raw Ridge. Returns (actual, predicted) in dollars.
    norm = scaler.transform(test)
    X, y = create_windows(norm, lag=lag)
    pred_norm = model.predict(X)
    return scaler.inverse_transform(y), scaler.inverse_transform(pred_norm)


def train_decomposed_ridge(imfs, residue, val_imfs, val_residue, lag=6):
    """Train one Ridge model per ICEEMDAN component with alpha tuning.
 
    For each component (IMF or residue):
        1. Normalize to [0,1] using training min/max
        2. Create lag-6 supervised windows
        3. Search alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
           using validation RMSE to pick the best
        4. Retrain final model with best alpha on training data
 
    Handles IMF count mismatch between train and val: if train has more
    components than val, extra components default to alpha=0.01.
 
    Returns list of (model, scaler) tuples, one per component.
    """
    from evaluate import rmse

    train_components = list(imfs) + [residue]
    val_components = list(val_imfs) + [val_residue]
    fitted = []
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]  # paper's range: [0.001, 0.2]

    for i, comp in enumerate(train_components):
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(comp)
        X_tr, y_tr = create_windows(norm, lag=lag)

        if i < len(val_components):
            val_norm = scaler.transform(val_components[i])
            X_val, y_val = create_windows(val_norm, lag=lag)
            best_alpha, best_score = 1.0, float('inf')
            for a in alphas:
                m = Ridge(alpha=a)
                m.fit(X_tr, y_tr)
                score = rmse(y_val, m.predict(X_val))
                if score < best_score:
                    best_alpha, best_score = a, score
        else:
            best_alpha = 0.01
        X_val, y_val = create_windows(val_norm, lag=lag)

        best_alpha, best_score = 1.0, float('inf')
        for a in alphas:
            m = Ridge(alpha=a)
            m.fit(X_tr, y_tr)
            score = rmse(y_val, m.predict(X_val))
            if score < best_score:
                best_alpha, best_score = a, score

        model = Ridge(alpha=best_alpha)
        model.fit(X_tr, y_tr)
        fitted.append((model, scaler))
        print(f"    Component {i+1}: alpha={best_alpha}")

    return fitted


def predict_decomposed_ridge(fitted, test_imfs, test_residue, lag=6):
    """Predict each test component independently, then sum for final forecast.
 
    Each component's model predicts in normalized space, then inverse-transforms
    back to dollars. The final price forecast is the sum of all component
    predictions (the paper's Stage 3: ensemble by addition).
 
    Handles IMF count mismatch: if fitted has more models than test components,
    extra models are skipped. If test has more components, extra components
    are ignored.
 
    Returns (actual_prices, predicted_prices) in dollars.
    """
    test_components = list(test_imfs) + [test_residue]

    component_preds = []
    actual_components = []
    min_len = float('inf')

    for i, (model, scaler) in enumerate(fitted):
        if i >= len(test_components):
            break
        norm = scaler.transform(test_components[i])
        X, y = create_windows(norm, lag=lag)
        pred_norm = model.predict(X)
        pred_dollars = scaler.inverse_transform(pred_norm)
        actual_dollars = scaler.inverse_transform(y)
        component_preds.append(pred_dollars)
        actual_components.append(actual_dollars)
        min_len = min(min_len, len(pred_dollars))

    trimmed_pred = [p[:min_len] for p in component_preds]
    trimmed_actual = [a[:min_len] for a in actual_components]

    return np.sum(trimmed_actual, axis=0), np.sum(trimmed_pred, axis=0)


if __name__ == "__main__":
    from data import load_wti, split_data
    from evaluate import mape, rmse

    prices, _ = load_wti()
    train, val, test = split_data(prices)

    rw_actual, rw_pred = random_walk(test)
    print(f"Random Walk:  MAPE={mape(rw_actual, rw_pred):.4f}%  RMSE={rmse(rw_actual, rw_pred):.4f}")

    raw_model, raw_scaler = train_raw_ridge(train)
    raw_actual, raw_pred = predict_raw_ridge(raw_model, raw_scaler, test)
    print(f"Raw Ridge:    MAPE={mape(raw_actual, raw_pred):.4f}%  RMSE={rmse(raw_actual, raw_pred):.4f}")
