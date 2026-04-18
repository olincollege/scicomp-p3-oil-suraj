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


def train_decomposed_ridge(imfs, residue, lag=6):
    """
    Train one Ridge model per component.

    Takes the ICEEMDAN output (imfs array and residue), normalizes each
    component independently, creates lag windows, fits Ridge.

    Returns list of (model, scaler) tuples, one per component.
    """
    components = list(imfs) + [residue]
    fitted = []

    for comp in components:
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(comp)
        X, y = create_windows(norm, lag=lag)
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        fitted.append((model, scaler))

    return fitted


def predict_decomposed_ridge(fitted, test_imfs, test_residue, lag=6):
    """
    Predict each test component with its Ridge model, sum to get final forecast.

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
