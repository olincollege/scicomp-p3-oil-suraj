"""
data.py — Load, split, normalize, and window WTI crude oil price data.

Data source: FRED series DCOILWTICO
    https://fred.stlouisfed.org/series/DCOILWTICO

    Date range: January 2, 1986 to February 4, 2019
    Total trading days: 8,342
    Train/test split: 80/20
    Within training: 80% train, 20% validation
    Normalization: min-max to [0, 1]
    Lag order: 6
"""

import numpy as np
import pandas as pd
import os


def _download_wti(csv_path="data/DCOILWTICO.csv"):
    import urllib.request

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    urls = [
        ("https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv", "github"),
        ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO&cosd=1986-01-02&coed=2019-02-04", "fred"),
    ]

    for url, source in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            raw = resp.read().decode("utf-8")

            if source == "github":
                lines = raw.strip().split("\n")
                out = ["DATE,DCOILWTICO"]
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) == 2 and parts[1]:
                        out.append(f"{parts[0]},{parts[1]}")
                raw = "\n".join(out)

            with open(csv_path, "w") as f:
                f.write(raw)
            return
        except Exception as e:
            print(f"  {source} failed: {e}")

    raise RuntimeError(
        f"Could not download."
        f"https://fred.stlouisfed.org/series/DCOILWTICO\n"
        f"   Save as {csv_path}"
    )


def load_wti(csv_path="data/DCOILWTICO.csv", start="1986-01-02", end="2019-02-04"):
    """Load WTI daily spot prices from FRED CSV. Auto-downloads if missing."""
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found")
        _download_wti(csv_path)
        print(f"Saved to {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    mask = (df[date_col] >= start) & (df[date_col] <= end)
    df = df.loc[mask].reset_index(drop=True)

    return df[val_col].values.astype(np.float64), df[date_col].values


def split_data(prices, train_frac=0.8, val_frac=0.2):
    """
    Split into train, validation, test.
    """
    n = len(prices)
    train_block_end = int(n * train_frac)
    train_block = prices[:train_block_end]
    test = prices[train_block_end:]

    val_start = int(len(train_block) * (1 - val_frac))
    train = train_block[:val_start]
    val = train_block[val_start:]

    return train, val, test


class MinMaxScaler:

    def __init__(self):
        self.x_min = None
        self.x_max = None

    def fit(self, x):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        return self

    def transform(self, x):
        denom = self.x_max - self.x_min
        if denom == 0:
            return np.zeros_like(x)
        return (x - self.x_min) / denom

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_scaled):
        return x_scaled * (self.x_max - self.x_min) + self.x_min


def create_windows(series, lag=6, horizon=1):
    """
    Convert 1-D series into (X, y) pairs for supervised learning.

    X[i] = [series[i], series[i+1], ..., series[i+lag-1]]
    y[i] = series[i + lag + horizon - 1]
    """
    X, y = [], []
    for t in range(lag, len(series) - horizon + 1):
        X.append(series[t - lag:t])
        y.append(series[t + horizon - 1])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    prices, dates = load_wti()
    train, val, test = split_data(prices)

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train)
    val_norm = scaler.transform(val)
    test_norm = scaler.transform(test)

    X_train, y_train = create_windows(train_norm)
    X_val, y_val = create_windows(val_norm)
    X_test, y_test = create_windows(test_norm)

    print(f"Total samples:     {len(prices)}")
    print(f"  Train:           {len(train)}")
    print(f"  Validation:      {len(val)}")
    print(f"  Test:            {len(test)}")
    print(f"  Price range:     ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  X_train shape:   {X_train.shape}")
    print(f"  X_test shape:    {X_test.shape}")
    print(f"  Norm range:      [{train_norm.min():.4f}, {train_norm.max():.4f}]")
