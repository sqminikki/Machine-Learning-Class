import numpy as np
import pandas as pd

_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "auto-mpg/auto-mpg.data")
_COLS = ["mpg", "cylinders", "displacement", "horsepower",
         "weight", "acceleration", "model_year", "origin", "car_name"]
_FEATURES = ["cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model_year", "origin"]


def _load_raw():
    df = pd.read_csv(_URL, sep=r"\s+", names=_COLS, na_values="?")
    df = df.dropna(subset=["horsepower"]).reset_index(drop=True)
    return df


def standardize_rows(M_train, M_test):
    # M shape: (n_features, n_samples) — standardize each row using training stats
    mu = M_train.mean(axis=1, keepdims=True)
    sigma = M_train.std(axis=1, keepdims=True)     # population std (divisor n)
    return (M_train - mu) / sigma, (M_test - mu) / sigma, mu.squeeze(), sigma.squeeze()


def load_dataset_b():
    df = _load_raw()
    np.random.seed(42)
    perm = np.random.permutation(len(df))

    train_idx = perm[:300]
    test_idx  = perm[300:392]

    X = df[_FEATURES].to_numpy(dtype=float).T   # shape (7, N)
    y = df["mpg"].to_numpy(dtype=float)          # shape (N,)

    TRI_raw = X[:, train_idx]   # (7, 300)
    TEI_raw = X[:, test_idx]    # (7, 92)
    TRO = y[train_idx]          # (300,)
    TEO = y[test_idx]           # (92,)

    TRI, TEI, mu_rows, sigma_rows = standardize_rows(TRI_raw, TEI_raw)
    return TRI, TRO, TEI, TEO, mu_rows, sigma_rows


if __name__ == "__main__":
    df = _load_raw()
    TRI, TRO, TEI, TEO, mu_rows, sigma_rows = load_dataset_b()

    print("=== Dataset B: Auto-MPG ===")
    print(f"Total samples (after dropping missing horsepower): {len(df)}")
    print(f"Features (7):        {_FEATURES}")
    print(f"Target:              mpg")
    print(f"Train size:          {TRO.shape[0]}")
    print(f"Test size:           {TEO.shape[0]}")
    print()
    print(f"TRI shape:           {TRI.shape}  (features x train_samples)")
    print(f"TRO shape:           {TRO.shape}  (train targets)")
    print(f"TEI shape:           {TEI.shape}  (features x test_samples)")
    print(f"TEO shape:           {TEO.shape}  (test targets)")
    print()
    print("Per-feature training stats (before standardization):")
    print(f"  {'Feature':<15} {'mu':>10} {'sigma':>10} {'min':>10} {'max':>10}")
    for i, feat in enumerate(_FEATURES):
        print(f"  {feat:<15} {mu_rows[i]:>10.4f} {sigma_rows[i]:>10.4f} "
              f"{TRI[i].min():>10.4f} {TRI[i].max():>10.4f}")
    print()
    print(f"TRO (mpg) range:     [{TRO.min():.4f}, {TRO.max():.4f}]")
    print(f"TEO (mpg) range:     [{TEO.min():.4f}, {TEO.max():.4f}]")
