import numpy as np
from sklearn.datasets import fetch_california_housing

# Global permutation computed once with seed 42
_housing = fetch_california_housing(as_frame=False)
_x_all = _housing.data[:, 0]   # MedInc (column 0)
_y_all = _housing.target        # median house value in $100k

np.random.seed(42)
_perm = np.random.permutation(len(_x_all))
_x_all = _x_all[_perm]
_y_all = _y_all[_perm]


def standardize(v):
    mu = np.mean(v)
    sigma = np.std(v)   # population std (divisor n)
    return (v - mu) / sigma, mu, sigma


def split_data(x, y, train_size):
    if train_size > 10000:
        raise ValueError("train_size must be <= 10000")
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test  = x[10000:15000]
    y_test  = y[10000:15000]
    return x_train, x_test, y_train, y_test


def load_dataset_a(train_size=5000):
    x_train, x_test, y_train, y_test = split_data(_x_all, _y_all, train_size)
    x_train_std, mu_x, sigma_x = standardize(x_train)
    x_test_std = (x_test - mu_x) / sigma_x
    return x_train_std, x_test_std, y_train, y_test, mu_x, sigma_x


if __name__ == "__main__":
    print("=== Dataset A: California Housing ===")
    print(f"Total samples:       {len(_x_all)}")
    print(f"Feature:             MedInc (median income, $100k)")
    print(f"Target:              Median house value ($100k)")
    print(f"Fixed test indices:  [10000:15000] (size 5000)")
    print()

    for ts in [1000, 5000, 10000]:
        x_tr, x_te, y_tr, y_te, mu_x, sigma_x = load_dataset_a(train_size=ts)
        print(f"--- train_size = {ts} ---")
        print(f"  x_train shape:   {x_tr.shape}")
        print(f"  x_test  shape:   {x_te.shape}")
        print(f"  y_train shape:   {y_tr.shape}")
        print(f"  y_test  shape:   {y_te.shape}")
        print(f"  x_train mean:    {x_tr.mean():.4f}  (should be ~0 after standardization)")
        print(f"  x_train std:     {x_tr.std():.4f}   (should be ~1 after standardization)")
        print(f"  training mu_x:   {mu_x:.4f}")
        print(f"  training sigma_x:{sigma_x:.4f}")
        print(f"  y_train range:   [{y_tr.min():.4f}, {y_tr.max():.4f}]")
        print(f"  y_test  range:   [{y_te.min():.4f}, {y_te.max():.4f}]")
        print()
