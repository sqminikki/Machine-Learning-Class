import numpy as np
import matplotlib.pyplot as plt
from dataset_a import _x_all, _y_all, split_data, standardize

np.random.seed(42)


def radius_predict(x_train, y_train, x_test, C):
    """Radius-based regression using sorted sliding window (O(n log n))."""
    order = np.argsort(x_train)
    xs = x_train[order]
    ys = y_train[order]
    fallback = y_train.mean()

    preds = np.empty(len(x_test))
    fallback_count = 0
    lo = 0
    hi = 0
    n = len(xs)

    # Use two-pointer on sorted test points mapped back
    test_order = np.argsort(x_test)
    x_test_sorted = x_test[test_order]
    preds_sorted = np.empty(len(x_test))

    lo = 0
    hi = 0
    for i, xi in enumerate(x_test_sorted):
        # advance lo so xs[lo] >= xi - C
        while lo < n and xs[lo] < xi - C:
            lo += 1
        # advance hi so xs[hi] <= xi + C
        while hi < n and xs[hi] <= xi + C:
            hi += 1
        # neighbors are xs[lo:hi]
        if hi > lo:
            preds_sorted[i] = ys[lo:hi].mean()
        else:
            preds_sorted[i] = fallback
            fallback_count += 1

    # map back to original test order
    preds[test_order] = preds_sorted
    return preds, fallback_count


def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# --- ii. Fix train_size=5000, sweep C ---
x_train_raw, x_test_raw, y_train, y_test = split_data(_x_all, _y_all, 5000)
x_train_std, mu_x, sigma_x = standardize(x_train_raw)
x_test_std = (x_test_raw - mu_x) / sigma_x

C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
mse_C = []
fallback_fracs = []
for C in C_values:
    preds, fb = radius_predict(x_train_std, y_train, x_test_std, C)
    mse_C.append(compute_mse(preds, y_test))
    fallback_fracs.append(fb / len(y_test))
    print(f"C={C:6.3f}  MSE={mse_C[-1]:.4f}  fallback_frac={fallback_fracs[-1]:.4f}")

# --- iii. Fig.1: MSE vs C ---
plt.figure(figsize=(8, 5))
plt.plot(C_values, mse_C, 'o-', color='steelblue')
plt.xlabel("Radius C (standardized x-scale)")
plt.ylabel("Test MSE")
plt.title("Fig.1: Radius Regression — Test MSE vs C (train_size=5000)")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig1.png", dpi=120)
plt.show()
print("Saved fig1.png")

# --- iv. Fix C=0.1, sweep train_size ---
C_fixed = 0.1
train_sizes = list(range(1000, 11000, 1000))
mse_size = []
for ts in train_sizes:
    xtr_r, xte_r, ytr, yte = split_data(_x_all, _y_all, ts)
    xtr_s, mu, sig = standardize(xtr_r)
    xte_s = (xte_r - mu) / sig
    preds, _ = radius_predict(xtr_s, ytr, xte_s, C_fixed)
    mse_size.append(compute_mse(preds, yte))
    print(f"train_size={ts}  MSE={mse_size[-1]:.4f}")

# --- v. Fig.2: MSE vs training size ---
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, mse_size, 's-', color='darkorange')
plt.xlabel("Training Size")
plt.ylabel("Test MSE")
plt.title("Fig.2: Radius Regression — Test MSE vs Training Size (C=0.1)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig2.png", dpi=120)
plt.show()
print("Saved fig2.png")

# --- vi. Discussion printout ---
print("\n--- Discussion ---")
print("Fig.1: As C increases from very small values, MSE first decreases (bias reduces)")
print("       then rises again as the neighborhood becomes too large (high bias).")
print("       Very small C causes high variance (few neighbors); very large C averages")
print("       nearly all training points (underfitting).")
print("Fig.2: MSE decreases with training size because denser coverage reduces fallback")
print("       rate and improves local averaging accuracy.")
print("\nFallback fractions per C:")
for C, ff in zip(C_values, fallback_fracs):
    print(f"  C={C}: fallback_frac={ff:.4f}")
