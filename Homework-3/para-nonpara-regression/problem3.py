import numpy as np
import matplotlib.pyplot as plt
from dataset_a import _x_all, _y_all, split_data, standardize

np.random.seed(42)


def knn_predict(x_train, y_train, x_test, K):
    """KNN regression using sorted two-pointer (O(n log n + m log n))."""
    K_eff = min(K, len(x_train))
    order = np.argsort(x_train)
    xs = x_train[order]
    ys = y_train[order]
    n = len(xs)

    test_order = np.argsort(x_test)
    x_test_sorted = x_test[test_order]
    preds_sorted = np.empty(len(x_test))

    # sliding window of size K_eff: keep a sorted window and track sum
    # For 1D we use the standard two-pointer expanding window approach
    for i, xi in enumerate(x_test_sorted):
        # find insertion point
        pos = np.searchsorted(xs, xi)
        lo = pos - 1
        hi = pos
        neighbors = []
        while len(neighbors) < K_eff:
            use_lo = lo >= 0
            use_hi = hi < n
            if not use_lo and not use_hi:
                break
            if use_lo and use_hi:
                if abs(xs[lo] - xi) <= abs(xs[hi] - xi):
                    neighbors.append(ys[lo]); lo -= 1
                else:
                    neighbors.append(ys[hi]); hi += 1
            elif use_lo:
                neighbors.append(ys[lo]); lo -= 1
            else:
                neighbors.append(ys[hi]); hi += 1
        preds_sorted[i] = np.mean(neighbors)

    preds = np.empty(len(x_test))
    preds[test_order] = preds_sorted
    return preds


def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# --- ii. Fix train_size=5000, sweep K ---
x_train_raw, x_test_raw, y_train, y_test = split_data(_x_all, _y_all, 5000)
x_train_std, mu_x, sigma_x = standardize(x_train_raw)
x_test_std = (x_test_raw - mu_x) / sigma_x

K_values = [1, 3, 5, 10, 20, 50, 100, 500, 1000]
mse_K = []
for K in K_values:
    preds = knn_predict(x_train_std, y_train, x_test_std, K)
    mse_K.append(compute_mse(preds, y_test))
    print(f"K={K:5d}  MSE={mse_K[-1]:.4f}")

# --- iii. Fig.3: MSE vs K ---
plt.figure(figsize=(8, 5))
plt.plot(K_values, mse_K, 'o-', color='steelblue')
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test MSE")
plt.title("Fig.3: KNN Regression — Test MSE vs K (train_size=5000)")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig3.png", dpi=120)
plt.show()
print("Saved fig3.png")

# --- iv. Fix K=100, sweep train_size ---
K_fixed = 100
train_sizes = list(range(1000, 11000, 1000))
mse_size = []
for ts in train_sizes:
    xtr_r, xte_r, ytr, yte = split_data(_x_all, _y_all, ts)
    xtr_s, mu, sig = standardize(xtr_r)
    xte_s = (xte_r - mu) / sig
    preds = knn_predict(xtr_s, ytr, xte_s, K_fixed)
    mse_size.append(compute_mse(preds, yte))
    print(f"train_size={ts}  MSE={mse_size[-1]:.4f}")

# --- v. Fig.4: MSE vs training size ---
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, mse_size, 's-', color='darkorange')
plt.xlabel("Training Size")
plt.ylabel("Test MSE")
plt.title("Fig.4: KNN Regression — Test MSE vs Training Size (K=100)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig4.png", dpi=120)
plt.show()
print("Saved fig4.png")

# --- vi. Discussion ---
print("\n--- Discussion ---")
print("Fig.3: Small K (e.g., K=1) produces low bias but high variance (overfitting).")
print("       Large K over-smooths the predictions (underfitting). Optimal K balances both.")
print("Fig.4: More training data improves accuracy because neighbors are closer in feature space.")
print("Compared to radius regression, KNN always uses exactly K neighbors so it never")
print("needs a fallback, making it more robust when data density varies.")
