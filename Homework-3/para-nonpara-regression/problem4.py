import numpy as np
import matplotlib.pyplot as plt
from dataset_a import _x_all, _y_all, split_data, standardize

np.random.seed(42)


def poly_features(x, degree, row_mu=None, row_sigma=None):
    """
    Build design matrix Xpoly (degree+1, n).
    Row 0: bias (ones). Rows 1..degree: x^k, row-standardized using training stats.
    If row_mu/row_sigma are None, compute from x (training call).
    Returns Xpoly, row_mu, row_sigma.
    """
    n = len(x)
    Xpoly = np.ones((degree + 1, n))
    for k in range(1, degree + 1):
        Xpoly[k] = x ** k

    if row_mu is None:
        row_mu = np.zeros(degree + 1)
        row_sigma = np.ones(degree + 1)
        for k in range(1, degree + 1):
            row_mu[k] = Xpoly[k].mean()
            row_sigma[k] = Xpoly[k].std()
            if row_sigma[k] == 0:
                row_sigma[k] = 1.0

    for k in range(1, degree + 1):
        Xpoly[k] = (Xpoly[k] - row_mu[k]) / row_sigma[k]

    return Xpoly, row_mu, row_sigma


def solve_LS(X_poly, y):
    """SVD-based least-squares. X_poly shape: (d+1, n); y shape: (n,)."""
    # Transpose so A is (n, d+1) — standard overdetermined system A w = y
    U, s, Vt = np.linalg.svd(X_poly.T, full_matrices=False)
    tol = 1e-10 * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    w = Vt.T @ (s_inv * (U.T @ y))
    return w


def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# --- iv. Fig.5: scatter standardized xtrain vs ytrain (train_size=5000) ---
x_train_raw, x_test_raw, y_train, y_test = split_data(_x_all, _y_all, 5000)
x_train_std, mu_x, sigma_x = standardize(x_train_raw)
x_test_std = (x_test_raw - mu_x) / sigma_x

plt.figure(figsize=(7, 5))
plt.scatter(x_train_std, y_train, s=5, alpha=0.4, color='steelblue')
plt.xlabel("Standardized MedInc")
plt.ylabel("Median House Value ($100k)")
plt.title("Fig.5: Standardized MedInc vs Target (train_size=5000)")
plt.tight_layout()
plt.savefig("fig5.png", dpi=120)
plt.show()
print("Saved fig5.png")

# --- iv. Sweep degrees, train_size=5000 ---
degrees = [1, 2, 3, 4, 5, 10, 15, 20]
train_mse_d = []
test_mse_d = []
print("\ndegree | train MSE | test MSE")
for d in degrees:
    Xtr, rmu, rsig = poly_features(x_train_std, d)
    Xte, _, _ = poly_features(x_test_std, d, rmu, rsig)
    w = solve_LS(Xtr, y_train)
    tr_mse = compute_mse(w @ Xtr, y_train)
    te_mse = compute_mse(w @ Xte, y_test)
    train_mse_d.append(tr_mse)
    test_mse_d.append(te_mse)
    print(f"  d={d:2d}  train={tr_mse:.4f}  test={te_mse:.4f}")

# --- iv. Fig.6: train and test MSE vs degree ---
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_mse_d, 'o-', color='blue', label='Train MSE')
plt.plot(degrees, test_mse_d, 's-', color='red', label='Test MSE')
plt.xlabel("Polynomial Degree d")
plt.ylabel("MSE")
plt.title("Fig.6: Polynomial Regression — Train & Test MSE vs Degree (train_size=5000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig6.png", dpi=120)
plt.show()
print("Saved fig6.png")

# --- iv. Fix d=9, sweep train_sizes ---
d_fixed = 9
train_sizes = [1000, 3000, 5000, 7000, 10000]
test_mse_sizes = []
print(f"\nd={d_fixed} — train_size sweep:")
print("train_size | train MSE | test MSE")
for ts in train_sizes:
    xtr_r, xte_r, ytr, yte = split_data(_x_all, _y_all, ts)
    xtr_s, mu, sig = standardize(xtr_r)
    xte_s = (xte_r - mu) / sig
    Xtr, rmu, rsig = poly_features(xtr_s, d_fixed)
    Xte, _, _ = poly_features(xte_s, d_fixed, rmu, rsig)
    w = solve_LS(Xtr, ytr)
    tr_mse = compute_mse(w @ Xtr, ytr)
    te_mse = compute_mse(w @ Xte, yte)
    test_mse_sizes.append(te_mse)
    print(f"  {ts:6d}  train={tr_mse:.4f}  test={te_mse:.4f}")

# --- iv. Fig.7: test MSE vs training size (d=9) ---
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, test_mse_sizes, 's-', color='darkorange')
plt.xlabel("Training Size")
plt.ylabel("Test MSE")
plt.title("Fig.7: Polynomial Regression (d=9) — Test MSE vs Training Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig7.png", dpi=120)
plt.show()
print("Saved fig7.png")

# --- iv. Fig.8: fitted curve overlay (d=9, train_size=5000) ---
x_train_raw, x_test_raw, y_train, y_test = split_data(_x_all, _y_all, 5000)
x_train_std, mu_x, sigma_x = standardize(x_train_raw)
x_test_std = (x_test_raw - mu_x) / sigma_x

Xtr, rmu, rsig = poly_features(x_train_std, d_fixed)
w = solve_LS(Xtr, y_train)

sort_idx = np.argsort(x_train_std)
x_sorted = x_train_std[sort_idx]
X_sorted, _, _ = poly_features(x_sorted, d_fixed, rmu, rsig)
y_fit = w @ X_sorted

plt.figure(figsize=(8, 5))
plt.scatter(x_train_std, y_train, s=5, alpha=0.3, color='steelblue', label='Training data')
plt.plot(x_sorted, y_fit, color='red', linewidth=2, label=f'Fitted curve (d={d_fixed})')
plt.xlabel("Standardized MedInc")
plt.ylabel("Median House Value ($100k)")
plt.title("Fig.8: Polynomial Fit (d=9, train_size=5000)")
plt.legend()
plt.tight_layout()
plt.savefig("fig8.png", dpi=120)
plt.show()
print("Saved fig8.png")

print("\n--- Discussion ---")
print("Fig.5: Data shows a roughly positive trend with considerable spread, suggesting")
print("       polynomial features can capture nonlinear structure.")
print("Fig.6: Train MSE decreases monotonically with degree (more expressive model).")
print("       Test MSE first decreases then increases — classic bias-variance tradeoff;")
print("       high-degree polynomials overfit (especially d>=10).")
print("Fig.7: More data reduces test MSE for d=9, stabilizing overfitting.")
print("Fig.8: d=9 fits the trend well but may show slight oscillation at extremes.")
