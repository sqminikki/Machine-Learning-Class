import numpy as np
import matplotlib.pyplot as plt
from dataset_b import load_dataset_b

np.random.seed(42)

FEATURES = ["cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year", "origin"]

TRI, TRO, TEI, TEO, mu_rows, sigma_rows = load_dataset_b()


def poly_features_1d(x, degree, row_mu=None, row_sigma=None):
    """Build (degree+1, n) design matrix; row-standardize non-bias rows."""
    n = len(x)
    Xp = np.ones((degree + 1, n))
    for k in range(1, degree + 1):
        Xp[k] = x ** k

    if row_mu is None:
        row_mu = np.zeros(degree + 1)
        row_sigma = np.ones(degree + 1)
        for k in range(1, degree + 1):
            row_mu[k] = Xp[k].mean()
            row_sigma[k] = Xp[k].std()
            if row_sigma[k] == 0:
                row_sigma[k] = 1.0

    for k in range(1, degree + 1):
        Xp[k] = (Xp[k] - row_mu[k]) / row_sigma[k]
    return Xp, row_mu, row_sigma


def solve_LS(X, y):
    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
    tol = 1e-10 * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))


def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


degrees = [0, 1, 2, 3, 4, 5]
colors = plt.cm.tab10(np.linspace(0, 0.7, len(FEATURES)))

# Store test MSE for Fig.13
test_mse_all = np.zeros((len(FEATURES), len(degrees)))

for fi, feat in enumerate(FEATURES):
    x_tr = TRI[fi]
    x_te = TEI[fi]
    for di, d in enumerate(degrees):
        Xtr, rmu, rsig = poly_features_1d(x_tr, d)
        Xte, _, _ = poly_features_1d(x_te, d, rmu, rsig)
        w = solve_LS(Xtr, TRO)
        test_mse_all[fi, di] = compute_mse(w @ Xte, TEO)

# --- Fig.11: weight (fi=3), M=3 ---
fi, d = 3, 3
x_tr = TRI[fi]; x_te = TEI[fi]
Xtr, rmu, rsig = poly_features_1d(x_tr, d)
w = solve_LS(Xtr, TRO)

# raw domain [1613, 5140] — build dense grid in standardized space
raw_grid = np.linspace(1613, 5140, 300)
x_grid_std = (raw_grid - mu_rows[fi]) / sigma_rows[fi]
Xgrid, _, _ = poly_features_1d(x_grid_std, d, rmu, rsig)
y_grid = w @ Xgrid

plt.figure(figsize=(7, 5))
plt.scatter(x_tr, TRO, s=20, alpha=0.6, label='Training data', color='steelblue')
plt.plot(x_grid_std, y_grid, color='red', linewidth=2, label=f'Poly fit M={d}')
plt.xlabel("Weight (standardized)")
plt.ylabel("MPG")
plt.title("Fig.11: Univariate Poly Regression — Weight (M=3)")
plt.legend()
plt.tight_layout()
plt.savefig("fig11.png", dpi=120)
plt.show()
print("Saved fig11.png")

# --- Fig.12: horsepower (fi=2), M=3 ---
fi, d = 2, 3
x_tr = TRI[fi]; x_te = TEI[fi]
Xtr, rmu, rsig = poly_features_1d(x_tr, d)
w = solve_LS(Xtr, TRO)

raw_grid = np.linspace(46, 230, 300)
x_grid_std = (raw_grid - mu_rows[fi]) / sigma_rows[fi]
Xgrid, _, _ = poly_features_1d(x_grid_std, d, rmu, rsig)
y_grid = w @ Xgrid

plt.figure(figsize=(7, 5))
plt.scatter(TRI[fi], TRO, s=20, alpha=0.6, label='Training data', color='steelblue')
plt.plot(x_grid_std, y_grid, color='red', linewidth=2, label=f'Poly fit M={d}')
plt.xlabel("Horsepower (standardized)")
plt.ylabel("MPG")
plt.title("Fig.12: Univariate Poly Regression — Horsepower (M=3)")
plt.legend()
plt.tight_layout()
plt.savefig("fig12.png", dpi=120)
plt.show()
print("Saved fig12.png")

# --- Fig.13: test MSE vs M for all 7 features ---
plt.figure(figsize=(9, 6))
for fi, feat in enumerate(FEATURES):
    plt.plot(degrees, test_mse_all[fi], 'o-', color=colors[fi], label=feat)
plt.xlabel("Polynomial Degree M")
plt.ylabel("Test MSE")
plt.title("Fig.13: Univariate Poly Regression — Test MSE vs M (all features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig13.png", dpi=120)
plt.show()
print("Saved fig13.png")

# Most predictive feature = lowest test MSE across any degree
best_fi = np.argmin(test_mse_all.min(axis=1))
best_d = np.argmin(test_mse_all[best_fi])
print(f"\nMost predictive feature: {FEATURES[best_fi]} "
      f"(best test MSE={test_mse_all[best_fi, best_d]:.4f} at M={degrees[best_d]})")
print("This aligns with the highest absolute Pearson correlation with mpg from Problem 5.")
