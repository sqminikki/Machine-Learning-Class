import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from dataset_b import load_dataset_b

np.random.seed(42)

FEATURES = ["cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year", "origin"]

TRI, TRO, TEI, TEO, mu_rows, sigma_rows = load_dataset_b()


def multivariate_poly_features(X, degree, col_mu=None, col_sigma=None):
    """
    Build design matrix for total-degree <= M polynomial.
    X shape: (n_features, n_samples) — each column is a sample.
    Returns Phi (n_terms, n_samples), col_mu, col_sigma (for non-bias terms).
    """
    p, n = X.shape
    # Generate all multi-index tuples with total degree <= M
    terms = [(0,) * p]  # bias
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(p), deg):
            exponents = [0] * p
            for idx in combo:
                exponents[idx] += 1
            terms.append(tuple(exponents))

    Phi = np.ones((len(terms), n))
    for ti, exponents in enumerate(terms):
        for fi, exp in enumerate(exponents):
            if exp > 0:
                Phi[ti] *= X[fi] ** exp

    # Row-standardize non-bias rows using training stats
    if col_mu is None:
        col_mu = np.zeros(len(terms))
        col_sigma = np.ones(len(terms))
        for ti in range(1, len(terms)):
            col_mu[ti] = Phi[ti].mean()
            col_sigma[ti] = Phi[ti].std()
            if col_sigma[ti] == 0:
                col_sigma[ti] = 1.0

    for ti in range(1, len(terms)):
        Phi[ti] = (Phi[ti] - col_mu[ti]) / col_sigma[ti]

    return Phi, col_mu, col_sigma, terms


def solve_LS(X, y):
    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
    tol = 1e-10 * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))


def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def format_term(exponents, feat_names):
    if all(e == 0 for e in exponents):
        return "1"
    parts = []
    for fi, e in enumerate(exponents):
        if e == 1:
            parts.append(feat_names[fi])
        elif e > 1:
            parts.append(f"{feat_names[fi]}^{e}")
    return "·".join(parts)


M_values = [1, 2, 3]
train_mse_M = []
test_mse_M = []

for M in M_values:
    Phi_tr, cmu, csig, terms = multivariate_poly_features(TRI, M)
    Phi_te, _, _, _ = multivariate_poly_features(TEI, M, cmu, csig)
    w = solve_LS(Phi_tr, TRO)
    tr_mse = compute_mse(w @ Phi_tr, TRO)
    te_mse = compute_mse(w @ Phi_te, TEO)
    train_mse_M.append(tr_mse)
    test_mse_M.append(te_mse)

    feat_labels = [f"x{i+1}" for i in range(7)]
    term_strs = [format_term(t, feat_labels) for t in terms]
    print(f"\nM={M}: {len(terms)} terms")
    print("  Terms: " + ", ".join(term_strs))
    print(f"  Train MSE: {tr_mse:.4f}")
    print(f"  Test MSE:  {te_mse:.4f}")

# --- Fig.14: MSE vs M ---
plt.figure(figsize=(7, 5))
plt.plot(M_values, train_mse_M, 'o-', color='red', label='Train MSE')
plt.plot(M_values, test_mse_M, 's-', color='blue', label='Test MSE')
plt.xlabel("Total Degree M")
plt.ylabel("MSE")
plt.title("Fig.14: Multivariate Poly Regression — Train & Test MSE vs M")
plt.xticks(M_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig14.png", dpi=120)
plt.show()
print("Saved fig14.png")

print("\n--- Discussion ---")
print("M=1 (linear): 8 terms. Underfits if the relationship is nonlinear.")
print("M=2: 36 terms. Captures pairwise interactions; test MSE typically improves.")
print("M=3: 120 terms. Very large model relative to ntrain=300; risk of overfitting.")
print("Train MSE decreases monotonically with M (more capacity).")
print("Test MSE should improve from M=1->2 but may worsen at M=3 due to overfitting.")
print("Multivariate regression leverages all feature interactions, likely beating")
print("the best univariate result from Problem 6 at M=2, but M=3 may overfit.")
