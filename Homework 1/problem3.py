import numpy as np
import matplotlib.pyplot as plt
from problem0 import X_tilde

# SVD of centered data matrix: X_tilde = U Σ V^T
# full_matrices=False keeps sizes manageable
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# Build K_set = {1, 51, 101, ..., 784}
K_set = list(range(1, 785, 50))     # 1, 51, 101, ..., 751
if K_set[-1] != 784:
    K_set.append(784)

errors = {}   # store e(X_tilde, Y_K) for each K

for K in K_set:
    # Q_K = [u1, u2, ..., uK]  (784 x K)
    Q_K = U[:, :K]

    # Y_K = Q_K Q_K^T X_tilde
    Y_K = Q_K @ (Q_K.T @ X_tilde)

    # e(X_tilde, Y_K) = ||X_tilde - Y_K||_F
    e = np.linalg.norm(X_tilde - Y_K, ord='fro')

    errors[K] = e

    print(f"K = {K:3d}   error = {e:.6f}")

# Plot e(X_tilde, Y_K) versus K
plt.figure()
plt.plot(list(errors.keys()), list(errors.values()), marker='o')
plt.xlabel("K")
plt.ylabel(r"$e(\tilde{X}, Y_K) = \|\tilde{X} - Y_K\|_F$")
plt.title("Low-Rank Approximation Error vs K")
plt.grid(True)
plt.savefig("problem3_ERvsK.png", dpi=300, bbox_inches="tight")
print("Plot saved as problem3_ERvsK.png")