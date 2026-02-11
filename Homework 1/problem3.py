import numpy as np
import matplotlib.pyplot as plt
from problem0 import X_tilde

# --- Compute Singular Value Decomposition (SVD) --- 
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# --- Build K_set = {1, 51, 101, ..., 784} ---
K_set = list(range(1, 785, 50))
if K_set[-1] != 784:
    K_set.append(784)

# --- Compute Low-Rank Approximations ---
errors = {}

for K in K_set:
    Q_K = U[:, :K]
    Y_K = Q_K @ (Q_K.T @ X_tilde)
    e = np.linalg.norm(X_tilde - Y_K, ord='fro')

    errors[K] = e
    print(f"K = {K:3d}   error = {e:.6f}")

# ~~~~ Plotting Figure 3: e(X_tilde, Y_K) vs. K ~~~~
plt.figure(figsize=(8,6), constrained_layout=True)
plt.plot(list(errors.keys()), list(errors.values()), marker='o')
plt.xlabel("K")
plt.ylabel(r"$e(\tilde{X}, Y_K) = \|\tilde{X} - Y_K\|_F$")
plt.title("Low-Rank Approximation Error vs K")
plt.grid(True)
plt.savefig("Fig_3_ERvsK.png")
plt.close()