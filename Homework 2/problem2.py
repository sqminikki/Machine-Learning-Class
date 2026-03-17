import numpy as np

# Load data
A = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_A.csv')
y = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_y.csv')

N, K = A.shape  # N=100, K=50

# --- Task 1: Compute H and C ---
# Hessian of L(x) = ||Ax - y||^2_2  is  H = 2 * A^T A
H = 2 * A.T @ A  # shape (K, K)

# Condition number of H: C = lambda_max / lambda_min
eigenvalues = np.linalg.eigvalsh(H)  # eigvalsh for symmetric matrix
C = eigenvalues.max() / eigenvalues.min()

print("=== Task 1: H and C ===")
print(f"H shape: {H.shape}")
print(f"H (first 3x3 block):\n{H[:3, :3]}")
print(f"Condition number C = {C:.6f}")

# --- Task 2: LS solution via SVD ---
# A = U S V^T  =>  xopt = V S^+ U^T y
U, s, Vt = np.linalg.svd(A, full_matrices=False)  # economy SVD

# Pseudoinverse: S^+ = diag(1/s_i)
s_inv = 1.0 / s
xopt = Vt.T @ (s_inv * (U.T @ y))  # V @ diag(s_inv) @ U^T @ y

# Optimal loss
Lopt = np.linalg.norm(A @ xopt - y) ** 2

print("\n=== Task 2: SVD-based LS solution ===")
print(f"Singular values (first 5): {s[:5]}")
print(f"xopt shape: {xopt.shape}")
print(f"xopt (first 5): {xopt[:5]}")
print(f"Lopt = L(xopt) = {Lopt:.6f}")
