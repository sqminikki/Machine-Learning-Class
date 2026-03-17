import numpy as np

# Load data
A     = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_A.csv')
y     = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_y.csv')
xinit = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_xinit.csv')

# Dimensions
N, K = A.shape  # N=100 rows, K=50 columns

# Least squares: xopt = argmin_x ||Ax - y||^2_2
# Closed-form solution: xopt = (A^T A)^{-1} A^T y
xopt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

print(f"N={N}, K={K}")
print(f"xopt shape: {xopt.shape}")
print(f"Optimal loss L(xopt) = {np.linalg.norm(A @ xopt - y)**2:.6f}")
