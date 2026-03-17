import numpy as np
import matplotlib.pyplot as plt

# Load data
A     = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_A.csv')
y     = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_y.csv')
xinit = np.loadtxt('S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_xinit.csv')

# Hessian and Lipschitz constant
H = 2 * A.T @ A
C = np.linalg.eigvalsh(H).max()  # Lipschitz constant = lambda_max(H)

# Optimal solution and loss (from SVD)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
xopt = Vt.T @ ((1.0 / s) * (U.T @ y))
Lopt = np.linalg.norm(A @ xopt - y) ** 2

def loss(x):
    return np.linalg.norm(A @ x - y) ** 2

def grad(x):
    return 2 * A.T @ (A @ x - y)

# GD with fixed step size
p_values = [0.1, 0.5, 1, 1.5, 2]
n_iters  = 30

plt.figure(figsize=(8, 5))

for p in p_values:
    gamma = p / C
    x = xinit.copy()
    losses = []
    for _ in range(n_iters):
        x = x - gamma * grad(x)
        losses.append(loss(x))
    plt.plot(range(1, n_iters + 1), losses, marker='o', markersize=3, label=f'p={p}')

# Benchmark
plt.axhline(Lopt, color='black', linestyle='--', linewidth=1.5, label=f'Lopt={Lopt:.2f}')

plt.xlabel('Iteration n')
plt.ylabel('L(xn)')
plt.title('GD with Fixed Step Size: L(xn) vs Iteration')
plt.legend()
plt.tight_layout()
plt.savefig('fig3_gd_fss.png', dpi=150)
plt.show()
print(f"Lipschitz constant C = {C:.4f}")
print(f"Lopt = {Lopt:.6f}")
