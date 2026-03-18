import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_A.csv")
y = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_y.csv")
x0 = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_xinit.csv")

def L(x):
    r = y - A @ x
    return r @ r

def g(x):
    return 2 * (A.T @ (A @ x - y))

# --- Closed-form LS solution ---
x_opt = np.linalg.pinv(A) @ y
L_opt = L(x_opt)

# --- Lipschitz constant ---
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
C = 2 * sigma_max**2
invC = 1 / C

# --- GD with Exact Line Search ---
x = x0.copy()
L_vals = []
gammas = []

for n in range(30):
    grad = g(x)
    gamma = (grad @ grad) / (2 * ((A @ grad) @ (A @ grad)))
    x = x - gamma * grad

    L_vals.append(L(x))
    gammas.append(gamma)

print("L_opt =", L_opt)
print("C =", C)
print("1/C =", invC)
print("L(x_30) =", L_vals[-1])

# --- Task 1: Plot L(x_n) vs iteration index Figure 1 ---
iters = np.arange(1, 31)
plt.figure()
plt.plot(iters, L_vals, 'b-', label=r'$L(x_n)$')
plt.axhline(L_opt, color='k', linestyle='--', label=r'$L_{\mathrm{opt}}$')
plt.xlabel("iteration index n")
plt.ylabel(r"$L(x_n)$")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_gd_els_L(x_n).png")

# --- Task 3: Plot L(x_n) vs iteration index Figure 2 ---
plt.figure()
plt.plot(iters, gammas, 'b-', label=r'$\gamma_n$')
plt.axhline(invC, color='k', linestyle='--', label=r'$1/C$')
plt.xlabel("iteration index n")
plt.ylabel(r'$\gamma_n$')
plt.legend()
plt.tight_layout()
plt.savefig("fig2_gd_els_gamma.png")