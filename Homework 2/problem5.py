import numpy as np
import matplotlib.pyplot as plt
import time

A = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_A.csv")
y = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_y.csv")
x0 = np.loadtxt("S2026_HW2_CSV_DATA/HW2_CSV_DATA/HW2_xinit.csv")

def L(x):
    r = y - A @ x
    return r @ r

def g(x):
    return 2 * (A.T @ (A @ x - y))

# --- Closed-form LS optimum and benchmark ---
x_opt = np.linalg.pinv(A) @ y
L_opt = L(x_opt)

sigma_max = np.linalg.svd(A, compute_uv=False)[0]
C = 2 * sigma_max**2
invC = 1 / C

def gd_bls(x0, eta, c, iters=30, gamma0=1.0):
    x = x0.copy()
    L_vals = []
    gammas = []

    start = time.perf_counter()

    for _ in range(iters):
        grad = g(x)
        gamma = gamma0
        Lx = L(x)

        # --- Backtracking --- 
        while L(x - gamma * grad) > Lx - c * gamma * (grad @ grad):
            gamma *= eta

        x = x - gamma * grad
        L_vals.append(L(x))
        gammas.append(gamma)

    runtime = time.perf_counter() - start
    return x, np.array(L_vals), np.array(gammas), runtime

# --- Fig. 4, 5, 6: vary eta, c=0.1

etas = [0.1, 0.2, 0.5, 0.9]
c_fixed = 0.1
results_eta = {}

for eta in etas:
    x_end, L_vals, gammas, runtime = gd_bls(x0, eta, c_fixed)
    results_eta[eta] = {
        "x_end": x_end,
        "L_vals": L_vals,
        "gammas": gammas,
        "runtime": runtime
    }

iters = np.arange(1, 31)

# --- Task 1: Plot L(x_n) vs iteration index Figure 4 ---
plt.figure()
for eta in etas:
    plt.plot(iters, results_eta[eta]["L_vals"], label=fr"$\eta={eta}$")
plt.axhline(L_opt, color="k", linestyle="--", label=r"$L_{\mathrm{opt}}$")
plt.xlabel("iteration n")
plt.ylabel(r"$L(x_n)$")
plt.legend()
plt.tight_layout()
plt.savefig("problem5_fig4.png")
plt.close()

# --- Task 3: Plot runtime bars vs n Figure 5 ---
plt.figure()
plt.bar([str(eta) for eta in etas], [results_eta[eta]["runtime"] for eta in etas])
plt.xlabel(r"$\eta$")
plt.ylabel("runtime (s)")
plt.tight_layout()
plt.savefig("problem5_fig5.png")
plt.close()

# --- Task 5: Plot gamma_n vs iteration index n Figure 6 ---
plt.figure()
for eta in etas:
    plt.plot(iters, results_eta[eta]["gammas"], label=fr"$\eta={eta}$")
plt.axhline(invC, color="k", linestyle="--", label=r"$1/C$")
plt.xlabel("iteration n")
plt.ylabel(r"$\gamma_n$")
plt.legend()
plt.tight_layout()
plt.savefig("problem5_fig6.png")
plt.close()

# --- Fig. 7, 8, 9: vary c, eta=0.5 ---

cs = [0.01, 0.1]
eta_fixed = 0.5
results_c = {}

for c in cs:
    x_end, L_vals, gammas, runtime = gd_bls(x0, eta_fixed, c)
    results_c[c] = {
        "x_end": x_end,
        "L_vals": L_vals,
        "gammas": gammas,
        "runtime": runtime
    }

# --- Task 7: Plot L(X_n) vs iteration index n Figure 7 ---
plt.figure()
for c in cs:
    plt.plot(iters, results_c[c]["L_vals"], label=fr"$c={c}$")
plt.axhline(L_opt, color="k", linestyle="--", label=r"$L_{\mathrm{opt}}$")
plt.xlabel("iteration n")
plt.ylabel(r"$L(x_n)$")
plt.legend()
plt.tight_layout()
plt.savefig("problem5_fig7.png")
plt.close()

# --- Task 9: runtime bars vs c Figure 8 ---
plt.figure()
plt.bar([str(c) for c in cs], [results_c[c]["runtime"] for c in cs])
plt.xlabel(r"$c$")
plt.ylabel("runtime (s)")
plt.tight_layout()
plt.savefig("problem5_fig8.png")
plt.close()

# --- Task 11: Plot gamma_n vs iteration index n Figure 9 ---
plt.figure()
for c in cs:
    plt.plot(iters, results_c[c]["gammas"], label=fr"$c={c}$")
plt.axhline(invC, color="k", linestyle="--", label=r"$1/C$")
plt.xlabel("iteration n")
plt.ylabel(r"$\gamma_n$")
plt.legend()
plt.tight_layout()
plt.savefig("problem5_fig9.png")
plt.close()

# --- Print summary ---
print(f"L_opt = {L_opt}")
print(f"C = {C}")
print(f"1/C = {invC}")

print("\nVary eta, c=0.1")
for eta in etas:
    print(
        f"eta={eta}: "
        f"L(x_30)={results_eta[eta]['L_vals'][-1]}, "
        f"runtime={results_eta[eta]['runtime']}"
    )

print("\nVary c, eta=0.5")
for c in cs:
    print(
        f"c={c}: "
        f"L(x_30)={results_c[c]['L_vals'][-1]}, "
        f"runtime={results_c[c]['runtime']}"
    )