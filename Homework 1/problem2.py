import numpy as np
import matplotlib.pyplot as plt
from problem0 import X_tilde

# ~~~~~ Singular Value Decomposition ~~~~
U, SIGMA, V_T = np.linalg.svd(X_tilde, full_matrices=False)
sigma_i = SIGMA 

# ~~~~ Plotting Figure 1: simga_i vs. index_i ~~~~
plt.figure()
plt.plot(sigma_i, marker='o')
plt.xlabel("index_i")
plt.ylabel("Singular value sigma_i")
plt.title("Singular value decay")
plt.grid(True)
plt.savefig("singular_values.png")
plt.close()

# ~~~~ Defining cumulative energy (variance explained) ratio for K in {1, 2, ..., r} ~~~~
sigma_squared = sigma_i**2
cum_energy = ( np.cumsum(sigma_squared) / np.sum(sigma_squared))

# ~~~~ Plotting Figure 2: E(K) vs. K (Energy Curve) ~~~~

plt.figure()
plt.plot(cum_energy, marker='o')
plt.xlabel("K")
plt.ylabel("E(K)")
plt.title("Cumulative energy (variance explained)")
plt.grid(True)
plt.savefig("energy_curve.png")
plt.close()

# --- Sanity checks ---
print("Rank r:", len(sigma_i))
print("Energy at K=10:", cum_energy[9])
print("Energy at K=50:", cum_energy[49])
print("Energy at K=100:", cum_energy[99])