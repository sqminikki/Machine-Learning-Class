# ~~~~ Storage Efficiency ~~~~

import numpy as np
import matplotlib.pyplot as plt
from problem0 import X_tilde

# ~~~ Establish dimensions ~~~~
d, N = X_tilde.shape # Where d = 28*28 = 784

# ~~~ K value range ~~~~
K_values = np.arange(1, d + 1)

# ~~~ Storage costs ~~~~
full_cost = d * N
lowrank_cost = K_values * (d + N)

# ~~~~ Figure 4 Plot: Storage cost vs. K, including a horizontal reference for storing X ~~~~
plt.figure()
plt.plot(K_values, lowrank_cost, label="Low-rank storage: K(784 + N)")
plt.axhline(full_cost, color='r', linestyle='--', label="Full storage: 784·N")
plt.xlabel("Rank K")
plt.ylabel("Number of stored scalars")
plt.title("Storage cost vs rank K")
plt.legend()
plt.grid(True)
plt.savefig("storage_cost.png")
plt.close()