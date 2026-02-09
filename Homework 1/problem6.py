# ~~~~ Low-Rank Denoising ~~~~

import numpy as np
import matplotlib.pyplot as plt
from problem0 import X
from problem2 import U

# ~~~~ Corruption probability noise parameter ~~~~
corrupted_pixel = 0.05 

# ~~~~ Noisy data matrix ~~~~
X_noise = X.copy()
mask = np.random.rand(*X.shape) < corrupted_pixel   # This independently corrupts each pixel.
noise = np.random.choice([0.0, 255.0], size = X.shape)
X_noise[mask] = noise[mask]

# ~~~~ Error computations ~~~~
def errors(A, B):
    return np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro')

# ~~~~ K values ~~~~
K_values = [1, 5, 10, 20, 50, 100, 200]     # "For each K in K_set." Chooses which ranks to test.

error_X_YK = []
error_X_noise_YK = []   # These prepare containers to store error values as K changes.

# ~~~~ Main loop ~~~~
for K in K_values:
    QK = U[: , :K]  # Q_K
    YK = QK @ (QK.T @ X_noise)  # Y_K = (Q_K) * (Q_K_Transpose) * (X_noise)

    error_X_YK.append(errors(X, YK))
    error_X_noise_YK.append(errors(X_noise, YK))

# ~~~~ Constant noise error, e(X, X_noise) ~~~~
error_X_X_noise = errors(X, X_noise)

# ~~~~ Plotting Figure 7: All errors vs. K ~~~~
plt.figure()
plt.plot(K_values, error_X_noise_YK, '-s', label='e(X_noise, Y_K)')
plt.axhline(error_X_X_noise, linestyle='--', label='e(X, X_noise)')
plt.xlabel('K')
plt.ylabel('Errors')
plt.title('LRD Errors vs. K')
plt.legend()
plt.grid(True)
plt.savefig('low-rank_denoising.png')
plt.close()