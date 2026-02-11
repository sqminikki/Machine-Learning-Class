import numpy as np
from problem0 import X_tilde, y

digit_class_c = [0, 1, 9]
centroids = {}
norms = {}

for c in digit_class_c:
    # --- Length N_c ---
    idx = np.where(y == c)[0]

    # --- Submatrix X_tilde_c is element of R^{784 x N_c} ---
    X_tilde_c = X_tilde[:, idx]
    N_c = X_tilde_c.shape[1]

    # --- Centroid: m_c = (1/N_c) X_tilde_c 1 ---
    m_c = (1.0 / N_c) * (X_tilde_c @ np.ones((N_c, 1), dtype=X_tilde.dtype))

    # --- Norms of Centroid ---
    m_flat = m_c[:, 0]
    n1 = np.linalg.norm(m_flat, ord=1)
    n2 = np.linalg.norm(m_flat, ord=2)
    n3 = np.linalg.norm(m_flat, ord=3)

    centroids[c] = m_c
    norms[c] = (n1, n2, n3)

    # --- Sanity checks ---
    print(f"\nClass c = {c}, N_c = {N_c}")
    print(f"m_c shape: {m_c.shape}")
    print(f"||m_c||_1 = {n1:.6f}")
    print(f"||m_c||_2 = {n2:.6f}")
    print(f"||m_c||_3 = {n3:.6f}")
