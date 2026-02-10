import numpy as np
import matplotlib.pyplot as plt
from problem0 import X, X_tilde, y, myu   # uses your Problem 0 outputs

# ----------------------------
# Problem 5: Low-Dimensional Geometry and Visualization
# ----------------------------

# 1) SVD of centered data
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# 2) Project onto first two left singular vectors
# Q2 = [u1, u2] ∈ R^{784x2}
Q2 = U[:, :2]                       # (784, 2)

# P2 = Q2^T X_tilde ∈ R^{2xN}
P2 = Q2.T @ X_tilde                 # (2, N)

print("Q2 shape:", Q2.shape)
print("P2 shape:", P2.shape)

# 3) Compute class centroids in projected space
classes = [0, 1, 9]
proj_centroids = {}

for c in classes:
    idx = np.where(y == c)[0]       # which columns belong to class c
    P2_c = P2[:, idx]               # (2, N_c)
    m2_c = np.mean(P2_c, axis=1, keepdims=True)  # (2,1)
    proj_centroids[c] = m2_c
    print(f"Class {c}: projected centroid shape {m2_c.shape}, N_c={len(idx)}")

# 4) Plot scatter of projected points, color-coded by digit + centroid markers
plt.figure()
for c in classes:
    idx = np.where(y == c)[0]
    plt.scatter(P2[0, idx], P2[1, idx], s=8, alpha=0.4, label=f"digit {c}")
    # centroid marker
    mc = proj_centroids[c]
    plt.scatter(mc[0, 0], mc[1, 0], s=200, marker="X", edgecolors="k")

plt.xlabel("Projection onto u1")
plt.ylabel("Projection onto u2")
plt.title("Problem 5: Projection onto First Two Left Singular Vectors")
plt.legend()
plt.grid(True)
plt.savefig("problem5_scatter_P2.png", dpi=300, bbox_inches="tight")
print("Saved scatter plot: problem5_scatter_P2.png")

# 5) Reconstruct representative samples using K = 5, 20, 100
K_list = [5, 20, 100]

# Choose one representative sample index for each digit (first occurrence)
rep_idx = {}
for c in classes:
    rep_idx[c] = int(np.where(y == c)[0][0])

# Make a figure: rows = digits, columns = original + reconstructions
# columns: original, K=5, K=20, K=100  => 4 columns
fig, axes = plt.subplots(nrows=len(classes), ncols=1 + len(K_list), figsize=(10, 7))

for r, c in enumerate(classes):
    j = rep_idx[c]

    # Original image vector (uncentered) is X[:, j]
    x_orig = X[:, j].reshape(28, 28)

    axes[r, 0].imshow(x_orig, cmap="gray")
    axes[r, 0].set_title(f"Digit {c}\nOriginal")
    axes[r, 0].axis("off")

    # Centered sample vector
    x_tilde = X_tilde[:, j]  # (784,)

    for col, K in enumerate(K_list, start=1):
        QK = U[:, :K]                           # (784, K)
        y_tilde = QK @ (QK.T @ x_tilde)         # (784,)  rank-K reconstruction in centered space
        y_rec = (y_tilde.reshape(784, 1) + myu) # add mean back to compare to original
        y_img = y_rec.reshape(28, 28)

        # Optional: clip to display range (MNIST is 0..255)
        y_img = np.clip(y_img, 0, 255)

        axes[r, col].imshow(y_img, cmap="gray")
        axes[r, col].set_title(f"K={K}")
        axes[r, col].axis("off")

plt.tight_layout()
plt.savefig("problem5_reconstructions.png", dpi=300, bbox_inches="tight")
print("Saved reconstruction figure: problem5_reconstructions.png")

# 6) (Optional) A simple classification rule based on projection + reconstruction error
# Rule idea:
#   - Compute projected point p = Q2^T x_tilde
#   - Assign class whose projected centroid is closest in Euclidean distance
#   - Use reconstruction error with some K (say K=20) as a confidence measure
def classify_with_projection_and_recon(x_col_index, K_recon=20):
    x_t = X_tilde[:, x_col_index]                # (784,)
    p = Q2.T @ x_t                               # (2,)

    # nearest centroid in projected space
    best_c = None
    best_dist = None
    for c in classes:
        mc = proj_centroids[c].reshape(2,)       # (2,)
        d = np.linalg.norm(p - mc, ord=2)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_c = c

    # reconstruction error as extra information
    QK = U[:, :K_recon]
    x_hat_t = QK @ (QK.T @ x_t)
    recon_err = np.linalg.norm(x_t - x_hat_t, ord=2)

    return best_c, best_dist, recon_err

# Example usage on one sample from each class
for c in classes:
    j = rep_idx[c]
    pred, dist, rerr = classify_with_projection_and_recon(j, K_recon=20)
    print(f"Example digit {c}: predicted={pred}, proj_dist={dist:.4f}, recon_err={rerr:.4f}")