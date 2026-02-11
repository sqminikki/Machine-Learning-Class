import numpy as np
import matplotlib.pyplot as plt
from problem0 import X, X_tilde, y, myu

# --- SVD of Centered Data Matrix ---
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# --- Project onto first two left singular vectors ---

Q_2 = U[:, :2]
P_2 = Q_2.T @ X_tilde

print("Q_2 shape:", Q_2.shape)
print("P_2 shape:", P_2.shape)

# --- Compute Class Centroids in Projected Space ---
classes = [0, 1, 9]
proj_centroids = {}

for c in classes:
    idx = np.where(y == c)[0]
    P2_c = P_2[:, idx]
    m2_c = np.mean(P2_c, axis=1, keepdims=True)
    proj_centroids[c] = m2_c
    print(f"Class {c}: projected centroid shape {m2_c.shape}, N_c={len(idx)}")

# ~~~~ Plotting Figure 5: Scatter Plot of the Columns of P_2 ~~~~
plt.figure()
for c in classes:
    idx = np.where(y == c)[0]
    plt.scatter(P_2[0, idx], P_2[1, idx], s=8, alpha=0.4, label=f"digit {c}")
    mc = proj_centroids[c]
    plt.scatter(mc[0, 0], mc[1, 0], s=200, marker="X", edgecolors="k")

plt.xlabel("Projection onto u1")
plt.ylabel("Projection onto u2")
plt.title("Scatter Plot of Columns of P_2")
plt.legend()
plt.grid(True)
plt.savefig("Fig_5_ScatterPlot.png")

# --- Reconstruct Representative Samples Using K = 5, 20, 100 ---
K_list = [5, 20, 100]

rep_idx = {}
for c in classes:
    rep_idx[c] = int(np.where(y == c)[0][0])
fig, axes = plt.subplots(nrows=len(classes), ncols=1 + len(K_list), figsize=(10, 7))

for r, c in enumerate(classes):
    j = rep_idx[c]

    # --- Original Image Vector ---
    x_orig = X[:, j].reshape(28, 28)

    axes[r, 0].imshow(x_orig, cmap="gray")
    axes[r, 0].set_title(f"Digit {c}\nOriginal")
    axes[r, 0].axis("off")

    # --- Centered Sample Vector ---
    x_tilde = X_tilde[:, j]

    for col, K in enumerate(K_list, start=1):
        QK = U[:, :K]
        y_tilde = QK @ (QK.T @ x_tilde)
        y_rec = (y_tilde.reshape(784, 1) + myu)
        y_img = y_rec.reshape(28, 28)


        y_img = np.clip(y_img, 0, 255)

        axes[r, col].imshow(y_img, cmap="gray")
        axes[r, col].set_title(f"K={K}")
        axes[r, col].axis("off")

plt.tight_layout()
plt.savefig("Fig_6_OrgvsRecons.png")
