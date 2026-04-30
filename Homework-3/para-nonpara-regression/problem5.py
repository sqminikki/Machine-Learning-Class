import numpy as np
import matplotlib.pyplot as plt
from dataset_b import load_dataset_b

np.random.seed(42)

FEATURES = ["cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year", "origin"]

TRI, TRO, TEI, TEO, mu_rows, sigma_rows = load_dataset_b()

# --- v. Pearson correlation matrix among {x1..x7, y} ---
# Build (8, ntrain) matrix: rows 0-6 = standardized features, row 7 = mpg
data_mat = np.vstack([TRI, TRO.reshape(1, -1)])   # (8, 300)
labels = FEATURES + ["mpg"]

n = data_mat.shape[1]
C = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        xi = data_mat[i] - data_mat[i].mean()
        xj = data_mat[j] - data_mat[j].mean()
        denom = np.sqrt((xi**2).sum() * (xj**2).sum())
        C[i, j] = abs(xi @ xj / denom) if denom > 0 else 0.0

print("Absolute Pearson Correlation Matrix (rows/cols: cylinders, displacement,")
print("horsepower, weight, acceleration, model_year, origin, mpg):\n")
header = "".join(f"{l:>13}" for l in labels)
print(f"{'':>13}" + header)
for i, li in enumerate(labels):
    row = "".join(f"{C[i,j]:>13.4f}" for j in range(8))
    print(f"{li:>13}" + row)

# Correlations of each feature with mpg (last row, cols 0-6)
mpg_corr = C[7, :7]
best_feat_idx = np.argmax(mpg_corr)
print(f"\nFeature most correlated with mpg: {FEATURES[best_feat_idx]} "
      f"(|r|={mpg_corr[best_feat_idx]:.4f})")

# --- Fig.9: mpg vs cylinders ---
plt.figure(figsize=(6, 5))
plt.scatter(TRI[0], TRO, s=20, alpha=0.6, color='steelblue')
plt.xlabel("Cylinders (standardized)")
plt.ylabel("MPG")
plt.title("Fig.9: MPG vs Cylinders")
plt.tight_layout()
plt.savefig("fig9.png", dpi=120)
plt.show()
print("Saved fig9.png")

# --- Fig.10: mpg vs weight ---
plt.figure(figsize=(6, 5))
plt.scatter(TRI[3], TRO, s=20, alpha=0.6, color='darkorange')
plt.xlabel("Weight (standardized)")
plt.ylabel("MPG")
plt.title("Fig.10: MPG vs Weight")
plt.tight_layout()
plt.savefig("fig10.png", dpi=120)
plt.show()
print("Saved fig10.png")

# --- Additional 3x3 grid: all 7 features vs mpg with jitter on discrete features ---
discrete_idx = {0, 5, 6}   # cylinders, model_year, origin
rng = np.random.default_rng(42)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    xi = TRI[i].copy()
    if i in discrete_idx:
        xi += rng.uniform(-0.05, 0.05, size=xi.shape)
    ax.scatter(xi, TRO, s=10, alpha=0.5)
    ax.set_xlabel(f"{feat} (std)")
    ax.set_ylabel("MPG")
    ax.set_title(feat)
# hide unused subplot
axes[7].set_visible(False)
axes[8].set_visible(False)
plt.suptitle("All 7 Features vs MPG (jitter on discrete features)", y=1.01)
plt.tight_layout()
plt.savefig("fig_grid.png", dpi=120, bbox_inches='tight')
plt.show()
print("Saved fig_grid.png")
