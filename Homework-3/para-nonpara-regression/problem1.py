import numpy as np
import matplotlib.pyplot as plt
from dataset_a import _x_all, _y_all, split_data, standardize

np.random.seed(42)

# --- i. Verify split sizes ---
for ts in [1000, 5000, 10000]:
    x_tr, x_te, y_tr, y_te = split_data(_x_all, _y_all, ts)
    print(f"train_size={ts}: x_train={x_tr.shape}, x_test={x_te.shape}, "
          f"y_train={y_tr.shape}, y_test={y_te.shape}")

# --- iii. Standardize features for train_size=5000 ---
x_train_raw, x_test_raw, y_train, y_test = split_data(_x_all, _y_all, 5000)
x_train_std, mu_x, sigma_x = standardize(x_train_raw)
x_test_std = (x_test_raw - mu_x) / sigma_x

print(f"\nStandardized x_train: mean={x_train_std.mean():.6f}, std={x_train_std.std():.6f}")

# --- iv. Fig.0: side-by-side scatter plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x_train_raw, y_train, s=5, alpha=0.4, color='steelblue')
axes[0].set_xlabel("MedInc ($100k)")
axes[0].set_ylabel("Median House Value ($100k)")
axes[0].set_title("Unstandardized (train_size=5000)")

axes[1].scatter(x_train_std, y_train, s=5, alpha=0.4, color='darkorange')
axes[1].set_xlabel("Standardized MedInc")
axes[1].set_ylabel("Median House Value ($100k)")
axes[1].set_title("Standardized MedInc vs Target (train_size=5000)")

plt.tight_layout()
plt.savefig("fig0.png", dpi=120)
plt.show()
print("Saved fig0.png")
