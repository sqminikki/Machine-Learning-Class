import numpy as np
from ucimlrepo import fetch_ucirepo

np.random.seed(42)

wine = fetch_ucirepo(id=109)

X = wine.data.features.to_numpy()
y = wine.data.targets.to_numpy().flatten()

# Convert labels from 1,2,3 → 0,1,2
y = y - 1

# Keep first 40 samples from each class
X_filtered = []
y_filtered = []

for c in range(3):
    idx = np.where(y == c)[0][:40]   # first 40 of class c
    X_filtered.append(X[idx])
    y_filtered.append(y[idx])

X = np.vstack(X_filtered)
y = np.hstack(y_filtered)

# Z-score normalization (using population std)

mean = np.mean(X, axis=0)

std = np.sqrt(np.mean((X - mean) ** 2, axis=0))

X = (X - mean) / std