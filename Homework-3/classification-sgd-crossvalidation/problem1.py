import numpy as np

np.random.seed(42)

# Load dataset from file
data = np.loadtxt("wine.data", delimiter=",")

# First column = labels, convert 1,2,3 → 0,1,2
y = data[:, 0].astype(int) - 1

# Remaining columns = features
X = data[:, 1:]

# Keep first 40 samples from each class
X_filtered = []
y_filtered = []

for c in range(3):
    idx = np.where(y == c)[0][:40]
    X_filtered.append(X[idx])
    y_filtered.append(y[idx])

X = np.vstack(X_filtered)
y = np.hstack(y_filtered)

# Z-score normalization using population std
mean = np.mean(X, axis=0)
std = np.sqrt(np.mean((X - mean) ** 2, axis=0))
X = (X - mean) / std
