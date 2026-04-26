import numpy as np
import matplotlib.pyplot as plt

from problem1 import X, y
from problem3 import run_cross_validation

# Hyperparameters
lr = 0.1
epochs = 500
batch_sizes = [1, 36, 72]
K = 10

results = {}

# Run experiments for each mini-batch size
for batch_size in batch_sizes:
    print(f"Running mini-batch size: {batch_size}")

    train_acc, test_acc, times, cm = run_cross_validation(
        X, y,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        K=K
    )

    results[batch_size] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "times": times,
        "cm": cm
    }


# Fig. 15: Accuracy vs Epoch

plt.figure(figsize=(10, 6))

for batch_size in batch_sizes:
    plt.plot(
        results[batch_size]["train_acc"],
        label=f"Train Accuracy, batch={batch_size}"
    )

    plt.plot(
        results[batch_size]["test_acc"],
        linestyle="--",
        label=f"Test Accuracy, batch={batch_size}"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fig. 15: Training and Testing Accuracy vs Epoch Index")
plt.legend()
plt.grid(True)
plt.savefig("fig15_train_and_test_accuracy_vs_epochs.png", dpi=300)


# Fig. 16: Time vs Epoch

plt.figure(figsize=(10, 6))

for batch_size in batch_sizes:
    plt.plot(
        results[batch_size]["times"],
        label=f"Batch size={batch_size}"
    )

plt.xlabel("Epoch")
plt.ylabel("Wall-Clock Time per Epoch (seconds)")
plt.title("Fig. 16: Training Time vs Epoch Index")
plt.legend()
plt.grid(True)
plt.savefig("fig16_training_time_vs_epoch.png", dpi=300)


# Fig. 17: Combined Confusion Matrices

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, batch_size in enumerate(batch_sizes):
    cm = results[batch_size]["cm"]
    ax = axes[idx]

    im = ax.imshow(cm, cmap="Greens")

    ax.set_title(f"Batch = {batch_size}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    # Add numbers in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.1f}",
                    ha="center", va="center")

plt.suptitle("Fig. 17: Final Average Confusion Matrices for Each Mini-Batch Size")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("fig17_avg_confusion_matrices.png", dpi=300)
