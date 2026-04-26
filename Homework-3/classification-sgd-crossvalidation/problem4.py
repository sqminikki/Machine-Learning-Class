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
plt.show()


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
plt.show()


# Fig. 17: Confusion Matrices

for batch_size in batch_sizes:
    cm = results[batch_size]["cm"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)

    plt.title(f"Fig. 17: Final Average Confusion Matrix, Batch Size = {batch_size}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar(label="Number of Instances")

    plt.xticks([0, 1, 2], ["Class 0", "Class 1", "Class 2"])
    plt.yticks([0, 1, 2], ["Class 0", "Class 1", "Class 2"])

    # Show numbers inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.1f}",
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"fig17_confusion_matrix_batch_{batch_size}.png", dpi=300)
    plt.show()