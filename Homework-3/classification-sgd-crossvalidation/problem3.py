import numpy as np
import time
np.random.seed(42)

from problem1 import X, y
from problem2 import softmax, kfold_split, accuracy, confusion_matrix

def one_hot(y, num_classes=3):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y

def train_sgd(X_train, y_train, X_test, y_test, batch_size, epochs, lr):
    n, d = X_train.shape
    num_classes = 3

    # Initialize weight matrix W with zeros
    W = np.zeros((d, num_classes))

    train_accs = []
    test_accs = []
    times = []

    for epoch in range(epochs):
        start_time = time.time()

        # Shuffle training data each epoch
        perm = np.random.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        # Mini-batch SGD
        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i + batch_size]
            y_batch = y_shuf[i:i + batch_size]

            Y_batch = one_hot(y_batch, num_classes)

            logits = X_batch @ W
            probs = softmax(logits)

            grad = X_batch.T @ (probs - Y_batch) / len(X_batch)

            W = W - lr * grad

        # Record accuracy after each epoch
        train_pred = np.argmax(X_train @ W, axis=1)
        test_pred = np.argmax(X_test @ W, axis=1)

        train_accs.append(accuracy(y_train, train_pred))
        test_accs.append(accuracy(y_test, test_pred))

        # Record wall-clock time for this epoch
        times.append(time.time() - start_time)

    final_test_pred = np.argmax(X_test @ W, axis=1)
    cm = confusion_matrix(y_test, final_test_pred, num_classes)

    return train_accs, test_accs, times, cm

def run_cross_validation(X, y, lr, epochs, batch_size, K):
    splits = kfold_split(len(X), K)

    all_train_accs = []
    all_test_accs = []
    all_times = []
    all_cms = []

    for train_idx, test_idx in splits:
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        train_accs, test_accs, times, cm = train_sgd(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size,
            epochs,
            lr
        )

        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_times.append(times)
        all_cms.append(cm)

    avg_train_accs = np.mean(all_train_accs, axis=0)
    avg_test_accs = np.mean(all_test_accs, axis=0)
    avg_times = np.mean(all_times, axis=0)
    avg_cm = np.mean(all_cms, axis=0)

    return avg_train_accs, avg_test_accs, avg_times, avg_cm
