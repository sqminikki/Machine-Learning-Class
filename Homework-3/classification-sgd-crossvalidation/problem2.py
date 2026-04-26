import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def kfold_split(n, K):
    indices = np.arange(n)
    np.random.shuffle(indices)

    folds = np.array_split(indices, K)

    splits = []
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
        splits.append((train_idx, test_idx))

    return splits

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    return cm