# ============================================================
# Cell 3: Decision Tree Classifier — Multiclass From Scratch
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# Node Structure
# ─────────────────────────────────────────────
class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, *, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None


def majority_class(y, n_classes=10):
    return int(np.bincount(y, minlength=n_classes).argmax())


# ─────────────────────────────────────────────
# Vectorized Best Split — Multiclass Gini Index
# ─────────────────────────────────────────────
def best_split_fast(X, y, n_classes=10):
    """
    Multiclass Gini:
        Gini(D) = 1 - sum(p_k^2)

    Weighted Gini:
        (|L|/|D|)Gini(L) + (|R|/|D|)Gini(R)
    """

    n_samples, n_features = X.shape
    total_counts = np.bincount(y, minlength=n_classes)

    best_feature = None
    best_threshold = None
    best_gini_val = float('inf')

    for feat_idx in range(n_features):
        col = X[:, feat_idx]

        order = np.argsort(col, kind='mergesort')
        col_sorted = col[order]
        y_sorted = y[order]

        one_hot = np.zeros((n_samples, n_classes), dtype=np.int32)
        one_hot[np.arange(n_samples), y_sorted] = 1

        cum_counts = np.cumsum(one_hot, axis=0)

        left_counts = cum_counts[:-1]
        left_n = np.arange(1, n_samples)

        right_counts = total_counts - left_counts
        right_n = n_samples - left_n

        valid = col_sorted[:-1] != col_sorted[1:]

        if not np.any(valid):
            continue

        left_counts = left_counts[valid]
        right_counts = right_counts[valid]
        left_n_valid = left_n[valid]
        right_n_valid = right_n[valid]

        p_left = left_counts / left_n_valid[:, None]
        p_right = right_counts / right_n_valid[:, None]

        gini_left = 1.0 - np.sum(p_left ** 2, axis=1)
        gini_right = 1.0 - np.sum(p_right ** 2, axis=1)

        weighted = (
            left_n_valid * gini_left +
            right_n_valid * gini_right
        ) / n_samples

        best_idx = np.argmin(weighted)

        if weighted[best_idx] < best_gini_val:
            valid_positions = np.where(valid)[0]
            cut = valid_positions[best_idx]

            best_gini_val = weighted[best_idx]
            best_feature = feat_idx
            best_threshold = (col_sorted[cut] + col_sorted[cut + 1]) / 2.0

    return best_feature, best_threshold, best_gini_val


# ─────────────────────────────────────────────
# Recursive Tree Builder
# ─────────────────────────────────────────────
def build_tree(X, y, depth=0, max_depth=10, min_samples_split=30, n_classes=10):
    n_unique_classes = len(np.unique(y))

    if (
        n_unique_classes == 1 or
        depth >= max_depth or
        len(y) < min_samples_split
    ):
        return Node(value=majority_class(y, n_classes))

    feat, thresh, _ = best_split_fast(X, y, n_classes)

    if feat is None:
        return Node(value=majority_class(y, n_classes))

    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask

    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return Node(value=majority_class(y, n_classes))

    return Node(
        feature=feat,
        threshold=thresh,
        left=build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            n_classes
        ),
        right=build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            n_classes
        )
    )


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict_one(tree, x):
    node = tree

    while not node.is_leaf():
        node = node.left if x[node.feature] <= node.threshold else node.right

    return node.value


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


print("Cell 3 complete — Multiclass Decision Tree defined successfully.")
