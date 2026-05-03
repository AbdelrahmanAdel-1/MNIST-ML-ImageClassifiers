# ============================================================
# Cell 3: Decision Tree Classifier — Implemented from Scratch
# Binary Classification on MNIST (Gini Index)
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


# ─────────────────────────────────────────────
# Vectorized Best Split (Gini Index)
# ─────────────────────────────────────────────
def best_split_fast(X, y):
    """
    For each feature, sort samples once then use prefix sums
    to evaluate ALL thresholds in one vectorized pass.
    Gini(D) = 1 - sum(p_k^2)
    Weighted Gini = (|L|/|D|)*Gini(L) + (|R|/|D|)*Gini(R)
    """
    n_samples, n_features = X.shape
    n_pos_total = np.sum(y == 1)
    best_feature, best_threshold, best_gini_val = None, None, float('inf')

    for feat_idx in range(n_features):
        col        = X[:, feat_idx]
        order      = np.argsort(col, kind='mergesort')
        col_sorted = col[order]
        y_sorted   = y[order]

        # Prefix sums — count positives on the left at every cut point
        cum_pos = np.cumsum(y_sorted == 1)
        cum_n   = np.arange(1, n_samples + 1)

        left_pos  = cum_pos[:-1]
        left_n    = cum_n[:-1]
        right_pos = n_pos_total - left_pos
        right_n   = n_samples   - left_n

        # Only consider cuts where the feature value actually changes
        valid = col_sorted[:-1] != col_sorted[1:]
        if not np.any(valid):
            continue

        left_pos  = left_pos[valid];  left_n  = left_n[valid]
        right_pos = right_pos[valid]; right_n = right_n[valid]

        # Vectorized Gini for all valid thresholds at once
        p_left  = left_pos  / left_n
        p_right = right_pos / right_n

        gini_left  = 1.0 - (p_left**2  + (1 - p_left)**2)
        gini_right = 1.0 - (p_right**2 + (1 - p_right)**2)
        weighted   = (left_n * gini_left + right_n * gini_right) / n_samples

        best_idx = np.argmin(weighted)
        if weighted[best_idx] < best_gini_val:
            best_gini_val  = weighted[best_idx]
            best_feature   = feat_idx
            cut            = np.where(valid)[0][best_idx]
            best_threshold = (col_sorted[cut] + col_sorted[cut + 1]) / 2.0

    return best_feature, best_threshold, best_gini_val


# ─────────────────────────────────────────────
# Recursive Tree Builder
# ─────────────────────────────────────────────
def build_tree(X, y, depth=0, max_depth=15, min_samples_split=10):
    """
    Stopping conditions:
      1. Pure node (all same class)
      2. Reached max_depth
      3. Too few samples to split
    Leaf value = majority class (argmax of bincount)
    """
    n_classes = len(np.unique(y))

    if (n_classes == 1) or (depth >= max_depth) or (len(y) < min_samples_split):
        return Node(value=int(np.bincount(y).argmax()))

    feat, thresh, _ = best_split_fast(X, y)

    if feat is None:
        return Node(value=int(np.bincount(y).argmax()))

    left_mask  = X[:, feat] <= thresh
    right_mask = ~left_mask

    return Node(
        feature   = feat,
        threshold = thresh,
        left  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth, min_samples_split),
        right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_split)
    )


# ─────────────────────────────────────────────
# Prediction (iterative — avoids recursion limit)
# ─────────────────────────────────────────────
def predict_one(tree, x):
    node = tree
    while not node.is_leaf():
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.value

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


print("✅ Cell 3 complete — Decision Tree defined successfully.")
