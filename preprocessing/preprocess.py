
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(" MNIST Dataset Loaded Successfully!")
    print(f"   Training images  : {X_train.shape}")
    print(f"   Training labels  : {y_train.shape}")
    print(f"   Test images      : {X_test.shape}")
    print(f"   Test labels      : {y_test.shape}")
    return X_train, y_train, X_test, y_test

def binary_encode_labels(y_train, y_test, positive_class=1):
    y_train_binary = np.where(y_train == positive_class, 1, 0)
    y_test_binary  = np.where(y_test  == positive_class, 1, 0)
    print(f" Labels encoded — Positive class: Digit {positive_class}")
    print(f"   Training — Positive samples : {np.sum(y_train_binary == 1)}")
    print(f"   Training — Negative samples : {np.sum(y_train_binary == 0)}")
    print(f"   Test     — Positive samples : {np.sum(y_test_binary  == 1)}")
    print(f"   Test     — Negative samples : {np.sum(y_test_binary  == 0)}")
    return y_train_binary, y_test_binary

def normalize_pixels(X_train, X_test):
    X_train_norm = X_train / 255.0
    X_test_norm  = X_test  / 255.0
    print(" Pixel values normalized to [0, 1]")
    return X_train_norm, X_test_norm

def flatten(X_train, X_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    print(" Images flattened successfully!")
    print(f"   Training shape : {X_train_flat.shape}")
    print(f"   Test shape     : {X_test_flat.shape}")
    return X_train_flat, X_test_flat

def apply_pca(X_train, X_test, n_components=50):
    mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean
    X_test_centered  = X_test  - mean
    cov_matrix = np.dot(X_train_centered.T, X_train_centered) / (X_train_centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices   = np.argsort(eigenvalues)[::-1]
    eigenvalues      = eigenvalues[sorted_indices]
    eigenvectors     = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :n_components]
    X_train_pca = np.dot(X_train_centered, top_eigenvectors)
    X_test_pca  = np.dot(X_test_centered,  top_eigenvectors)
    variance_explained = np.sum(eigenvalues[:n_components]) / np.sum(eigenvalues) * 100
    print(f"PCA applied successfully!")
    print(f"   Components kept    : {n_components}")
    print(f"   Training shape     : {X_train_pca.shape}")
    print(f"   Test shape         : {X_test_pca.shape}")
    print(f"   Variance explained : {variance_explained:.2f}%")
    return X_train_pca, X_test_pca, mean, top_eigenvectors

def extract_hog(X, cell_size=4, num_bins=9):
    n_samples = X.shape[0]
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(X)
    grad_x[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]
    grad_y[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180
    n_cells_row = X.shape[1] // cell_size
    n_cells_col = X.shape[2] // cell_size
    bin_width   = 180 / num_bins
    hog_features = []
    for row in range(n_cells_row):
        for col in range(n_cells_col):
            cell_mag = magnitude[:, row*cell_size:(row+1)*cell_size,
                                    col*cell_size:(col+1)*cell_size]
            cell_dir = direction[:, row*cell_size:(row+1)*cell_size,
                                    col*cell_size:(col+1)*cell_size]
            for bin_idx in range(num_bins):
                bin_start = bin_idx * bin_width
                bin_end   = bin_start + bin_width
                mask      = (cell_dir >= bin_start) & (cell_dir < bin_end)
                hog_features.append(np.sum(cell_mag * mask, axis=(1, 2)))
    hog_features = np.array(hog_features).T
    print(f" HOG features extracted successfully!")
    print(f"   Input shape  : {X.shape}")
    print(f"   Output shape : {hog_features.shape}")
    return hog_features

def train_val_split(X_train, y_train, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n_samples  = X_train.shape[0]
    indices    = np.random.permutation(n_samples)
    val_size   = int(n_samples * val_ratio)
    val_indices   = indices[:val_size]
    train_indices = indices[val_size:]
    X_tr  = X_train[train_indices]
    X_val = X_train[val_indices]
    y_tr  = y_train[train_indices]
    y_val = y_train[val_indices]
    print(" Train/Validation split done!")
    print(f"   Training   samples : {X_tr.shape[0]}")
    print(f"   Validation samples : {X_val.shape[0]}")
    return X_tr, X_val, y_tr, y_val

def standardize(X_tr, X_val, X_test):
    mean = np.mean(X_tr, axis=0)
    std  = np.std(X_tr, axis=0)
    std[std == 0] = 1
    X_tr_std   = (X_tr   - mean) / std
    X_val_std  = (X_val  - mean) / std
    X_test_std = (X_test - mean) / std
    print(" Features standardized successfully!")
    print(f"   Training mean (should be ~0) : {X_tr_std.mean():.4f}")
    print(f"   Training std  (should be ~1) : {X_tr_std.std():.4f}")
    return X_tr_std, X_val_std, X_test_std

def preprocess(feature_method='flatten', binary=True, positive_class=1, pca_components=50):
    print(" Starting Preprocessing Pipeline...")
    print(f"   Feature method : {feature_method}")
    print(f"   Mode           : {'Binary' if binary else 'Multi-class'}")
    print("="*50)
    print("\n Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print("\n  Encoding labels...")
    if binary:
        y_train, y_test = binary_encode_labels(y_train, y_test, positive_class)
    else:
        print(" Multi-class mode — labels kept as original (0-9)")
    print("\n Normalizing pixels...")
    X_train, X_test = normalize_pixels(X_train, X_test)
    print(f"\n Extracting features using {feature_method.upper()}...")
    if feature_method == 'flatten':
        X_train, X_test = flatten(X_train, X_test)
    elif feature_method == 'pca':
        X_train, X_test = flatten(X_train, X_test)
        X_train, X_test, _, _ = apply_pca(X_train, X_test, pca_components)
    elif feature_method == 'hog':
        X_train = extract_hog(X_train)
        X_test  = extract_hog(X_test)
    else:
        raise ValueError(" feature_method must be 'flatten', 'pca', or 'hog'")
    print("\n  Splitting data...")
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train)
    print("\n Standardizing features...")
    X_tr, X_val, X_test = standardize(X_tr, X_val, X_test)
    print("\n" + "="*50)
    print(" Preprocessing Pipeline Complete!")
    print(f"   X_tr   : {X_tr.shape}  y_tr   : {y_tr.shape}")
    print(f"   X_val  : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"   X_test : {X_test.shape}  y_test : {y_test.shape}")
    return X_tr, X_val, X_test, y_tr, y_val, y_test
