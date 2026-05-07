import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix_multiclass(y_true, y_pred, num_classes=10):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        cm[true_label, predicted_label] += 1

    return cm


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_recall_f1_per_class(y_true, y_pred, num_classes=10):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0

    return precision, recall, f1


def macro_average(values):
    return np.mean(values)


def weighted_average(values, y_true, num_classes=10):
    y_true = np.asarray(y_true).astype(int)

    weights = np.zeros(num_classes)

    for c in range(num_classes):
        weights[c] = np.sum(y_true == c)

    total = np.sum(weights)

    if total == 0:
        return 0.0

    return np.sum(values * weights) / total


def classification_report_multiclass(y_true, y_pred, model_name="Model", num_classes=10):
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1 = precision_recall_f1_per_class(
        y_true,
        y_pred,
        num_classes=num_classes
    )

    macro_precision = macro_average(precision)
    macro_recall = macro_average(recall)
    macro_f1 = macro_average(f1)

    weighted_precision = weighted_average(precision, y_true, num_classes)
    weighted_recall = weighted_average(recall, y_true, num_classes)
    weighted_f1 = weighted_average(f1, y_true, num_classes)

    print("=" * 70)
    print(f"Multiclass Classification Report — {model_name}")
    print("=" * 70)
    print(f"Accuracy           : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro Precision    : {macro_precision:.4f}")
    print(f"Macro Recall       : {macro_recall:.4f}")
    print(f"Macro F1-score     : {macro_f1:.4f}")
    print(f"Weighted Precision : {weighted_precision:.4f}")
    print(f"Weighted Recall    : {weighted_recall:.4f}")
    print(f"Weighted F1-score  : {weighted_f1:.4f}")
    print("-" * 70)

    for c in range(num_classes):
        support = np.sum(np.asarray(y_true).astype(int) == c)
        print(
            f"Class {c}: "
            f"Precision={precision[c]:.4f} | "
            f"Recall={recall[c]:.4f} | "
            f"F1={f1[c]:.4f} | "
            f"Support={support}"
        )

    print("=" * 70)

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
    }


def plot_confusion_matrix_multiclass(
    y_true,
    y_pred,
    model_name="Model",
    num_classes=10,
    save_path=None
):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image)

    class_labels = np.arange(num_classes)

    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def save_results_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    file_exists = os.path.exists(save_path)

    with open(save_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {key: results[key] for key in fieldnames}
        writer.writerow(row)


def evaluate_multiclass(
    y_true,
    y_pred,
    model_name="Model",
    num_classes=10,
    plot=True,
    save_confusion_path=None,
    save_csv_path=None
):
    results = classification_report_multiclass(
        y_true,
        y_pred,
        model_name=model_name,
        num_classes=num_classes
    )

    if plot:
        plot_confusion_matrix_multiclass(
            y_true,
            y_pred,
            model_name=model_name,
            num_classes=num_classes,
            save_path=save_confusion_path
        )

    if save_csv_path is not None:
        save_results_csv(results, save_csv_path)

    return results
