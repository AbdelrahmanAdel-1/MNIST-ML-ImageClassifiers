import numpy as np
import matplotlib.pyplot as plt
import csv


# =========================================================
# MULTICLASS CONFUSION MATRIX
# =========================================================

def confusion_matrix_multiclass(y_true, y_pred, num_classes=10):

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual)][int(predicted)] += 1

    return cm


# =========================================================
# ACCURACY
# =========================================================

def accuracy_score(y_true, y_pred):

    correct = np.sum(y_true == y_pred)

    return correct / len(y_true)


# =========================================================
# PRECISION / RECALL / F1 PER CLASS
# =========================================================

def precision_recall_f1_per_class(y_true, y_pred, num_classes=10):

    cm = confusion_matrix_multiclass(
        y_true,
        y_pred,
        num_classes
    )

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):

        tp = cm[i, i]

        fp = np.sum(cm[:, i]) - tp

        fn = np.sum(cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        np.array(precision_list),
        np.array(recall_list),
        np.array(f1_list)
    )


# =========================================================
# MACRO AVERAGE
# =========================================================

def macro_average(metric_values):

    return np.mean(metric_values)


# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(
    cm,
    title="Confusion Matrix",
    save_path=None
):

    plt.figure(figsize=(8, 6))

    plt.imshow(cm, cmap='Blues')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha='center',
                va='center',
                fontsize=7
            )

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(cm.shape[0]))
    plt.yticks(range(cm.shape[0]))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# =========================================================
# SAVE RESULTS CSV
# =========================================================

def save_results_csv(results_dict, save_path):

    with open(save_path, mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        for key, value in results_dict.items():
            writer.writerow([key, value])


# =========================================================
# COMPLETE MULTICLASS EVALUATION
# =========================================================

def evaluate_multiclass(
    y_true,
    y_pred,
    model_name="Model",
    num_classes=10,
    plot=True,
    save_confusion_path=None,
    save_csv_path=None
):

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1 = precision_recall_f1_per_class(
        y_true,
        y_pred,
        num_classes
    )

    macro_precision = macro_average(precision)

    macro_recall = macro_average(recall)

    macro_f1 = macro_average(f1)

    cm = confusion_matrix_multiclass(
        y_true,
        y_pred,
        num_classes
    )

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print("Accuracy :", acc)
    print("Macro Precision :", macro_precision)
    print("Macro Recall    :", macro_recall)
    print("Macro F1-score  :", macro_f1)

    print("\nConfusion Matrix:\n")
    print(cm)

    if plot:

        plot_confusion_matrix(
            cm,
            title=f"{model_name} Confusion Matrix",
            save_path=save_confusion_path
        )

    results = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

    if save_csv_path is not None:

        save_results_csv(
            results,
            save_csv_path
        )

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }


# =========================================================
# SIMPLE WRAPPER FUNCTIONS
# Used for cleaner notebook evaluation
# =========================================================

def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


def precision_macro(y_true, y_pred, num_classes=10):

    precision, _, _ = precision_recall_f1_per_class(
        y_true,
        y_pred,
        num_classes
    )

    return macro_average(precision)


def recall_macro(y_true, y_pred, num_classes=10):

    _, recall, _ = precision_recall_f1_per_class(
        y_true,
        y_pred,
        num_classes
    )

    return macro_average(recall)


def f1_macro(y_true, y_pred, num_classes=10):

    _, _, f1 = precision_recall_f1_per_class(
        y_true,
        y_pred,
        num_classes
    )

    return macro_average(f1)


def confusion_matrix(y_true, y_pred, num_classes=10):

    return confusion_matrix_multiclass(
        y_true,
        y_pred,
        num_classes
    )