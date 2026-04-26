
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    cm = np.array([[TN, FP],
                   [FN, TP]])
    return cm

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    if (TP + FP) == 0:
        return 0.0
    prec = TP / (TP + FP)
    return prec

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    if (TP + FN) == 0:
        return 0.0
    rec = TP / (TP + FN)
    return rec

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec  = recall(y_true, y_pred)
    if (prec + rec) == 0:
        return 0.0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def classification_report(y_true, y_pred, model_name="Model"):
    acc  = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec  = recall(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print("="*50)
    print(f" Classification Report — {model_name}")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("-"*50)
    print(f"  TP : {TP}  |  FP : {FP}")
    print(f"  FN : {FN}  |  TN : {TN}")
    print("="*50)
    return acc, prec, rec, f1

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    classes    = ['Not Digit 1 (0)', 'Digit 1 (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate(y_true, y_pred, model_name="Model"):
    acc, prec, rec, f1 = classification_report(y_true, y_pred, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    results = {
        'model'    : model_name,
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1_score' : f1
    }
    return results
