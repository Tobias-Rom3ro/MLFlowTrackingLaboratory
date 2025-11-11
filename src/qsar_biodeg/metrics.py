import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict
import numpy as np


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


def get_classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(
        y_true, y_pred, target_names=['NRB', 'RB'], output_dict=True
    )


def save_json(obj: dict, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_confusion_matrix_png(y_true, y_pred, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['NRB', 'RB'])
    ax.set_yticklabels(['NRB', 'RB'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()