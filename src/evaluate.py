"""Held-out multiclass evaluation helpers."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .preprocessing import CLASS_ORDER


def evaluate_model(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision), "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_ORDER).tolist(),
        "class_order": CLASS_ORDER,
        "classification_report": classification_report(y_test, predictions, labels=CLASS_ORDER, zero_division=0),
        "test_size": int(len(y_test)),
        "feature_importance": dict(zip(x_test.columns, model.feature_importances_.tolist())),
    }
