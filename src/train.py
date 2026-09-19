"""Train and persist the required Decision Tree classifier."""
from pathlib import Path
import sys

import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate_model
from src.preprocessing import FEATURES, add_performance_target, load_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Student_Performance.csv"
MODEL_PATH = ROOT / "models" / "decision_tree.pkl"
RANDOM_STATE = 42


def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH) -> dict:
    data = add_performance_target(load_data(data_path))
    x, y = data[FEATURES], data["Performance_Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    basic_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    basic_tree.fit(x_train, y_train)
    model = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    comparison = {
        "basic_train_accuracy": float(basic_tree.score(x_train, y_train)),
        "basic_test_accuracy": float(basic_tree.score(x_test, y_test)),
        "limited_train_accuracy": float(model.score(x_train, y_train)),
        "limited_test_accuracy": float(model.score(x_test, y_test)),
    }
    bundle = {
        "model": model, "features": FEATURES, "metrics": metrics,
        "comparison": comparison,
        "class_distribution": data["Performance_Class"].value_counts().reindex(
            ["Excellent", "Good", "Average", "Poor"], fill_value=0
        ).to_dict(),
        "parameters": model.get_params(),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


if __name__ == "__main__":
    saved = train_and_save()
    print("Saved model to", MODEL_PATH)
    print("Class distribution:", saved["class_distribution"])
    print("Metrics:", {key: round(value, 4) for key, value in saved["metrics"].items() if isinstance(value, float)})
    print("Overfitting comparison:", saved["comparison"])
