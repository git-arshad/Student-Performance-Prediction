from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src.predict import predict_student
from src.recommendations import generate_recommendations

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "decision_tree.pkl"


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file is missing. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def input_panel(prefix: str, defaults: dict) -> dict:
    return {
        "Attendance_Percentage": st.number_input("Attendance (%)", 0.0, 100.0, float(defaults["Attendance_Percentage"]), 0.5, key=f"attendance_{prefix}"),
        "Assignment_Score": st.number_input("Assignment marks", 0.0, 100.0, float(defaults["Assignment_Score"]), 0.5, key=f"assignment_{prefix}"),
        "Internal_Marks": st.number_input("Internal marks", 0.0, 100.0, float(defaults["Internal_Marks"]), 0.5, key=f"internal_{prefix}"),
        "Study_Hours_Per_Week": st.number_input("Study hours per week", 0.0, None, float(defaults["Study_Hours_Per_Week"]), 0.5, key=f"study_{prefix}"),
        "Previous_GPA": st.number_input("Previous GPA (0–4)", 0.0, 4.0, float(defaults["Previous_GPA"]), 0.01, key=f"gpa_{prefix}"),
    }


def metrics_and_charts(bundle):
    metrics = bundle["metrics"]
    st.subheader("Model Evaluation / Metrics")
    cols = st.columns(4)
    for column, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"], ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]):
        column.metric(label, f"{metrics[key]:.2%}")
    st.caption("Precision, recall, and F1 use weighted averaging, which accounts for the number of examples in each class.")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(pd.DataFrame(metrics["confusion_matrix"], index=metrics["class_order"], columns=metrics["class_order"]).values, display_labels=metrics["class_order"]).plot(ax=ax, colorbar=False)
        ax.set_title("Held-out test confusion matrix")
        st.pyplot(fig)
    with right:
        importance = pd.Series(metrics["feature_importance"]).sort_values()
        fig, ax = plt.subplots()
        importance.plot.barh(ax=ax, color="#2a6f97")
        ax.set_title("Decision Tree feature importance")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    st.subheader("Class Distribution")
    st.bar_chart(pd.Series(bundle["class_distribution"], name="Students"))
    st.caption(f"Depth control comparison — basic test accuracy: {bundle['comparison']['basic_test_accuracy']:.2%}; limited-tree test accuracy: {bundle['comparison']['limited_test_accuracy']:.2%}.")


st.set_page_config(page_title="Student Performance Prediction", page_icon="📘", layout="wide")
st.title("Intelligent Student Performance Prediction System")
st.write("A Decision Tree estimates one of four project-defined performance categories. Results are decision support, not a guarantee.")
bundle = load_bundle()
defaults = {"Attendance_Percentage": 75, "Assignment_Score": 65, "Internal_Marks": 65, "Study_Hours_Per_Week": 10, "Previous_GPA": 3.0}

with st.sidebar:
    st.header("Student Information / Prediction")
    current = input_panel("current", defaults)
    predict_clicked = st.button("Predict performance", type="primary")

if predict_clicked:
    prediction = predict_student(bundle, current)
    recommendation = generate_recommendations(current, prediction)
    st.session_state["prediction"] = prediction
    st.session_state["recommendation"] = recommendation

if "prediction" in st.session_state:
    st.subheader("Predicted Performance")
    st.success(st.session_state["prediction"].upper())
    st.subheader("Areas Needing Improvement")
    areas = st.session_state["recommendation"]["areas"]
    st.write(", ".join(areas) if areas else "No areas fall below the project guidance thresholds.")
    st.subheader("Improvement Suggestions")
    for suggestion in st.session_state["recommendation"]["suggestions"]:
        st.write("• " + suggestion)

st.divider()
st.subheader("What-If Analysis")
st.caption("Change values below to see the trained model's new estimate. A change is not a guarantee of improvement.")
with st.expander("Try another scenario", expanded=False):
    scenario = input_panel("whatif", defaults)
    if st.button("Run what-if analysis"):
        st.info(f"What-if predicted performance: **{predict_student(bundle, scenario).upper()}**")

st.divider()
st.subheader("Model Information")
st.write("DecisionTreeClassifier with max_depth=5 and min_samples_leaf=2; trained with a reproducible 75/25 stratified split (random_state=42).")
metrics_and_charts(bundle)
