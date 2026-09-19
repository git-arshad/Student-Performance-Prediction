"""Prediction API used by the Streamlit interface."""
import pandas as pd

from .preprocessing import validate_student_input


def predict_student(bundle: dict, student_data: dict) -> str:
    clean = validate_student_input(student_data)
    frame = pd.DataFrame([clean], columns=bundle["features"])
    return str(bundle["model"].predict(frame)[0])
