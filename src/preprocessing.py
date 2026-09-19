"""Data validation and project-defined four-class target construction."""
from pathlib import Path

import pandas as pd


FEATURES = [
    "Attendance_Percentage", "Assignment_Score", "Internal_Marks",
    "Study_Hours_Per_Week", "Previous_GPA",
]
CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]
RANGES = {
    "Attendance_Percentage": (0, 100), "Assignment_Score": (0, 100),
    "Internal_Marks": (0, 100), "Study_Hours_Per_Week": (0, None),
    "Previous_GPA": (0, 4.0),
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load, rename the internal-assessment field, and validate the raw CSV."""
    data = pd.read_csv(path).rename(columns={"Midterm_Marks": "Internal_Marks"})
    required = set(FEATURES + ["Final_Result"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data = data.dropna(subset=FEATURES + ["Final_Result"]).copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="raise")
        low, high = RANGES[feature]
        if (data[feature] < low).any() or (high is not None and (data[feature] > high).any()):
            raise ValueError(f"{feature} contains values outside its permitted range.")
    return data.drop_duplicates().reset_index(drop=True)


def add_performance_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create transparent project-defined categories without using Final_Result as a feature.

    The index combines current assessments and engagement; the legacy Pass/Fail
    outcome supplies only a small historical-outcome adjustment.  It is excluded
    from the classifier inputs, so the model must learn patterns from the five
    student inputs rather than receiving an outcome column at prediction time.
    """
    result = data.copy()
    study_component = result["Study_Hours_Per_Week"].clip(upper=20) * 5
    gpa_component = ((result["Previous_GPA"] - 2.0) / 2.0 * 100).clip(0, 100)
    index = (
        0.25 * result["Internal_Marks"] + 0.20 * result["Assignment_Score"]
        + 0.15 * result["Attendance_Percentage"] + 0.15 * study_component
        + 0.25 * gpa_component
    )
    # The original outcome is historical evidence only; it is never an input feature.
    index = index + result["Final_Result"].map({"Pass": 4, "Fail": -6}).fillna(0)
    result["Performance_Index"] = index.round(2)
    result["Performance_Class"] = pd.cut(
        index, bins=[-float("inf"), 45, 60, 75, float("inf")],
        labels=CLASS_ORDER, right=False,
    ).astype(str)
    return result


def validate_student_input(student_data: dict) -> dict:
    """Convert dashboard values to floats and enforce documented ranges."""
    clean = {}
    for feature in FEATURES:
        if feature not in student_data:
            raise ValueError(f"Missing input: {feature}")
        value = float(student_data[feature])
        low, high = RANGES[feature]
        if value < low or (high is not None and value > high):
            maximum = "no maximum" if high is None else high
            raise ValueError(f"{feature} must be between {low} and {maximum}.")
        clean[feature] = value
    return clean
