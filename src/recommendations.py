"""Deterministic, input-based improvement guidance kept separate from the model."""


def generate_recommendations(student_data: dict, predicted_class: str) -> dict:
    """Return relevant weak areas and practical suggestions for one student."""
    areas, suggestions = [], []
    checks = [
        ("Attendance", student_data["Attendance_Percentage"] < 75,
         "Aim for at least 75% attendance by planning for every scheduled class."),
        ("Assignment marks", student_data["Assignment_Score"] < 60,
         "Review assignment feedback and set an earlier weekly deadline for each task."),
        ("Internal marks", student_data["Internal_Marks"] < 60,
         "Spend regular preparation time on internal assessments and practise weak topics."),
        ("Study hours", student_data["Study_Hours_Per_Week"] < 8,
         "Build toward at least 8 focused study hours per week in short regular sessions."),
        ("Previous GPA", student_data["Previous_GPA"] < 2.8,
         "Use a weekly revision plan and seek course support for subjects that lowered GPA."),
    ]
    for area, weak, suggestion in checks:
        if weak:
            areas.append(area)
            suggestions.append(suggestion)
    if not areas:
        suggestions.append("Maintain these habits and review goals before the next assessment.")
    if predicted_class in {"Poor", "Average"}:
        suggestions.append("Track progress after each assessment; the model result is guidance, not a guarantee.")
    return {"areas": areas, "suggestions": suggestions}
