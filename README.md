# Intelligent Student Performance Prediction System (P-8)

## Problem statement and objective

This project gives academic staff a transparent local tool for estimating student performance and discussing practical improvement actions. It uses a Decision Tree to predict **Excellent**, **Good**, **Average**, or **Poor** from attendance, assignment marks, internal marks, weekly study hours, and previous GPA.

## Features

- Validated interactive Streamlit inputs and four-class Decision Tree prediction.
- Deterministic areas-to-improve and suggestions, separate from ML prediction.
- What-if analysis, class distribution, feature importance, and held-out metrics.
- Saved model bundle at `models/decision_tree.pkl`.

## Dataset and four-class target

The 200-record source is `data/Student_Performance.csv`. The source field `Midterm_Marks` is renamed to **Internal Marks** throughout the app; no records are needlessly discarded. Loading validates required columns, numeric types, ranges, missing values, and duplicates.

The source only contains a Pass/Fail result. Therefore the four labels are transparent **project-defined proxy categories**, not supplied ground truth. `src.preprocessing.add_performance_target` calculates this index:

`0.25 × internal + 0.20 × assignment + 0.15 × attendance + 0.15 × min(study hours, 20)/20×100 + 0.25 × GPA-normalised + historical outcome adjustment`

The historical Pass/Fail value provides only a small adjustment (+4 / -6) and is never a model feature. Fixed bands are Poor `<45`, Average `45–<60`, Good `60–<75`, Excellent `≥75`. This is not a renaming of Pass/Fail; it is reproducible and retains a modest historical signal while requiring the classifier to learn from the five available student inputs. These proxy categories should be validated against institutional outcomes before any real use.

## Model, preprocessing, and evaluation

`sklearn.tree.DecisionTreeClassifier` is the only prediction model. Training uses a 75/25 stratified held-out split (`random_state=42`), `max_depth=5`, `min_samples_leaf=2`, and balanced class weights. An unrestricted tree is included only as a small overfitting comparison. Scaling is not used because Decision Trees split using thresholds and do not need feature scaling.

Training dynamically prints the final class distribution and held-out accuracy, weighted precision, recall, F1-score, classification report, confusion matrix, and feature importances. Weighted multiclass averages are used because classes are not exactly equally sized. No metric values are hard-coded.

## Recommendation system

`src/recommendations.py` is conceptually separate from the classifier. It checks actual inputs against clear guidance thresholds: attendance 75%, assignment/internal marks 60, study hours 8/week, and GPA 2.8/4. It returns every applicable weak area and suggestion, without random content, APIs, or guarantees.

## Project structure

```text
├── app.py
├── data/Student_Performance.csv
├── models/decision_tree.pkl
├── notebooks/CIT_23_02_0162_Lab04_DecisionTree_.ipynb
├── src/ (preprocessing, train, evaluate, predict, recommendations)
├── requirements.txt
└── README.md
```

## Installation and use

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Open the matching experiment notebook with:

```powershell
python -m jupyter notebook notebooks/CIT_23_02_0162_Lab04_DecisionTree_.ipynb
```

## Limitations

The dataset is small and the labels are project-defined proxies, so this is demonstration decision support, not a sole basis for high-stakes student decisions.

## Team contribution

| Member | Student ID | Contribution |
| --- | --- | --- |
| Team member 1 |  |  |
| Team member 2 |  |  |
| Team member 3 |  |  |
| Team member 4 |  |  |
