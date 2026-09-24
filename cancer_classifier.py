"""
Breast Cancer Diagnosis Classifier
==================================
End-to-end ML project: load data -> explore -> compare models ->
tune the best one -> evaluate -> inspect feature importance -> save model.

Install:  pip install scikit-learn pandas matplotlib joblib
Run:      python cancer_classifier.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up on screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Load & explore the data
# ----------------------------------------------------------------------
def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
    print("=" * 60)
    print("1. DATA OVERVIEW")
    print("=" * 60)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print("Class balance:")
    print(y.map({0: "malignant", 1: "benign"}).value_counts(), "\n")
    print("Missing values:", int(X.isna().sum().sum()), "\n")
    return X, y, data.target_names


# ----------------------------------------------------------------------
# 2. Compare several models with cross-validation
# ----------------------------------------------------------------------
def compare_models(X_train, y_train):
    print("=" * 60)
    print("2. MODEL COMPARISON (5-fold CV, ROC-AUC)")
    print("=" * 60)

    # Scaling lives INSIDE the pipeline so each CV fold is scaled using only
    # its own training portion -> no data leakage.
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores
        print(f"{name:<22} AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best = max(results, key=lambda k: results[k].mean())
    print(f"\nBest baseline model: {best}\n")
    return models, best, cv


# ----------------------------------------------------------------------
# 3. Hyperparameter tuning
# ----------------------------------------------------------------------
def tune_model(pipeline, best_name, X_train, y_train, cv):
    print("=" * 60)
    print("3. HYPERPARAMETER TUNING")
    print("=" * 60)

    param_grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "SVM (RBF)": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
        "Random Forest": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    }

    search = GridSearchCV(
        pipeline, param_grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}\n")
    return search.best_estimator_


# ----------------------------------------------------------------------
# 4. Final evaluation on the held-out test set
# ----------------------------------------------------------------------
def evaluate(model, X_test, y_test, target_names):
    print("=" * 60)
    print("4. TEST SET EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=target_names, ax=axes[0], cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    plt.close()
    print("Saved evaluation.png")


# ----------------------------------------------------------------------
# 5. Feature importance (model-agnostic: permutation importance)
# ----------------------------------------------------------------------
def feature_importance(model, X_test, y_test):
    from sklearn.inspection import permutation_importance

    print("\n" + "=" * 60)
    print("5. FEATURE IMPORTANCE")
    print("=" * 60)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    imp = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print(imp.round(4).to_string())

    plt.figure(figsize=(8, 5))
    imp[::-1].plot(kind="barh", color="steelblue")
    plt.xlabel("Drop in AUC when feature is shuffled")
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")


# ----------------------------------------------------------------------
# 6. Save the model & show how to use it for prediction
# ----------------------------------------------------------------------
def save_and_predict(model, X_test, target_names):
    joblib.dump(model, "cancer_model.joblib")
    print("\nSaved cancer_model.joblib")

    loaded = joblib.load("cancer_model.joblib")
    sample = X_test.iloc[[0]]
    pred = loaded.predict(sample)[0]
    prob = loaded.predict_proba(sample)[0]
    print(f"Sample prediction: {target_names[pred]} "
          f"(P(malignant)={prob[0]:.3f}, P(benign)={prob[1]:.3f})")


def main():
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models, best_name, cv = compare_models(X_train, y_train)
    tuned = tune_model(models[best_name], best_name, X_train, y_train, cv)
    evaluate(tuned, X_test, y_test, target_names)
    feature_importance(tuned, X_test, y_test)
    save_and_predict(tuned, X_test, target_names)


if __name__ == "__main__":
    main()
