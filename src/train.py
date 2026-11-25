"""Fetch, evaluate, and save a reproducible Cleveland heart-disease model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

if __package__:
    from .features import FEATURE_NAMES
else:
    from features import FEATURE_NAMES

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "heart_model.joblib"


def fetch_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the UCI Cleveland data and return selected features and a binary target."""
    heart_disease = fetch_ucirepo(id=45)
    if heart_disease.data is None:
        raise ValueError("UCI did not return feature and target data.")
    features = heart_disease.data.features
    targets = heart_disease.data.targets

    missing_features = [name for name in FEATURE_NAMES if name not in features.columns]
    if missing_features:
        raise ValueError(f"UCI dataset is missing expected features: {missing_features}")
    if "num" not in targets.columns:
        raise ValueError("UCI dataset is missing the expected 'num' target.")

    X = features.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(targets["num"], errors="coerce")
    if target.isna().any():
        raise ValueError("The UCI target contains missing or non-numeric values.")
    unexpected_values = set(target.unique()) - {0, 1, 2, 3, 4}
    if unexpected_values:
        raise ValueError(f"Unexpected values in UCI target 'num': {unexpected_values}")

    y = target.gt(0).astype(int).rename("has_heart_disease")
    return X, y


def build_models() -> dict[str, Pipeline]:
    """Create preprocessing-and-classifier pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    np.random.seed(RANDOM_SEED)
    X, y = fetch_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    models = build_models()
    evaluations = {}
    for model_name, pipeline in models.items():
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        cv_metrics = {
            metric: {
                "mean": float(np.mean(cv_result[f"test_{metric}"])),
                "std": float(np.std(cv_result[f"test_{metric}"], ddof=1)),
            }
            for metric in scoring
        }
        pipeline.fit(X_train, y_train)
        test_predictions = pipeline.predict(X_test)
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "precision": float(
                precision_score(y_test, test_predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        }
        evaluations[model_name] = {
            "cross_validation": cv_metrics,
            "test": test_metrics,
        }

    best_model_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["cross_validation"]["recall"]["mean"],
            evaluations[name]["cross_validation"]["f1"]["mean"],
        ),
    )
    best_pipeline = models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "feature_names": FEATURE_NAMES,
        "model_name": best_model_name,
        "random_seed": RANDOM_SEED,
        "metrics": evaluations[best_model_name],
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Selected model: {best_model_name}")
    print("Selection criteria: mean cross-validation recall, then mean F1.")
    print(f"All model evaluation metrics: {evaluations}")
    print(f"Saved model artifact to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
