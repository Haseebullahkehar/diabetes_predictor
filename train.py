"""
Train the Intelligent Diabetes Risk Predictor (production training pipeline).

What this does:
  1. Load the Pima Indians Diabetes dataset.
  2. Treat physiologically-impossible zeros (Glucose, BloodPressure, SkinThickness,
     Insulin, BMI) as missing and impute with the training-set median (no leakage).
  3. Standardize features and hyperparameter-tune SEVEN algorithms with GridSearchCV,
     handling class imbalance via class_weight / scale_pos_weight.
  4. Select the best pipeline by cross-validated ROC-AUC.
  5. Optimize the decision threshold (maximize F1 on out-of-fold predictions) instead
     of a naive 0.5 -- important for a medical screener where recall matters.
  6. Evaluate on a held-out test set, compute permutation feature importance.
  7. Save model.joblib, metadata.json (used by the API) and metrics.json (report).

Run from the project root:  python train.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
META_PATH = BASE_DIR / "model" / "metadata.json"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_search_space():
    """Return {name: (pipeline, param_grid)} for every candidate algorithm."""
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    space = {
        "LogisticRegression": (
            pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"clf__C": [0.05, 0.1, 0.5, 1, 5],
             "clf__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            pipe(RandomForestClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [4, 6, 8, None],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__class_weight": [None, "balanced"]},
        ),
        "ExtraTrees": (
            pipe(ExtraTreesClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [6, 10, None],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__class_weight": [None, "balanced"]},
        ),
        "GradientBoosting": (
            pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [100, 250],
             "clf__learning_rate": [0.03, 0.1],
             "clf__max_depth": [2, 3]},
        ),
        "SVM_RBF": (
            pipe(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {"clf__C": [0.5, 1, 5, 10],
             "clf__gamma": ["scale", 0.05, 0.1],
             "clf__class_weight": [None, "balanced"]},
        ),
        "KNN": (
            pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [11, 15, 21, 31],
             "clf__weights": ["uniform", "distance"]},
        ),
    }
    if HAS_XGB:
        space["XGBoost"] = (
            pipe(XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist",
            )),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.03, 0.1],
             "clf__scale_pos_weight": [1, 1.9]},  # ~ n_neg/n_pos for class imbalance
        )
    return space


def best_threshold(y_true, proba):
    """Threshold that maximizes F1, clamped to a sensible screening range."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    # thr has len-1 vs prec/rec; align by dropping the last point
    best = thr[np.argmax(f1[:-1])] if len(thr) else 0.5
    return float(min(max(best, 0.15), 0.60))


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}"
          f"  |  XGBoost: {'yes' if HAS_XGB else 'no'}")

    X = df[FEATURES].astype(float).copy()
    y = df["Outcome"].astype(int)
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # --- tune every algorithm, keep the best by CV ROC-AUC ---------------
    print("\nTuning algorithms (5-fold CV, scoring = ROC-AUC):")
    leaderboard, best = [], None
    for name, (pipe, grid) in build_search_space().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        leaderboard.append((name, search.best_score_, search.best_params_, search.best_estimator_))
        print(f"  {name:<18} CV ROC-AUC = {search.best_score_:.4f}")

    leaderboard.sort(key=lambda r: r[1], reverse=True)
    best_name, best_cv, best_params, best_model = leaderboard[0]
    print(f"\n>> Best model: {best_name}  (CV ROC-AUC = {best_cv:.4f})")
    print(f"   Params: {best_params}")

    # --- optimize decision threshold on out-of-fold predictions ----------
    oof_proba = cross_val_predict(
        best_model, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = best_threshold(y_train.values, oof_proba)
    print(f"   Optimized decision threshold (max F1): {threshold:.3f}")

    # --- evaluate on the held-out test set -------------------------------
    proba_test = best_model.predict_proba(X_test)[:, 1]
    preds = (proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba_test), 4),
        "test_precision": round(precision_score(y_test, preds), 4),
        "test_recall": round(recall_score(y_test, preds), 4),
        "test_f1": round(f1_score(y_test, preds), 4),
        "cv_roc_auc": round(best_cv, 4),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }
    print("\nHeld-out test performance (at optimized threshold):")
    for k in ["test_accuracy", "test_roc_auc", "test_precision", "test_recall", "test_f1"]:
        print(f"   {k:<15} = {metrics[k]}")
    print("\n" + classification_report(y_test, preds, target_names=["No diabetes", "Diabetes"]))

    # --- feature importance (permutation, model-agnostic) ----------------
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    importances = sorted(
        ({"feature": f, "importance": round(float(m), 4)}
         for f, m in zip(FEATURES, perm.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )
    print("Feature importance (permutation, drop in ROC-AUC):")
    for row in importances:
        print(f"   {row['feature']:<26} {row['importance']}")

    # --- persist artifacts ------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "features": FEATURES,
        "zero_as_missing": ZERO_AS_MISSING,
        "medians": {k: float(v) for k, v in medians.items()},
        "decision_threshold": round(threshold, 4),
        "risk_bands": {"low": 0.30, "high": 0.60},
        "metrics": {k: metrics[k] for k in
                    ["cv_roc_auc", "test_accuracy", "test_roc_auc",
                     "test_precision", "test_recall", "test_f1"]},
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    report = {
        "chosen_model": best_name,
        "leaderboard": [{"model": n, "cv_roc_auc": round(s, 4)} for n, s, _, _ in leaderboard],
        "metrics": metrics,
        "decision_threshold": round(threshold, 4),
        "feature_importance": importances,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nSaved model    -> {MODEL_PATH}")
    print(f"Saved metadata -> {META_PATH}")
    print(f"Saved metrics  -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
