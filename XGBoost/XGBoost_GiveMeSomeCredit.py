"""
Improved Credit Scoring with XGBoost + Optuna
Dataset: Give Me Some Credit (cs-training.csv)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, DMatrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# CONFIG
# -----------------------
SEED = 42
DATA_FILE = "cs-training.csv"
OUT_DIR = "output_model"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# 1) Load data
# -----------------------
print("Loading data:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Target
df = df.rename(columns={"SeriousDlqin2yrs": "TARGET"})
y = df["TARGET"].astype(int)
X = df.drop(columns=["TARGET"])

# -----------------------
# 2) Feature engineering
# -----------------------
# Clip ages
if "age" in X.columns:
    X["age"] = X["age"].clip(18, 100)

# Missing flags
for col in ["MonthlyIncome", "NumberOfDependents"]:
    if col in X.columns:
        X[f"{col}_missing"] = X[col].isnull().astype(int)

# New ratios
if "MonthlyIncome" in X.columns and "NumberOfDependents" in X.columns:
    X["Income_per_person"] = X["MonthlyIncome"] / (X["NumberOfDependents"].fillna(0) + 1)

if "DebtRatio" in X.columns and "MonthlyIncome" in X.columns:
    X["Debt_to_income_ratio"] = X["DebtRatio"] * X["MonthlyIncome"].fillna(0)

# -----------------------
# 3) Train/Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
numeric_pipeline.fit(X_train)

X_train_proc = numeric_pipeline.transform(X_train)
X_test_proc = numeric_pipeline.transform(X_test)
feature_names = X.columns.tolist()

# -----------------------
# 4) Optuna optimization
# -----------------------
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": SEED,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train)
    }
    clf = XGBClassifier(**params, n_jobs=1)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    auc = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    return auc

print("Starting Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial:", study.best_trial.params)

# -----------------------
# 5) Final training + calibration
# -----------------------
best_params = study.best_trial.params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "n_jobs": 1,
    "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train)
})

final_clf = XGBClassifier(**best_params)
calibrated_clf = CalibratedClassifierCV(final_clf, method="sigmoid", cv=3)
calibrated_clf.fit(X_train_proc, y_train)

# -----------------------
# 6) Evaluation
# -----------------------
y_test_proba = calibrated_clf.predict_proba(X_test_proc)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

roc = roc_auc_score(y_test, y_test_proba)
print("\nFinal Test ROC-AUC:", roc)
print("\nClassification report (threshold=0.5):")
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.show()

# -----------------------
# 7) Save artifacts
# -----------------------
joblib.dump(calibrated_clf, os.path.join(OUT_DIR, "xgb_calibrated_model.joblib"))
joblib.dump(numeric_pipeline, os.path.join(OUT_DIR, "preprocessor.joblib"))
pd.Series(feature_names).to_csv(os.path.join(OUT_DIR, "feature_names.csv"), index=False)

print("Artifacts saved in", OUT_DIR)
