# train_kidney_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# ---------- CONFIG ----------
DATA_PATH = Path("/users/shreyash/Desktop/hj/Chronic-Kidney-Disease-Prediction-main/data/kidney.csv")   # <-- put your CSV hereDesktop/hj/Chronic-Kidney-Disease-Prediction-main/data/kidney_disease-Copy1.csv
MODEL_OUT = Path("models/kidney.pkl")
RANDOM_STATE = 42
# ----------------------------

if not DATA_PATH.exists():
    raise SystemExit(f"Dataset not found at {DATA_PATH}. Put your CSV there or edit DATA_PATH in the script.")

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["id"])

# Basic inspection (prints)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ---------- Assumptions about dataset ----------
# The old CKD datasets commonly use a target column named 'class' or 'target' or 'label'
# If your target column has a different name, change TARGET_COL below.
TARGET_COL = "classification"

print("Using target column:", TARGET_COL)

# Drop rows that are completely empty (if any)
df = df.dropna(how="all")

# ---------- Preprocessing ----------
# Identify numeric vs categorical features
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Remove target from lists if present
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
if TARGET_COL in cat_cols:
    cat_cols.remove(TARGET_COL)

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# Simple strategy: impute numeric with median, categorical with most frequent
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
], remainder="drop", sparse_threshold=0)

# ---------- Create pipeline with classifier ----------
clf = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
])

# Prepare X, y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].copy()

# If target is strings like 'ckd'/'notckd', convert to 0/1
if y.dtype == "object" or y.dtype.name == "category":
    y = y.astype(str).str.strip().str.lower()
    mapping = {val: i for i, val in enumerate(sorted(y.unique()))}
    print("Target mapping:", mapping)
    y = y.map(mapping)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Fit
print("Training model...")
clf.fit(X_train, y_train)

# Evaluate
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy:  {test_score:.4f}")
print(f"CV scores:      {cv_scores}")
print(f"CV mean:        {cv_scores.mean():.4f}")

# Save pipeline (preprocessing + model)
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODEL_OUT)
print("Saved model to:", MODEL_OUT)
