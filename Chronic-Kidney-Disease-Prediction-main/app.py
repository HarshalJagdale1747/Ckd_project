'''from flask import Flask, render_template, request
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("models/kidney.pkl")
if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {MODEL_PATH}. Train it first or copy kidney.pkl there.")

model = joblib.load(MODEL_PATH)   # pipeline: preprocessing + classifier

@app.route("/")
def home():
    return render_template("kidney.html")

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        form_data = request.form.to_dict()
        # Use the same column order as used during training:
        keys = sorted(form_data.keys())
        values = [float(form_data[k]) for k in keys]
        X = np.array(values).reshape(1, -1)
        pred = model.predict(X)[0]


        # probability if classifier supports it
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).max()
            proba = float(proba)

        return render_template("predict.html", pred=pred, proba=proba)
    except Exception as e:
        return render_template("kidney.html", message=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
'''
'''
from flask import Flask, render_template, request
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

# ---------------------------------------------
# Load trained model (pipeline: preprocessing + classifier)
# ---------------------------------------------
MODEL_PATH = Path("models/kidney.pkl")

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {MODEL_PATH}. Place kidney.pkl in models/ folder.")

model = joblib.load(MODEL_PATH)


# ---------------------------------------------
# FEATURE ORDER (MATCHES YOUR DATASET EXACTLY)
# ---------------------------------------------
FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
    "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm",
    "cad", "appet", "pe", "ane"
]
# (We exclude "id" and "classification")


# ---------------------------------------------
# ROUTES
# ---------------------------------------------
@app.route("/")
def home():
    return render_template("kidney.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # build input row using correct feature ordering
        values = []
        for col in FEATURES:
            if col not in form_data:
                return render_template("kidney.html", message=f"Missing field: {col}")
            values.append(form_data[col])   # KEEP RAW STRINGS — DO NOT FLOAT CONVERT

        X = np.array(values, dtype=object).reshape(1, -1)

        # prediction
        pred = model.predict(X)[0]

        # probability (optional)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X).max())

        return render_template("predict.html", pred=pred, proba=proba)

    except Exception as e:
        return render_template("kidney.html", message=f"Error: {e}")


# ---------------------------------------------
# RUN SERVER
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


'''

from flask import Flask, render_template, request
import numpy as np
import joblib
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# ---------------------------------------------
# Load trained model (pipeline: preprocessing + classifier)
# ---------------------------------------------
MODEL_PATH = Path("models/kidney.pkl")

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {MODEL_PATH}. Place kidney.pkl in models/ folder.")

model = joblib.load(MODEL_PATH)

# ---------------------------------------------
# FEATURE ORDER (MATCHES YOUR DATASET EXACTLY)
# ---------------------------------------------
FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
    "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm",
    "cad", "appet", "pe", "ane"
]
# Numeric columns in your dataset (so we coerce them to numeric)
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

# ---------------------------------------------
# ROUTES
# ---------------------------------------------
@app.route("/")
def home():
    return render_template("kidney.html")

'''
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # build input row using correct feature ordering
        row = {}
        for col in FEATURES:
            if col not in form_data:
                return render_template("kidney.html", message=f"Missing field: {col}")
            row[col] = form_data[col]  # keep raw input for now

        # Create a DataFrame with a single row. ColumnTransformer expects dataframe when using string column names.
        X_df = pd.DataFrame([row], columns=FEATURES)

        # Coerce numeric columns to numeric dtype (invalid parsing -> NaN), pipeline imputer will handle NaNs.
        for num_col in NUMERIC_COLS:
            if num_col in X_df.columns:
                X_df[num_col] = pd.to_numeric(X_df[num_col], errors="coerce")

        # Now predict using the pipeline (preprocessing + model)
        pred = model.predict(X_df)[0]

        # probability (optional)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_df).max())

        return render_template("predict.html", pred=pred, proba=proba)

    except Exception as e:
        # show a helpful error on the form page
        return render_template("kidney.html", message=f"Error: {e}")
'''
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # respect feature order
        row = {col: form_data[col] for col in FEATURES}

        # dataframe for pipeline
        X_df = pd.DataFrame([row], columns=FEATURES)

        # numeric conversion
        for col in NUMERIC_COLS:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        # prediction
        pred = int(model.predict(X_df)[0])

        # probability
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_df).max() * 100)

        # label mapping
        LABEL_MAP = {
            0: "Chronic Kidney Disease (CKD) Detected",
            1: "No Kidney Disease (Healthy)"
        }

        label_text = LABEL_MAP.get(pred, f"Class {pred}")

        return render_template(
            "predict.html",
            pred_label=label_text,
            pred_value=pred,
            proba=proba
        )

    except Exception as e:
        return render_template("kidney.html", message=f"Error: {e}")

# ---------------------------------------------
# RUN SERVER
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

