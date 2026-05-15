from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import time
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
#  META-FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_meta_features(X, y):
    df_X = pd.DataFrame(X)
    class_counts = np.bincount(y.astype(int))

    mean_skewness = float(np.mean(np.abs(df_X.skew())))
    mean_variance = float(np.mean(np.var(X, axis=0)))

    # Correlation: mean of upper triangle (excluding diagonal)
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        mean_corr = float(np.mean(np.abs(upper_tri)))
    else:
        mean_corr = 0.0

    return {
        "n_samples":             int(X.shape[0]),
        "n_features":            int(X.shape[1]),
        "n_classes":             int(len(np.unique(y))),
        "class_balance":         round(float(min(class_counts) / max(class_counts)), 3),
        "mean_variance":         round(mean_variance, 4),
        "mean_skewness":         round(mean_skewness, 4),
        "feature_to_sample_ratio": round(float(X.shape[1] / X.shape[0]), 4),
        "mean_correlation":      round(mean_corr, 4),
    }


# ─────────────────────────────────────────────
#  ALGORITHM RUNNER
# ─────────────────────────────────────────────
def run_algorithms(X, y):
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    # Stratify only if every class has ≥ 2 samples
    do_stratify = all(c >= 2 for c in np.bincount(y.astype(int)))
    stratify = y if do_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    algorithms = {
        "Random Forest":      (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "SVM":                (SVC(kernel="rbf", probability=True, random_state=42),       True),
        "KNN":                (KNeighborsClassifier(n_neighbors=min(5, n_samples // 2)),   True),
        "Decision Tree":      (DecisionTreeClassifier(random_state=42),                    False),
        "Logistic Regression":(LogisticRegression(max_iter=1000, random_state=42),         True),
        "Gradient Boosting":  (GradientBoostingClassifier(random_state=42),                False),
        "Naive Bayes":        (GaussianNB(),                                               False),
    }

    avg_method = "weighted" if n_classes > 2 else "binary"
    results, feature_importances, confusion_matrices = [], None, {}

    for name, (model, use_scale) in algorithms.items():
        X_tr = X_train_sc if use_scale else X_train
        X_te = X_test_sc  if use_scale else X_test

        t0 = time.time()
        model.fit(X_tr, y_train)
        train_ms = round((time.time() - t0) * 1000, 2)

        t0 = time.time()
        y_pred = model.predict(X_te)
        pred_ms = round((time.time() - t0) * 1000, 2)

        accuracy  = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, average=avg_method, zero_division=0) * 100, 2)
        recall    = round(recall_score(y_test, y_pred, average=avg_method, zero_division=0) * 100, 2)
        f1        = round(f1_score(y_test, y_pred, average=avg_method, zero_division=0) * 100, 2)

        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()

        if name == "Random Forest":
            feature_importances = model.feature_importances_.tolist()

        results.append({
            "name":       name,
            "accuracy":   accuracy,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "train_time": train_ms,
            "pred_time":  pred_ms,
            "is_best":    False,
        })

    results.sort(key=lambda r: (r["f1"], r["accuracy"]), reverse=True)
    results[0]["is_best"] = True
    return results, feature_importances, confusion_matrices


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample/<name>")
def sample_dataset(name):
    """Serve built-in sklearn sample datasets as CSV."""
    loaders = {
        "iris":          load_iris,
        "wine":          load_wine,
        "breast_cancer": load_breast_cancer,
    }
    if name not in loaders:
        return jsonify({"error": "Unknown sample"}), 404
    dataset = loaders[name](as_frame=True)
    df = dataset.frame          # includes target column
    csv_data = df.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}.csv"})


@app.route("/columns", methods=["POST"])
def get_columns():
    """Return column names so the user can pick the target."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    try:
        df = pd.read_csv(file, nrows=5)
        return jsonify({"columns": list(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file       = request.files["file"]
    target_col = request.form.get("target_col", "").strip()

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        df = pd.read_csv(file)

        # Auto-detect target column
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        # Drop rows with NaN
        df = df.dropna()

        if df.shape[0] < 20:
            return jsonify({"error": "Dataset too small – need at least 20 rows"}), 400

        # Encode all object columns
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.drop(columns=[target_col]).values.astype(float)
        y = le.fit_transform(df[target_col].values)

        feature_names = list(df.drop(columns=[target_col]).columns)

        meta          = extract_meta_features(X, y)
        results, fi, cms = run_algorithms(X, y)

        preview = df.head(6).to_dict(orient="records")

        return jsonify({
            "success":            True,
            "meta_features":      meta,
            "results":            results,
            "feature_importances": fi,
            "feature_names":      feature_names,
            "confusion_matrices": cms,
            "target_col":         target_col,
            "dataset_preview":    preview,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
