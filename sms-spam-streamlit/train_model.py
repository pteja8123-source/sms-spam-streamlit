"""Reproduce the Week 2 winning pipeline and save it for local inference."""

import csv
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    archive_path = data_dir / "sms-spam-collection.zip"
    if not archive_path.exists():
        print("Downloading the official UCI SMS dataset…", flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_path.write_bytes(response.read())
    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
        with archive.open(name) as source:
            df = pd.read_csv(source, sep="\t", header=None, names=["label", "message"],
                             encoding="utf-8", quoting=csv.QUOTE_NONE)
    raw_count = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.dropna(subset=["label", "message"]).drop_duplicates().reset_index(drop=True)
    assert set(df.label.unique()) == {"ham", "spam"}
    x_train, x_test, y_train, y_test = train_test_split(
        df.message.astype(str), df.label, test_size=0.20, stratify=df.label, random_state=42
    )
    assert set(x_train).isdisjoint(set(x_test))
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  max_features=10000, min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "spam_recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "spam_f1": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
    }
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "sms_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    restored = joblib.load(model_path)
    assert np.array_equal(restored.predict(x_test), predictions)
    metadata = {
        "model": "TF-IDF + balanced Logistic Regression", "random_state": 42,
        "raw_rows": raw_count, "duplicate_rows_removed": duplicate_count, "clean_rows": len(df),
        "training_rows": len(x_train), "test_rows": len(x_test),
        "test_metrics": metrics,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "dataset_url": DATA_URL, "dataset_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print("Saved and reloaded pipeline verified.", flush=True)


if __name__ == "__main__":
    main()
