from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
MODEL_PATH = MODELS_DIR / "model_pipeline.joblib"
OUT_JSON = REPORTS_DIR / "final_metrics.json"


def main():
    if not FAKE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FAKE_PATH}")
    if not TRUE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRUE_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)

    X = df["text"].astype(str).fillna("")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = joblib.load(MODEL_PATH)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n✅ FINAL EVALUATION (Balanced Model)")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"\nSaved metrics to: {OUT_JSON}")


if __name__ == "__main__":
    main()