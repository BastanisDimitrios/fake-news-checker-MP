from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import basic_clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model_pipeline.joblib")
    p.add_argument("--fake", default="data/raw/Fake.csv")
    p.add_argument("--true", default="data/raw/True.csv")
    p.add_argument("--fig_out", default="reports/figures/confusion_matrix.png")
    p.add_argument("--metrics_out", default="reports/metrics.json")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pipe = joblib.load(args.model)

    fake = pd.read_csv(Path(args.fake))
    true = pd.read_csv(Path(args.true))
    fake["label"] = 0  # fake
    true["label"] = 1  # real

    df = pd.concat([fake, true], ignore_index=True)

    # preprocessing same as training
    df["text"] = df["text"].fillna("").map(basic_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ROC-AUC (IMPORTANT: use proba for class "1" = real)
    auc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        # column index for class=1
        if hasattr(pipe, "classes_"):
            classes = list(pipe.classes_)
            if 1 in classes:
                idx = classes.index(1)
            else:
                idx = 1  # fallback
        else:
            idx = 1
        y_score = proba[:, idx]
        auc = roc_auc_score(y_test, y_score)

    print("\n✅ Evaluation complete")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=["fake", "real"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["fake", "real"]
    )
    Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {args.fig_out}")

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(auc) if auc is not None else None,
        "test_size": args.test_size,
        "seed": args.seed,
        "model_path": str(args.model),
        "fake_path": str(args.fake),
        "true_path": str(args.true),
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()