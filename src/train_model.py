from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from preprocessing import basic_clean

def load_and_merge(fake_csv: Path, true_csv: Path) -> pd.DataFrame:
    fake = pd.read_csv(fake_csv)
    true = pd.read_csv(true_csv)
    fake["label"] = 0  # fake
    true["label"] = 1  # real
    df = pd.concat([fake, true], ignore_index=True)
    # use 'text' column if available; fallback to 'title'
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the CSV files.")
    df["text"] = df["text"].fillna("").map(basic_clean)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fake", default="data/raw/Fake.csv")
    p.add_argument("--true", default="data/raw/True.csv")
    p.add_argument("--model_out", default="models/model_pipeline.joblib")
    args = p.parse_args()

    df = load_and_merge(Path(args.fake), Path(args.true))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(classification_report(y_test, preds, target_names=["fake", "real"]))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"Saved model pipeline to: {args.model_out}")

if __name__ == "__main__":
    main()
