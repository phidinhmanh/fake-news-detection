"""
baseline_logreg.py — TF-IDF + Logistic Regression Baseline Model
===============================================================
Implemented for Task IMPL-B-002.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

from config import MODELS_ARTIFACTS_DIR, TARGET_BASELINE_F1

class BaselineLogReg:
    """Baseline model using TF-IDF and Logistic Regression."""

    def __init__(self, max_features=5000):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.model_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"

    def train(self, train_df: pd.DataFrame):
        """Trains the model on the provided dataframe."""
        X = train_df['text']
        y = train_df['label']
        self.pipeline.fit(X, y)
        print("Training complete.")

    def evaluate(self, val_df: pd.DataFrame):
        """Evaluates the model on the provided validation dataframe."""
        X = val_df['text']
        y = val_df['label']
        y_pred = self.pipeline.predict(X)

        # Mapping string labels to binary for F1 calculation if needed
        # Assuming LABELS[0] is 'fake' (positive class)
        f1 = f1_score(y, y_pred, pos_label='fake', average='binary')
        report = classification_report(y, y_pred)

        print(f"Validation F1-score: {f1:.4f}")
        print("Classification Report:")
        print(report)

        return f1

    def save(self):
        """Saves the trained pipeline."""
        MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """Loads the trained pipeline."""
        self.pipeline = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict_with_score(self, text: str) -> dict:
        """
        Predict label and return confidence scores.

        Args:
            text: Raw input text.

        Returns:
            dict with keys:
              - label (str): predicted class ('fake' or 'real')
              - confidence (float 0-1): probability of predicted class
              - fake_proba (float): P(fake)
              - real_proba (float): P(real)
        """
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        pred_idx = proba.argmax()          # index of highest probability
        pred_label = classes[pred_idx]
        confidence = float(proba.max())

        # Map probabilities to class names
        if len(classes) == 2 and set(classes) == {'fake', 'real'}:
            fake_proba = float(proba[list(classes).index('fake')])
            real_proba = float(proba[list(classes).index('real')])
        else:
            fake_proba = float(proba[0]) if len(proba) >= 1 else 0.0
            real_proba = float(proba[-1]) if len(proba) >= 2 else 0.0

        return {
            "label": pred_label,
            "confidence": confidence,
            "fake_proba": fake_proba,
            "real_proba": real_proba,
        }

if __name__ == "__main__":
    # Example usage (requires data)
    import sys
    import os
    from data.datamodule import FakeNewsDataModule

    try:
        # Load datasets using our FakeNewsDataset class
        train_dataset = FakeNewsDataset(split="train")
        val_dataset = FakeNewsDataset(split="val")

        if len(train_dataset) == 0:
            print("No data found in normalized directory. Please run IMPL-B-001 first.")
            sys.exit(0)

        train_df = train_dataset.df
        val_df = val_dataset.df

        model = BaselineLogReg()
        model.train(train_df)
        f1 = model.evaluate(val_df)

        if f1 >= TARGET_BASELINE_F1:
            print(f"Success! F1-score {f1:.4f} >= Target {TARGET_BASELINE_F1}")
            model.save()
        else:
            print(f"Warning: F1-score {f1:.4f} < Target {TARGET_BASELINE_F1}")

    except Exception as e:
        print(f"Error: {e}")
