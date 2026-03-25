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

from config import MODELS_ARTIFACTS_DIR, TARGET_BASELINE_F1, LABELS

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

        # pos_label=1 vì 1 = fake (positive class)
        f1 = f1_score(y, y_pred, pos_label=1, average='binary')
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

def train(max_features=5000):
    """Train TF-IDF + Logistic Regression baseline model.

    Args:
        max_features: Max TF-IDF features (default 5000)

    Returns:
        BaselineLogReg: Trained model
    """
    from data.dataset import FakeNewsDataset

    print("Loading data...")
    train_dataset = FakeNewsDataset(split="train")
    val_dataset = FakeNewsDataset(split="val")

    if len(train_dataset) == 0:
        raise RuntimeError("No training data found. Run create_mock_data.py first.")

    train_df = train_dataset.df
    val_df = val_dataset.df

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    model = BaselineLogReg(max_features=max_features)
    model.train(train_df)
    f1 = model.evaluate(val_df)

    model.save()
    print(f"\n{'='*50}")
    print(f"Baseline F1: {f1:.4f} | Target: {TARGET_BASELINE_F1}")
    print(f"{'='*50}")

    return model


if __name__ == "__main__":
    train()
