"""
baseline_logreg.py — TF-IDF + Logistic Regression Baseline Model
===============================================================
A simple, robust baseline for fake news detection.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

# Try to import from config, with safe defaults
try:
    from config import MODELS_ARTIFACTS_DIR, TARGET_BASELINE_F1, TFIDF_MAX_FEATURES
except ImportError:
    MODELS_ARTIFACTS_DIR = Path("saved_models")
    TARGET_BASELINE_F1 = 0.72
    TFIDF_MAX_FEATURES = 5000

class BaselineLogReg:
    """Baseline model using TF-IDF and Logistic Regression."""

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.model_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"

    def train(self, train_df: pd.DataFrame):
        """Trains the model on 'text' and 'label' columns."""
        X = train_df['text']
        y = train_df['label']
        self.pipeline.fit(X, y)
        print("Training complete.")

    def evaluate(self, val_df: pd.DataFrame):
        X = val_df['text']
        y = val_df['label']
        y_pred = self.pipeline.predict(X)
        f1 = f1_score(y, y_pred, pos_label='fake', average='binary')
        print(f"Validation F1-score: {f1:.4f}")
        print(classification_report(y, y_pred))
        return f1

    def save(self):
        MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict_with_score(self, text: str) -> dict:
        """Predict and return structured probabilities."""
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        pred_idx = proba.argmax()
        
        res = {
            "label": classes[pred_idx],
            "confidence": float(proba.max()),
            "fake_proba": 0.0,
            "real_proba": 0.0
        }
        
        for i, cls in enumerate(classes):
            if cls == 'fake': res["fake_proba"] = float(proba[i])
            if cls == 'real': res["real_proba"] = float(proba[i])
            
        return res

if __name__ == "__main__":
    # Minimal training script example
    from config import DATASET_PROCESSED_DIR
    data_path = DATASET_PROCESSED_DIR / "preprocessed_all.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Simple split
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df[:int(0.8*len(df))]
        val_df = df[int(0.8*len(df)):]
        
        model = BaselineLogReg()
        model.train(train_df)
        model.evaluate(val_df)
        model.save()
    else:
        print(f"Data not found at {data_path}. Run preprocessing first.")
