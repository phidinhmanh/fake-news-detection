"""
baseline_logreg.py — TF-IDF + Logistic Regression Baseline
=============================================================
Người B phát triển (Tuần 1-2).
Reference point để so sánh với model transformer.

TODO:
    - [ ] TF-IDF vectorization (unigram + bigram)
    - [ ] Logistic Regression training
    - [ ] Export models/baseline_logreg.pkl
    - [ ] Target: F1 ≥ 0.72
    - [ ] Export metrics.json
"""

from __future__ import annotations

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline


# def create_baseline_pipeline():
#     """Create TF-IDF + LogReg pipeline."""
#     return Pipeline([
#         ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
#         ("clf", LogisticRegression(max_iter=1000, C=1.0)),
#     ])
