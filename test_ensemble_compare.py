"""
test_ensemble_compare.py - Compare Baseline vs Ensemble with FakeNewsNet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import sys
sys.path.append(".")

from model.ensemble import EnsembleClassifier

# Load raw data
print("📥 Loading FakeNewsNet data...")
df = pd.read_csv("data/raw/fakenewsnet_clean.csv")
print(f"Total samples: {len(df)}")

# Sample 10%
sample_size = int(len(df) * 0.10)
df_sample = df.sample(n=sample_size, random_state=42)
print(f"Using 10% sample: {len(df_sample)} samples")

# Prepare data
X = df_sample['text'].fillna("")
y = df_sample['label']

# Split: 80% train, 20% val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ================ METHOD 1: Direct TF-IDF LogReg (Baseline) ================
print("\n" + "="*60)
print("🚀 METHOD 1: Baseline TF-IDF + LogisticRegression (Direct)")
print("="*60)

tfidf1 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_val_tfidf1 = tfidf1.transform(X_val)

clf1 = LogisticRegression(max_iter=1000, class_weight='balanced')
clf1.fit(X_train_tfidf1, y_train)

y_pred1 = clf1.predict(X_val_tfidf1)
y_proba1 = clf1.predict_proba(X_val_tfidf1)

# Calculate metrics
f1_baseline = f1_score(y_val, y_pred1, pos_label='fake')
acc_baseline = accuracy_score(y_val, y_pred1)
y_val_binary = (y_val == 'fake').astype(int)
y_proba_fake1 = y_proba1[:, list(clf1.classes_).index('fake')]
auc_baseline = roc_auc_score(y_val_binary, y_proba_fake1)

print(f"✅ Baseline Results:")
print(f"   - Accuracy: {acc_baseline:.4f}")
print(f"   - F1-score (fake): {f1_baseline:.4f}")
print(f"   - AUC: {auc_baseline:.4f}")

# ================ METHOD 2: Ensemble with Baseline Only ================
print("\n" + "="*60)
print("🚀 METHOD 2: EnsembleClassifier (Baseline only)")
print("="*60)

# Create a temporary joblib file for baseline model
import joblib
import os
os.makedirs("saved_models", exist_ok=True)
baseline_path = "saved_models/baseline_logreg.joblib"

# Create proper pipeline for ensemble
pipeline = Pipeline([
    ('tfidf', tfidf1),
    ('clf', clf1)
])
joblib.dump(pipeline, baseline_path)

# Load ensemble with baseline only
ensemble = EnsembleClassifier(
    baseline_model_path=baseline_path,
    weights={"baseline": 1.0}
)

# Predict using ensemble
y_pred2 = []
y_proba2 = []
for text in X_val:
    label, conf, prob_dict = ensemble.predict(text)
    y_pred2.append(label)
    y_proba2.append(prob_dict['fake'])

y_pred2 = np.array(y_pred2)
y_proba2 = np.array(y_proba2)

# Calculate metrics
f1_ensemble = f1_score(y_val, y_pred2, pos_label='fake')
acc_ensemble = accuracy_score(y_val, y_pred2)
auc_ensemble = roc_auc_score(y_val_binary, y_proba2)

print(f"✅ Ensemble (Baseline) Results:")
print(f"   - Accuracy: {acc_ensemble:.4f}")
print(f"   - F1-score (fake): {f1_ensemble:.4f}")
print(f"   - AUC: {auc_ensemble:.4f}")

# ================ COMPARISON ================
print("\n" + "="*60)
print("📊 COMPARISON: Baseline vs Ensemble")
print("="*60)
print(f"{'Metric':<20} {'Baseline':<15} {'Ensemble':<15} {'Diff':<10}")
print("-" * 60)
print(f"{'Accuracy':<20} {acc_baseline:<15.4f} {acc_ensemble:<15.4f} {acc_ensemble-acc_baseline:+.4f}")
print(f"{'F1 (fake)':<20} {f1_baseline:<15.4f} {f1_ensemble:<15.4f} {f1_ensemble-f1_baseline:+.4f}")
print(f"{'AUC':<20} {auc_baseline:<15.4f} {auc_ensemble:<15.4f} {auc_ensemble-auc_baseline:+.4f}")

print("\n📝 Note: Ensemble with baseline only = Same results as direct baseline")
print("   Ensemble will be better when LoRA model is added!")

# Classification reports
print("\n" + "="*60)
print("📋 Detailed Classification Reports")
print("="*60)

print("\n--- Baseline ---")
print(classification_report(y_val, y_pred1))

print("\n--- Ensemble ---")
print(classification_report(y_val, y_pred2))
