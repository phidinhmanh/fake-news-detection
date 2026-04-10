"""
test_ensemble_difference.py - Demonstrate ensemble with multiple model variants
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load data
print("📥 Loading FakeNewsNet data...")
df = pd.read_csv("data/raw/fakenewsnet_clean.csv")
sample_size = int(len(df) * 0.10)
df_sample = df.sample(n=sample_size, random_state=42)

X = df_sample['text'].fillna("")
y = df_sample['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ================ Create 2 Different Model Variants ================

print("\n" + "="*60)
print("🔧 Creating 2 Different Model Variants for Ensemble")
print("="*60)

# Model 1: TF-IDF (5k features, ngram 1-2)
print("\n📌 Model 1: TF-IDF (5k features, ngram 1-2)")
tfidf1 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_val_tfidf1 = tfidf1.transform(X_val)
clf1 = LogisticRegression(max_iter=1000, class_weight='balanced')
clf1.fit(X_train_tfidf1, y_train)
y_proba1 = clf1.predict_proba(X_val_tfidf1)

# Model 2: TF-IDF (10k features, ngram 1-3)
print("📌 Model 2: TF-IDF (10k features, ngram 1-3)")
tfidf2 = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_tfidf2 = tfidf2.fit_transform(X_train)
X_val_tfidf2 = tfidf2.transform(X_val)
clf2 = LogisticRegression(max_iter=1000, C=0.5)  # Different regularization
clf2.fit(X_train_tfidf2, y_train)
y_proba2 = clf2.predict_proba(X_val_tfidf2)

# ================ Individual Model Results ================
y_val_binary = (y_val == 'fake').astype(int)
y_proba_fake1 = y_proba1[:, list(clf1.classes_).index('fake')]
y_proba_fake2 = y_proba2[:, list(clf2.classes_).index('fake')]

acc1 = accuracy_score(y_val, clf1.predict(X_val_tfidf1))
acc2 = accuracy_score(y_val, clf2.predict(X_val_tfidf2))
f1_1 = f1_score(y_val, clf1.predict(X_val_tfidf1), pos_label='fake')
f1_2 = f1_score(y_val, clf2.predict(X_val_tfidf2), pos_label='fake')
auc1 = roc_auc_score(y_val_binary, y_proba_fake1)
auc2 = roc_auc_score(y_val_binary, y_proba_fake2)

print("\n📊 Individual Model Results:")
print(f"Model 1 - Acc: {acc1:.4f} | F1: {f1_1:.4f} | AUC: {auc1:.4f}")
print(f"Model 2 - Acc: {acc2:.4f} | F1: {f1_2:.4f} | AUC: {auc2:.4f}")

# ================ Ensemble with Different Weights ================
print("\n" + "="*60)
print("🔄 Ensemble: Weighted Average of 2 Models")
print("="*60)

def ensemble_predict(proba1, proba2, w1):
    """Combine predictions with weights."""
    w2 = 1 - w1
    return w1 * proba1 + w2 * proba2

# Test different weight combinations
weight_configs = [
    (1.0, 0.0, "Model 1 Only"),
    (0.0, 1.0, "Model 2 Only"),
    (0.5, 0.5, "Equal Weights"),
    (0.7, 0.3, "70% M1 + 30% M2"),
    (0.3, 0.7, "30% M1 + 70% M2"),
]

print(f"\n{'Config':<20} | {'Acc':>8} | {'F1':>8} | {'AUC':>8} | vs Best")
print("-" * 65)

best_f1 = 0
best_config = ""
best_idx = -1

for i, (w1, w2, name) in enumerate(weight_configs):
    # Ensemble probabilities
    ens_proba_fake = ensemble_predict(y_proba_fake1, y_proba_fake2, w1)
    ens_pred = ['fake' if p > 0.5 else 'real' for p in ens_proba_fake]
    
    ens_acc = accuracy_score(y_val, ens_pred)
    ens_f1 = f1_score(y_val, ens_pred, pos_label='fake')
    ens_auc = roc_auc_score(y_val_binary, ens_proba_fake)
    
    marker = ""
    if ens_f1 > best_f1:
        marker = " ⭐ BEST"
        best_f1 = ens_f1
        best_config = name
        best_idx = i
    
    print(f"{name:<20} | {ens_acc:>8.4f} | {ens_f1:>8.4f} | {ens_auc:>8.4f} |{marker}")

# ================ Summary ================
print("\n" + "="*60)
print("📝 Key Insight")
print("="*60)
print(f"""
Ensemble CHỈ khác Baseline khi:
1. Có ÍT NHẤT 2 models khác nhau
2. Các models có LEAKING patterns khác nhau (để bù互补)

Trong test này:
- Model 1: TF-IDF 5k, ngram (1,2), balanced
- Model 2: TF-IDF 10k, ngram (1,3), C=0.5

Kết quả ensemble phụ thuộc vào:
- Chất lượng từng model thành phần
- Correlation giữa các models (càng ít correlation = ensemble càng tốt)
- Weight optimization

💡 Muốn ensemble tốt hơn → Cần thêm LoRA/XLM-R model với features khác biệt!
""")
