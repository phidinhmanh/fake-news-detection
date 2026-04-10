"""
test_baseline_quick.py - Quick test of Baseline LogReg with 10% sample
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Load raw data
print("📥 Loading FakeNewsNet data...")
df = pd.read_csv("data/raw/fakenewsnet_clean.csv")
print(f"Total samples: {len(df)}")

# Sample 10%
sample_size = int(len(df) * 0.10)
df_sample = df.sample(n=sample_size, random_state=42)
print(f"Using 10% sample: {len(df_sample)} samples")

# Check class distribution
print(f"\nClass distribution in sample:")
print(df_sample['label'].value_counts())

# Prepare data
X = df_sample['text'].fillna("")
y = df_sample['label']

# Split: 80% train, 20% val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

# Create and train pipeline
print("\n🚀 Training Baseline LogReg (TF-IDF + LogisticRegression)...")
pipeline = [
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
]

# Manual pipeline training
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

# Predict
print("\n📊 Evaluating on validation set...")
y_pred = clf.predict(X_val_tfidf)

# Metrics
f1 = f1_score(y_val, y_pred, pos_label='fake')
acc = accuracy_score(y_val, y_pred)

print(f"\n✅ Results (10% sample):")
print(f"- Accuracy: {acc:.4f}")
print(f"- F1-score (fake): {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_val, y_pred))

# Test prediction example
print("\n🔍 Sample predictions:")
test_texts = [
    "Breaking: Scientists discover cure for cancer",
    "Government hides truth about aliens",
]
for text in test_texts:
    text_tfidf = tfidf.transform([text])
    pred = clf.predict(text_tfidf)[0]
    proba = clf.predict_proba(text_tfidf)[0]
    print(f"  Text: '{text[:50]}...'")
    print(f"  -> Predicted: {pred}, Confidence: {max(proba):.4f}\n")
