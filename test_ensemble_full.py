"""
test_ensemble_full.py - Full test with Baseline + XLM-R + Ensemble
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

# Load data
print("\n📥 Loading FakeNewsNet data...")
df = pd.read_csv("data/raw/fakenewsnet_clean.csv")
sample_size = int(len(df) * 0.10)
df_sample = df.sample(n=sample_size, random_state=42)
print(f"Using 10% sample: {len(df_sample)} samples")

X = df_sample['text'].fillna("").tolist()
y = df_sample['label'].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ================ MODEL 1: Baseline TF-IDF ================
print("\n" + "="*60)
print("🚀 MODEL 1: Baseline TF-IDF + LogisticRegression")
print("="*60)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

y_pred_baseline = clf.predict(X_val_tfidf)
y_proba_baseline = clf.predict_proba(X_val_tfidf)

acc_baseline = accuracy_score(y_val, y_pred_baseline)
f1_baseline = f1_score(y_val, y_pred_baseline, pos_label='fake')
y_proba_fake_baseline = y_proba_baseline[:, list(clf.classes_).index('fake')]
auc_baseline = roc_auc_score((np.array(y_val) == 'fake').astype(int), y_proba_fake_baseline)

print(f"✅ Baseline Results:")
print(f"   Accuracy: {acc_baseline:.4f} | F1: {f1_baseline:.4f} | AUC: {auc_baseline:.4f}")

# ================ MODEL 2: XLM-RoBERTa ================
print("\n" + "="*60)
print("🚀 MODEL 2: XLM-RoBERTa-base (Fine-tuned)")
print("="*60)

# Load XLM-RoBERTa
model_name = "xlm-roberta-base"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
xlm_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True
)
xlm_model.to(device)
xlm_model.train()  # Training mode for few epochs

# Prepare data for XLM-R
print("Tokenizing data for XLM-R...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=256, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=256, return_tensors='pt')

# Convert labels
label_map = {'real': 0, 'fake': 1}
train_labels = torch.tensor([label_map[l] for l in y_train])
val_labels = torch.tensor([label_map[l] for l in y_val])

# Quick fine-tune (2 epochs for demo)
print("Fine-tuning XLM-R (2 epochs on 10% sample)...")
optimizer = torch.optim.AdamW(xlm_model.parameters(), lr=2e-5)

xlm_model.train()
batch_size = 16
for epoch in range(2):
    total_loss = 0
    for i in range(0, len(train_encodings['input_ids']), batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in train_encodings.items()}
        labels = train_labels[i:i+batch_size].to(device)
        
        outputs = xlm_model(**batch, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/2 - Loss: {total_loss:.4f}")

# Evaluate XLM-R
print("Evaluating XLM-R on validation set...")
xlm_model.eval()
val_preds = []
val_probas = []

with torch.no_grad():
    for i in range(0, len(val_encodings['input_ids']), batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in val_encodings.items()}
        outputs = xlm_model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(outputs.logits, dim=-1)
        val_preds.extend(preds.cpu().numpy())
        val_probas.extend(probs[:, 1].cpu().numpy())  # P(fake)

y_pred_xlm = ['fake' if p == 1 else 'real' for p in val_preds]
y_proba_fake_xlm = np.array(val_probas)

acc_xlm = accuracy_score(y_val, y_pred_xlm)
f1_xlm = f1_score(y_val, y_pred_xlm, pos_label='fake')
auc_xlm = roc_auc_score((np.array(y_val) == 'fake').astype(int), y_proba_fake_xlm)

print(f"✅ XLM-R Results:")
print(f"   Accuracy: {acc_xlm:.4f} | F1: {f1_xlm:.4f} | AUC: {auc_xlm:.4f}")

# ================ ENSEMBLE: Baseline + XLM-R ================
print("\n" + "="*60)
print("🔄 ENSEMBLE: Baseline + XLM-R (Weighted Average)")
print("="*60)

def ensemble_predict(p1, p2, w1):
    """Combine predictions with weights."""
    w2 = 1 - w1
    return w1 * p1 + w2 * p2

# Test different weights
weight_configs = [
    (1.0, 0.0, "Baseline Only"),
    (0.0, 1.0, "XLM-R Only"),
    (0.5, 0.5, "50/50"),
    (0.3, 0.7, "30% Base + 70% XLM"),
    (0.7, 0.3, "70% Base + 30% XLM"),
]

y_val_binary = (np.array(y_val) == 'fake').astype(int)

print(f"\n{'Config':<20} | {'Acc':>8} | {'F1':>8} | {'AUC':>8} | vs Best")
print("-" * 65)

best_f1 = 0
best_config = ""

for w1, w2, name in weight_configs:
    ens_proba = ensemble_predict(y_proba_fake_baseline, y_proba_fake_xlm, w1)
    ens_pred = ['fake' if p > 0.5 else 'real' for p in ens_proba]
    
    acc_ens = accuracy_score(y_val, ens_pred)
    f1_ens = f1_score(y_val, ens_pred, pos_label='fake')
    auc_ens = roc_auc_score(y_val_binary, ens_proba)
    
    marker = ""
    if f1_ens > best_f1:
        marker = " ⭐ BEST"
        best_f1 = f1_ens
        best_config = name
    
    print(f"{name:<20} | {acc_ens:>8.4f} | {f1_ens:>8.4f} | {auc_ens:>8.4f} |{marker}")

# ================ SUMMARY ================
print("\n" + "="*60)
print("📊 FINAL COMPARISON")
print("="*60)

print(f"""
┌─────────────────┬──────────┬──────────┬──────────┐
│ Model           │ Accuracy │   F1     │   AUC    │
├─────────────────┼──────────┼──────────┼──────────┤
│ Baseline TF-IDF │  {acc_baseline:.4f}   │  {f1_baseline:.4f}   │  {auc_baseline:.4f}   │
│ XLM-RoBERTa     │  {acc_xlm:.4f}   │  {f1_xlm:.4f}   │  {auc_xlm:.4f}   │
│ Best Ensemble   │    -     │  {best_f1:.4f}   │    -     │
└─────────────────┴──────────┴──────────┴──────────┘

💡 KEY INSIGHTS:
1. XLM-RoBERTa leverages semantic understanding vs TF-IDF's word frequency
2. Ensemble combines different error patterns
3. Best ensemble config: {best_config}

🎯 ENSEMBLE works when combining models with DIVERSE features!
""")
