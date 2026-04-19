"""
train_cv_comparison.py - Cross-Validation Training: Baseline vs LoRA XLM-RoBERTa
================================================================================

Chạy: python train_cv_comparison.py
"""

import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Torch imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

from config import TFIDF_MAX_FEATURES

# ============== CONFIG ==============
DATA_PATH = "data/raw/fakenewsnet_clean.csv"
N_SPLITS = 5
EPOCHS_LORA = 3
BATCH_SIZE = 16
MAX_LENGTH = 256
MODEL_NAME = "xlm-roberta-base"

# ============== LOAD DATA ==============
print("="*70)
print("📥 Loading FakeNewsNet data...")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"Total samples: {len(df)}")

X = df['text'].fillna("").values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # fake=0, real=1

print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
print(f"Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ============== BASELINE CV ==============
print("\n" + "="*70)
print("🚀 MODEL 1: Baseline TF-IDF + LogisticRegression")
print(f"   {N_SPLITS}-Fold Cross-Validation")
print("="*70)

def train_baseline_cv(X, y, n_splits=5):
    """Train Baseline TF-IDF + LogReg với Cross-Validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'fold': [], 'accuracy': [], 'f1': [], 'precision': [], 
        'recall': [], 'auc': [], 'time': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_splits}...")
        start_time = time.time()
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, pos_label=0)
        prec = precision_score(y_val, y_pred, pos_label=0)
        rec = recall_score(y_val, y_pred, pos_label=0)
        auc = roc_auc_score(y_val, y_proba)
        elapsed = time.time() - start_time
        
        results['fold'].append(fold)
        results['accuracy'].append(acc)
        results['f1'].append(f1)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['auc'].append(auc)
        results['time'].append(elapsed)
        
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.1f}s")
    
    return pd.DataFrame(results)

baseline_results = train_baseline_cv(X, y_encoded, n_splits=N_SPLITS)

print("\n" + "="*70)
print("📊 BASELINE CV SUMMARY")
print("="*70)
for col in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
    mean = baseline_results[col].mean()
    std = baseline_results[col].std()
    print(f"  {col.capitalize():12s}: {mean:.4f} ± {std:.4f}")
print(f"  Total time: {baseline_results['time'].sum():.1f}s")

# ============== LORA CV ==============
print("\n" + "="*70)
print("🚀 MODEL 2: LoRA XLM-RoBERTa")
print(f"   {N_SPLITS}-Fold Cross-Validation, {EPOCHS_LORA} epochs")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def train_lora_cv(X, y, n_splits=5, epochs=3, batch_size=16, max_length=256):
    """Train LoRA XLM-RoBERTa với Cross-Validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'fold': [], 'accuracy': [], 'f1': [], 'precision': [], 
        'recall': [], 'auc': [], 'time': []
    }
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        start_time = time.time()
        
        X_train_text, X_val_text = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Tokenize
        train_encodings = tokenizer(
            X_train_text.tolist(), truncation=True, padding=True, 
            max_length=max_length, return_tensors='pt'
        )
        val_encodings = tokenizer(
            X_val_text.tolist(), truncation=True, padding=True, 
            max_length=max_length, return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(y_train)
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Load model & apply LoRA
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Training
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        all_preds, all_probas, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probas.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds, all_probas, all_labels = np.array(all_preds), np.array(all_probas), np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, pos_label=0)
        prec = precision_score(all_labels, all_preds, pos_label=0)
        rec = recall_score(all_labels, all_preds, pos_label=0)
        auc = roc_auc_score(all_labels, all_probas)
        elapsed = time.time() - start_time
        
        results['fold'].append(fold)
        results['accuracy'].append(acc)
        results['f1'].append(f1)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['auc'].append(auc)
        results['time'].append(elapsed)
        
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.1f}s")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

lora_results = train_lora_cv(X, y_encoded, n_splits=N_SPLITS, epochs=EPOCHS_LORA, batch_size=BATCH_SIZE)

print("\n" + "="*70)
print("📊 LORA CV SUMMARY")
print("="*70)
for col in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
    mean = lora_results[col].mean()
    std = lora_results[col].std()
    print(f"  {col.capitalize():12s}: {mean:.4f} ± {std:.4f}")
print(f"  Total time: {lora_results['time'].sum():.1f}s")

# ============== COMPARISON ==============
print("\n" + "="*70)
print("📊 FINAL COMPARISON")
print("="*70)

print(f"\n{'Metric':<15} | {'Baseline':<20} | {'LoRA XLM-R':<20}")
print("-" * 60)
for col in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
    base = f"{baseline_results[col].mean():.4f} ± {baseline_results[col].std():.4f}"
    lora = f"{lora_results[col].mean():.4f} ± {lora_results[col].std():.4f}"
    print(f"{col.capitalize():<15} | {base:<20} | {lora:<20}")

# Save results
print("\n" + "="*70)
print("💾 Saving results...")
print("="*70)

os.makedirs("results", exist_ok=True)
baseline_results.to_csv("results/baseline_cv_results.csv", index=False)
lora_results.to_csv("results/lora_cv_results.csv", index=False)

# Create comparison summary
summary = pd.DataFrame({
    'Metric': ['accuracy', 'f1', 'precision', 'recall', 'auc'],
    'Baseline_Mean': [baseline_results[c].mean() for c in ['accuracy', 'f1', 'precision', 'recall', 'auc']],
    'Baseline_Std': [baseline_results[c].std() for c in ['accuracy', 'f1', 'precision', 'recall', 'auc']],
    'LoRA_Mean': [lora_results[c].mean() for c in ['accuracy', 'f1', 'precision', 'recall', 'auc']],
    'LoRA_Std': [lora_results[c].std() for c in ['accuracy', 'f1', 'precision', 'recall', 'auc']],
})
summary['Diff'] = summary['LoRA_Mean'] - summary['Baseline_Mean']
summary.to_csv("results/cv_comparison_summary.csv", index=False)

print("✅ Results saved to results/")
print("\n🎉 Training Complete!")
