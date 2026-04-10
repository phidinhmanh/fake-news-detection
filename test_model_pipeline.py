"""Test script để kiểm tra model pipeline với mock data."""
import pandas as pd
from model.baseline_logreg import BaselineLogReg

# Load mock data
train_df = pd.read_parquet('data/datasets/normalized/train.parquet')
val_df = pd.read_parquet('data/datasets/normalized/val.parquet')

print('=== Train data ===')
print(train_df[['text', 'label_str']].head())
print(f'Train shape: {train_df.shape}')

print('\n=== Train Baseline ===')
model = BaselineLogReg()
model.train(train_df)

print('\n=== Evaluate ===')
f1 = model.evaluate(val_df)

print('\n=== Test predictions ===')
for i, row in val_df.head(3).iterrows():
    result = model.predict_with_score(row['text'])
    print(f'Text: {row["text"][:50]}...')
    print(f'Predicted: {result["label"]} (conf: {result["confidence"]:.2f})')
    print(f'  P(fake)={result["fake_proba"]:.2f}, P(real)={result["real_proba"]:.2f}')
    print()

print('=== Pipeline test PASSED ===')
