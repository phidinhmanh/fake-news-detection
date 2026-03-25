"""
ENSEMBLE_README.md — Documentation for Ensemble Meta-learner
============================================================

Task: IMPL-B-005 - Develop Meta-learner for model ensemble
Status: COMPLETE
Target: AUC >= 0.90

## Overview

The ensemble meta-learner combines predictions from multiple models using a
weighted sum approach. This implementation provides:

1. **Weighted Sum Meta-learner**: Combines probability outputs from multiple models
2. **Flexible Model Integration**: Supports LoRA, baseline, and future model types
3. **Weight Optimization**: Automatic grid search for optimal ensemble weights
4. **Graceful Degradation**: Works with any subset of available models

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  EnsembleClassifier                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: text, lang                                      │
│     │                                                   │
│     ├──> LoRA Model (XLM-RoBERTa)                       │
│     │      └──> P_lora = [P(fake), P(real)]            │
│     │                                                   │
│     └──> Baseline Model (TF-IDF + LogReg)              │
│            └──> P_baseline = [P(fake), P(real)]        │
│                                                         │
│  Weighted Sum:                                          │
│    P_ensemble = w1 * P_lora + w2 * P_baseline          │
│                                                         │
│  Output: (label, confidence, probabilities)             │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Weighted Sum Meta-learner
- Combines probability outputs from multiple models
- Weights are normalized to sum to 1.0
- Default weights: LoRA=0.7, Baseline=0.3

### 2. Model Support
- **LoRA Model**: XLM-RoBERTa with LoRA fine-tuning
- **Baseline Model**: TF-IDF + Logistic Regression
- **Extensible**: Easy to add more models to the ensemble

### 3. Weight Optimization
- Grid search over weight combinations (0.0 to 1.0, step 0.1)
- Supports multiple metrics: accuracy, F1-score, AUC
- Validates on held-out validation set

### 4. Robust Design
- Handles missing models gracefully
- GPU/CPU device selection
- Proper error handling and logging

## Usage

### Basic Usage

```python
from model.ensemble import EnsembleClassifier

# Initialize ensemble
ensemble = EnsembleClassifier(
    lora_model_path="models/xlmr_lora",
    baseline_model_path="models/baseline_logreg.joblib",
    weights={"lora": 0.7, "baseline": 0.3}
)

# Make prediction
text = "Vaccine gây tự kỷ, chính phủ che giấu!"
label, confidence, probs = ensemble.predict(text, lang="vi")

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")
print(f"Probabilities: {probs}")
```

### Using Presets

```python
from model.ensemble import load_ensemble

# Load default configuration
ensemble = load_ensemble(config="default")

# Load baseline only
ensemble = load_ensemble(config="baseline_only")

# Load LoRA only
ensemble = load_ensemble(config="lora_only")
```

### Weight Optimization

```python
# Load validation data
from data.dataset import FakeNewsDataset

val_dataset = FakeNewsDataset(split="val")
val_texts = [val_dataset.df.iloc[i]["text"] for i in range(len(val_dataset))]
val_labels = [val_dataset.df.iloc[i]["label"] for i in range(len(val_dataset))]

# Optimize weights
optimal_weights = ensemble.optimize_weights(
    val_texts,
    val_labels,
    metric="auc"
)

print(f"Optimal weights: {optimal_weights}")
```

## Implementation Details

### Class: EnsembleClassifier

#### Constructor Parameters
- `lora_model_path`: Path to LoRA model directory
- `baseline_model_path`: Path to baseline .joblib file
- `weights`: Dictionary of model weights
- `device`: 'cuda' or 'cpu'

#### Key Methods

**predict(text, lang) -> (label, confidence, probabilities)**
- Runs ensemble prediction on input text
- Returns predicted label, confidence score, and probability dict

**set_weights(weights)**
- Updates ensemble weights
- Automatically normalizes to sum to 1.0

**optimize_weights(val_texts, val_labels, metric)**
- Optimizes weights via grid search
- Supports 'accuracy', 'f1', 'auc' metrics
- Returns optimal weight configuration

**Private Methods:**
- `_load_lora_model(path)`: Loads LoRA model
- `_load_baseline_model(path)`: Loads baseline model
- `_predict_lora(text)`: Gets LoRA probabilities
- `_predict_baseline(text)`: Gets baseline probabilities

### Function: load_ensemble(config, device)

Convenience function for loading pre-configured ensembles:
- `"default"`: LoRA + Baseline (0.7/0.3)
- `"baseline_only"`: Baseline only
- `"lora_only"`: LoRA only

## Performance Tuning

### Weight Selection Strategies

1. **Default (0.7/0.3)**:
   - LoRA is primary classifier
   - Baseline provides complementary signal

2. **Equal Weight (0.5/0.5)**:
   - Balanced ensemble
   - Use when both models have similar performance

3. **Optimized**:
   - Use `optimize_weights()` on validation set
   - Automatically find best weight combination

### Achieving Target AUC >= 0.90

Steps to achieve target:

1. **Train High-Quality Base Models**:
   - Ensure baseline F1 >= 0.72 (IMPL-B-002)
   - Ensure LoRA F1 >= 0.85 (IMPL-B-004)

2. **Weight Optimization**:
   - Run grid search on validation set
   - Use AUC as optimization metric
   - Test range: 0.0 to 1.0 in 0.1 steps

3. **Fine-tuning**:
   - If AUC < 0.90, adjust weight granularity
   - Consider adding more models to ensemble
   - Experiment with different base model checkpoints

4. **Validation**:
   - Cross-validate on multiple splits
   - Ensure no overfitting on validation set
   - Final evaluation on held-out test set

## Integration Points

### Dependencies (Upstream)
- **IMPL-B-001**: FakeNewsDataset class (completed)
- **IMPL-B-002**: Baseline LogReg model (completed)
- **IMPL-B-004**: LoRA training (pending)

### Downstream Integration
- **IMPL-B-006**: Predictor interface will use EnsembleClassifier
- **API Server**: Predictor will be called by FastAPI endpoints
- **UI**: Predictions displayed in Streamlit dashboard

## Testing

### Run Demo
```bash
python model/demo_ensemble.py
```

### Run Unit Tests (when models are available)
```bash
python model/test_ensemble.py
```

### Manual Testing
```python
from model.ensemble import EnsembleClassifier

# Test with baseline only
ensemble = EnsembleClassifier(
    baseline_model_path="models/baseline_logreg.joblib"
)

# Test prediction
label, conf, probs = ensemble.predict("Test text", lang="vi")
assert label in ["fake", "real"]
assert 0.0 <= conf <= 1.0
assert sum(probs.values()) - 1.0 < 1e-6  # Sum to 1
```

## Future Enhancements

Potential improvements for Phase 2:

1. **Stacking Meta-learner**:
   - Train a meta-classifier on model outputs
   - Use logistic regression or neural network

2. **Dynamic Weighting**:
   - Adjust weights based on input characteristics
   - Domain-specific weight configurations

3. **Confidence Calibration**:
   - Temperature scaling for better calibrated probabilities
   - Platt scaling for probability calibration

4. **Additional Models**:
   - Add rule-based heuristics scorer
   - Integrate domain-specific models
   - Include source credibility scores

## References

- **Task JSON**: .workflow/active/WFS-fake-news-detection/.task/IMPL-B-005.json
- **Context Package**: .workflow/active/WFS-fake-news-detection/.process/context-package.json
- **Config**: config.py, model/config.yaml
- **Schemas**: schemas.py

---

**Implemented**: 2026-03-24
**Developer**: Person B (Model Training)
**Task**: IMPL-B-005
"""
