# Training Module Documentation

## Overview
This document describes the training pipeline implementation for the Fake News Detection model.

## File: `model/train.py`

### Purpose
Complete training pipeline with LoRA fine-tuning and Curriculum Learning support for achieving F1 ≥ 0.85.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Main Training Flow                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              1. Configuration Loading                │
│  - Load YAML config (model/config.yaml)            │
│  - Parse CLI arguments                              │
│  - Override settings                                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│            2. DataModule Initialization              │
│  - FakeNewsDataModule (from IMPL-B-001)            │
│  - Setup train/val/test splits                      │
│  - Tokenization handled by dataset                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│         3. Model Initialization (LoRA)               │
│  - Load XLM-RoBERTa base model                     │
│  - Apply LoRA adapters via PEFT                     │
│  - Configure optimizer with warmup                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────┴─────────────────┐
        │                                    │
        ▼                                    ▼
┌──────────────────┐            ┌──────────────────────┐
│ Standard Training│            │ Curriculum Learning   │
│                  │            │                      │
│ - All samples    │            │ - Easy → Hard        │
│ - Random shuffle │            │ - 50% → 100%         │
└──────────────────┘            │ - Dynamic difficulty │
                                └──────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              4. Training Loop                        │
│  - PyTorch Lightning Trainer                        │
│  - Model checkpointing (top-K by val_f1)           │
│  - Early stopping (patience=3)                      │
│  - Gradient clipping                                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              5. Evaluation & Saving                  │
│  - Test on held-out set                             │
│  - Save final model                                 │
│  - Save LoRA adapters                               │
└─────────────────────────────────────────────────────┘
```

## Key Components

### 1. FakeNewsClassifier (PyTorch Lightning Module)

**Responsibilities:**
- Wraps HuggingFace XLM-RoBERTa with LoRA
- Implements training/validation/test steps
- Computes and logs metrics (F1, Accuracy, Precision, Recall)
- Manages optimizer and scheduler

**Key Methods:**
- `forward()`: Forward pass through model
- `training_step()`: Training step with loss computation
- `validation_step()`: Validation step
- `test_step()`: Test step
- `configure_optimizers()`: Setup AdamW + warmup scheduler
- `on_*_epoch_end()`: Compute epoch-level metrics

**LoRA Configuration:**
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                    # Low-rank dimension
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Dropout
    target_modules=["query", "value"],  # Attention layers
)
```

### 2. CurriculumLearningScheduler

**Strategy:** Easy-to-Hard sample ordering

**Algorithm:**
1. **Difficulty Scoring**: Compute loss/confidence for each sample
2. **Sample Ranking**: Sort by difficulty (lower = easier)
3. **Curriculum Pacing**:
   - Epoch 0: 50% easiest samples
   - Linear increase to 100% by final epoch
4. **Dynamic Adaptation**: Recompute difficulties after each epoch

**Benefits:**
- Faster initial convergence
- Better generalization
- Reduced overfitting on hard samples

**Methods:**
- `compute_difficulties()`: Score all samples using model
- `get_curriculum_indices()`: Return sample indices for current epoch

### 3. Training Pipeline

**Function:** `train_with_curriculum()`

**Features:**
- Automatic device selection (GPU/CPU)
- Checkpointing with ModelCheckpoint callback
- Early stopping with EarlyStopping callback
- Gradient clipping (max_norm=1.0)
- TensorBoard logging integration

**Outputs:**
- Checkpoints: `models/checkpoints/fake-news-epoch=XX-val_f1=0.XXXX.ckpt`
- Final model: `models/final_model.pt`
- LoRA adapter: `models/lora_adapter/`

## Usage

### Basic Training (Standard)
```bash
python model/train.py --config model/config.yaml
```

**What it does:**
- Loads config from `model/config.yaml`
- Uses all training samples
- Standard random shuffling
- Trains for specified epochs

### Curriculum Learning (Recommended)
```bash
python model/train.py --config model/config.yaml --curriculum
```

**What it does:**
- Enables curriculum learning strategy
- Starts with 50% easiest samples
- Gradually increases to 100% by final epoch
- Recomputes sample difficulties after each epoch

### Custom Hyperparameters
```bash
python model/train.py \
    --config model/config.yaml \
    --curriculum \
    --batch-size 16 \
    --epochs 15
```

**What it does:**
- Overrides batch size to 16
- Overrides epochs to 15
- Other params from config.yaml

## Configuration File Structure

`model/config.yaml`:
```yaml
model:
  name: "xlm-roberta-base"      # HuggingFace model ID
  num_labels: 2                  # Binary classification

training:
  batch_size: 32
  learning_rate: 2.0e-5
  epochs: 10
  max_seq_length: 256
  warmup_ratio: 0.1              # 10% warmup steps
  weight_decay: 0.01             # AdamW weight decay

lora:
  r: 16                          # LoRA rank
  alpha: 32                      # LoRA alpha
  dropout: 0.1                   # LoRA dropout
  target_modules:                # Layers to apply LoRA
    - query
    - value

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

logging:
  log_every_n_steps: 50
  save_top_k: 3                  # Keep top 3 checkpoints
  monitor: "val_f1"              # Metric to monitor
  mode: "max"                    # Maximize val_f1

targets:
  baseline_f1: 0.72
  lora_f1: 0.85                  # Target for this task
  ensemble_auc: 0.90
  latency_seconds: 3.0
```

## Metrics Tracked

### Training Metrics
- `train_loss` (per step and per epoch)
- `train_acc` (per epoch)
- `train_f1` (per epoch)
- `train_precision` (per epoch)
- `train_recall` (per epoch)

### Validation Metrics
- `val_loss` (per epoch)
- `val_acc` (per epoch)
- `val_f1` (per epoch) ← **Used for checkpointing**
- `val_precision` (per epoch)
- `val_recall` (per epoch)

### Test Metrics
- `test_acc`
- `test_f1` ← **Target: ≥ 0.85**
- `test_precision`
- `test_recall`

## Output Artifacts

### 1. Checkpoints
Location: `models/checkpoints/`

Files:
- `fake-news-epoch=XX-val_f1=0.XXXX.ckpt` (top-K models)
- `last.ckpt` (latest checkpoint for resuming)

**Usage:**
```python
# Load checkpoint
model = FakeNewsClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### 2. Final Model
Location: `models/final_model.pt`

**Usage:**
```python
# Load state dict
model = FakeNewsClassifier(...)
model.load_state_dict(torch.load("models/final_model.pt"))
```

### 3. LoRA Adapter
Location: `models/lora_adapter/`

Files:
- `adapter_config.json`
- `adapter_model.bin`

**Usage:**
```python
# Load base model + adapter
from peft import PeftModel
base_model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
model = PeftModel.from_pretrained(base_model, "models/lora_adapter")
```

### 4. TensorBoard Logs
Location: `lightning_logs/version_X/`

**View with:**
```bash
tensorboard --logdir lightning_logs/
```

## Dependencies

### Internal Dependencies
- `config.py`: Paths and constants
- `schemas.py`: API contracts (indirectly via datamodule)
- `data/datamodule.py`: FakeNewsDataModule
- `data/dataset.py`: FakeNewsDataset

### External Dependencies
- `torch`: PyTorch framework
- `lightning`: PyTorch Lightning
- `transformers`: HuggingFace Transformers
- `peft`: Parameter-Efficient Fine-Tuning (LoRA)
- `scikit-learn`: Metrics computation
- `numpy`: Numerical operations
- `pyyaml`: Config loading

## Performance Considerations

### Memory Usage
- **LoRA**: Reduces trainable params by ~90%
- **Gradient Accumulation**: Can be added for larger effective batch size
- **Mixed Precision**: Compatible via `Trainer(precision="16-mixed")`

### Training Speed
- **Curriculum Learning**: ~10-15% overhead for difficulty computation
- **Multi-GPU**: Compatible via DDP (set `devices="auto"`)
- **Checkpointing**: May slow down first epoch

### Convergence
- **With Curriculum**: Typically 20-30% faster convergence
- **Without Curriculum**: May require more epochs
- **Early Stopping**: Prevents overfitting, saves time

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
python model/train.py --config model/config.yaml --batch-size 16

# Or use gradient accumulation in config.yaml
trainer:
  accumulate_grad_batches: 2
```

### Issue: Training Too Slow
**Solution:**
```bash
# Disable curriculum learning
python model/train.py --config model/config.yaml

# Or use smaller model
# Edit config.yaml: model.name = "distilbert-base-multilingual-cased"
```

### Issue: Poor F1 Score
**Solutions:**
1. Enable curriculum learning: `--curriculum`
2. Increase epochs: `--epochs 15`
3. Adjust learning rate in config.yaml
4. Check data quality (IMPL-B-001)
5. Tune LoRA hyperparameters (IMPL-B-003)

### Issue: Checkpoints Not Saving
**Check:**
1. Directory exists: `models/checkpoints/`
2. Sufficient disk space
3. Write permissions

## Integration with Other Tasks

### IMPL-B-001 (Dataset)
- **Dependency**: Requires normalized parquet files
- **Location**: `data/datasets/normalized/{train,val,test}.parquet`
- **Interface**: Via `FakeNewsDataModule`

### IMPL-B-003 (LoRA Config)
- **Dependency**: LoRA hyperparameters
- **Location**: `model/config.yaml` (lora section)
- **Tunable**: r, alpha, dropout, target_modules

### IMPL-B-005 (Ensemble)
- **Output**: Multiple checkpoints for ensemble
- **Selection**: Use top-K models by val_f1
- **Loading**: Via checkpoint paths

### IMPL-B-006 (Predictor)
- **Output**: Trained model for inference
- **Format**: State dict or LoRA adapter
- **Interface**: Compatible with predictor.py

## Testing Recommendations

### Unit Tests
```python
def test_model_initialization():
    model = FakeNewsClassifier(model_name="xlm-roberta-base")
    assert model is not None
    assert model.hparams.num_labels == 2

def test_curriculum_scheduler():
    scheduler = CurriculumLearningScheduler(dataset, num_epochs=10)
    scheduler.sample_difficulties = np.random.rand(100)
    indices = scheduler.get_curriculum_indices(epoch=0)
    assert len(indices) == 50  # 50% of 100
```

### Integration Tests
```python
def test_training_pipeline():
    config = load_config("model/config.yaml")
    config["training"]["epochs"] = 1  # Quick test
    datamodule = FakeNewsDataModule(batch_size=4)
    model, trainer = train_with_curriculum(config, datamodule, use_curriculum=False)
    assert (MODELS_ARTIFACTS_DIR / "final_model.pt").exists()
```

## Best Practices

1. **Always use curriculum learning** for initial training
2. **Monitor validation F1** as primary metric
3. **Save top-3 checkpoints** for ensemble
4. **Use early stopping** to prevent overfitting
5. **Log to TensorBoard** for monitoring
6. **Validate on held-out test set** only once
7. **Save LoRA adapters** for efficient deployment

## Future Enhancements

### Planned
- Multi-GPU distributed training
- Hyperparameter optimization (Optuna)
- Advanced curriculum strategies
- W&B/MLflow integration
- Automated learning rate finder

### Experimental
- Knowledge distillation
- Self-training on unlabeled data
- Domain adaptation techniques
- Few-shot learning capabilities

## Contact & Support

For issues or questions:
1. Check this documentation
2. Review task summary: `.summaries/IMPL-B-004-summary.md`
3. Check demo script: `model/demo_train.py`
4. Refer to TODO_LIST.md for dependencies

---

**Task**: IMPL-B-004
**Status**: ✅ Completed
**Date**: 2026-03-24
**Target**: F1 ≥ 0.85
