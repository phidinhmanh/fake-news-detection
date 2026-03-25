"""
test_ensemble.py — Test and evaluate ensemble model
===================================================
Test script for IMPL-B-005.

Usage:
    python model/test_ensemble.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.ensemble import EnsembleClassifier, load_ensemble
from config import MODELS_ARTIFACTS_DIR, TARGET_ENSEMBLE_AUC


def test_ensemble_basic():
    """Test basic ensemble functionality."""
    print("=" * 60)
    print("Test 1: Basic Ensemble Initialization")
    print("=" * 60)

    # Test with baseline model only (since LoRA may not be trained yet)
    baseline_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"

    if not baseline_path.exists():
        print(f"Warning: Baseline model not found at {baseline_path}")
        print("Please run IMPL-B-002 first to train the baseline model.")
        return None

    ensemble = EnsembleClassifier(
        baseline_model_path=baseline_path,
        weights={"baseline": 1.0}
    )

    # Test prediction
    test_texts = [
        "Vaccine gây ra bệnh tự kỷ, đây là sự thật mà chính phủ đang che giấu!",
        "Nghiên cứu khoa học mới cho thấy vaccine an toàn và hiệu quả.",
    ]

    print("\nRunning predictions:")
    for i, text in enumerate(test_texts, 1):
        label, confidence, probs = ensemble.predict(text, lang="vi")
        print(f"\nText {i}: {text[:60]}...")
        print(f"  Prediction: {label} (confidence: {confidence:.4f})")
        print(f"  Probabilities: {probs}")

    return ensemble


def test_ensemble_with_weights():
    """Test ensemble with different weight configurations."""
    print("\n" + "=" * 60)
    print("Test 2: Ensemble with Different Weights")
    print("=" * 60)

    baseline_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"

    if not baseline_path.exists():
        print("Skipping: Baseline model not found")
        return

    # Test different configurations
    configs = [
        {"baseline": 1.0},
        {"lora": 0.7, "baseline": 0.3},
        {"lora": 0.5, "baseline": 0.5},
    ]

    test_text = "Chính phủ đang che giấu sự thật về vaccine!"

    for i, weights in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {weights}")
        try:
            ensemble = EnsembleClassifier(
                baseline_model_path=baseline_path,
                weights=weights
            )
            label, confidence, probs = ensemble.predict(test_text, lang="vi")
            print(f"  Prediction: {label} (confidence: {confidence:.4f})")
        except Exception as e:
            print(f"  Error: {e}")


def test_weight_optimization():
    """Test weight optimization on validation set."""
    print("\n" + "=" * 60)
    print("Test 3: Weight Optimization (Mock)")
    print("=" * 60)

    baseline_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"

    if not baseline_path.exists():
        print("Skipping: Baseline model not found")
        return

    # Create mock validation data
    val_texts = [
        "Vaccine gây tự kỷ, chính phủ che giấu sự thật!",
        "Nghiên cứu khoa học chứng minh vaccine an toàn.",
        "Tin nóng: Vaccine có chứa chip theo dõi!",
        "WHO khuyến cáo tiêm vaccine đầy đủ.",
    ]
    val_labels = ["fake", "real", "fake", "real"]

    ensemble = EnsembleClassifier(
        baseline_model_path=baseline_path,
        weights={"baseline": 1.0}
    )

    # Note: This will only optimize over baseline since LoRA is not loaded
    print("\nOptimizing weights on validation set...")
    print("(Note: Only baseline model is used in this test)")

    try:
        optimal_weights = ensemble.optimize_weights(
            val_texts, val_labels, metric="accuracy"
        )
        print(f"\nOptimal weights: {optimal_weights}")
    except Exception as e:
        print(f"Error during optimization: {e}")


def test_load_ensemble_presets():
    """Test loading pre-configured ensemble presets."""
    print("\n" + "=" * 60)
    print("Test 4: Pre-configured Ensemble Presets")
    print("=" * 60)

    presets = ["baseline_only", "default", "lora_only"]

    for preset in presets:
        print(f"\nLoading preset: {preset}")
        try:
            ensemble = load_ensemble(config=preset, device="cpu")
            print(f"  ✓ Successfully loaded {preset} configuration")
        except Exception as e:
            print(f"  ✗ Error loading {preset}: {e}")


def main():
    """Run all ensemble tests."""
    print("Testing Ensemble Meta-learner (IMPL-B-005)")
    print("=" * 60)

    # Run tests
    ensemble = test_ensemble_basic()

    if ensemble is not None:
        test_ensemble_with_weights()
        test_weight_optimization()
        test_load_ensemble_presets()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
    print(f"\nTarget AUC: {TARGET_ENSEMBLE_AUC}")
    print("\nNext steps:")
    print("1. Train LoRA model (IMPL-B-004)")
    print("2. Optimize ensemble weights on validation set")
    print("3. Evaluate ensemble AUC on test set")
    print("4. Fine-tune weights to achieve target AUC >= 0.90")


if __name__ == "__main__":
    main()
