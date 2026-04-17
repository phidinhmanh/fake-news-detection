"""
ablation_study.py — Ablation Study: đo ảnh hưởng của từng component
======================================================================
Thí nghiệm bỏ từng component để đo mức giảm performance.

Experiments:
  1. Full system (PhoBERT + features + agents) → baseline score
  2. Remove stylistic features → delta
  3. Remove agent pipeline → delta
  4. Remove RAG evidence → delta

Chạy:
    python evaluation/ablation_study.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_PROCESSED_DIR, EVAL_RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ablation_study() -> list[dict]:
    """Run ablation study.

    Returns:
        List of experiment results.
    """
    experiments = []

    # Load test data
    test_path = DATASET_PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        logger.error(f"❌ Test data not found: {test_path}")
        return []

    test_df = pd.read_csv(test_path, encoding="utf-8")
    logger.info(f"📂 Test set: {len(test_df)} samples")

    # Experiment 1: Full system
    logger.info("\n📊 Experiment 1: Full system (PhoBERT + Features)")
    full_result = _evaluate_variant(test_df, "full")
    if full_result:
        experiments.append({
            "experiment": "Full System (PhoBERT + Features)",
            "description": "PhoBERT + 9 stylistic features",
            **full_result,
        })

    # Experiment 2: PhoBERT only (no features)
    logger.info("\n📊 Experiment 2: PhoBERT only (no features)")
    no_features_result = _evaluate_variant(test_df, "no_features")
    if no_features_result:
        experiments.append({
            "experiment": "PhoBERT Only (no features)",
            "description": "PhoBERT without stylistic features",
            **no_features_result,
        })

    # Experiment 3: TF-IDF only
    logger.info("\n📊 Experiment 3: TF-IDF Baseline")
    tfidf_result = _evaluate_variant(test_df, "tfidf")
    if tfidf_result:
        experiments.append({
            "experiment": "TF-IDF + LogReg",
            "description": "Traditional ML baseline",
            **tfidf_result,
        })

    # Compute deltas
    if len(experiments) >= 2 and "f1" in experiments[0]:
        baseline_f1 = experiments[0]["f1"]
        for exp in experiments[1:]:
            if "f1" in exp:
                exp["f1_delta"] = round(exp["f1"] - baseline_f1, 4)
                exp["f1_delta_pct"] = round((exp["f1"] - baseline_f1) / baseline_f1 * 100, 2)

    return experiments


def _evaluate_variant(test_df: pd.DataFrame, variant: str) -> dict | None:
    """Evaluate a specific variant.

    Args:
        test_df: Test DataFrame.
        variant: "full", "no_features", "tfidf".

    Returns:
        Metrics dict or None.
    """
    try:
        from evaluation.evaluate_models import (
            evaluate_phobert_features,
            evaluate_phobert_baseline,
            evaluate_tfidf_baseline,
        )

        if variant == "full":
            return evaluate_phobert_features(test_df)
        elif variant == "no_features":
            return evaluate_phobert_baseline(test_df)
        elif variant == "tfidf":
            return evaluate_tfidf_baseline(test_df)

    except Exception as exc:
        logger.warning(f"⚠️ Variant '{variant}' failed: {exc}")
        return {"error": str(exc)}


def generate_ablation_table(experiments: list[dict]) -> str:
    """Generate Markdown ablation table.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Experiment | F1-Score | Δ F1 | Δ F1 (%) | Accuracy | Description |",
        "|------------|---------|------|----------|----------|-------------|",
    ]

    for exp in experiments:
        if "error" in exp:
            lines.append(f"| {exp['experiment']} | ERROR | - | - | - | {exp.get('description', '')} |")
        else:
            f1 = exp.get("f1", 0)
            delta = exp.get("f1_delta", "-")
            delta_pct = exp.get("f1_delta_pct", "-")
            acc = exp.get("accuracy", 0)
            desc = exp.get("description", "")

            delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "-"
            delta_pct_str = f"{delta_pct:+.2f}%" if isinstance(delta_pct, (int, float)) else "-"

            lines.append(
                f"| {exp['experiment']} | {f1:.4f} | {delta_str} | {delta_pct_str} | {acc:.4f} | {desc} |"
            )

    return "\n".join(lines)


def main() -> None:
    """Run ablation study and save results."""
    logger.info("=" * 60)
    logger.info("🔬 ABLATION STUDY")
    logger.info("=" * 60)

    experiments = run_ablation_study()

    if not experiments:
        logger.warning("⚠️ No experiments completed.")
        return

    # Save results
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = EVAL_RESULTS_DIR / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    # Generate table
    table = generate_ablation_table(experiments)
    table_path = EVAL_RESULTS_DIR / "ablation_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(table)
        f.write("\n\n## Key Findings\n\n")
        f.write("- **Full System** = PhoBERT + stylistic features (proposed method)\n")
        f.write("- Removing features shows impact of stylistic analysis\n")
        f.write("- TF-IDF baseline shows minimum viable performance\n")

    logger.info(f"\n{table}")
    logger.info(f"\n💾 Results: {results_path}")
    logger.info(f"📊 Table: {table_path}")


if __name__ == "__main__":
    main()
