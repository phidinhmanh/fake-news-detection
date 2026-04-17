"""
confusion_matrix.py — Vẽ Confusion Matrix cho từng model
==========================================================
Chạy:
    python evaluation/confusion_matrix.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVAL_RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str,
    output_dir: Path,
    labels: list[str] | None = None,
) -> str | None:
    """Vẽ confusion matrix cho 1 model.

    Args:
        cm: Confusion matrix (2D list).
        model_name: Tên model.
        output_dir: Thư mục output.
        labels: Label names (default: ["Real", "Fake"]).

    Returns:
        Path to saved image or None.
    """
    if labels is None:
        labels = ["Real", "Fake"]

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(figsize=(6, 5))

        cm_array = np.array(cm)
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 16, "weight": "bold"},
        )

        ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=12)

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace(" ", "_").replace("+", "plus").lower()
        out_path = output_dir / f"cm_{safe_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Confusion matrix saved: {out_path}")
        return str(out_path)

    except ImportError:
        logger.warning("⚠️ matplotlib/seaborn not installed. Skipping plot.")
        return None


def plot_all_confusion_matrices() -> None:
    """Vẽ confusion matrices cho tất cả models từ kết quả evaluation."""
    results_path = EVAL_RESULTS_DIR / "comparison_results.json"

    if not results_path.exists():
        logger.error(f"❌ Results not found: {results_path}")
        logger.info("💡 Chạy: python evaluation/evaluate_models.py trước")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    cm_dir = EVAL_RESULTS_DIR / "confusion_matrices"

    for r in results:
        if "confusion_matrix" in r and "error" not in r:
            plot_confusion_matrix(
                cm=r["confusion_matrix"],
                model_name=r["model"],
                output_dir=cm_dir,
            )

    logger.info(f"\n✅ All confusion matrices saved to {cm_dir}")


if __name__ == "__main__":
    plot_all_confusion_matrices()
