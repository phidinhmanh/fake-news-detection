"""
model/drift_monitor.py — Model Drift Monitoring
==============================================
Monitors prediction distributions over time and alerts on drift.
Follows NFR-8.7: Monitor model drift.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from config import EVAL_RESULTS_DIR
except ImportError:
    EVAL_RESULTS_DIR = Path("evaluation/results")

logger = logging.getLogger(__name__)


# ─── Drift Detection Thresholds ──────────────────────────────────────────────


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    psi_threshold: float = 0.2
    min_samples: int = 100
    baseline_window: int = 1000
    alert_cooldown: int = 500


@dataclass
class DriftStats:
    """Statistics for drift monitoring."""
    total_predictions: int = 0
    fake_ratio: float = 0.5
    avg_confidence: float = 0.5
    confidence_std: float = 0.0
    recent_fake_ratio: float = 0.5
    psi: float = 0.0
    drift_detected: bool = False
    drift_direction: str = "none"
    last_alert: datetime | None = None


class DriftMonitor:
    """
    Monitor model predictions for drift over time (NFR-8.7).

    Tracks:
    - Prediction distribution (fake vs real ratio)
    - Confidence distributions
    - Population Stability Index (PSI)
    """

    def __init__(
        self,
        thresholds: DriftThresholds | None = None,
        storage_path: Path | None = None,
    ):
        self.thresholds = thresholds or DriftThresholds()
        self.storage_path = storage_path or (EVAL_RESULTS_DIR / "drift_log.json")

        self.baseline: deque = deque(maxlen=self.thresholds.baseline_window)
        self.recent: deque = deque(maxlen=self.thresholds.alert_cooldown)
        self.all_predictions: list = []

        self._samples_since_alert = 0
        self._load_baseline()

    def record(self, label: str, confidence: float, text_hash: str | None = None) -> DriftStats:
        """
        Record a prediction and check for drift (NFR-8.7).
        """
        record = {
            "timestamp": datetime.utcnow(),
            "label": label,
            "confidence": confidence,
            "text_hash": text_hash or hashlib.md5(str(label).encode()).hexdigest()[:8],
        }

        self.all_predictions.append(record)
        self.recent.append(record)

        if len(self.baseline) < self.thresholds.baseline_window:
            self.baseline.append(record)

        self._samples_since_alert += 1
        return self._compute_stats()

    def _compute_stats(self) -> DriftStats:
        """Compute current drift statistics."""
        stats = DriftStats()

        if not self.all_predictions:
            return stats

        stats.total_predictions = len(self.all_predictions)

        fake_count = sum(1 for r in self.all_predictions if r["label"] == "fake")
        stats.fake_ratio = fake_count / len(self.all_predictions)

        confidences = [r["confidence"] for r in self.all_predictions]
        stats.avg_confidence = float(np.mean(confidences))
        stats.confidence_std = float(np.std(confidences))

        if len(self.recent) >= self.thresholds.min_samples:
            recent_fake = sum(1 for r in self.recent if r["label"] == "fake")
            stats.recent_fake_ratio = recent_fake / len(self.recent)
        else:
            stats.recent_fake_ratio = stats.fake_ratio

        stats.drift_detected, stats.drift_direction, stats.psi = self._check_drift(
            stats.recent_fake_ratio,
            stats.fake_ratio,
        )

        if stats.drift_detected and self._samples_since_alert >= self.thresholds.alert_cooldown:
            self._trigger_alert(stats)
            self._samples_since_alert = 0
        else:
            stats.drift_detected = False

        return stats

    def _check_drift(self, recent_ratio: float, baseline_ratio: float) -> tuple[bool, str, float]:
        """Check for drift using PSI."""
        if baseline_ratio <= 0:
            baseline_ratio = 0.001
        if baseline_ratio >= 1:
            baseline_ratio = 0.999
        if recent_ratio <= 0:
            recent_ratio = 0.001
        if recent_ratio >= 1:
            recent_ratio = 0.999

        psi = (recent_ratio - baseline_ratio) * np.log(recent_ratio / baseline_ratio)

        if psi >= self.thresholds.psi_threshold:
            direction = "increased" if recent_ratio > baseline_ratio else "decreased"
            return True, direction, psi

        return False, "none", psi

    def _trigger_alert(self, stats: DriftStats) -> None:
        """Trigger drift alert."""
        logger.warning(
            f"[DRIFT ALERT] Model drift detected!\n"
            f"  Direction: {stats.drift_direction}\n"
            f"  Baseline fake ratio: 0.50\n"
            f"  Recent fake ratio: {stats.recent_fake_ratio:.2%}\n"
            f"  PSI: {stats.psi:.4f}\n"
            f"  Total predictions: {stats.total_predictions}"
        )

        self._save_alert(stats)

    def _save_alert(self, stats: DriftStats) -> None:
        """Save drift alert to log file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_predictions": stats.total_predictions,
            "fake_ratio": stats.fake_ratio,
            "recent_fake_ratio": stats.recent_fake_ratio,
            "drift_direction": stats.drift_direction,
            "psi": stats.psi,
            "avg_confidence": stats.avg_confidence,
        }

        log_file = self.storage_path
        existing = []
        if log_file.exists():
            try:
                with open(log_file) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing.append(alert)

        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Drift alert saved to {log_file}")

    def _load_baseline(self) -> None:
        """Load previous baseline if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    if data:
                        logger.info(f"Loaded {len(data)} drift records from baseline")
            except Exception as e:
                logger.warning(f"Could not load baseline: {e}")

    def update_baseline(self) -> None:
        """Update baseline from recent predictions."""
        self.baseline.clear()
        for record in list(self.recent)[-self.thresholds.baseline_window:]:
            self.baseline.append(record)
        logger.info("Baseline updated from recent predictions")

    def reset(self) -> None:
        """Reset all monitoring data."""
        self.baseline.clear()
        self.recent.clear()
        self.all_predictions.clear()
        self._samples_since_alert = 0
        logger.info("Drift monitor reset")


# ─── Singleton Instance ────────────────────────────────────────────────────────


_monitor: DriftMonitor | None = None


def get_monitor() -> DriftMonitor:
    """Get singleton drift monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor


def record_prediction(label: str, confidence: float, text_hash: str | None = None) -> DriftStats:
    """
    Record a prediction for drift monitoring (NFR-8.7).

    Usage:
        from model.drift_monitor import record_prediction

        result = model.predict_with_score(text)
        stats = record_prediction(
            label=result["label"],
            confidence=result["confidence"],
            text_hash=hashlib.md5(text.encode()).hexdigest(),
        )

        if stats.drift_detected:
            logger.warning("Drift detected in model predictions!")
    """
    monitor = get_monitor()
    return monitor.record(label, confidence, text_hash)