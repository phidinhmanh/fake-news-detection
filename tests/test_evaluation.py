import pytest
import numpy as np
from evaluation.evaluate_models import _compute_metrics
from evaluation.ablation_study import generate_ablation_table

def test_compute_metrics_logic():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    
    metrics = _compute_metrics(y_true, y_pred, y_prob, "Test Model")
    assert metrics["model"] == "Test Model"
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

def test_compute_metrics_imperfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0]) # 1 correct, 1 false pos, 1 false neg, 1 correct
    y_prob = np.array([0.9, 0.6, 0.4, 0.1])
    
    metrics = _compute_metrics(y_true, y_pred, y_prob, "Imperfect Model")
    assert metrics["accuracy"] == 0.5
    # precision = 1/2, recall = 1/2, F1 = 0.5
    assert metrics["f1"] == 0.5

def test_ablation_table_generation():
    experiments = [
        {
            "experiment": "Exp 1",
            "f1": 0.85,
            "accuracy": 0.88,
            "description": "Desc 1"
        },
        {
            "experiment": "Exp 2",
            "f1": 0.82,
            "f1_delta": -0.03,
            "f1_delta_pct": -3.5,
            "accuracy": 0.84,
            "description": "Desc 2"
        },
        {
            "experiment": "Exp 3",
            "error": "Failed"
        }
    ]
    
    table = generate_ablation_table(experiments)
    assert "| Exp 1 |" in table
    assert "0.8500" in table
    assert "-0.0300" in table
    assert "ERROR" in table
    assert "| Exp 3 |" in table

import evaluation.ablation_study as as_mod
from unittest.mock import patch, MagicMock
import pandas as pd
def test_run_ablation_study_mock():
    # Mocking _evaluate_variant to return static values
    with patch("evaluation.ablation_study._evaluate_variant") as mock_eval:
        mock_eval.return_value = {"f1": 0.8}
        # Mock DATASET_PROCESSED_DIR to skip file checking error
        with patch("evaluation.ablation_study.DATASET_PROCESSED_DIR") as mock_dir:
            mock_dir.__truediv__.return_value.exists.return_value = True
            with patch("pandas.read_csv") as mock_csv:
                mock_csv.return_value = pd.DataFrame({"text": ["T"], "label": ["fake"]})
                results = as_mod.run_ablation_study()
                assert len(results) > 0
                assert results[0]["f1"] == 0.8

import evaluation.evaluate_models as em
def test_evaluate_tfidf_baseline_mock(sample_vn_df):
    from model.baseline_logreg import BaselineLogReg
    with patch("model.baseline_logreg.BaselineLogReg.load") as mock_load:
        with patch("model.baseline_logreg.BaselineLogReg.predict_with_score") as mock_pred:
            mock_pred.return_value = {"label": "fake", "fake_proba": 0.8}
            report = em.evaluate_tfidf_baseline(sample_vn_df)
            assert report["model"] == "TF-IDF Baseline"
            assert report["accuracy"] == 0.25 # Only 1 correct out of 4 (all predicted fake)

def test_evaluate_phobert_mock(sample_vn_df):
    # Test file missing path
    report = em.evaluate_phobert(sample_vn_df, variant="baseline")
    assert "error" in report or "not found" in report["error"].lower()

def test_evaluate_pipeline_mock(full_mock_pipeline, sample_vn_df):
    # Patch the class where it's defined and used
    with patch("sequential_adversarial.pipeline.SequentialAdversarialPipeline") as mock_pipe_cls:
        mock_pipe_cls.return_value = full_mock_pipeline
        # Mocking compute_metrics (internal private function)
        with patch("evaluation.evaluate_models._compute_metrics") as mock_metrics:
            mock_metrics.return_value = {"accuracy": 0.9}
            report = em.evaluate_pipeline(sample_vn_df, mock=True)
            assert report["accuracy"] == 0.9

def test_evaluate_models_main_mock(sample_vn_df):
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = MagicMock(real=False)
        with patch("pandas.read_csv") as mock_csv:
            mock_csv.return_value = sample_vn_df
            with patch("evaluation.evaluate_models.EVAL_RESULTS_DIR") as mock_dir:
                mock_dir.__truediv__.return_value = MagicMock()
                with patch("builtins.open", MagicMock()):
                    # Mocking the evaluation functions to speed up and avoid artifact errors
                    with patch("evaluation.evaluate_models.evaluate_tfidf_baseline") as m1:
                        m1.return_value = {"acc": 0.8}
                        with patch("evaluation.evaluate_models.evaluate_phobert") as m2:
                            m2.return_value = {"acc": 0.8}
                            with patch("evaluation.evaluate_models.evaluate_pipeline") as m3:
                                m3.return_value = {"acc": 0.8}
                                em.main()
                                assert m1.called
                                assert m2.called
                                assert m3.called
