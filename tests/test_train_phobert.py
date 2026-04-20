import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from model.phobert import PhoBERTDataset
from model.train_phobert import WeightedTrainer
from model.phobert import PhoBERTBaseline
from transformers import TrainingArguments, AutoTokenizer

def test_phobert_dataset():
    df = pd.DataFrame({
        "text": ["Tin gi 1", "Tin th-t 1"],
        "label_binary": [1, 0]
    })
    from transformers import AutoTokenizer
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
        mock_tok.return_value = MagicMock()
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        dataset = PhoBERTDataset(df["text"].tolist(), df["label_binary"].tolist(), tokenizer, max_length=20)
        assert len(dataset) == 2

def test_weighted_trainer_init():
    with patch("transformers.AutoModel.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        model = PhoBERTBaseline(num_labels=2)
        args = TrainingArguments(
            output_dir="./tmp_out",
            do_train=True,
            max_steps=1
        )
        # Just test if it initializes without error
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=None
        )
        assert trainer is not None

def test_compute_metrics_train():
    from model.train_phobert import compute_metrics
    eval_pred = (np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([1, 0]))
    metrics = compute_metrics(eval_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

def test_weighted_trainer_compute_loss():
    with patch("transformers.AutoModel.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        model = PhoBERTBaseline(num_labels=2)
        args = TrainingArguments(output_dir="./tmp", label_smoothing_factor=0.1)
        trainer = WeightedTrainer(model=model, args=args)
        
        inputs = {
            "input_ids": torch.zeros((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long)
        }
        loss = trainer.compute_loss(model, inputs)
        assert loss > 0
        assert isinstance(loss, torch.Tensor)

def test_load_data_mock(tmp_path):
    from model.train_phobert import load_data
    csv_file = tmp_path / "train.csv"
    pd.DataFrame({"text": ["T"], "label_binary": [1]}).to_csv(csv_file, index=False)
    
    with patch("model.train_phobert.DATASET_PROCESSED_DIR", tmp_path):
        df = load_data("train")
        assert len(df) == 1

import model.train_phobert as mt
def test_phobert_train_main_mock(tmp_path):
    # This is a very heavy mock to cover the main logic without running it
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # baseline variant
        mock_args.return_value = MagicMock(variant="baseline", seed=42, model_name="test", epochs=1, batch_size=1, lr=1e-5, max_seq_len=10)
        with patch("model.train_phobert.load_data") as mock_load:
            mock_load.return_value = pd.DataFrame({"text": ["T"], "label_binary": [1]})
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
                with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model:
                    with patch("model.train_phobert.WeightedTrainer") as mock_trainer:
                        instance = mock_trainer.return_value
                        # trainer.train() returns a TrainOutput (global_step, training_loss, metrics)
                        from collections import namedtuple
                        TrainOutput = namedtuple("TrainOutput", ["global_step", "training_loss", "metrics"])
                        instance.train.return_value = TrainOutput(1, 0.5, {"eval_f1": 0.8})
                        instance.evaluate.return_value = {"eval_f1": 0.8}
                        with patch("builtins.open", MagicMock()):
                            mt.main()
                            assert mock_trainer.called

def test_evaluate_variant_mock():
    # Cover _evaluate_variant logic
    from evaluation.ablation_study import _evaluate_variant
    # Now it's at top level, patch it there
    with patch("evaluation.ablation_study.SequentialAdversarialPipeline") as mock_pipe_cls:
        instance = mock_pipe_cls.return_value
        instance.stages = [MagicMock(), MagicMock(), MagicMock()]
        instance.run.return_value = MagicMock()
        # Mocking compute_metrics
        with patch("evaluation.ablation_study.compute_metrics") as mock_metrics:
            mock_metrics.return_value = {"f1": 0.85}
            res = _evaluate_variant(df=pd.DataFrame({"text":["T"], "label_binary":[1]}), removed_stage=1)
            assert res["f1"] == 0.85

def test_phobert_train_features_main_mock(tmp_path):
    # Cover the features variant branch in main
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # features variant
        mock_args.return_value = MagicMock(
            variant="features", seed=42, model_name="test", epochs=1, 
            batch_size=1, lr=1e-5, max_seq_len=10, dropout=0.1,
            lr_scheduler_type="linear" # Prevent transformers ValidationError
        )
        with patch("model.train_phobert.load_data") as mock_load:
            # Mock columns to have features
            mock_load.return_value = pd.DataFrame({"text": ["T"], "label_binary": [1], "f1": [0.1], "f2": [0.2]})
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
                # Patch at top level reference in train_phobert
                with patch("model.train_phobert.PhoBERTWithFeatures") as mock_model_cls:
                    with patch("model.train_phobert.WeightedTrainer") as mock_trainer:
                        with patch("model.train_phobert.DATASET_PROCESSED_DIR", tmp_path):
                            with patch("builtins.open", MagicMock()):
                                with patch("model.train_phobert.FEATURE_NAMES", ["f1", "f2"]):
                                    mt.main()
                                    assert mock_model_cls.called
