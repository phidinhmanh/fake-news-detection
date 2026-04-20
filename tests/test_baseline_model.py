import pytest
import pandas as pd
from pathlib import Path
from model.baseline_logreg import BaselineLogReg

def test_baseline_predict_structure(sample_vn_df):
    model = BaselineLogReg()
    # Mocking trained pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    model.pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    
    # Fit on sample data
    model.pipeline.fit(sample_vn_df['text'], sample_vn_df['label'])
    
    res = model.predict_with_score("Vaccine rất an toàn.")
    assert "label" in res
    assert "confidence" in res
    assert "fake_proba" in res
    assert "real_proba" in res
    assert res["fake_proba"] + res["real_proba"] == pytest.approx(1.0)

def test_baseline_save_load(sample_vn_df, tmp_path):
    # Set model artifacts dir to tmp
    import model.baseline_logreg as bl
    original_dir = bl.MODELS_ARTIFACTS_DIR
    bl.MODELS_ARTIFACTS_DIR = tmp_path
    
    model = BaselineLogReg()
    model.pipeline.fit(sample_vn_df['text'], sample_vn_df['label'])
    model.model_path = tmp_path / "baseline_logreg.joblib"
    
    model.save()
    assert model.model_path.exists()
    
    # New instance
    new_model = BaselineLogReg()
    new_model.model_path = model.model_path
    new_model.load()
    
    res = new_model.predict_with_score("Test load.")
    assert "label" in res
    
    # Restore original dir
    bl.MODELS_ARTIFACTS_DIR = original_dir

def test_load_missing_file_raises(tmp_path):
    model = BaselineLogReg()
    model.model_path = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        model.load()

def test_baseline_train_eval(sample_vn_df):
    model = BaselineLogReg()
    # Test full training cycle - returns None
    model.train(sample_vn_df)
    assert model.pipeline is not None

def test_baseline_evaluate_method(sample_vn_df):
    model = BaselineLogReg()
    model.train(sample_vn_df)
    # evaluate returns F1 score (float)
    f1 = model.evaluate(sample_vn_df)
    assert isinstance(f1, float)
    assert f1 >= 0
