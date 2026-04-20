import os
from pathlib import Path
import config

def test_config_paths():
    # Verify PROJECT_ROOT exists and contains typical files
    assert config.PROJECT_ROOT.exists()
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    
    # Check key directories are defined as Path objects
    assert isinstance(config.DATASET_DIR, Path)
    assert isinstance(config.AGENTS_DIR, Path)
    assert isinstance(config.SA_DIR, Path)

def test_config_values():
    # Basic sanity checks on default values
    assert config.LABELS == ("fake", "real")
    assert config.MAX_TEXT_LENGTH == 2048
    assert config.PHOBERT_MODEL_NAME == "vinai/phobert-base-v2"
    assert config.EMBEDDING_MODEL_NAME == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def test_env_override_simulation():
    # Simulate env override
    os.environ["SA_MODEL_NAME"] = "test-model"
    # We might need to reload if config already imported, 
    # but since it's a test we can just check if it was picked up initially 
    # or how the module handles it.
    import importlib
    importlib.reload(config)
    assert config.SA_MODEL_NAME == "test-model"
    
    # Cleanup
    del os.environ["SA_MODEL_NAME"]
    importlib.reload(config)
