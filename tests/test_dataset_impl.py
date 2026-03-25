import pandas as pd
from pathlib import Path
import torch
from data.dataset import FakeNewsDataset
from config import NORMALIZED_DIR, LABELS

def test_dataset_loading():
    # Create directory if it doesn't exist
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Create a dummy parquet file
    data = {
        "text": ["Nội dung tin giả", "Nội dung tin thật"],
        "label": ["fake", "real"],
        "domain": ["politics", "social"],
        "lang": ["vi", "vi"]
    }
    df = pd.DataFrame(data)
    file_path = NORMALIZED_DIR / "train.parquet"
    df.to_parquet(file_path)

    # Test FakeNewsDataset
    dataset = FakeNewsDataset(data_type="normalized", split="train", max_length=128)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input ids shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label']}")

    assert len(dataset) == 2
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample
    assert sample["label"].item() == 0  # 'fake' is 0 in LABELS

    # Cleanup
    file_path.unlink()
    print("Test passed!")

if __name__ == "__main__":
    test_dataset_loading()
