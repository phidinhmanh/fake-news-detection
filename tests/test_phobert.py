import pytest
import torch
from model.phobert import PhoBERTBaseline

def test_phobert_architecture():
    # Test model initialization
    model = PhoBERTBaseline(num_labels=2)
    assert model.backbone.config._name_or_path == "vinai/phobert-base-v2"
    assert isinstance(model.classifier, torch.nn.Sequential)

def test_phobert_forward_pass():
    model = PhoBERTBaseline(num_labels=2)
    # Mock input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    assert "logits" in output
    assert output["logits"].shape == (batch_size, 2)
    # Check if probabilities sum to ~1 after softmax (implicitly check logits)
    probs = torch.softmax(output["logits"], dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0]))
