import numpy as np
import re

"""
dataset/feature_extraction.py — Stylistic feature extraction for Vietnamese news.
"""

FEATURE_NAMES = [
    "length", 
    "avg_word_len", 
    "num_punc", 
    "num_excl", 
    "num_ques", 
    "num_caps", 
    "num_digits", 
    "num_stop", 
    "num_emojis"
]

def extract_features_batch(texts: list[str]) -> np.ndarray:
    """Extracts 9 stylistic features from a batch of texts.
    
    Returns:
        np.ndarray of shape (len(texts), 9)
    """
    all_features = []
    for text in texts:
        text = str(text)
        words = text.split()
        
        length = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        num_punc = len(re.findall(r'[.,!?;:]', text))
        num_excl = text.count('!')
        num_ques = text.count('?')
        num_caps = len(re.findall(r'[A-Z]', text))
        num_digits = len(re.findall(r'\d', text))
        num_stop = 0  # Placeholder or implement with a stopword list
        num_emojis = len(re.findall(r'[^\w\s,.]', text)) - num_punc # Crude approximation
        
        features = [
            float(length), 
            float(avg_word_len), 
            float(num_punc), 
            float(num_excl), 
            float(num_ques), 
            float(num_caps), 
            float(num_digits), 
            float(num_stop), 
            float(num_emojis)
        ]
        all_features.append(features)
        
    return np.array(all_features, dtype=np.float32)
