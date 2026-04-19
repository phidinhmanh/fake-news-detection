"""prediction_cache.py — Model Instance Caching
===============================================
Singleton cache for model instances to avoid redundant loading.

Usage:
    cache = PredictionCache()
    cache.set_lora("path", (model, tokenizer))
    model, tokenizer = cache.get_lora("path")
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedTokenizer


ModelTokenizerPair = Tuple["nn.Module", "PreTrainedTokenizer"]
BaselineModel = object


class PredictionCache:
    """Singleton cache for model instances.

    Responsibility: Caching only (SRP fix for ARCH-001).
    Uses class-level dicts for singleton pattern.
    """

    _lora_instances: Dict[str, ModelTokenizerPair] = {}
    _baseline_instances: Dict[str, BaselineModel] = {}

    def get_lora(self, model_path: str) -> Optional[ModelTokenizerPair]:
        """Get cached LoRA model and tokenizer.

        Args:
            model_path: Absolute path string as cache key.

        Returns:
            Tuple of (model, tokenizer) or None if not cached.
        """
        return self._lora_instances.get(model_path)

    def set_lora(
        self,
        model_path: str,
        instance: ModelTokenizerPair,
    ) -> None:
        """Cache LoRA model and tokenizer.

        Args:
            model_path: Absolute path string as cache key.
            instance: Tuple of (model, tokenizer).
        """
        self._lora_instances[model_path] = instance

    def get_baseline(self, model_path: str) -> Optional[BaselineModel]:
        """Get cached baseline model.

        Args:
            model_path: Absolute path string as cache key.

        Returns:
            Baseline model instance or None if not cached.
        """
        return self._baseline_instances.get(model_path)

    def set_baseline(
        self,
        model_path: str,
        instance: BaselineModel,
    ) -> None:
        """Cache baseline model.

        Args:
            model_path: Absolute path string as cache key.
            instance: Baseline model instance.
        """
        self._baseline_instances[model_path] = instance

    def clear(self) -> None:
        """Clear all cached instances."""
        self._lora_instances.clear()
        self._baseline_instances.clear()

    @classmethod
    def get_lora_cls(cls, model_path: str) -> Optional[ModelTokenizerPair]:
        """Class-level getter for backwards compatibility."""
        return cls._lora_instances.get(model_path)

    @classmethod
    def set_lora_cls(
        cls,
        model_path: str,
        instance: ModelTokenizerPair,
    ) -> None:
        """Class-level setter for backwards compatibility."""
        cls._lora_instances[model_path] = instance

    @classmethod
    def get_baseline_cls(cls, model_path: str) -> Optional[BaselineModel]:
        """Class-level getter for backwards compatibility."""
        return cls._baseline_instances.get(model_path)

    @classmethod
    def set_baseline_cls(
        cls,
        model_path: str,
        instance: BaselineModel,
    ) -> None:
        """Class-level setter for backwards compatibility."""
        cls._baseline_instances[model_path] = instance