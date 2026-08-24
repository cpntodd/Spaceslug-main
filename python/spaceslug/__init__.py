"""Spaceslug-main host contracts and integration helpers."""

from .backend import BackendCapabilities, BackendError, BackendSession, ExecutionResult
from .dataset import DatasetBundle, create_bundle, validate_manifest, verify_bundle
from .tiny_model import TinyBigramModel
from .tiny_dense_model import TinyDenseCausalModel
from .tiny_training import TinyTrainingConfig, load_training_checkpoint, save_training_checkpoint, train_tiny
from .tiny_artifact import load_tiny_artifact, write_tiny_artifact
from .tokenizer import ByteTokenizer, default_tokenizer
from .reference import vector_add
from .gpu_lora_training import PersistentTinyTrainer

__all__ = [
    "BackendCapabilities", "BackendError", "BackendSession", "ExecutionResult",
    "DatasetBundle", "create_bundle", "validate_manifest", "verify_bundle", "TinyBigramModel",
    "TinyDenseCausalModel", "TinyTrainingConfig", "train_tiny", "save_training_checkpoint",
    "load_training_checkpoint", "write_tiny_artifact", "load_tiny_artifact", "ByteTokenizer",
    "default_tokenizer", "vector_add", "PersistentTinyTrainer",
]
