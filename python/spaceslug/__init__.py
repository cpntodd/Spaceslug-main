"""Spaceslug-main host contracts and integration helpers."""

from .backend import BackendCapabilities, BackendError, BackendSession, ExecutionResult
from .dataset import DatasetBundle, create_bundle, verify_bundle
from .tiny_model import TinyBigramModel

__all__ = [
    "BackendCapabilities", "BackendError", "BackendSession", "ExecutionResult",
    "DatasetBundle", "create_bundle", "verify_bundle", "TinyBigramModel",
]
