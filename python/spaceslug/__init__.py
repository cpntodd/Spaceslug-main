"""Spaceslug-main host contracts and integration helpers."""

from .backend import BackendCapabilities, BackendError, BackendSession, ExecutionResult
from .dataset import DatasetBundle, create_bundle, validate_manifest, verify_bundle
from .tiny_model import TinyBigramModel
from .reference import vector_add

__all__ = [
    "BackendCapabilities", "BackendError", "BackendSession", "ExecutionResult",
    "DatasetBundle", "create_bundle", "validate_manifest", "verify_bundle", "TinyBigramModel", "vector_add",
]
