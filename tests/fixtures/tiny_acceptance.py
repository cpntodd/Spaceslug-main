"""Small deterministic fixture shared by the Tiny acceptance tests."""

from __future__ import annotations

from pathlib import Path

from spaceslug.dataset import DatasetBundle, create_bundle, verify_bundle


def create_tiny_acceptance_bundle(root: str | Path) -> DatasetBundle:
    return verify_bundle(create_bundle(Path(root), "tiny-acceptance-v1", {
        "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}, {"record_id": "b", "prompt": "Q: ", "target": "b"}],
        "validation": [{"record_id": "v", "prompt": "Q: ", "target": "a"}],
        "test": [{"record_id": "t", "prompt": "Q: ", "target": "b"}],
    }, tokenizer_id="spaceslug-byte", tokenizer_revision="v1" ).root)
