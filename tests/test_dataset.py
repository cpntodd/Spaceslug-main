import tempfile
from pathlib import Path
import unittest

from spaceslug.dataset import create_bundle, verify_bundle


class DatasetBundleTest(unittest.TestCase):
    def test_round_trip_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "fixture.dts"
            bundle = create_bundle(
                bundle_path,
                "fixture",
                {
                    "train": [{"record_id": "b", "text": "second"}, {"record_id": "a", "text": "first"}],
                    "validation": [{"record_id": "v", "text": "validate"}],
                    "test": [],
                },
                tokenizer_id="fixture-tokenizer",
                tokenizer_revision="sha256:fixture",
            )
            verified = verify_bundle(bundle_path)
            self.assertEqual(verified.records("train"), [
                {"record_id": "a", "text": "first"}, {"record_id": "b", "text": "second"}
            ])
            self.assertEqual(bundle.manifest["revision"], verified.manifest["revision"])
            self.assertEqual(verified.manifest["record_count"], 3)
            self.assertEqual(verified.stats(), {"train": 2, "validation": 1, "test": 0})
            self.assertEqual(verified.manifest["provenance"]["sources"], [])

            (bundle_path / "records" / "train.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_bundle(bundle_path)

    def test_repeated_writes_are_byte_identical(self):
        splits = {
            "train": [{"record_id": "b", "text": "second"}, {"record_id": "a", "text": "first"}],
            "validation": [{"record_id": "v", "text": "validate"}],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.dts"
            second = Path(directory) / "second.dts"
            create_bundle(first, "fixture", splits, sources=["source-b", "source-a"], licenses=["license-b", "license-a"])
            create_bundle(second, "fixture", splits, sources=["source-b", "source-a"], licenses=["license-b", "license-a"])
            first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
            self.assertEqual(first_files, second_files)
            for relative_path in first_files:
                self.assertEqual((first / relative_path).read_bytes(), (second / relative_path).read_bytes())

    def test_manifest_contract_rejects_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "fixture.dts"
            create_bundle(bundle_path, "fixture", {"train": [], "validation": [], "test": []})
            manifest_path = bundle_path / "manifest.json"
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["splits"]["test"]
            manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_bundle(bundle_path)


if __name__ == "__main__":
    unittest.main()
