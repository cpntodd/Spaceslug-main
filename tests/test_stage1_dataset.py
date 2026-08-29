import json
import tempfile
import unittest
from pathlib import Path

from spaceslug.stage1_dataset import build_stage1_bundle, deduplicate, grouped_split, normalize_text, validate_record


class Stage1DatasetTest(unittest.TestCase):
    def test_validate_and_normalize(self):
        row = validate_record({"record_id": "a", "format": "coding", "prompt": "  explain\nthis ", "target": " answer "})
        self.assertEqual(row["prompt"], "explain\nthis")
        self.assertEqual(row["topic"], "general")
        self.assertEqual(normalize_text("a\r\n b"), "a\nb")

    def test_deduplicate_and_grouped_split(self):
        rows = [{"record_id": str(i), "prompt": f"q{i}", "target": "answer", "group_id": f"g{i}"} for i in range(20)]
        rows.append({"record_id": "duplicate", "prompt": "q0", "target": "answer", "group_id": "g0"})
        unique, duplicates = deduplicate(rows)
        self.assertEqual(len(unique), 20)
        self.assertEqual(duplicates, ["duplicate"])
        splits = grouped_split(unique, seed=7)
        self.assertEqual(sum(map(len, splits.values())), 20)
        self.assertTrue(all(set(r["group_id"] for r in splits[a]).isdisjoint(set(r["group_id"] for r in splits[b])) for a in splits for b in splits if a != b))

    def test_build_bundle_keeps_evaluation_out_of_train(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, report = build_stage1_bundle(Path(directory) / "data.dts", "agent", [
                {"record_id": "train", "prompt": "question", "target": "answer", "group_id": "train"},
                {"record_id": "eval", "format": "evaluation", "prompt": "held out", "target": "expected", "group_id": "eval"},
            ], seed=1)
            self.assertEqual(bundle.manifest["record_count"], 2)
            self.assertEqual(len(bundle.records("train")), 1)
            self.assertEqual(len(bundle.records("test")), 1)
            self.assertEqual(report["records"], 2)
            self.assertTrue((Path(directory) / "data.dts" / "stage1-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
