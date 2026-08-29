import tempfile
import unittest
from pathlib import Path

from spaceslug.stage1_dataset import import_folder


class Stage1FolderImportTest(unittest.TestCase):
    def test_recursively_imports_text_and_skips_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "guide.txt").write_text("Use the test command.", encoding="utf-8")
            (root / "image.png").write_bytes(b"not training text")
            records, report = import_folder(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["format"], "instruction")
            self.assertEqual(report["files_seen"], 2)
            self.assertEqual(report["files_imported"], 1)
            self.assertEqual(report["files_skipped"][0]["reason"], "unsupported-extension")

    def test_missing_folder_rejected(self):
        with self.assertRaises(ValueError):
            import_folder("/definitely/missing/stage1-folder")


if __name__ == "__main__":
    unittest.main()
