import tempfile
from pathlib import Path
import unittest

from spaceslug.filesystem_picker import pick_files
from spaceslug.model_profiles import get_profile, profile_names, resolve_profile


class ProfilesPickerTest(unittest.TestCase):
    def test_model_profiles_validate_and_override(self):
        self.assertIn("Spaceslug-0.1B", profile_names())
        profile = resolve_profile("Spaceslug-0.1B", training_mode="lora")
        self.assertEqual(profile["target_parameters"], 100_000_000)
        self.assertEqual(profile["training_mode"], "lora")
        with self.assertRaises(ValueError):
            get_profile("unknown")

    def test_filesystem_picker_filters_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.md").write_text("hello", encoding="utf-8")
            (root / "skip.py").write_text("pass", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "ignored.md").write_text("ignored", encoding="utf-8")
            selected = pick_files(root)
            self.assertEqual([path.name for path in selected.files], ["ok.md"])
            self.assertEqual([path.name for path in selected.skipped], ["ignored.md", "skip.py"])


if __name__ == "__main__":
    unittest.main()
