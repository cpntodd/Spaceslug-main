import tempfile
from pathlib import Path
import unittest

from spaceslug.tui import SpaceslugTui


class TuiTest(unittest.TestCase):
    def test_headless_workflow_and_worm_graph(self):
        tui = SpaceslugTui()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "notes.md").write_text("hello", encoding="utf-8")
            selection = tui.select_directory(directory)
            self.assertEqual(len(selection.files), 1)
        config = tui.set_model("Spaceslug-0.1B", training_mode="lora")
        self.assertEqual(config["training_mode"], "lora")
        tui.set_training(steps=3, epochs=2)
        tui.add_loss(2.0)
        tui.add_loss(1.0)
        self.assertEqual(len(tui.worm_graph()), 8)
        rendered = tui.render()
        self.assertIn("worm graph", rendered)
        self.assertIn("training", rendered)
        self.assertEqual(tui.state.model_id, "Spaceslug-0.1B")

    def test_invalid_training_controls_fail(self):
        with self.assertRaises(ValueError):
            SpaceslugTui().set_training(steps=0)


if __name__ == "__main__":
    unittest.main()
