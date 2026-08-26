import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spaceslug.desktop import (
    DesktopController,
    default_workspace_paths,
    load_workspace_paths,
    save_workspace_paths,
    tiny_profile_id,
)
from spaceslug.desktop.settings import WorkspacePaths


class WorkspacePathsTest(unittest.TestCase):
    def test_default_derivation(self):
        paths = default_workspace_paths("/tmp/ws")
        self.assertEqual(paths.workspace_root, Path("/tmp/ws"))
        self.assertEqual(paths.dataset_dir, Path("/tmp/ws/datasets"))
        self.assertEqual(paths.checkpoint_dir, Path("/tmp/ws/checkpoints"))
        self.assertEqual(paths.artifact_dir, Path("/tmp/ws/artifacts"))
        self.assertEqual(paths.experiment_dir, Path("/tmp/ws/experiments"))
        self.assertEqual(paths.temp_dir, Path("/tmp/ws/tmp"))

    def test_dict_round_trip(self):
        paths = default_workspace_paths("/tmp/ws")
        restored = WorkspacePaths.from_dict(paths.to_dict())
        self.assertEqual(restored, paths)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config" / "desktop.json"
            paths = default_workspace_paths(Path(directory) / "ws")
            save_workspace_paths(paths, config)
            self.assertTrue(config.is_file())
            self.assertEqual(load_workspace_paths(config), paths)

    def test_load_falls_back_to_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_workspace_paths(Path(directory) / "absent.json")
            self.assertEqual(loaded.workspace_root, default_workspace_paths().workspace_root)

    def test_load_tolerates_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "desktop.json"
            config.write_text("{not json", encoding="utf-8")
            loaded = load_workspace_paths(config)
            self.assertEqual(loaded.workspace_root, default_workspace_paths().workspace_root)


class DesktopPathControlTest(unittest.TestCase):
    def test_explicit_workspace_root_wins_over_persisted_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "desktop.json"
            save_workspace_paths(default_workspace_paths(Path(directory) / "persisted"), config)
            controller = DesktopController(workspace_root=Path(directory) / "explicit", config_path=config)
            self.assertEqual(controller.paths.workspace_root, Path(directory) / "explicit")

    def test_set_workspace_root_rebases_subdirs_and_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "desktop.json"
            controller = DesktopController(workspace_root=Path(directory) / "old", config_path=config)
            controller.set_workspace_root(Path(directory) / "new")
            self.assertEqual(controller.paths.workspace_root, Path(directory) / "new")
            self.assertEqual(controller.paths.dataset_dir, Path(directory) / "new" / "datasets")
            self.assertEqual(controller.paths.checkpoint_dir, Path(directory) / "new" / "checkpoints")
            self.assertEqual(load_workspace_paths(config), controller.paths)

    def test_set_individual_subdir_persists_without_resetting_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "desktop.json"
            controller = DesktopController(workspace_root=Path(directory) / "ws", config_path=config)
            controller.set_dataset_license("MIT")
            controller.import_local_source(_write_text(directory, "a.txt", "hello\n"))
            controller.set_artifact_dir(Path(directory) / "custom-artifacts")
            self.assertEqual(controller.paths.artifact_dir, Path(directory) / "custom-artifacts")
            self.assertEqual(len(controller.imports), 1)  # imports are not cleared
            self.assertEqual(load_workspace_paths(config).artifact_dir, Path(directory) / "custom-artifacts")

    def test_set_workspace_root_resets_dataset_state(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            controller.set_dataset_license("MIT")
            controller.import_local_source(_write_text(directory, "a.txt", "hello\n"))
            controller.create_dataset("ds")
            controller.set_workspace_root(Path(directory) / "moved")
            self.assertEqual(controller.imports, [])
            self.assertEqual(controller.created_bundle, "")

    def test_snapshot_exposes_persisted_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            snapshot = controller.snapshot()
            self.assertEqual(snapshot["workspace_root"], str(Path(directory) / "ws"))
            self.assertEqual(snapshot["workspace_paths"]["checkpoint_dir"], str(Path(directory) / "ws" / "checkpoints"))


class DesktopRankTest(unittest.TestCase):
    def test_rank_must_be_4_or_8(self):
        controller = DesktopController()
        controller.set_training_rank(4)
        self.assertEqual(controller.training_rank, 4)
        controller.set_training_rank(8)
        self.assertEqual(controller.training_rank, 8)
        for bad in (0, 1, 5, 16, "4", 4.0, True, None):
            with self.assertRaises(ValueError):
                controller.set_training_rank(bad)

    def test_tiny_profile_id_constrained(self):
        self.assertEqual(tiny_profile_id(4), "tiny_h64_v259_vp320_t128_rank4")
        self.assertEqual(tiny_profile_id(8), "tiny_h64_v259_vp320_t128_rank8")
        with self.assertRaises(ValueError):
            tiny_profile_id(6)

    def test_training_records_rank_in_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_text(directory, "notes.txt", "hello world\nsecond line\n")
            controller = DesktopController(workspace_root=root / "ws", code_revision="test-rev")
            controller.set_dataset_license("CC-BY-4.0")
            controller.import_local_source(root / "notes.txt")
            controller.create_dataset("rank-ds")
            controller.build_training_bundle()
            controller.set_training_steps(2)
            controller.set_training_rank(8)
            controller.start_training()
            final = controller.wait_training(timeout=30)
            self.assertEqual(final["state"], "finished")
            self.assertEqual(final["rank"], 8)
            self.assertEqual(final["tiny_profile"], "tiny_h64_v259_vp320_t128_rank8")
            record = json.loads(Path(final["result"]["experiment"]).read_text(encoding="utf-8"))
            self.assertEqual(record["config"]["rank"], 8)

    def test_training_outputs_land_in_configured_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_text(directory, "notes.txt", "hello world\nsecond line\n")
            controller = DesktopController(workspace_root=root / "ws")
            controller.set_dataset_license("CC-BY-4.0")
            controller.import_local_source(root / "notes.txt")
            controller.create_dataset("paths-ds")
            controller.build_training_bundle()
            controller.set_training_steps(2)
            controller.set_checkpoint_dir(root / "ck")
            controller.set_artifact_dir(root / "art")
            controller.set_experiment_dir(root / "exp")
            controller.start_training()
            final = controller.wait_training(timeout=30)
            self.assertEqual(final["state"], "finished")
            paths = final["paths"]
            self.assertTrue(str(paths["checkpoint"]).startswith(str(root / "ck")))
            self.assertTrue(str(paths["artifact"]).startswith(str(root / "art")))
            self.assertTrue(str(paths["experiment"]).startswith(str(root / "exp")))


class DesktopTkFreeTest(unittest.TestCase):
    def test_controller_does_not_import_tkinter(self):
        code = (
            "import sys\n"
            "import spaceslug.desktop.controller\n"
            "assert 'tkinter' not in sys.modules, 'controller imported tkinter'\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=_pythonpath_env(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


def _pythonpath_env():
    import os

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1] / "python")
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _write_text(directory, name, text) -> Path:
    root = Path(directory)
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
