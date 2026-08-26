import os
import sys
import unittest

from spaceslug.desktop import (
    TAB_NAMES,
    DesktopController,
    LossWormGraph,
    TrainingPlacement,
    default_runtime_probe,
    fixed_tiny_profile,
    fixed_tiny_profile_lines,
    parse_api_address,
    resolve_placement,
    resolve_training_placement,
)


class FakeCanvas:
    def __init__(self):
        self.deleted = 0
        self.lines = []
        self.ovals = []

    def delete(self, tag):
        self.deleted += 1

    def create_line(self, *args, **kwargs):
        self.lines.append((args, kwargs))
        return len(self.lines)

    def create_oval(self, *args, **kwargs):
        self.ovals.append((args, kwargs))
        return len(self.ovals)


class DesktopControllerTest(unittest.TestCase):
    def test_default_snapshot_is_render_safe(self):
        controller = DesktopController()
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["active_tab"], "Home")
        self.assertEqual(snapshot["profile"]["model_id"], "Spaceslug-Tiny")
        self.assertEqual(snapshot["loss_history"], [])
        self.assertIn("training", snapshot["capabilities"])
        # Snapshot must not touch the runtime probe.
        self.assertEqual(snapshot["placement"]["reason"], "runtime not probed yet")
        self.assertEqual(snapshot["training_placement"]["actual"], "cpu-fallback")
        self.assertEqual(snapshot["training"]["state"], "idle")
        self.assertFalse(snapshot["api"]["running"])

    def test_tabs_validate(self):
        controller = DesktopController()
        for name in TAB_NAMES:
            self.assertEqual(controller.set_active_tab(name), name)
            self.assertEqual(controller.active_tab, name)
        with self.assertRaises(ValueError):
            controller.set_active_tab("nope")

    def test_loss_recording_and_clearing(self):
        controller = DesktopController(history=[2.0, 1.0, 1.5])
        self.assertEqual(controller.loss.history, [2.0, 1.0, 1.5])
        self.assertEqual(controller.record_loss(0.5), 4)
        controller.clear_loss()
        self.assertEqual(controller.loss.history, [])

    def test_dataset_and_chat_api_settings(self):
        controller = DesktopController()
        controller.set_dataset_file("/tmp/x.dts")
        controller.set_dataset_url("https://example.test/data.dts")
        controller.set_dataset_search("tiny")
        controller.set_dataset_license("MIT")
        controller.set_dataset_id("my-ds")
        controller.set_searxng_base_url("http://127.0.0.1:9999")
        controller.set_chat_prompt("hello")
        controller.set_api_address("127.0.0.1:9000")
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["dataset_file"], "/tmp/x.dts")
        self.assertEqual(snapshot["dataset_url"], "https://example.test/data.dts")
        self.assertEqual(snapshot["dataset_search"], "tiny")
        self.assertEqual(snapshot["dataset_license"], "MIT")
        self.assertEqual(snapshot["dataset_id"], "my-ds")
        self.assertEqual(snapshot["searxng_base_url"], "http://127.0.0.1:9999")
        self.assertEqual(snapshot["chat_prompt"], "hello")
        self.assertEqual(snapshot["api_address"], "127.0.0.1:9000")

    def test_training_steps_validate(self):
        controller = DesktopController()
        controller.set_training_steps(20)
        self.assertEqual(controller.training_steps, 20)
        with self.assertRaises(ValueError):
            controller.set_training_steps(0)

    def test_training_hyperparameters_validate(self):
        controller = DesktopController()
        controller.set_training_learning_rate(0.5)
        controller.set_training_batch_size(4)
        self.assertEqual(controller.training_learning_rate, 0.5)
        self.assertEqual(controller.training_batch_size, 4)
        with self.assertRaises(ValueError):
            controller.set_training_learning_rate(0.0)
        with self.assertRaises(ValueError):
            controller.set_training_batch_size(0)

    def test_capability_boundaries_describe_wired_workflows(self):
        controller = DesktopController()
        for area in ("training", "chat", "api", "datasets"):
            text = controller.capability_boundary(area)
            self.assertTrue(text.strip())
        self.assertIn("CPU", controller.capability_boundary("training"))
        self.assertIn("checkpoint", controller.capability_boundary("training"))
        self.assertIn("responder", controller.capability_boundary("chat"))
        self.assertIn("TinyCpuEchoResponder", controller.capability_boundary("chat"))
        self.assertIn("loopback", controller.capability_boundary("api"))
        self.assertIn("OpenAICompatibleServer", controller.capability_boundary("api"))
        self.assertIn("workspace", controller.capability_boundary("datasets"))

    def test_structured_capabilities_are_assembled_from_metadata(self):
        controller = DesktopController()
        structured = controller.capabilities()
        self.assertIn("gpu_lora", structured)
        self.assertIn("persistent_tiny", structured)
        self.assertIn("native_fp32_base", structured)
        self.assertIn("dataset_training", structured["gpu_lora"])
        self.assertIn("optimizers", structured["gpu_lora"])

    def test_refresh_runtime_uses_injected_probe(self):
        def probe():
            return {
                "backend": "spaceslug",
                "device": "AMD Radeon RX 580",
                "runtime_revision": "runtime",
                "operations": ["sgemm", "tiny_forward_abi"],
                "software_vulkan": False,
            }

        controller = DesktopController(runtime_probe=probe)
        placement = controller.refresh_runtime()
        self.assertEqual(placement.mode, "gpu-primary")
        self.assertTrue(placement.gpu_primary)
        self.assertTrue(placement.cpu_fallback)
        self.assertEqual(controller.snapshot()["placement"]["device"], "AMD Radeon RX 580")

    def test_gpu_primary_requested_is_never_actual_until_available(self):
        def probe():
            return {"device": "AMD Radeon RX 580", "operations": ["sgemm"], "software_vulkan": False}

        controller = DesktopController(runtime_probe=probe)
        controller.refresh_runtime()
        controller.set_gpu_primary_requested(True)
        placement = controller.training_placement()
        self.assertEqual(placement.requested, "gpu-primary")
        self.assertEqual(placement.actual, "cpu-fallback")
        self.assertEqual(placement.hardware, "gpu-primary")
        self.assertIn("dataset-integrated GPU training", placement.reason)

    def test_gpu_primary_can_be_not_requested(self):
        controller = DesktopController()
        controller.set_gpu_primary_requested(False)
        placement = controller.training_placement()
        self.assertEqual(placement.requested, "cpu")
        self.assertEqual(placement.actual, "cpu-fallback")


class DesktopProfileTest(unittest.TestCase):
    def test_fixed_tiny_profile(self):
        profile = fixed_tiny_profile()
        self.assertEqual(profile["model_id"], "Spaceslug-Tiny")
        self.assertEqual(profile["hidden_size"], 64)
        self.assertEqual(profile["lora_rank"], 4)
        self.assertIn("training_mode", profile)

    def test_profile_lines(self):
        lines = fixed_tiny_profile_lines()
        self.assertTrue(any("Spaceslug-Tiny" in line for line in lines))
        self.assertTrue(any("hidden_size: 64" in line for line in lines))


class LossWormGraphTest(unittest.TestCase):
    def test_record_truncates_to_max_points(self):
        graph = LossWormGraph(max_points=3)
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            graph.record(value)
        self.assertEqual(graph.history, [3.0, 4.0, 5.0])
        self.assertEqual(graph.latest(), 5.0)
        self.assertEqual(graph.bounds(), (3.0, 5.0))

    def test_rejects_non_positive_max_points(self):
        with self.assertRaises(ValueError):
            LossWormGraph(max_points=0)

    def test_scaled_points_stay_inside_canvas(self):
        graph = LossWormGraph(history=[1.0, 2.0, 1.5])
        points = graph.scaled_points(100, 50, padding=5)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0][0], 5)
        self.assertEqual(points[-1][0], 95)
        for x, y in points:
            self.assertGreaterEqual(x, 5)
            self.assertLessEqual(x, 95)
            self.assertGreaterEqual(y, 5)
            self.assertLessEqual(y, 45)

    def test_single_point_does_not_divide_by_zero(self):
        graph = LossWormGraph(history=[3.0])
        points = graph.scaled_points(100, 50, padding=5)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0], (5.0, 45.0))

    def test_empty_series_returns_no_points(self):
        self.assertEqual(LossWormGraph().scaled_points(100, 50), [])

    def test_draw_emits_line_and_markers(self):
        canvas = FakeCanvas()
        graph = LossWormGraph(history=[1.0, 2.0])
        graph.draw(canvas, 100, 50, padding=5)
        self.assertEqual(canvas.deleted, 1)
        self.assertEqual(len(canvas.lines), 1)
        self.assertEqual(len(canvas.ovals), 2)


class RuntimePlacementTest(unittest.TestCase):
    def test_default_probe_is_static(self):
        self.assertEqual(default_runtime_probe()["device"], None)

    def test_gpu_primary_placement(self):
        placement = resolve_placement(
            {"device": "AMD Radeon RX 580", "operations": ["sgemm"], "software_vulkan": False}
        )
        self.assertEqual(placement.mode, "gpu-primary")
        self.assertTrue(placement.gpu_primary)
        self.assertTrue(placement.cpu_fallback)

    def test_software_vulkan_falls_back(self):
        placement = resolve_placement(
            {"device": "llvmpipe", "operations": ["sgemm"], "software_vulkan": True}
        )
        self.assertEqual(placement.mode, "cpu-fallback")
        self.assertFalse(placement.gpu_primary)

    def test_no_device_falls_back(self):
        placement = resolve_placement({"device": None, "operations": [], "software_vulkan": False})
        self.assertEqual(placement.mode, "cpu-fallback")
        self.assertFalse(placement.gpu_primary)

    def test_training_placement_is_cpu_fallback_until_integrated_gpu(self):
        runtime = resolve_placement(
            {"device": "AMD Radeon RX 580", "operations": ["sgemm"], "software_vulkan": False}
        )
        placement = resolve_training_placement(True, runtime)
        self.assertIsInstance(placement, TrainingPlacement)
        self.assertEqual(placement.requested, "gpu-primary")
        self.assertEqual(placement.actual, "cpu-fallback")
        self.assertEqual(placement.hardware, "gpu-primary")

    def test_parse_api_address(self):
        self.assertEqual(parse_api_address("127.0.0.1:8123"), ("127.0.0.1", 8123))
        self.assertEqual(parse_api_address("127.0.0.1:0"), ("127.0.0.1", 0))
        self.assertEqual(parse_api_address("localhost"), ("localhost", 8123))
        for bad in ("0.0.0.0:8123", "127.0.0.1:notaport", "127.0.0.1:99999", ""):
            with self.assertRaises(ValueError):
                parse_api_address(bad)


class RunDesktopProbeRegressionTest(unittest.TestCase):
    def test_backend_probe_factory_returns_callable_mapping_to_gpu_primary(self):
        from unittest import mock

        from spaceslug.desktop.app import runtime_probe_factory

        calls = []

        def fake_backend_probe(runtime_root, runtime_revision):
            calls.append((runtime_root, runtime_revision))
            return {
                "backend": "spaceslug",
                "device": "AMD Radeon RX 580",
                "runtime_revision": runtime_revision,
                "operations": ["sgemm", "tiny_forward_abi"],
                "software_vulkan": False,
            }

        # Keep the patch active through the deferred probe call: the factory only
        # binds the callable, while ``refresh_runtime`` invokes it later.
        with mock.patch("spaceslug.desktop.runtime.backend_runtime_probe", fake_backend_probe):
            probe = runtime_probe_factory("/tmp/fake-runtime-root", "rev-123")

            # The controller contract is a callable, never a pre-computed dict.
            self.assertTrue(callable(probe))
            self.assertNotIsInstance(probe, dict)

            controller = DesktopController(runtime_probe=probe)
            placement = controller.refresh_runtime()

        self.assertEqual(calls, [("/tmp/fake-runtime-root", "rev-123")])

        self.assertEqual(placement.mode, "gpu-primary")
        self.assertTrue(placement.gpu_primary)
        self.assertTrue(placement.cpu_fallback)
        self.assertEqual(controller.snapshot()["placement"]["device"], "AMD Radeon RX 580")


def _display_available() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@unittest.skipUnless(_display_available(), "requires a display")
class DesktopAppTest(unittest.TestCase):
    def test_app_builds_tabs_and_renders_loss(self):
        import tkinter as tk

        from spaceslug.desktop import DesktopApp

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no usable Tk display")
        root.withdraw()
        try:
            app = DesktopApp(controller=DesktopController(), root=root)
            self.assertEqual(len(app.notebook.tabs()), 5)
            app.controller.record_loss(2.0)
            app.controller.record_loss(1.0)
            app.refresh()
            self.assertGreater(len(app._loss_canvas.find_all()), 0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
