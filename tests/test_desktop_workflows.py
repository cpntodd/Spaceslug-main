import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import ClassVar

from spaceslug.dataset import verify_bundle
from spaceslug.desktop import DesktopController
from spaceslug.openai_api import ModelResponder, ResponderResult
from spaceslug.workspace import IngestionError, LicenseRequiredError


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: ClassVar[dict] = {}
    requests: ClassVar[list] = []

    def do_GET(self):
        self.__class__.requests.append(self.path)
        handler = self.routes.get(self.path)
        if handler is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        handler(self)

    def log_message(self, *args, **kwargs):
        pass


class LocalServer:
    def __init__(self):
        handler_class = type("Handler", (_Handler,), {"routes": {}, "requests": []})
        self.handler_class = handler_class
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    def serve(self, path, body, content_type="text/plain"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")

        def responder(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            try:
                handler.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        self.handler_class.routes[path] = responder

    def url(self, path):
        return self.base + path

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class _FixedResponder(ModelResponder):
    backend_id = "test-backend"
    model_id = "test-model"

    def respond(self, messages):
        return ResponderResult(
            content=f"echo:{messages[-1]['content']}",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


class DesktopDatasetWorkflowTest(unittest.TestCase):
    def test_local_import_create_dts_and_training_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "notes.txt"
            src.write_text("hello world\nsecond line\n", encoding="utf-8")
            controller = DesktopController(workspace_root=root / "ws")
            controller.set_dataset_license("CC-BY-4.0")

            imported = controller.import_local_source(src)
            self.assertEqual(imported.records[0]["text"], "hello world\nsecond line\n")
            self.assertEqual(len(controller.imports), 1)

            bundle = controller.create_dataset("my-ds")
            verified = verify_bundle(bundle.root)
            self.assertEqual(verified.stats()["train"], 1)
            self.assertEqual(verified.manifest["provenance"]["licenses"], ["CC-BY-4.0"])
            self.assertEqual(controller.dataset_revision, verified.manifest["revision"])

            training_bundle = controller.build_training_bundle()
            self.assertEqual(training_bundle.stats()["train"], 1)
            self.assertEqual(training_bundle.manifest["preprocessing"]["pipeline"], "spaceslug-prompt-target")
            self.assertTrue(controller.training_bundle_path.endswith(".dts"))

    def test_license_required_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "notes.txt"
            src.write_text("data\n", encoding="utf-8")
            controller = DesktopController(workspace_root=root / "ws")
            with self.assertRaises(LicenseRequiredError):
                controller.import_local_source(src)

    def test_create_dataset_requires_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            with self.assertRaises(IngestionError):
                controller.create_dataset("empty-ds")

    def test_build_training_bundle_requires_created_bundle(self):
        controller = DesktopController()
        with self.assertRaises(IngestionError):
            controller.build_training_bundle()

    def test_choose_local_sources_includes_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.md").write_text("hello", encoding="utf-8")
            (root / "paper.pdf").write_bytes(b"%PDF-1.4 placeholder")
            (root / "skip.py").write_text("pass", encoding="utf-8")
            selection = DesktopController().choose_local_sources(root)
            self.assertEqual([path.name for path in selection.files], ["ok.md", "paper.pdf"])
            self.assertEqual([path.name for path in selection.skipped], ["skip.py"])


class DesktopNetworkWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.server = LocalServer()

    def tearDown(self):
        self.server.close()

    def test_approved_url_import(self):
        self.server.serve("/doc.txt", "fetched body\n")
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            controller.set_dataset_license("MIT")
            imported = controller.import_approved_url(self.server.url("/doc.txt"))
            self.assertEqual(imported.retrieval, "http")
            self.assertEqual(imported.records[0]["text"], "fetched body\n")

    def test_search_select_and_import_selected(self):
        target = self.server.url("/selected.txt")
        payload = {"results": [{"url": target, "title": "Selected", "content": "pick me"}]}
        self.server.serve("/search?q=needle&format=json", json.dumps(payload), content_type="application/json")
        self.server.serve("/selected.txt", "selected body\n")
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            controller.set_dataset_license("MIT")
            results = controller.run_search("needle", base_url=self.server.base)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].url, target)
            selected = controller.select_search_result(0)
            self.assertEqual(selected.title, "Selected")
            imported = controller.import_selected_search_result()
            self.assertEqual(imported.records[0]["text"], "selected body\n")

    def test_search_requires_selection_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            with self.assertRaises(IngestionError):
                controller.import_selected_search_result()


class DesktopTrainingWorkflowTest(unittest.TestCase):
    def _trained_controller(self, directory):
        root = Path(directory)
        src = root / "notes.txt"
        src.write_text("hello world\nsecond line\n", encoding="utf-8")
        controller = DesktopController(workspace_root=root / "ws", code_revision="test-rev")
        controller.set_dataset_license("CC-BY-4.0")
        controller.import_local_source(src)
        controller.create_dataset("train-ds")
        controller.build_training_bundle()
        controller.set_training_steps(3)
        return controller

    def test_training_runs_in_background_with_live_loss_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self._trained_controller(directory)
            status = controller.start_training()
            self.assertEqual(status["state"], "running")
            self.assertIsNotNone(controller._training_thread)
            final = controller.wait_training(timeout=30)
            self.assertEqual(final["state"], "finished")
            self.assertIsNone(final["error"])
            self.assertEqual(final["completed_steps"], 3)
            self.assertEqual(len(final["loss_history"]), 3)
            paths = final["paths"]
            self.assertTrue(Path(paths["checkpoint"]).is_file())
            self.assertTrue((Path(paths["artifact"]) / "manifest.json").is_file())
            self.assertTrue(Path(paths["experiment"]).is_dir())
            self.assertTrue(final["result"]["artifact_revision"].startswith("sha256:"))
            self.assertEqual(final["result"]["stopped_reason"], "steps")

    def test_training_records_cpu_placement_not_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self._trained_controller(directory)
            controller.set_gpu_primary_requested(True)
            controller.start_training()
            final = controller.wait_training(timeout=30)
            placement = final["placement"]
            self.assertEqual(placement["requested"], "gpu-primary")
            self.assertEqual(placement["actual"], "cpu-fallback")

    def test_training_rejects_second_start_while_running(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self._trained_controller(directory)
            controller.set_training_steps(100000)
            controller.start_training()
            try:
                with self.assertRaises(RuntimeError):
                    controller.start_training()
            finally:
                controller.stop_training()
                controller.wait_training(timeout=30)

    def test_training_reports_error_on_missing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DesktopController(workspace_root=Path(directory) / "ws")
            controller.start_training(bundle_path=Path(directory) / "missing.dts")
            final = controller.wait_training(timeout=30)
            self.assertEqual(final["state"], "error")
            self.assertIsNotNone(final["error"])

    def test_stop_training_is_idempotent_when_idle(self):
        controller = DesktopController()
        self.assertFalse(controller.stop_training())


class DesktopChatApiWorkflowTest(unittest.TestCase):
    def test_chat_uses_injected_responder(self):
        controller = DesktopController(responder=_FixedResponder())
        reply = controller.send_chat("hello")
        self.assertEqual(reply, "echo:hello")
        self.assertEqual(controller.chat_history[-1], ("assistant", "echo:hello"))
        identity = controller.responder_identity()
        self.assertEqual(identity["backend"], "test-backend")
        self.assertEqual(identity["model"], "test-model")

    def test_chat_uses_default_responder_and_reports_identity(self):
        controller = DesktopController()
        reply = controller.send_chat("hi")
        self.assertIn("cpu-tiny", reply)
        self.assertIn("spaceslug-tiny-attention-v1", reply)

    def test_api_start_stop_lifecycle(self):
        controller = DesktopController()
        status = controller.start_api("127.0.0.1:0")
        self.assertTrue(status["running"])
        self.assertTrue(status["base_url"].startswith("http://127.0.0.1:"))
        self.assertEqual(status["backend"], "cpu-tiny")
        controller.stop_api()
        self.assertFalse(controller.api_status()["running"])

    def test_api_start_twice_raises(self):
        controller = DesktopController()
        controller.start_api("127.0.0.1:0")
        try:
            with self.assertRaises(RuntimeError):
                controller.start_api("127.0.0.1:0")
        finally:
            controller.stop_api()

    def test_api_rejects_non_loopback_address(self):
        controller = DesktopController()
        with self.assertRaises(ValueError):
            controller.start_api("0.0.0.0:8123")


if __name__ == "__main__":
    unittest.main()
