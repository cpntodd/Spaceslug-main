import http.server
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from spaceslug.dataset import verify_bundle
from spaceslug.workspace import (
    ContentTooLargeError,
    FetchNotApprovedError,
    FetchTimeoutError,
    IngestionError,
    InvalidURLError,
    LicenseRequiredError,
    SearchResult,
    UnsupportedKindError,
    WorkspaceService,
    extract_text,
    read_local_file,
    searxng_search,
    sha256_bytes,
    validate_http_url,
)


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}
    requests: list = []

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

    def route(self, path, handler):
        self.handler_class.routes[path] = handler

    def serve(self, path, body, content_type="text/plain", status=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")

        def responder(handler):
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            try:
                handler.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        self.route(path, responder)

    def url(self, path):
        return self.base + path

    @property
    def requests(self):
        return list(self.handler_class.requests)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class WorkspaceServiceTest(unittest.TestCase):
    def setUp(self):
        self.server = LocalServer()

    def tearDown(self):
        self.server.close()

    def test_local_txt_import_and_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "notes.txt"
            src.write_text("hello world\n", encoding="utf-8")
            ws = WorkspaceService(root / "ws")
            imported = ws.import_local(src, license="CC-BY-4.0")
            self.assertEqual(imported.sha256, sha256_bytes(src.read_bytes()))
            self.assertEqual(imported.retrieval, "local")
            self.assertEqual(imported.records[0]["text"], "hello world\n")
            self.assertEqual(imported.records[0]["kind"], "text")

            bundle = ws.create_dataset("local-ds", [imported])
            verified = verify_bundle(bundle.root)
            self.assertEqual(verified.stats()["train"], 1)
            self.assertEqual(verified.manifest["provenance"]["licenses"], ["CC-BY-4.0"])
            self.assertEqual(verified.manifest["provenance"]["sources"], [str(src.resolve())])

    def test_md_import_records_markdown_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "guide.md"
            src.write_text("# Title\n\nBody text.\n", encoding="utf-8")
            imported = WorkspaceService(root / "ws").import_local(src, license="MIT")
            self.assertEqual(imported.kind, "md")
            self.assertEqual(imported.records[0]["kind"], "markdown")
            self.assertEqual(imported.records[0]["text"], "# Title\n\nBody text.\n")

    def test_jsonl_import_extracts_text_and_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "samples.jsonl"
            lines = [
                {"record_id": "ignored", "text": "first line"},
                {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
            ]
            src.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
            imported = WorkspaceService(root / "ws").import_local(src, license="CC0")
            self.assertEqual(imported.kind, "jsonl")
            self.assertEqual(len(imported.records), 2)
            self.assertEqual(imported.records[0]["text"], "first line")
            self.assertEqual(imported.records[1]["text"], "user: hello\nassistant: hi")

    def test_unsupported_extension_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "script.py"
            src.write_text("print('hi')\n", encoding="utf-8")
            with self.assertRaises(UnsupportedKindError):
                WorkspaceService(root / "ws").import_local(src, license="MIT")

    def test_license_confirmation_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "notes.txt"
            src.write_text("data\n", encoding="utf-8")
            ws = WorkspaceService(root / "ws")
            with self.assertRaises(LicenseRequiredError):
                ws.import_local(src, license="   ")

    def test_local_file_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "big.txt"
            src.write_bytes(b"x" * 2048)
            with self.assertRaises(ContentTooLargeError):
                read_local_file(src, max_bytes=1024)

    def test_fetch_requires_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            with self.assertRaises(FetchNotApprovedError):
                ws.import_url(self.server.url("/doc.txt"), license="MIT", approved=False)

    def test_approved_fetch_over_local_http(self):
        self.server.serve("/doc.txt", "fetched text\n")
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            imported = ws.import_url(self.server.url("/doc.txt"), license="MIT", approved=True)
            self.assertEqual(imported.retrieval, "http")
            self.assertEqual(imported.sha256, sha256_bytes(b"fetched text\n"))
            self.assertEqual(imported.records[0]["text"], "fetched text\n")
            self.assertIsNotNone(imported.fetched_at)

    def test_fetch_size_limit(self):
        self.server.serve("/big.txt", b"x" * 64 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            with self.assertRaises(ContentTooLargeError):
                ws.import_url(self.server.url("/big.txt"), license="MIT", approved=True, max_bytes=1024)

    def test_fetch_time_limit(self):
        def slow(handler):
            time.sleep(0.5)
            body = b"late"
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            try:
                handler.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        self.server.route("/slow.txt", slow)
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            with self.assertRaises(FetchTimeoutError):
                ws.import_url(self.server.url("/slow.txt"), license="MIT", approved=True, timeout=0.05)

    def test_url_validation(self):
        for bad in ("file:///etc/passwd", "ftp://example.com/x", "http://", "http://user:pass@example.com/x"):
            with self.assertRaises(InvalidURLError):
                validate_http_url(bad)
        self.assertEqual(validate_http_url("https://example.com/a"), "https://example.com/a")

    def test_non_http_scheme_import_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            with self.assertRaises(InvalidURLError):
                ws.import_url("file:///etc/passwd", license="MIT", approved=True)

    def test_searxng_query_returns_validated_results_without_fetching(self):
        payload = {
            "results": [
                {"url": self.server.url("/never-fetched"), "title": "Valid", "content": "desc"},
                {"url": "ftp://example.com/file", "title": "Invalid scheme", "content": "skip"},
                {"url": "not-a-url", "title": "Invalid", "content": "skip"},
                {"title": "Missing url", "content": "skip"},
            ]
        }
        self.server.serve("/search?q=test&format=json", json.dumps(payload), content_type="application/json")
        results = searxng_search("test", base_url=self.server.base)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], SearchResult(self.server.url("/never-fetched"), "Valid", "desc"))
        # The search endpoint was hit, but no result document was fetched.
        self.assertIn("/search?q=test&format=json", self.server.requests)
        self.assertNotIn("/never-fetched", self.server.requests)

    def test_selected_search_result_goes_through_approved_import(self):
        target = self.server.url("/selected.txt")
        payload = {"results": [{"url": target, "title": "Selected", "content": "pick me"}]}
        self.server.serve("/search?q=needle&format=json", json.dumps(payload), content_type="application/json")
        self.server.serve("/selected.txt", "selected body\n")
        with tempfile.TemporaryDirectory() as directory:
            ws = WorkspaceService(Path(directory) / "ws")
            results = searxng_search("needle", base_url=self.server.base)
            self.assertEqual(results[0].url, target)
            # Selecting the URL still requires an explicit approved import.
            with self.assertRaises(FetchNotApprovedError):
                ws.import_url(results[0].url, license="MIT", approved=False)
            imported = ws.import_url(results[0].url, license="MIT", approved=True)
            self.assertEqual(imported.records[0]["text"], "selected body\n")

    def test_deterministic_bundle_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "notes.txt"
            src.write_text("deterministic\n", encoding="utf-8")
            first = WorkspaceService(root / "ws-a").import_local(src, license="CC-BY-4.0")
            second = WorkspaceService(root / "ws-b").import_local(src, license="CC-BY-4.0")
            bundle_a = WorkspaceService(root / "ws-a").create_dataset("det-ds", [first])
            bundle_b = WorkspaceService(root / "ws-b").create_dataset("det-ds", [second])
            self.assertEqual(bundle_a.manifest["revision"], bundle_b.manifest["revision"])
            files_a = sorted(p.relative_to(bundle_a.root) for p in bundle_a.root.rglob("*") if p.is_file())
            files_b = sorted(p.relative_to(bundle_b.root) for p in bundle_b.root.rglob("*") if p.is_file())
            self.assertEqual(files_a, files_b)
            for rel in files_a:
                self.assertEqual((bundle_a.root / rel).read_bytes(), (bundle_b.root / rel).read_bytes())

    def test_import_bytes_round_trip_via_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ws = WorkspaceService(root / "ws")
            ws.import_bytes(b"line\n", source="http://example.com/x", kind="txt", content_type="text/plain", license="MIT", retrieval="http")
            imports = ws.list_imports()
            self.assertEqual(len(imports), 1)
            self.assertEqual(imports[0].records[0]["text"], "line\n")

    def test_extract_text_unknown_kind(self):
        with self.assertRaises(UnsupportedKindError):
            extract_text(b"x", "binary", source_id="sha")


if __name__ == "__main__":
    unittest.main()
