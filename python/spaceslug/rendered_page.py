"""Optional isolated JavaScript page-renderer adapter (explicit opt-in only)."""
from __future__ import annotations
import os, subprocess
from dataclasses import dataclass
from pathlib import Path
from .workspace import InvalidURLError, validate_http_url

@dataclass(frozen=True)
class RenderResult:
    status: str
    text: str = ""
    error: str | None = None

class RenderUnavailable(RuntimeError): pass

class SubprocessPageRenderer:
    """Run a user-configured renderer with no inherited credentials or cookies.

    The executable must accept ``--url URL --output PATH`` and write extracted
    UTF-8 text. It is never used implicitly by the static crawler.
    """
    def __init__(self, executable: str | Path, *, timeout: float = 30, max_output_bytes: int = 16*1024*1024, allowed_origins=()):
        self.executable = str(executable); self.timeout = timeout; self.max_output_bytes = max_output_bytes
        self.allowed_origins = tuple(allowed_origins)
    def capability(self) -> dict:
        exists = Path(self.executable).is_file()
        return {"enabled": False, "configured": True, "available": exists, "status": "available-opt-in" if exists else "unavailable", "reason": "explicit opt-in required; no inherited cookies or credentials" if exists else "renderer executable is missing"}
    def render(self, url: str) -> RenderResult:
        try: url = validate_http_url(url)
        except InvalidURLError as exc: return RenderResult("rejected", error=str(exc))
        origin = url.split("/", 3)[:3]
        normalized = "/".join(origin)
        if self.allowed_origins and normalized not in self.allowed_origins:
            return RenderResult("rejected", error="URL origin is not allowlisted")
        if not Path(self.executable).is_file(): return RenderResult("unavailable", error="renderer executable is missing")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"rendered.txt"
            env={"PATH": os.environ.get("PATH", ""), "LANG":"C.UTF-8", "HOME":td}
            try:
                completed=subprocess.run([self.executable,"--url",url,"--output",str(out)], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired: return RenderResult("error", error="renderer timed out")
            if completed.returncode: return RenderResult("error", error=f"renderer exited {completed.returncode}")
            if not out.is_file(): return RenderResult("error", error="renderer produced no output")
            data=out.read_bytes()
            if len(data)>self.max_output_bytes: return RenderResult("error", error="renderer output exceeds byte limit")
            try: return RenderResult("ok", text=data.decode("utf-8"))
            except UnicodeDecodeError: return RenderResult("error", error="renderer output is not UTF-8")
