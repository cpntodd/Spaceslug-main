"""Bounded, approval-controlled documentation crawler for native dataset ingestion."""
from __future__ import annotations
import json, time, urllib.error, urllib.parse, urllib.request, urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from .workspace import DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT_SECONDS, IngestionError, WorkspaceService, infer_kind, sha256_bytes, validate_http_url

class CrawlPolicyError(IngestionError): pass

class DocumentationParser(HTMLParser):
    BLOCK={"article","blockquote","br","code","div","h1","h2","h3","h4","h5","h6","li","p","pre","section","td","th","title","tr"}
    IGNORE={"script","style","noscript"}
    def __init__(self):
        super().__init__(convert_charrefs=True); self.links=[]; self.parts=[]; self.title=[]; self._ignore=0; self._tag=""
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag in self.IGNORE: self._ignore+=1
        if tag=="a" and attrs.get("href"): self.links.append((attrs["href"], attrs.get("title", "")))
        if tag=="title": self._tag="title"
        if self._ignore==0 and tag in self.BLOCK: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in self.IGNORE: self._ignore=max(0,self._ignore-1)
        if tag=="title": self._tag=""
        if self._ignore==0 and tag in self.BLOCK: self.parts.append("\n")
    def handle_data(self,data):
        if self._ignore==0:
            self.parts.append(data)
            if self._tag=="title": self.title.append(data)
    def result(self):
        return {"title":" ".join("".join(self.title).split()),"text":"\n".join(x.strip() for x in "".join(self.parts).splitlines() if x.strip()),"links":list(self.links)}

def parse_html(data: bytes) -> dict[str,Any]:
    try: text=data.decode("utf-8")
    except UnicodeDecodeError as exc: raise CrawlPolicyError(f"HTML is not UTF-8: {exc}") from exc
    p=DocumentationParser(); p.feed(text); p.close(); return p.result()

def same_origin(a,b):
    x,y=urllib.parse.urlsplit(a),urllib.parse.urlsplit(b)
    return (x.scheme,x.hostname,x.port or (443 if x.scheme=="https" else 80))==(y.scheme,y.hostname,y.port or (443 if y.scheme=="https" else 80))

def normalize_link(base, href):
    href=href.strip()
    if not href or href.startswith(("#","mailto:","javascript:","data:")): return None
    return validate_http_url(urllib.parse.urljoin(base,href).split("#",1)[0])

@dataclass
class CrawlSession:
    seed: str; policy: dict[str,Any]; queue: list[dict[str,Any]]=field(default_factory=list); fetched:list[dict[str,Any]]=field(default_factory=list); rejected:list[dict[str,Any]]=field(default_factory=list); errors:list[dict[str,Any]]=field(default_factory=list)
    def save(self,path): Path(path).write_text(json.dumps(asdict(self),sort_keys=True,indent=2)+"\n")
    @classmethod
    def load(cls,path): return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

class DocumentationCrawler:
    def __init__(self, workspace: WorkspaceService, *, max_pages=10,max_depth=1,max_page_bytes=DEFAULT_MAX_BYTES,max_total_bytes=64*1024*1024,delay=0.0,same_origin_only=True,respect_robots=True,opener=None):
        self.workspace=workspace; self.max_pages=max_pages; self.max_depth=max_depth; self.max_page_bytes=max_page_bytes; self.max_total_bytes=max_total_bytes; self.delay=delay; self.same_origin_only=same_origin_only; self.respect_robots=respect_robots; self.opener=opener or urllib.request.urlopen
    def start(self, seed, *, approve=lambda url: False, session_path=None):
        seed=validate_http_url(seed); session=CrawlSession(seed, {"max_pages":self.max_pages,"max_depth":self.max_depth,"same_origin_only":self.same_origin_only,"respect_robots":self.respect_robots}, [{"url":seed,"depth":0,"parent":None}]); return self.run(session,approve=approve,session_path=session_path)
    def run(self,session,*,approve,session_path=None):
        seen={x["url"] for x in session.fetched}|{x["url"] for x in session.rejected}; total=sum(x.get("bytes",0) for x in session.fetched)
        while session.queue and len(session.fetched)<self.max_pages:
            item=session.queue.pop(0); url=item["url"]; depth=item["depth"]
            if url in seen: continue
            seen.add(url)
            if not approve(url): session.rejected.append({**item,"reason":"approval required"}); continue
            if depth>self.max_depth or (self.same_origin_only and not same_origin(session.seed,url)): session.rejected.append({**item,"reason":"crawl policy"}); continue
            try:
                if self.respect_robots:
                    rp=urllib.robotparser.RobotFileParser(); rp.set_url(urllib.parse.urljoin(url,"/robots.txt")); rp.read()
                    if not rp.can_fetch("spaceslug-docs",url): session.rejected.append({**item,"reason":"robots.txt"}); continue
                if self.delay: time.sleep(self.delay)
                req=urllib.request.Request(url,headers={"User-Agent":"spaceslug-docs/1.0"})
                with self.opener(req,timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                    data=response.read(self.max_page_bytes+1); status=getattr(response,"status",None); ctype=response.headers.get("Content-Type")
                if len(data)>self.max_page_bytes or total+len(data)>self.max_total_bytes: raise CrawlPolicyError("byte limit exceeded")
                total+=len(data); kind=infer_kind(url,ctype); sha=sha256_bytes(data)
                if kind=="html":
                    parsed=parse_html(data); source=self.workspace.import_bytes(data,source=url,kind="html",content_type=ctype,retrieval="http",fetched_at=datetime.now(UTC).isoformat(),status=status); links=parsed["links"]
                    for href,title in links:
                        child=normalize_link(url,href)
                        if child and child not in seen: session.queue.append({"url":child,"depth":depth+1,"parent":url,"title":title})
                else:
                    source=self.workspace.import_bytes(data,source=url,kind=kind,content_type=ctype,retrieval="http",fetched_at=datetime.now(UTC).isoformat(),status=status); parsed={"title":"","links":[]}
                session.fetched.append({**item,"url":url,"bytes":len(data),"sha256":sha,"kind":kind,"status":status,"title":parsed.get("title","")})
            except Exception as exc: session.errors.append({**item,"error":f"{type(exc).__name__}: {exc}"})
            if session_path: session.save(session_path)
        if session_path: session.save(session_path)
        return session
