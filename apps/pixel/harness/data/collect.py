"""Resumable, parallel, idempotent fetcher. Every completed file is one line in
data/ledger.jsonl; a re-run skips what the ledger already has, resumes `.part` files
with HTTP Range, verifies sha256 when the manifest states one, and records it otherwise."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from . import data_root
from .manifest import FileSpec, Source

USER_AGENT = "harness-data/0.1 (+verifiably-better-than-any-lens)"
CHUNK = 1 << 20
RETRIES = 4
MIN_INTERVAL_PER_HOST_S = 0.05


@dataclass
class LedgerEntry:
    source: str
    path: str
    url: str
    bytes: int
    sha256: str
    status: str  # done | failed | skipped
    fetched_at: str
    error: str = ""


class _HostThrottle:
    def __init__(self, interval: float):
        self.interval, self._last, self._lock = interval, {}, threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            t = time.monotonic()
            wait = self._last.get(host, 0.0) + self.interval - t
            self._last[host] = max(t, self._last.get(host, 0.0) + self.interval)
        if wait > 0:
            time.sleep(wait)


def fs(p: Path | str) -> str:
    """Path as a string the OS will open: Windows needs the \\\\?\\ prefix past 260 chars."""
    s = str(p)
    if os.name == "nt" and len(s) > 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(s)
    return s


def _digest(path: Path | str, algo: str) -> str:
    h = hashlib.new(algo)
    with open(fs(path), "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of(path: Path | str) -> str:
    return _digest(path, "sha256")


class Collector:
    def __init__(self, root: Path | None = None, jobs: int = 8, dry_run: bool = False,
                 max_bytes: float | None = None, log=print):
        self.root = Path(root) if root else data_root()
        self.raw = self.root / "raw"
        self.ledger_path = self.root / "ledger.jsonl"
        self.jobs, self.dry_run, self.max_bytes, self.log = jobs, dry_run, max_bytes, log
        self._ledger_lock = threading.Lock()
        self._budget_lock = threading.Lock()
        self._spent = 0
        self._throttle = _HostThrottle(MIN_INTERVAL_PER_HOST_S)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self.done: dict[tuple[str, str], LedgerEntry] = {}
        self.failed: dict[tuple[str, str], LedgerEntry] = {}  # latest failure with no later success
        self._load_ledger()

    # ---- ledger -------------------------------------------------------------------
    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = LedgerEntry(**json.loads(line))
                self._index(e)

    def _record(self, e: LedgerEntry) -> None:
        with self._ledger_lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(e)) + "\n")
            self._index(e)

    def _index(self, e: LedgerEntry) -> None:
        key = (e.source, e.path)
        if e.status == "done":
            self.done[key] = e
            self.failed.pop(key, None)
        elif e.status == "failed":
            self.failed[key] = e
        else:  # skipped: a later successful listing clears an earlier "<expand>" failure
            self.failed.pop(key, None)

    def note_expand(self, source: Source, error: Exception | None = None) -> None:
        """Ledger a listing (expander) failure as the pseudo-file "<expand>" so gated or dead
        sources show up in `status`/`plan`; with no error, clear an earlier one. No-op in dry runs."""
        if self.dry_run:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if error is not None:
            self._record(LedgerEntry(source.id, "<expand>", source.url, 0, "", "failed", now,
                                     f"{type(error).__name__}: {error}"))
        elif (source.id, "<expand>") in self.failed:
            self._record(LedgerEntry(source.id, "<expand>", source.url, 0, "", "skipped", now))

    def status(self) -> dict[str, dict]:
        """Per source: done files/bytes and files whose last attempt failed (with the first error text)."""
        out: dict[str, dict] = {}
        for (src, _), e in self.done.items():
            d = out.setdefault(src, {"files": 0, "bytes": 0, "failed": 0, "error": ""})
            d["files"] += 1
            d["bytes"] += e.bytes
        for (src, _), e in self.failed.items():
            d = out.setdefault(src, {"files": 0, "bytes": 0, "failed": 0, "error": ""})
            d["failed"] += 1
            d["error"] = d["error"] or e.error
        return out

    # ---- fetching ----------------------------------------------------------------
    def target(self, source: Source, spec: FileSpec) -> Path:
        return self.raw / source.id / spec.path

    def fetch_source(self, source: Source, specs: list[FileSpec] | None = None) -> dict:
        specs = list(specs if specs is not None else source.files)
        todo = [s for s in specs
                if (source.id, s.path) not in self.done or not os.path.exists(fs(self.target(source, s)))]
        summary = {"source": source.id, "requested": len(specs), "already": len(specs) - len(todo),
                   "done": 0, "failed": 0, "bytes": 0, "errors": []}
        if self.dry_run:
            self.log(f"[dry-run] {source.id}: {len(todo)} file(s) to fetch, {summary['already']} present")
            summary["planned"] = [s.url for s in todo]
            return summary
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futs = {pool.submit(self._fetch_one, source, s): s for s in todo}
            for fut in as_completed(futs):
                e = fut.result()
                if e.status == "done":
                    summary["done"] += 1
                    summary["bytes"] += e.bytes
                else:
                    summary["failed"] += 1
                    summary["errors"].append(f"{e.path}: {e.error}")
        self.log(f"{source.id}: {summary['done']} fetched, {summary['already']} present, "
                 f"{summary['failed']} failed, {summary['bytes'] / 1e6:.1f} MB")
        return summary

    def _fetch_one(self, source: Source, spec: FileSpec) -> LedgerEntry:
        dst = self.target(source, spec)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            if spec.size and self.max_bytes is not None:
                with self._budget_lock:
                    if self._spent + spec.size > self.max_bytes:
                        raise RuntimeError("byte budget exhausted")
                    self._spent += spec.size
            os.makedirs(fs(dst.parent), exist_ok=True)
            scheme = urlparse(spec.url).scheme
            if scheme == "file":
                shutil.copyfile(url2pathname(urlparse(spec.url).path), fs(dst))
            elif scheme == "hf":
                self._download_columns(spec, dst)
            else:
                self._download(spec.url, dst)
            digest = sha256_of(dst)
            if spec.sha256 and digest != spec.sha256:
                os.unlink(fs(dst))
                raise RuntimeError(f"sha256 mismatch: got {digest[:12]} want {spec.sha256[:12]}")
            if spec.md5 and (got := _digest(dst, "md5")) != spec.md5:
                os.unlink(fs(dst))
                raise RuntimeError(f"md5 mismatch: got {got[:12]} want {spec.md5[:12]}")
            if spec.sha1 and (got := _digest(dst, "sha1")) != spec.sha1:
                os.unlink(fs(dst))
                raise RuntimeError(f"sha1 mismatch: got {got[:12]} want {spec.sha1[:12]}")
            e = LedgerEntry(source.id, spec.path, spec.url, os.stat(fs(dst)).st_size, digest, "done", now)
        except Exception as ex:  # noqa: BLE001 - every failure is data, not a crash
            e = LedgerEntry(source.id, spec.path, spec.url, 0, "", "failed", now, f"{type(ex).__name__}: {ex}")
        self._record(e)
        return e

    def _download_columns(self, spec: FileSpec, dst: Path) -> None:
        """Column-projected parquet over the Hub filesystem: only the requested columns' chunks travel
        (HTTP range requests, coalesced by pyarrow's pre_buffer), so a shard whose bulk is image bytes
        costs a few MB. Written atomically through a .part file; resume is per shard."""
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        if not spec.columns:
            raise RuntimeError("hf:// spec without columns; use the https shard url for a full download")
        part = fs(dst.with_suffix(dst.suffix + ".part"))
        hub = HfFileSystem(token=os.environ.get("HF_TOKEN"))
        hub_path = spec.url[len("hf://"):]
        self._throttle.wait("huggingface.co")
        for attempt in range(RETRIES):
            try:
                with hub.open(hub_path, "rb", block_size=4 << 20) as f:
                    table = pq.ParquetFile(f, pre_buffer=True).read(columns=list(spec.columns))
                pq.write_table(table, part, compression="zstd")
                break
            except Exception as ex:  # noqa: BLE001 - network and parquet errors alike are retried
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
                del ex
        os.replace(part, fs(dst))

    def _download(self, url: str, dst: Path) -> None:
        part = fs(dst.with_suffix(dst.suffix + ".part"))
        host = urlparse(url).netloc
        for attempt in range(RETRIES):
            have = os.stat(part).st_size if os.path.exists(part) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            self._throttle.wait(host)
            try:
                with self._session.get(url, headers=headers, stream=True, timeout=(15, 120)) as r:
                    if r.status_code == 416:  # range past the end: file was complete
                        break
                    r.raise_for_status()
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(part, mode) as f:
                        for block in r.iter_content(CHUNK):
                            f.write(block)
                break
            except (requests.RequestException, OSError) as ex:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(1.5 * 2 ** attempt)
        os.replace(part, fs(dst))

    # ---- small helpers used by expanders -------------------------------------------
    def get_text(self, url: str, params: dict | None = None, headers: dict | None = None) -> str:
        """HTML/text listing call (public folder views, directory indexes) with the same retry policy."""
        for attempt in range(RETRIES):
            self._throttle.wait(urlparse(url).netloc)
            try:
                r = self._session.get(url, params=params, headers=headers, timeout=(15, 60))
                if r.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                    raise requests.HTTPError(f"{r.status_code} from {urlparse(url).netloc}")
                r.raise_for_status()
                return r.text
            except (requests.RequestException, OSError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(1.5 * 2 ** attempt)

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        """Listing call with the same retry/backoff as downloads (APIs drop connections under bursts)."""
        for attempt in range(RETRIES):
            self._throttle.wait(urlparse(url).netloc)
            try:
                r = self._session.get(url, params=params, headers=headers, timeout=(15, 60))
                if r.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                    raise requests.HTTPError(f"{r.status_code} from {urlparse(url).netloc}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, OSError, ValueError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(1.5 * 2 ** attempt)
