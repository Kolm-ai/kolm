"""Content-addressed, verified, resumable, parallel collector.

* Every file lands at ``store/<sha256[:2]>/<sha256[2:4]>/<sha256>.<ext>``; a
  second download of the same bytes is a no-op, whatever its URL.
* Downloads stream to a temp file, hash on the fly, verify against the
  publisher's sha256/md5 when one exists, then rename atomically.
* ``manifest.jsonl`` is append-only, one line per stored item, and is the only
  index. Rerunning resumes: items already in the manifest are skipped.
* Licenses are checked per item before any bytes move.
* ``max_bytes`` caps the run so a 4 TB disk is a budget, not a surprise.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .sources import Item, USER_AGENT, expand, license_ok, load_registry


@dataclass
class CollectStats:
    listed: int = 0
    skipped_license: int = 0
    skipped_existing: int = 0
    downloaded: int = 0
    failed: int = 0
    bytes: int = 0
    budget_stop: bool = False
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "errors"} | {"errors": self.errors[:20]}


def _ext(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    return ext if ext and len(ext) <= 8 and ext.isalnum() else "bin"


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self._lock = threading.Lock()
        self._known_urls: set[str] = set()
        self._known_sha: set[str] = set()
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._known_urls.add(row["url"])
                        self._known_sha.add(row["sha256"])

    def path_for(self, sha256: str, ext: str) -> Path:
        return self.root / sha256[:2] / sha256[2:4] / f"{sha256}.{ext}"

    def has_url(self, url: str) -> bool:
        return url in self._known_urls

    def has_sha(self, sha256: str) -> bool:
        return sha256 in self._known_sha

    def record(self, row: dict) -> None:
        with self._lock:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            self._known_urls.add(row["url"])
            self._known_sha.add(row["sha256"])

    def rows(self):
        if not self.manifest_path.exists():
            return
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> dict:
        """Re-hash every stored file against the manifest. Returns counts and the bad paths."""
        ok, missing, corrupt = 0, [], []
        for row in self.rows():
            p = self.path_for(row["sha256"], row["ext"])
            if not p.exists():
                missing.append(str(p))
                continue
            if _sha256_file(p) != row["sha256"]:
                corrupt.append(str(p))
                continue
            ok += 1
        return {"ok": ok, "missing": missing, "corrupt": corrupt}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(session, item: Item, store: Store, retries: int = 4, chunk: int = 1 << 20) -> dict:
    """Stream one item into the store, verifying as it goes. Returns the manifest row."""
    ext = _ext(item.filename)
    tmp = store.root / "tmp" / f"{threading.get_ident()}_{abs(hash(item.url))}.part"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
            with session.get(item.url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk_size=chunk):
                        if not block:
                            continue
                        f.write(block)
                        sha.update(block)
                        md5.update(block)
                        size += len(block)
            digest = sha.hexdigest()
            if item.sha256 and item.sha256.lower() != digest:
                raise ValueError(f"sha256 mismatch for {item.url}: expected {item.sha256}, got {digest}")
            if item.md5 and item.md5.lower() != md5.hexdigest():
                raise ValueError(f"md5 mismatch for {item.url}")
            if item.bytes and int(item.bytes) != size:
                raise ValueError(f"size mismatch for {item.url}: expected {item.bytes}, got {size}")
            final = store.path_for(digest, ext)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                tmp.unlink(missing_ok=True)
            else:
                os.replace(tmp, final)
            return {
                "sha256": digest,
                "md5": md5.hexdigest(),
                "bytes": size,
                "ext": ext,
                "url": item.url,
                "source_id": item.source_id,
                "license": item.license,
                "filename": item.filename,
                "provenance": item.provenance,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duplicate_of_existing": final.exists() and store.has_sha(digest),
            }
        except Exception as e:  # noqa: BLE001 - retry anything transient, surface the last one
            last_err = e
            tmp.unlink(missing_ok=True)
            if isinstance(e, ValueError):
                break
            time.sleep(min(2.0**attempt, 16.0))
    raise RuntimeError(f"download failed after {retries} attempts: {item.url}: {last_err}")


def collect(
    registry_path: str | Path,
    store_dir: str | Path,
    workers: int = 8,
    max_bytes: int | None = None,
    max_items: int | None = None,
    dry_run: bool = False,
    source_ids: list[str] | None = None,
    session=None,
) -> CollectStats:
    import requests  # local import: the harness itself stays numpy+pillow only

    session = session or requests.Session()
    store = Store(store_dir)
    stats = CollectStats()
    entries = load_registry(registry_path)
    if source_ids:
        entries = [e for e in entries if e["id"] in set(source_ids)]
    budget = max_bytes if max_bytes is not None else float("inf")
    budget_lock = threading.Lock()

    def _work(item: Item) -> dict:
        row = download(session, item, store, retries=4)
        store.record(row)
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {}
        for entry in entries:
            for item in expand(entry, session):
                stats.listed += 1
                if not license_ok(item.license):
                    stats.skipped_license += 1
                    continue
                if store.has_url(item.url):
                    stats.skipped_existing += 1
                    continue
                if max_items is not None and stats.downloaded + len(pending) >= max_items:
                    break
                if item.bytes is not None and stats.bytes + int(item.bytes) > budget:
                    stats.budget_stop = True
                    break
                if dry_run:
                    stats.downloaded += 1
                    stats.bytes += int(item.bytes or 0)
                    continue
                pending[pool.submit(_work, item)] = item
                # keep the in-flight set bounded so expansion can't outrun disk accounting
                if len(pending) >= workers * 4:
                    for fut in as_completed(list(pending)):
                        _finish(fut, pending.pop(fut), stats)
                        if len(pending) < workers * 2:
                            break
                with budget_lock:
                    if stats.bytes > budget:
                        stats.budget_stop = True
                        break
            if stats.budget_stop or (max_items is not None and stats.downloaded + len(pending) >= max_items):
                break
        for fut in as_completed(list(pending)):
            _finish(fut, pending.pop(fut), stats)
    return stats


def _finish(fut, item: Item, stats: CollectStats) -> None:
    try:
        row = fut.result()
        stats.downloaded += 1
        stats.bytes += int(row["bytes"])
    except Exception as e:  # noqa: BLE001
        stats.failed += 1
        stats.errors.append(f"{item.url}: {e}")
