"""Tar shards in the WebDataset layout: ``<key>.<ext>`` next to ``<key>.json``.

Shards are the unit of everything downstream (labelling, training, dedup
passes): a worker takes one shard, streams it, and never needs an index.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

from .collect import Store

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"}


def write_shards(store: Store, out_dir: str | Path, shard_bytes: int = 1 << 30, prefix: str = "shard", exts: set[str] = IMAGE_EXTS) -> list[Path]:
    """Pack stored image files into tar shards of about ``shard_bytes``. Returns the shard paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tar: tarfile.TarFile | None = None
    size = 0

    def _open():
        p = out / f"{prefix}-{len(paths):06d}.tar"
        paths.append(p)
        return tarfile.open(p, "w")

    for row in store.rows():
        if row["ext"] not in exts:
            continue
        src = store.path_for(row["sha256"], row["ext"])
        if not src.exists():
            continue
        if tar is None or size >= shard_bytes:
            if tar is not None:
                tar.close()
            tar = _open()
            size = 0
        key = row["sha256"]
        data = src.read_bytes()
        meta = json.dumps(row, sort_keys=True).encode()
        for name, blob in ((f"{key}.{row['ext']}", data), (f"{key}.json", meta)):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
        size += len(data)
    if tar is not None:
        tar.close()
    return paths


def read_shard(path: str | Path) -> Iterator[tuple[str, str, bytes, dict]]:
    """Yield (key, ext, image bytes, meta) per sample, pairing files by key."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            blob = tar.extractfile(member).read()
            slot = pending.setdefault(key, {})
            if ext == "json":
                slot["meta"] = json.loads(blob.decode("utf-8"))
            else:
                slot["ext"] = ext
                slot["data"] = blob
            if "meta" in slot and "data" in slot:
                yield key, slot["ext"], slot["data"], slot["meta"]
                del pending[key]


def list_shards(dir_: str | Path, prefix: str = "shard") -> list[Path]:
    return sorted(Path(dir_).glob(f"{prefix}-*.tar"))
