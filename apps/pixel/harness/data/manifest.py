"""The source registry. One entry per external source; `files` are fixed URLs, `expander`
names an API walker that generates the file list at collect time (Poly Haven, HF rows)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("sources.json")

KINDS = ("reference", "dataset", "api", "generator", "spec", "package")
ACCESS = ("direct", "api", "hf", "pip", "docker", "manual", "page-only", "scrape-tos-unclear")
TIERS = ("S", "M", "L", "XL")  # <50 MB, <2 GB, <50 GB, beyond
TIER_BYTES = {"S": 50e6, "M": 2e9, "L": 50e9, "XL": float("inf")}
FETCHABLE = ("direct", "api")


@dataclass
class FileSpec:
    url: str
    path: str  # relative to data/raw/<source id>/
    sha256: str | None = None
    size: int | None = None
    md5: str | None = None  # some hosts (Zenodo) publish md5 only
    sha1: str | None = None  # Wikimedia Commons publishes sha1 only
    columns: list[str] | None = None  # hf:// parquet only: fetch these columns, skip the rest (image bytes)

    def to_dict(self) -> dict:
        d = {"url": self.url, "path": self.path}
        if self.columns:
            d["columns"] = list(self.columns)
        if self.sha256:
            d["sha256"] = self.sha256
        if self.md5:
            d["md5"] = self.md5
        if self.sha1:
            d["sha1"] = self.sha1
        if self.size is not None:
            d["size"] = self.size
        return d


@dataclass
class Source:
    id: str
    name: str
    kind: str
    access: str
    url: str  # landing page
    license: str = "unknown"
    size_bytes: float = 0.0
    gives: str = ""
    harness_use: str = ""
    files: list[FileSpec] = field(default_factory=list)
    expander: str | None = None
    params: dict = field(default_factory=dict)
    deriver: str | None = None
    requires: list[str] = field(default_factory=list)
    notes: str = ""
    clean: dict = field(default_factory=dict)  # how the labeller and trainer read it as a clean source:
    #   scale (box-downsample factor), low_quality_scale (for JPEGs below quality 90), exclude (fnmatch patterns)

    @property
    def tier(self) -> str:
        for t in TIERS:
            if self.size_bytes < TIER_BYTES[t]:
                return t
        return "XL"

    @property
    def fetchable(self) -> bool:
        return self.access in FETCHABLE and (bool(self.files) or self.expander is not None)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        d = dict(d)
        d["files"] = [FileSpec(**f) for f in d.get("files", [])]
        return cls(**d)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, "", [], {}, 0.0)}
        d["files"] = [f.to_dict() for f in self.files]
        if not d["files"]:
            del d["files"]
        return d


def validate(sources: list[Source]) -> None:
    seen = set()
    for s in sources:
        if s.id in seen:
            raise ValueError(f"duplicate source id {s.id}")
        seen.add(s.id)
        if s.kind not in KINDS:
            raise ValueError(f"{s.id}: unknown kind {s.kind}")
        if s.access not in ACCESS:
            raise ValueError(f"{s.id}: unknown access {s.access}")
        if s.access in FETCHABLE and not (s.files or s.expander):
            raise ValueError(f"{s.id}: access {s.access} but no files and no expander")
    for s in sources:
        for r in s.requires:
            if r not in seen:
                raise ValueError(f"{s.id} requires unknown source {r}")


def load_manifest(path: Path | str = MANIFEST_PATH) -> list[Source]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = [Source.from_dict(d) for d in raw["sources"]]
    validate(sources)
    return sources


def save_manifest(sources: list[Source], path: Path | str = MANIFEST_PATH) -> None:
    validate(sources)
    Path(path).write_text(json.dumps({"sources": [s.to_dict() for s in sources]}, indent=2),
                          encoding="utf-8")


def by_id(sources: list[Source]) -> dict[str, Source]:
    return {s.id: s for s in sources}
