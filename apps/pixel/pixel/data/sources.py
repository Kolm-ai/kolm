"""Source registry and expanders.

A registry entry names a *kind* and the parameters that kind needs. An expander
turns one entry into a stream of ``Item``s (one per file) with a URL, a license,
provenance and, where the host publishes one, an expected hash. Network calls
happen only inside expanders and only when ``collect`` runs.

Licensing is per item, never per registry entry, because hosts like Wikimedia
mix licenses inside one category. Anything that does not match the allowlist
is skipped and counted, not downloaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

# CC0, CC BY, CC BY-SA, public domain, permissive software licenses. NC and ND variants are refused.
LICENSE_ALLOW = re.compile(r"^(cc0|cc[ -]?by(?![ -]?n[cd])(?:[ -]?sa)?(?![ -]?n[cd])|public domain|pdm\b|pd\b|apache|mit\b|kodak)", re.I)

HF_API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
POLYHAVEN_ASSETS = "https://api.polyhaven.com/assets?t=hdris"
POLYHAVEN_FILES = "https://api.polyhaven.com/files/{id}"
AMBIENTCG = "https://ambientcg.com/api/v2/full_json?type={type}&include=downloadData&limit={limit}&offset={offset}"
ZENODO = "https://zenodo.org/api/records/{record}"
WIKIMEDIA = "https://commons.wikimedia.org/w/api.php"

USER_AGENT = "kolm-pixel/0.1 (data collector; dev@kolm.ai)"


@dataclass
class Item:
    source_id: str
    url: str
    license: str
    filename: str
    bytes: int | None = None
    sha256: str | None = None
    md5: str | None = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def license_ok(name: str | None, allow=LICENSE_ALLOW) -> bool:
    return bool(name) and bool(allow.match(name.strip()))


def load_registry(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["sources"] if isinstance(data, dict) else data
    return [e for e in entries if e.get("enabled", True)]


def _get_json(session, url: str, **kw):
    r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, **kw)
    r.raise_for_status()
    return r.json()


# -- expanders -----------------------------------------------------------------
def expand_http(entry: dict, session) -> Iterator[Item]:
    """Direct URLs. ``url`` or ``url_template`` + ``range`` [start, stop]."""
    lic = entry.get("license", "")
    if "url_template" in entry:
        start, stop = entry["range"]
        urls = [entry["url_template"].format(i=i) for i in range(int(start), int(stop) + 1)]
    else:
        urls = [entry["url"]] if "url" in entry else list(entry.get("urls", []))
    for u in urls:
        yield Item(entry["id"], u, lic, u.rsplit("/", 1)[-1], sha256=entry.get("sha256"), provenance={"kind": "http"})


def expand_hf_tree(entry: dict, session) -> Iterator[Item]:
    """Every file under a path of a Hugging Face dataset repo, with the LFS sha256 the API publishes."""
    repo, rev, path = entry["repo"], entry.get("revision", "main"), entry.get("path", "")
    pattern = re.compile(entry.get("pattern", r".*\.(parquet|jpe?g|png|webp|tiff?|exr|hdr)$"), re.I)
    lic = entry.get("license", "")
    for node in _get_json(session, HF_API.format(repo=repo, rev=rev, path=path)):
        if node.get("type") != "file" or not pattern.match(node["path"]):
            continue
        yield Item(
            entry["id"],
            HF_RESOLVE.format(repo=repo, rev=rev, path=node["path"]),
            lic,
            node["path"].rsplit("/", 1)[-1],
            bytes=node.get("size"),
            sha256=(node.get("lfs") or {}).get("oid"),
            provenance={"kind": "hf_tree", "repo": repo, "revision": rev, "path": node["path"]},
        )


def expand_polyhaven_hdris(entry: dict, session) -> Iterator[Item]:
    """Poly Haven HDRIs (CC0). ``resolution`` like '4k', ``format`` 'hdr' or 'exr'."""
    res, fmt = entry.get("resolution", "4k"), entry.get("format", "hdr")
    assets = _get_json(session, POLYHAVEN_ASSETS)
    for asset_id in sorted(assets):
        files = _get_json(session, POLYHAVEN_FILES.format(id=asset_id))
        node = ((files.get("hdri") or {}).get(res) or {}).get(fmt)
        if not node:
            continue
        yield Item(
            entry["id"],
            node["url"],
            "CC0",
            f"{asset_id}_{res}.{fmt}",
            bytes=node.get("size"),
            md5=node.get("md5"),
            provenance={"kind": "polyhaven", "asset": asset_id, "resolution": res},
        )


def expand_ambientcg(entry: dict, session) -> Iterator[Item]:
    """ambientCG assets (CC0). ``type`` 'HDRI' or 'Material', ``variant`` substring like '4K-JPG'."""
    kind, variant, limit = entry.get("type", "HDRI"), entry.get("variant", "4K"), int(entry.get("page_size", 100))
    offset = 0
    while True:
        page = _get_json(session, AMBIENTCG.format(type=kind, limit=limit, offset=offset))
        assets = page.get("foundAssets") or []
        if not assets:
            return
        for a in assets:
            folders = (a.get("downloadFolders") or {}).get("default") or {}
            cats = folders.get("downloadFiletypeCategories") or {}
            for cat in cats.values():
                for dl in cat.get("downloads") or []:
                    if variant.lower() in str(dl.get("attribute", "")).lower():
                        yield Item(
                            entry["id"],
                            dl["downloadLink"],
                            "CC0",
                            dl.get("fileName") or f"{a['assetId']}_{dl['attribute']}.zip",
                            bytes=dl.get("size"),
                            provenance={"kind": "ambientcg", "asset": a.get("assetId"), "attribute": dl.get("attribute")},
                        )
        offset += limit


def expand_zenodo(entry: dict, session) -> Iterator[Item]:
    """All files of a Zenodo record; Zenodo publishes md5 per file."""
    rec = _get_json(session, ZENODO.format(record=entry["record"]))
    lic = entry.get("license") or (rec.get("metadata", {}).get("license", {}) or {}).get("id", "")
    for f in rec.get("files", []):
        checksum = f.get("checksum", "")
        yield Item(
            entry["id"],
            f["links"]["self"],
            lic,
            f.get("key") or f.get("filename"),
            bytes=f.get("size"),
            md5=checksum.split(":", 1)[1] if checksum.startswith("md5:") else None,
            provenance={"kind": "zenodo", "record": entry["record"], "doi": rec.get("doi")},
        )


def expand_wikimedia_category(entry: dict, session) -> Iterator[Item]:
    """Files in a Commons category with per-file license from extmetadata and sha1 from imageinfo."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": entry["category"],
        "gcmtype": "file",
        "gcmlimit": str(int(entry.get("page_size", 200))),
        "prop": "imageinfo",
        "iiprop": "url|size|sha1|extmetadata|mime",
        "iiextmetadatafilter": "LicenseShortName|License|Artist",
    }
    mime_ok = re.compile(entry.get("mime", r"^image/(jpeg|png|tiff|webp)$"))
    while True:
        data = _get_json(session, WIKIMEDIA, params=params)
        for page in (data.get("query") or {}).get("pages", {}).values():
            for ii in page.get("imageinfo", []):
                if not mime_ok.match(ii.get("mime", "")):
                    continue
                meta = ii.get("extmetadata") or {}
                lic = (meta.get("LicenseShortName") or {}).get("value", "")
                yield Item(
                    entry["id"],
                    ii["url"],
                    lic,
                    page["title"].replace("File:", "").replace(" ", "_"),
                    bytes=ii.get("size"),
                    provenance={"kind": "wikimedia", "pageid": page.get("pageid"), "sha1": ii.get("sha1"),
                                "artist": (meta.get("Artist") or {}).get("value", "")[:200]},
                )
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)


EXPANDERS = {
    "http": expand_http,
    "hf_tree": expand_hf_tree,
    "polyhaven_hdris": expand_polyhaven_hdris,
    "ambientcg": expand_ambientcg,
    "zenodo": expand_zenodo,
    "wikimedia_category": expand_wikimedia_category,
}


def expand(entry: dict, session) -> Iterator[Item]:
    kind = entry["kind"]
    if kind not in EXPANDERS:
        raise KeyError(f"unknown source kind {kind!r}; one of {sorted(EXPANDERS)}")
    return EXPANDERS[kind](entry, session)
