"""Expanders: turn one manifest entry into its file list by walking an API.
Each takes (collector, source) and returns [FileSpec]; `source.params` holds the knobs,
overridable from the CLI with --param k=v. Keep them pure listings: no downloading here."""
from __future__ import annotations

import base64
import html as _html
import os
import re
from typing import Callable
from urllib.parse import quote

from .collect import Collector
from .manifest import FileSpec, Source

POLYHAVEN = "https://api.polyhaven.com"
HF_SERVER = "https://datasets-server.huggingface.co"
HF_HUB = "https://huggingface.co"
CIE = "https://cie.co.at"
ZENODO = "https://zenodo.org/api"
AMBIENTCG = "https://ambientcg.com/api/v2/full_json"
GDRIVE_FOLDER = "https://drive.google.com/embeddedfolderview?id={id}"
GDRIVE_FILE = "https://drive.google.com/uc?export=download&id={id}"
GITHUB_API = "https://api.github.com"
RPU = "https://raw.pixls.us"
DATAVERSE_DEFAULT = "https://dataverse.harvard.edu"
GCS = "https://storage.googleapis.com"


def _limit(items, source: Source):
    n = source.params.get("limit")
    return items if n in (None, 0, "0") else items[: int(n)]


# ---- Poly Haven (CC0) ----------------------------------------------------------------
def polyhaven_hdris(c: Collector, source: Source) -> list[FileSpec]:
    """Every HDRI at one resolution. Sizes per file: 1k 1.4 MB, 2k 5.5 MB, 4k 21 MB, 8k 75 MB, 16k 262 MB."""
    res, fmt = str(source.params.get("resolution", "1k")), str(source.params.get("format", "hdr"))
    assets = c.get_json(f"{POLYHAVEN}/assets", {"t": "hdris"})
    ids = sorted(assets)
    cats = source.params.get("categories")
    if cats:
        want = set(cats.split(",")) if isinstance(cats, str) else set(cats)
        ids = [i for i in ids if want & set(assets[i].get("categories", []))]
    specs = [FileSpec(url=f"{POLYHAVEN}/assets?t=hdris", path="index.json")]
    for i in _limit(ids, source):
        files = c.get_json(f"{POLYHAVEN}/files/{i}")
        entry = files.get("hdri", {}).get(res, {}).get(fmt)
        if entry:
            specs.append(FileSpec(url=entry["url"], path=f"{res}/{i}.{fmt}", size=entry.get("size"),
                                  sha256=entry.get("md5") and None))
    return specs


def polyhaven_textures(c: Collector, source: Source) -> list[FileSpec]:
    """Diffuse + normal + roughness maps per texture at one resolution (default 1k, PNG)."""
    res, fmt = str(source.params.get("resolution", "1k")), str(source.params.get("format", "png"))
    maps = str(source.params.get("maps", "Diffuse,nor_gl,Rough")).split(",")
    assets = c.get_json(f"{POLYHAVEN}/assets", {"t": "textures"})
    specs = [FileSpec(url=f"{POLYHAVEN}/assets?t=textures", path="index.json")]
    for i in _limit(sorted(assets), source):
        files = c.get_json(f"{POLYHAVEN}/files/{i}")
        for m in maps:
            entry = files.get(m, {}).get(res, {}).get(fmt)
            if entry:
                specs.append(FileSpec(url=entry["url"], path=f"{res}/{i}_{m}.{fmt}", size=entry.get("size")))
    return specs


def polyhaven_models(c: Collector, source: Source) -> list[FileSpec]:
    """glTF per model at one texture resolution (default 1k). Index only unless limit > 0."""
    res = str(source.params.get("resolution", "1k"))
    assets = c.get_json(f"{POLYHAVEN}/assets", {"t": "models"})
    specs = [FileSpec(url=f"{POLYHAVEN}/assets?t=models", path="index.json")]
    for i in _limit(sorted(assets), source):
        files = c.get_json(f"{POLYHAVEN}/files/{i}")
        entry = files.get("gltf", {}).get(res, {}).get("gltf")
        if entry:
            specs.append(FileSpec(url=entry["url"], path=f"{res}/{i}.gltf", size=entry.get("size")))
            for inc_path, inc in entry.get("include", {}).items():
                specs.append(FileSpec(url=inc["url"], path=f"{res}/{inc_path}", size=inc.get("size")))
    return specs


# ---- Hugging Face datasets-server: rows as JSON pages, no parquet download ----------------
def _hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def hf_rows(c: Collector, source: Source) -> list[FileSpec]:
    """Page through /rows for (dataset, config, split). Gated sets need HF_TOKEN in the env.
    Each page is one JSON file; `pages` caps the walk (0 = all)."""
    p = source.params
    dataset, config, split = p["dataset"], p.get("config", "default"), p.get("split", "train")
    length = int(p.get("length", 100))
    info = c.get_json(f"{HF_SERVER}/size", {"dataset": dataset}, headers=_hf_headers())
    total = None
    for s in info.get("size", {}).get("splits", []):
        if s["config"] == config and s["split"] == split:
            total = s["num_rows"]
    if total is None:
        raise RuntimeError(f"{dataset}: config {config!r} split {split!r} not found")
    n_pages = -(-total // length)
    cap = int(p.get("pages", 0))
    if cap:
        n_pages = min(n_pages, cap)
    specs = [FileSpec(url=f"{HF_SERVER}/size?dataset={dataset}", path="size.json")]
    for k in range(n_pages):
        url = (f"{HF_SERVER}/rows?dataset={dataset}&config={config}&split={split}"
               f"&offset={k * length}&length={length}")
        specs.append(FileSpec(url=url, path=f"{config}/{split}/rows_{k * length:09d}.json"))
    return specs


def hf_parquet(c: Collector, source: Source) -> list[FileSpec]:
    """Every parquet shard the Hub auto-converted for (dataset[, config][, split]). This is the
    bulk path: one request lists the shards, the collector streams them in parallel. With
    `columns=a,b,c` the specs become hf:// column projections (see Collector._download_columns)."""
    p = source.params
    dataset = p["dataset"]
    info = c.get_json(f"{HF_SERVER}/parquet", {"dataset": dataset}, headers=_hf_headers())
    want_cfg, want_split = p.get("config"), p.get("split")
    specs = [FileSpec(url=f"{HF_SERVER}/parquet?dataset={dataset}", path="parquet.json")]
    shards = [f for f in info.get("parquet_files", [])
              if (not want_cfg or f["config"] == want_cfg) and (not want_split or f["split"] == want_split)]
    columns = [x for x in p.get("columns", "").split(",") if x] if p.get("columns") else None
    for f in _limit(shards, source):
        rel = f"{f['config']}/{f['split']}/{f['filename']}"
        if columns:
            # column projection: the collector reads only these columns' chunks over HTTP ranges and
            # writes a small local parquet; size unknown up front, so the byte budget does not see it
            specs.append(FileSpec(url=f"hf://datasets/{dataset}@~parquet/{rel}", path=rel, columns=columns))
        else:
            specs.append(FileSpec(url=f["url"], path=rel, size=f.get("size")))
    return specs


def hf_hub_files(c: Collector, source: Source) -> list[FileSpec]:
    """Raw files of a Hub dataset repo (zips, images, csvs) via huggingface_hub's tree listing.
    `pattern` is a regex on the repo path; `revision` defaults to main."""
    from huggingface_hub import HfApi
    p = source.params
    repo, rev = p["repo"], p.get("revision", "main")
    pat = re.compile(p.get("pattern", ".*"))
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    specs = []
    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True):
        path, size = getattr(entry, "path", None), getattr(entry, "size", None)
        if path is None or size is None or not pat.search(path):
            continue
        lfs = getattr(entry, "lfs", None)
        specs.append(FileSpec(url=f"{HF_HUB}/datasets/{repo}/resolve/{rev}/{path}", path=path, size=size,
                              sha256=getattr(lfs, "sha256", None) if lfs else None))
    return _limit(sorted(specs, key=lambda s: s.path), source)


# ---- Zenodo: one record -> its files with md5 --------------------------------------------------
def zenodo_record(c: Collector, source: Source) -> list[FileSpec]:
    """`record` is the numeric Zenodo id; `pattern` optionally filters file names."""
    p = source.params
    rec = c.get_json(f"{ZENODO}/records/{p['record']}")
    pat = re.compile(p.get("pattern", ".*"))
    specs = [FileSpec(url=f"{ZENODO}/records/{p['record']}", path="record.json")]
    for f in rec.get("files", []):
        if not pat.search(f["key"]):
            continue
        md5 = f.get("checksum", "")
        md5 = md5.split(":", 1)[1] if md5.startswith("md5:") else None
        specs.append(FileSpec(url=f["links"]["self"], path=f["key"], size=f.get("size"), md5=md5))
    return _limit(specs, source)


# ---- ambientCG (CC0 PBR materials / HDRIs) --------------------------------------------------------
def ambientcg(c: Collector, source: Source) -> list[FileSpec]:
    """Walk the v2 full_json API for one asset type; `variant` picks the zip (e.g. 1K-JPG, 2K-PNG)."""
    p = source.params
    typ, variant = p.get("type", "Material"), p.get("variant", "1K-JPG")
    want = int(p.get("limit") or 0)
    specs, offset, page = [], 0, 100
    while True:
        j = c.get_json(AMBIENTCG, {"type": typ, "limit": page, "offset": offset, "include": "downloadData"})
        assets = j.get("foundAssets", [])
        for a in assets:
            cats = a.get("downloadFolders", {}).get("default", {}).get("downloadFiletypeCategories", {})
            for cat in (cats.values() if isinstance(cats, dict) else cats):  # dict or list, API varies
                for d in cat.get("downloads", []):
                    if d.get("attribute") == variant:
                        specs.append(FileSpec(url=d["fullDownloadPath"], path=f"{variant}/{a['assetId']}.zip",
                                              size=d.get("size")))
        if len(assets) < page or (want and len(specs) >= want):
            break
        offset += page
    specs.sort(key=lambda s: s.path)
    return [FileSpec(url=f"{AMBIENTCG}?type={typ}", path=f"index_{typ}.json")] + _limit(specs, source)


# ---- Lens prescriptions: a public Google Drive folder (Bill Claff's archive) -------------------------
def lens_designs(c: Collector, source: Source) -> list[FileSpec]:
    """Every file in a public Drive folder: the embedded folder view is plain HTML listing entry ids and
    titles; each entry downloads through uc?export=download. `folder` is the Drive folder id, `pattern`
    a regex on the file name (e.g. \\.zmx$), `limit` caps the count (the index page is always kept)."""
    p = source.params
    folder = p["folder"]
    index_url = GDRIVE_FOLDER.format(id=folder)
    page = c.get_text(index_url)
    pat = re.compile(p.get("pattern", ".*"), re.I)
    specs, seen = [], set()
    for fid, title in re.findall(r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?flip-entry-title">([^<]*)<',
                                 page, flags=re.S):
        title = _html.unescape(title).strip()
        if not title or title in seen or not pat.search(title):
            continue
        seen.add(title)
        specs.append(FileSpec(url=GDRIVE_FILE.format(id=fid), path=title))
    specs.sort(key=lambda s: s.path)
    return [FileSpec(url=index_url, path="folder.html")] + _limit(specs, source)


# ---- GitHub: one directory of a repository via the contents API --------------------------------------
def github_dir(c: Collector, source: Source) -> list[FileSpec]:
    """Files of `path` in `repo` (owner/name) at optional `ref`; `pattern` is a regex on the file name.
    Sizes come from the API, so --max-bytes budgets work before any download."""
    p = source.params
    url = f"{GITHUB_API}/repos/{p['repo']}/contents/{p.get('path', '').strip('/')}"
    j = c.get_json(url, {"ref": p["ref"]} if p.get("ref") else None, headers={"Accept": "application/vnd.github+json"})
    if isinstance(j, dict):
        raise RuntimeError(f"GitHub contents API: {j.get('message', 'unexpected response')}")
    pat = re.compile(p.get("pattern", ".*"))
    specs = [FileSpec(url=e["download_url"], path=e["name"], size=e.get("size"))
             for e in j if e.get("type") == "file" and e.get("download_url") and pat.search(e["name"])]
    return _limit(sorted(specs, key=lambda s: s.path), source)


# ---- CIE data tables: resolve each table page to its CSV ------------------------------------
def cie_tables(c: Collector, source: Source) -> list[FileSpec]:
    """`tables` lists datatable slugs; the CSV link is scraped from each page (the index has none)."""
    specs = []
    for slug in source.params["tables"]:
        c._throttle.wait("cie.co.at")
        html = c._session.get(f"{CIE}/datatable/{slug}", timeout=(15, 60)).text
        links = sorted(set(re.findall(r'href="([^"]+\.csv)"', html)))
        if not links:
            raise RuntimeError(f"no csv link on {slug}")
        url = links[0] if links[0].startswith("http") else CIE + links[0]
        specs.append(FileSpec(url=url, path=f"{slug}.csv"))
        if f'{links[0]}_metadata.json' in html:  # licence, citation, column units
            specs.append(FileSpec(url=url + "_metadata.json", path=f"{slug}.metadata.json"))
    return specs


# ---- raw.pixls.us (CC0 sample raws, one per camera) ----------------------------------------------
def rpu_index(c: Collector, source: Source) -> list[FileSpec]:
    """One sample raw per make/model from the raw.pixls.us JSON index (CC0 rows only unless
    `license=all`; `pattern` is a regex on "Make Model variant"; `one_per_model=0` keeps every
    variant). The index publishes sha256 and size per file, so every raw is checksum-verified."""
    p = source.params
    idx = c.get_json(f"{RPU}/json/getrepository.php", params={"set": "all"})
    pat = re.compile(str(p.get("pattern", ".*")), re.I)
    cc0_only = str(p.get("license", "cc0")).lower() == "cc0"
    one = str(p.get("one_per_model", "1")).lower() not in ("0", "false", "")
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "")).strip("_")
    specs = [FileSpec(url=f"{RPU}/json/getrepository.php?set=all", path="repository.json")]
    seen: set[tuple[str, str]] = set()
    for row in idx["data"]:
        make, model, variant, _mp, _, lic, _date, link = row[:8]
        if cc0_only and "publicdomain/zero" not in str(lic):
            continue
        href = re.search(r"href='([^']+)'", link)
        if not href or not pat.search(f"{make} {model} {variant or ''}"):
            continue
        if one and (make, model) in seen:
            continue
        seen.add((make, model))
        url = _html.unescape(href.group(1)).replace(" ", "%20")
        sha = re.search(r"sha256 Checksum'>([0-9a-f]{64})", link)
        size = re.search(r"[(]([0-9.]+)MB[)]", link)
        ext = url.rsplit(".", 1)[-1].lower() if "." in url.rsplit("/", 1)[-1] else "raw"
        specs.append(FileSpec(url=url, path=f"{safe(make)}/{safe(model)}/{safe(variant) or 'default'}.{ext}",
                              sha256=sha.group(1) if sha else None,
                              size=int(float(size.group(1)) * 1e6) if size else None))
    return _limit(specs, source)


# ---- Dataverse (Harvard, UCLouvain, DaRUS, ...): one dataset through the native API ----------------------
def dataverse_dataset(c: Collector, source: Source) -> list[FileSpec]:
    """Every file of a Dataverse dataset. `server` is the installation (https://dataverse.harvard.edu,
    https://dataverse.uclouvain.be, https://darus.uni-stuttgart.de), `doi` the persistent id
    (doi:10.7910/DVN/SGHDCP), `pattern` a regex on "directory/filename", `limit` a cap. The API publishes
    size and MD5 per file, so every download is verified; files download through /api/access/datafile/<id>
    and the dataset record is kept as dataset.json."""
    p = source.params
    server = str(p.get("server", DATAVERSE_DEFAULT)).rstrip("/")
    doi = str(p["doi"])
    rec = c.get_json(f"{server}/api/datasets/:persistentId/", params={"persistentId": doi})
    files = rec["data"]["latestVersion"]["files"]
    pat = re.compile(str(p.get("pattern", ".*")), re.I)
    specs = [FileSpec(url=f"{server}/api/datasets/:persistentId/?persistentId={doi}", path="dataset.json")]
    rows = []
    for f in files:
        df = f["dataFile"]
        rel = "/".join(x for x in (f.get("directoryLabel"), df["filename"]) if x)
        if not pat.search(rel):
            continue
        chk = df.get("checksum") or {}
        md5 = df.get("md5") or (chk.get("value") if str(chk.get("type", "")).upper() == "MD5" else None)
        rows.append(FileSpec(url=f"{server}/api/access/datafile/{df['id']}", path=rel,
                             size=df.get("filesize"), md5=md5))
    rows.sort(key=lambda s: s.path)
    return specs + _limit(rows, source)


# ---- public Google Cloud Storage bucket: anonymous JSON listing, no gsutil -------------------------------
def gcs_bucket(c: Collector, source: Source) -> list[FileSpec]:
    """Every object under `prefix` of the public bucket `bucket` (gs://hdrplusdata is one), through the
    storage JSON API's anonymous listing: `pattern` is a regex on the object name, `limit` stops the walk
    once that many objects are listed. Size and MD5 (base64 in the API, hex here) come with the listing;
    objects download as plain https://storage.googleapis.com/<bucket>/<object>. Paths drop the prefix."""
    p = source.params
    bucket, prefix = str(p["bucket"]), str(p.get("prefix", ""))
    pat = re.compile(str(p.get("pattern", ".*")), re.I)
    limit = int(p.get("limit") or 0)
    specs: list[FileSpec] = []
    token = None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        page = c.get_json(f"{GCS}/storage/v1/b/{bucket}/o", params=q)
        for it in page.get("items", []):
            name = it["name"]
            if name.endswith("/") or not pat.search(name):
                continue
            md5 = base64.b64decode(it["md5Hash"]).hex() if it.get("md5Hash") else None
            rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
            specs.append(FileSpec(url=f"{GCS}/{bucket}/{quote(name)}", path=rel.lstrip("/"),
                                  size=int(it["size"]), md5=md5))
            if limit and len(specs) >= limit:
                return specs
        token = page.get("nextPageToken")
        if not token:
            return specs


EXPANDERS: dict[str, Callable[[Collector, Source], list[FileSpec]]] = {
    "polyhaven_hdris": polyhaven_hdris,
    "polyhaven_textures": polyhaven_textures,
    "polyhaven_models": polyhaven_models,
    "hf_rows": hf_rows,
    "hf_parquet": hf_parquet,
    "hf_hub_files": hf_hub_files,
    "zenodo_record": zenodo_record,
    "ambientcg": ambientcg,
    "lens_designs": lens_designs,
    "github_dir": github_dir,
    "cie_tables": cie_tables,
    "rpu_index": rpu_index,
    "dataverse_dataset": dataverse_dataset,
    "gcs_bucket": gcs_bucket,
}


def expand(c: Collector, source: Source) -> list[FileSpec]:
    if source.expander is None:
        return list(source.files)
    return EXPANDERS[source.expander](c, source)
