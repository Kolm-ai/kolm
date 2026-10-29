"""Offline labels for synthetic degradation: python -m harness data label --sources <ids> --n <count> --out data/labels

Per clean tile (a random crop of a source image) draw parameter vectors with harness.data.degrade, render
each, score it against the prepared clean tile with harness.data.tilemetrics and write one row per draw:
source, file, crop box, seed, the parameter vector (p_*) and the metric vector (m_*). No image is stored:
(source, file, row, crop box, seed) reproduce the pair bit for bit. Rows go to parquet or JSONL shards of
`chunk` items each; the ledger (<out>/ledger.jsonl) records finished chunks so a re-run with the same
configuration skips them. Chunks run in a process pool; every draw is deterministic in its seed, so the
output does not depend on the pool size or on the order chunks finish.

Input domains. Every decoder hands the labeller linear RGB float64, so the metrics of a row mean the same
thing whatever file it came from; the row's `domain` column says what the file held and `is_hdr` how
prepare_tile treats its range:

  srgb8             8-bit sRGB files (PNG, JPEG, JPEG 2000, WebP, BMP, 8-bit TIFF; parquet image blobs): value / 255,
                    then the sRGB transfer function inverted. is_hdr False.
  srgb16            16-bit integer PNG/TIFF: value / 65535, then the sRGB transfer function inverted.
                    is_hdr False.
  linear16          camera raws through LibRaw (rawpy): demosaiced, camera white balance, sRGB primaries,
                    gamma (1, 1) so the output is linear, no auto-brightening, no sharpening, 16-bit output
                    / 65535, so 1.0 is the sensor white level. is_hdr True: the range is kept as captured.
                    Pixel-shift merges (.mat, PixelShift200's RGGB: four sensor-shifted shots sample R, G, G
                    and B at every site, so there is no demosaic) are rendered the way LibRaw renders their
                    camera's single shot (_decode_rggb_mat), into the same domain.
  linear_float_hdr  Radiance .hdr and OpenEXR .exr (plain files or members of a .zip): relative scene
                    radiance as stored, non-finite samples zeroed, negatives clipped to 0. is_hdr True.

LDR domains (is_hdr False) get the sampled scene peak-over-median stretch in prepare_tile; HDR domains keep
their own range. Every sampled draw re-exposes the tile (expose=1 puts the tile median at mid grey times the
drawn exposure offset), so the absolute scale of a domain never reaches the metrics, only its linearity and
its range do; a JPEG row and a raw row are therefore read the same way.

Archive members are items of their own: Item.file is "<zip path>::<member path>", so the per-item resume
(item_key, labelled_items, plan_items) works member by member. A file a decoder cannot read (optional
package missing, unsupported format, image smaller than a tile) is recorded under `skipped` in the ledger
entry of its chunk with a reason, and never stops the run. A permanent reason (decode_error, too_small)
counts the item as done for later runs (retry_skipped / --retry-skipped labels it again); a missing decoder
(no_decoder:<package>) does not, so installing it labels those items on the next run. The decoders are
optional imports: `pip install -e .[decode]` brings rawpy (LibRaw) and OpenEXR. A chunk that fails outright
(a worker exception that is not a decode error, exhausted memory) is logged as FAILED, counted in the summary
under `failed` and left for the next run, which plans its items again; a worker process that dies takes its
pool down and the unfinished chunks go to a fresh one (_run_pool). Neither stops the pass.

Clean sources. A source's manifest `clean` block says how it is read as a clean source (clean_settings):
`scale` box-downsamples every decoded image in linear light (a 50 MP JPEG at 2, a demosaiced raw at 4, so the
camera's own demosaic, noise and blur sit below the tile's pixel) and each row records the factor used in its
`scale` column; `low_quality_scale` is the factor for a JPEG whose quantisation tables put it below quality
LOW_QUALITY_JPEG; `exclude` lists fnmatch patterns (case-insensitive, on Item.file) of files that are not
pictures (lens-shading maps, scan containers), which list_items leaves out. Splits key on the scene
(scene_of), so two resolution tiers of one HDRI never straddle train and validation.

Memory: a chunk decodes one image at a time, so a worker holds one full image as float64 (a 50 Mpx raw is
about 1.2 GB); --jobs times that is the peak."""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import math
import os
import re
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import data_root
from .degrade import DegradeParams, Distributions, degrade, prepare_tile, sample_params
from .manifest import load_manifest
from .tilemetrics import tile_metrics
from ..colorspace import srgb_to_linear
from ..metrics import STILL_METRICS
from ..scene import box_downsample

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2", ".j2k"}   # JPEG 2000: openjpeg
TIFF_EXT = {".tif", ".tiff"}
# LibRaw's file formats (libraw.org/supported-cameras); one LibRaw rejects is recorded as skipped, not fatal
RAW_EXT = {".3fr", ".ari", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".crw", ".dcr", ".dcs", ".dng", ".drf", ".erf",
           ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
           ".ptx", ".pxn", ".raf", ".raw", ".rw2", ".rwl", ".rwz", ".sr2", ".srf", ".srw", ".x3f"}
HDR_EXT = {".hdr"}
EXR_EXT = {".exr"}
ZIP_EXT = {".zip"}
TAR_EXT = {".tar"}                              # uncompressed only: a member is one seek and read (_tar_index)
ARCHIVE_EXT = ZIP_EXT | TAR_EXT
MAT_EXT = {".mat"}                              # pixel-shift merges, _decode_rggb_mat
ZIP_MEMBER_EXT = HDR_EXT | EXR_EXT | RAW_EXT   # what is listed inside an archive (not its previews); plain images
#                                                 too for a source whose clean.archive_images is set (UHD-IQA's tar)
ZIP_SEP = "::"                                  # Item.file of a member: "<archive path>::<member path>"
NO_DECODER = "no_decoder:"                      # skip reason prefix that never counts as done
PARQUET_SOURCES = {"hf-kadid10k-images": ("image", "kadid"), "hf-tid2013": ("reference", "tid")}
SEED_MASK = (1 << 63) - 1
LEDGER = "ledger.jsonl"
CONFIG_KEYS = ("tile", "per_tile", "crops", "ss", "fmt")
_PIL_16BIT = {"I;16", "I;16B", "I;16L", "I;16N", "I"}
LOW_QUALITY_JPEG = 90    # a JPEG whose quantisation tables put it below this IJG quality takes low_quality_scale
_LUMA_TABLE_SUM = 3688   # sum of the ITU T.81 K.1 luminance table, what the IJG quality scale factor multiplies
_RES_DIR = re.compile("^[0-9]+k$", re.IGNORECASE)      # Poly Haven's resolution tiers: 1k/<name>.hdr, 2k/<name>.hdr
_RES_SUFFIX = re.compile("_[0-9]+k$", re.IGNORECASE)
_SRGB_LUT = srgb_to_linear(np.arange(256) / 255.0)     # the transfer curve of the 256 8-bit values, once
_XYZ_RGB = np.array([[0.412453, 0.357580, 0.180423],    # linear sRGB (D65) to XYZ, LibRaw's xyz_rgb
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])
RGGB_WHITE = 16383            # a pixel-shift merge's white: PixelShift200 subtracts the black and scales to 14 bits
RGGB_OFFSET = 2               # its site (i, j) is the raw's (i + 2, j + 2): the merge cuts 2 px from every side
_FLIP_TURNS = {0: 0, 3: 2, 5: 1, 6: 3}                  # LibRaw's sizes.flip as np.rot90 quarter turns


@dataclass(frozen=True)
class Item:
    """One clean image: a file under <root>/raw/<source> (or a member of a zip or tar there, see ZIP_SEP), or one row
    of a parquet shard there. is_hdr is what the extension promises; the decoder has the last word (Decoded)."""
    source: str
    file: str            # path relative to <root>/raw/<source>, posix; "<zip>::<member>" inside an archive
    row: int = -1        # parquet row (global over the sorted shards), -1 for a plain file
    column: str = ""
    is_hdr: bool = False


class Decoded(NamedTuple):
    """What a decoder returns: linear RGB float64 (H, W, 3), the input domain (module docstring), the range flag
    and the clean scale the image was box-downsampled by (clean_scale; 1 for the file's own pixels)."""
    img: np.ndarray
    domain: str
    is_hdr: bool
    scale: int = 1


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & SEED_MASK


def scene_of(item: Item) -> str:
    """The scene an item depicts, the key of the train/validation split. A source whose manifest `clean.scene`
    gives a regex takes its first group where it matches the path (RawNIND's scene name shared by the GT and the
    ISO sweep, an HDR+ burst id shared by bursts/ and results_*/). Otherwise the path with resolution directories
    (`1k/`, `2k/`) and resolution stem suffixes (`_4k`) removed, so every resolution tier of one Poly Haven HDRI is
    one scene, and any other path is its own scene."""
    pat = clean_settings(item.source).get("scene")
    if pat:
        m = re.search(pat, item.file)
        if m and m.groups():
            return m.group(1)
    parts = item.file.split("/")
    stem, ext = os.path.splitext(parts[-1])
    kept = [d for d in parts[:-1] if not _RES_DIR.match(d)]
    return "/".join(kept + [_RES_SUFFIX.sub("", stem) + ext])


@lru_cache(maxsize=None)
def _manifest_clean() -> dict[str, dict]:
    return {s.id: dict(s.clean) for s in load_manifest() if s.clean}


def clean_settings(source: str) -> dict:
    """The manifest's `clean` block for a source, empty for one it does not know (a test fixture): `scale`, the
    box-downsampling factor applied in linear light after decoding; `low_quality_scale`, the factor for a JPEG
    whose quantisation tables put it below LOW_QUALITY_JPEG; `exclude`, fnmatch patterns (case-insensitive, on
    the posix path relative to the source, "<zip>::<member>" for an archive member) of files that are not
    pictures; `include`, patterns of the only files that are (RawNIND's GT raws, the HDR+ merges); `scene`, a
    regex whose first group is the scene key (scene_of); `holdout`, patterns of scene keys the sampler never
    trains or validates on (model.data.split_of gives them "test"); `archive_images`, list an archive's plain
    images as items (they are its previews unless the source says otherwise)."""
    return _manifest_clean().get(source, {})


def split_member(file: str) -> tuple[str, str | None]:
    """Item.file -> (path on disk, member path inside it or None)."""
    if ZIP_SEP in file:
        z, m = file.split(ZIP_SEP, 1)
        if z.lower().endswith(tuple(ARCHIVE_EXT)):
            return z, m
    return file, None


@lru_cache(maxsize=8)
def _tar_index(path: str, mtime_ns: int) -> dict[str, tuple[int, int]]:
    """member -> (data offset, size) of an uncompressed tar's regular files, read once per process: tarfile finds
    a member by walking every header before it, so a 900-member tar would cost a walk per decode."""
    with tarfile.open(path, "r:") as t:
        return {m.name: (m.offset_data, m.size) for m in t if m.isreg()}


def _archive_members(path: Path, exts: set[str]) -> list[str] | None:
    """Members of an archive with an extension in `exts`, in a fixed order; None when it is not a readable zip
    or uncompressed tar. Members are read, never extracted, so their paths are only names."""
    try:
        if path.suffix.lower() in TAR_EXT:
            names = list(_tar_index(str(path), path.stat().st_mtime_ns))
        else:
            with zipfile.ZipFile(path) as z:
                names = [i.filename for i in z.infolist() if not i.is_dir()]
    except (zipfile.BadZipFile, tarfile.TarError, OSError):
        return None
    return sorted(n for n in names if Path(n).suffix.lower() in exts)


def read_member(path: Path, member: str) -> bytes:
    """One archive member's bytes (a zip's through zipfile, a tar's by its indexed offset)."""
    if path.suffix.lower() in TAR_EXT:
        off, size = _tar_index(str(path), path.stat().st_mtime_ns)[member]
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(size)
    with zipfile.ZipFile(path) as z:
        return z.read(member)


def file_filter(source: str):
    """The predicate on a source-relative file the source's `clean.include` and `clean.exclude` patterns make
    (fnmatch, case-insensitive): list_items, the real-pair listing and the real store all apply it."""
    settings = clean_settings(source)
    excluded = [pat.lower() for pat in settings.get("exclude", [])]
    included = [pat.lower() for pat in settings.get("include", [])]

    def keep(file: str) -> bool:
        f = file.lower()
        if included and not any(fnmatch.fnmatchcase(f, pat) for pat in included):
            return False
        return not any(fnmatch.fnmatchcase(f, pat) for pat in excluded)
    return keep


def list_items(source: str, root: Path | None = None) -> list[Item]:
    """Every clean image of a source on disk, in a fixed order: plain image, raw and HDR files, the decodable
    members of every zip or uncompressed tar (an unreadable archive is listed as one item so the run records why
    it was skipped; plain images inside one only with clean.archive_images); files matching the source's
    `clean.exclude` patterns are left out."""
    root = Path(root) if root else data_root()
    base = root / "raw" / source
    if not base.is_dir():
        raise FileNotFoundError(f"{source}: nothing under {base}")
    if source in PARQUET_SOURCES:
        column, kind = PARQUET_SOURCES[source]
        files = sorted(base.rglob("*.parquet"))
        if kind == "kadid":
            from .fit import N_REFS, kadid_index
            rows = [kadid_index(r, 0, 0) for r in range(1, N_REFS + 1)]
        else:
            import pyarrow.parquet as pq
            seen, rows, g = set(), [], 0
            for f in files:
                for rid in pq.read_table(str(f), columns=["reference_id"]).column("reference_id").to_pylist():
                    if rid not in seen:
                        seen.add(rid)
                        rows.append(g)
                    g += 1
        rel = "|".join(f.relative_to(base).as_posix() for f in files)
        return [Item(source, rel, r, column) for r in rows]
    keep = file_filter(source)
    member_ext = ZIP_MEMBER_EXT | (IMAGE_EXT if clean_settings(source).get("archive_images") else set())
    items = []
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        rel = f.relative_to(base).as_posix()
        if ext in ARCHIVE_EXT:
            members = _archive_members(f, member_ext)
            if members is None:
                items.append(Item(source, rel))
            else:
                items += [Item(source, f"{rel}{ZIP_SEP}{m}", is_hdr=Path(m).suffix.lower() not in IMAGE_EXT)
                          for m in members if keep(f"{rel}{ZIP_SEP}{m}")]
        elif ext in HDR_EXT or ext in EXR_EXT or ext in RAW_EXT or ext in MAT_EXT:
            items.append(Item(source, rel, is_hdr=True))
        elif ext in IMAGE_EXT:
            items.append(Item(source, rel))
    return [it for it in items if keep(it.file)]


# ---- decoders (optional packages; an ImportError means "no decoder", anything else "decode error") -----------

def _decode_exr(data: bytes) -> np.ndarray:
    """OpenEXR bytes -> float32 (H, W, 3) linear RGB: the R, G, B channels of the first part, whether stored
    as separate channels or as one grouped RGB/RGBA array; a Y-only file is replicated to three channels."""
    import OpenEXR
    ch = OpenEXR.File(io.BytesIO(data)).channels()
    if "RGBA" in ch or "RGB" in ch:
        rgb = np.asarray(ch["RGBA" if "RGBA" in ch else "RGB"].pixels)[..., :3]
    elif all(c in ch for c in ("R", "G", "B")):
        rgb = np.stack([np.asarray(ch[c].pixels) for c in ("R", "G", "B")], axis=-1)
    elif "Y" in ch:
        rgb = np.repeat(np.asarray(ch["Y"].pixels)[..., None], 3, axis=-1)
    else:
        raise ValueError(f"no RGB channels in {sorted(ch)}")
    rgb = np.array(rgb, dtype=np.float32)
    rgb[~np.isfinite(rgb)] = 0.0
    return np.clip(rgb, 0.0, None, out=rgb)


def _decode_raw(src: str | io.BytesIO) -> np.ndarray:
    """Camera raw through LibRaw -> float64 (H, W, 3): demosaiced, camera white balance, sRGB primaries, linear
    (gamma 1, 1), no auto-brightening, no sharpening; 16-bit / 65535, so 1.0 is the sensor white level."""
    import rawpy
    with rawpy.imread(src) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
    return np.asarray(rgb, dtype=np.float64) / 65535.0


def _decode_rggb_mat(src, companion: Path) -> np.ndarray:
    """A pixel-shift merge -> float64 (H, W, 3) in the linear16 domain. The MAT's `rggb` (R, Gr, Gb, B at every
    site, RGGB_WHITE the white, its site the raw's shifted by RGGB_OFFSET) rendered as LibRaw renders the scene's
    single-shot raw `companion`: the greens' mean; the raw's as-shot white balance over its smallest multiplier,
    each channel clipped at white after it; the raw's camera-to-sRGB matrix (a DNG's own, else the one LibRaw
    derives from the camera's XYZ-to-camera one: camera-from-sRGB with rows normalised to 1, inverted; neither, the
    camera's colour as it is); clipped to [0, 1]; cut to the raw's visible area and turned as LibRaw turns it. The
    MAT's own gains are LibRaw's daylight multipliers, not the shot's: the raw is read for them."""
    import rawpy
    import scipy.io as sio
    m = sio.loadmat(src, variable_names=["rggb"]).get("rggb")
    if m is None or m.ndim != 3 or m.shape[-1] != 4:
        raise ValueError("no (H, W, 4) rggb array: not a pixel-shift merge")
    with rawpy.imread(str(companion)) as raw:
        wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        if not (wb > 0).all():
            wb = np.asarray(raw.daylight_whitebalance[:3], dtype=np.float64)
        rgb_cam = np.asarray(raw.color_matrix, dtype=np.float64)[:3, :3]
        cam_xyz = np.asarray(raw.rgb_xyz_matrix[:3], dtype=np.float64)
        s = raw.sizes
    if not rgb_cam.any():
        cam_rgb = cam_xyz @ _XYZ_RGB
        rgb_cam = np.linalg.inv(cam_rgb / cam_rgb.sum(axis=1, keepdims=True)) if cam_xyz.any() else np.eye(3)
    r0, c0 = max(s.top_margin - RGGB_OFFSET, 0), max(s.left_margin - RGGB_OFFSET, 0)
    m = m[r0:s.top_margin + s.height - RGGB_OFFSET, c0:s.left_margin + s.width - RGGB_OFFSET]
    cam = np.empty(m.shape[:2] + (3,), dtype=np.float32)
    cam[..., 0], cam[..., 2] = m[..., 0], m[..., 3]
    cam[..., 1] = (m[..., 1].astype(np.float32) + m[..., 2]) * 0.5
    cam *= (wb / wb.min() / RGGB_WHITE).astype(np.float32)
    np.clip(cam, 0.0, 1.0, out=cam)
    out = np.clip(cam.reshape(-1, 3).astype(np.float64) @ rgb_cam.T, 0.0, 1.0).reshape(cam.shape)
    k = _FLIP_TURNS.get(int(s.flip), 0)
    return np.ascontiguousarray(np.rot90(out, k)) if k else out


def _companion_raw(it: Item, root: Path) -> Path:
    """The single-shot raw a pixel-shift merge is rendered by: the source's `clean.companion` template, {name} the
    merge's file stem less a trailing _rggb."""
    template = clean_settings(it.source).get("companion")
    if not template:
        raise ValueError(f"{it.source}: a pixel-shift merge needs the clean block's companion raw template")
    stem = Path(split_member(it.file)[1] or split_member(it.file)[0]).stem
    return root / "raw" / it.source / template.format(name=stem[:-5] if stem.endswith("_rggb") else stem)


def _decode_pil(src) -> Decoded:
    """8-bit (srgb8) or 16-bit integer (srgb16) image through Pillow; float images have no transfer function.
    The 8-bit path goes through the 256-entry table _SRGB_LUT: the same curve of the same 256 inputs to the bit,
    in a quarter of the time on a 50 MP JPEG, where the curve over 160 M values was the cost."""
    from PIL import Image
    with Image.open(src) as im:
        if im.mode in _PIL_16BIT:
            a = np.clip(np.asarray(im, dtype=np.float64) / 65535.0, 0.0, 1.0)
            return Decoded(srgb_to_linear(np.repeat(a[..., None], 3, axis=-1)), "srgb16", False)
        if im.mode == "F":
            raise ValueError("32-bit float image has no defined transfer function")
        return Decoded(_SRGB_LUT[np.asarray(im.convert("RGB"))], "srgb8", False)


def jpeg_quality(src) -> int | None:
    """IJG quality estimated from a JPEG's luminance quantisation table, which the encoder scaled from the standard
    table by (200 - 2q) % at q >= 50 and 5000 / q % below; None for a file that is not a JPEG."""
    from PIL import Image
    with Image.open(src) as im:
        tables = getattr(im, "quantization", None)
        if im.format != "JPEG" or not tables:
            return None
        table = tables.get(0) or next(iter(tables.values()))
    s = 100.0 * sum(table) / _LUMA_TABLE_SUM
    return int(round((200.0 - s) / 2.0 if s <= 100.0 else 5000.0 / s))


def clean_scale(it: Item, src) -> int:
    """The downsampling factor the manifest asks of this item: its source's `scale`, or the `low_quality_scale` for
    a JPEG below LOW_QUALITY_JPEG (at 4 the 8 x 8 blocking of a quality-85 web JPEG is below one pixel)."""
    c = clean_settings(it.source)
    scale = int(c.get("scale", 1) or 1)
    low = int(c.get("low_quality_scale", 0) or 0)
    if low > scale and it.file.lower().endswith((".jpg", ".jpeg")):
        q = jpeg_quality(src)
        if q is not None and q < LOW_QUALITY_JPEG:
            scale = low
    return max(scale, 1)


def decode_item(it: Item, root: Path | None = None, tile: int = 0) -> Decoded:
    """Linear RGB of one plain-file item (parquet rows go through load_items), box-downsampled by its source's
    clean scale (clean_scale, recorded in Decoded.scale). Raises ImportError when the decoder package is
    missing, anything else when the file cannot be read."""
    root = Path(root) if root else data_root()
    dec = _decode_file(it, root, tile)
    disk, member = split_member(it.file)
    p = root / "raw" / it.source / disk
    k = clean_scale(it, io.BytesIO(read_member(p, member)) if member is not None else p)
    return dec if k == 1 else Decoded(box_downsample(dec.img, k), dec.domain, dec.is_hdr, k)


def skip_reason(ex: BaseException) -> str:
    """Why an item has no image, as the ledger records it: no_decoder:<package> for a missing optional import
    (never counts as done), decode_error:<Exception>: <message> for a file the decoder rejects."""
    if isinstance(ex, ImportError):
        return f"{NO_DECODER}{ex.name or ex}"
    message = " ".join(str(ex).split())[:160]
    return f"decode_error:{type(ex).__name__}: {message}"


def _decode_file(it: Item, root: Path, tile: int) -> Decoded:
    """decode_item before the clean scale: the file's own pixels. A TIFF is tried through Pillow first; when
    that fails, or yields less than a tile (the first page of a TIFF-container raw is its thumbnail), LibRaw
    gets it."""
    disk, member = split_member(it.file)
    p = root / "raw" / it.source / disk
    ext = Path(member if member is not None else disk).suffix.lower()
    if member is not None:
        data = read_member(p, member)
    elif ext in ZIP_EXT:
        zipfile.ZipFile(p).close()  # listed as a single item only when it was unreadable: raise that
        raise ValueError("zip archive without decodable members")
    elif ext in TAR_EXT:
        tarfile.open(p, "r:").close()
        raise ValueError("tar archive without decodable members")
    else:
        data = None
    if ext in EXR_EXT:
        return Decoded(_decode_exr(data if data is not None else p.read_bytes()).astype(np.float64),
                       "linear_float_hdr", True)
    if ext in HDR_EXT:
        from .hdri import decode_rgbe, read_rgbe
        rgb = decode_rgbe(data, it.file) if data is not None else read_rgbe(p)
        rgb = np.array(rgb, dtype=np.float64)
        rgb[~np.isfinite(rgb)] = 0.0
        return Decoded(np.clip(rgb, 0.0, None, out=rgb), "linear_float_hdr", True)
    if ext in RAW_EXT:
        return Decoded(_decode_raw(io.BytesIO(data) if data is not None else str(p)), "linear16", True)
    if ext in MAT_EXT:
        return Decoded(_decode_rggb_mat(io.BytesIO(data) if data is not None else str(p), _companion_raw(it, root)),
                       "linear16", True)
    src = io.BytesIO(data) if data is not None else p
    try:
        dec = _decode_pil(src)
    except Exception as pil_error:  # noqa: BLE001 - a TIFF Pillow cannot open may be a raw container
        if ext not in TIFF_EXT:
            raise
        try:
            return Decoded(_decode_raw(io.BytesIO(data) if data is not None else str(p)), "linear16", True)
        except Exception:  # noqa: BLE001
            raise pil_error from None
    if ext in TIFF_EXT and tile and min(dec.img.shape[:2]) < tile:
        try:
            return Decoded(_decode_raw(io.BytesIO(data) if data is not None else str(p)), "linear16", True)
        except Exception:  # noqa: BLE001 - not a raw: the small image stands (and is skipped as too small)
            pass
    return dec


def load_items(items: list[Item], root: Path | None = None, errors: dict[Item, str] | None = None,
               scales: dict[Item, int] | None = None) -> dict[Item, np.ndarray]:
    """Linear RGB float64 (H, W, 3) per item, box-downsampled by its clean scale (decode_item; the factor used
    goes to `scales` when given); parquet rows of one shard set are read in a single pass. With `errors` given,
    an item its decoder rejects is left out of the result and recorded there with its skip_reason instead of
    raising."""
    root = Path(root) if root else data_root()
    out: dict[Item, np.ndarray] = {}
    by_parquet: dict[tuple[str, str, str], list[Item]] = {}

    def record(it: Item, e: Exception) -> None:
        if errors is None:
            raise e
        errors[it] = skip_reason(e)

    for it in items:
        if it.row >= 0:
            by_parquet.setdefault((it.source, it.file, it.column), []).append(it)
            continue
        try:
            dec = decode_item(it, root)
        except MemoryError:
            raise               # never a property of the file
        except Exception as e:  # noqa: BLE001 - recorded, or re-raised, by `record`
            record(it, e)
            continue
        out[it] = dec.img
        if scales is not None:
            scales[it] = dec.scale
    for (source, files, column), its in by_parquet.items():
        from .fit import decode_image, iter_parquet_rows
        paths = [root / "raw" / source / f for f in files.split("|")]
        want = {it.row: it for it in its}
        for g, blob in iter_parquet_rows(paths, set(want), column):
            try:
                out[want[g]] = decode_image(blob["bytes"] if isinstance(blob, dict) else blob)
            except MemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                record(want[g], e)
                continue
            if scales is not None:
                scales[want[g]] = 1
    return out


def crop_boxes(item: Item, h: int, w: int, tile: int, crops: int) -> list[tuple[int, int]]:
    """(y0, x0) of `crops` random tile positions, fixed by the item's identity; none if the image is small."""
    if h < tile or w < tile:
        return []
    rng = np.random.default_rng(_seed("crop", item.source, item.file, item.row, tile, crops))
    return [(int(rng.integers(0, h - tile + 1)), int(rng.integers(0, w - tile + 1))) for _ in range(crops)]


def row_seed(item: Item, y0: int, x0: int, tile: int, draw: int) -> int:
    return _seed("draw", item.source, item.file, item.row, y0, x0, tile, draw)


def label_tile(tile_img: np.ndarray, params: DegradeParams, seed: int, ss: int, is_hdr: bool) -> dict[str, float]:
    """Metric vector of one draw: the degraded tile against the prepared clean tile."""
    clean = prepare_tile(tile_img, params, is_hdr)
    return tile_metrics(degrade(tile_img, params, seed, ss, is_hdr), clean)


def label_item(item: Item, img: np.ndarray, dists: Distributions, tile: int, crops: int, per_tile: int,
               ss: int, domain: str = "srgb8", is_hdr: bool | None = None, scale: int = 1) -> list[dict]:
    """Rows of one decoded image; `domain`, `is_hdr` and `scale` (the clean scale it was read at; rows without
    the column are scale 1) are what the decoder reported (Decoded)."""
    is_hdr = item.is_hdr if is_hdr is None else is_hdr
    rows = []
    for y0, x0 in crop_boxes(item, img.shape[0], img.shape[1], tile, crops):
        t = np.ascontiguousarray(img[y0:y0 + tile, x0:x0 + tile])
        for k in range(per_tile):
            seed = row_seed(item, y0, x0, tile, k)
            params = sample_params(np.random.default_rng(seed), dists)
            t0 = time.perf_counter()
            m = label_tile(t, params, seed, ss, is_hdr)
            row = {"source": item.source, "file": item.file, "row": item.row, "y0": y0, "x0": x0, "size": tile,
                   "draw": k, "seed": seed, "is_hdr": is_hdr, "domain": domain, "scale": scale, "ss": ss}
            row.update({f"p_{n}": v for n, v in params.to_dict().items()})
            row.update({f"m_{n}": float(m[n]) for n in STILL_METRICS})
            row["ms"] = (time.perf_counter() - t0) * 1000.0
            rows.append(row)
    return rows


# ---- chunks, ledger, pool -------------------------------------------------------------------------------

def item_key(it: Item) -> tuple[str, str, int]:
    return (it.source, it.file, it.row)


def skip_class(reason: str) -> str:
    """A skip reason without its per-file detail: no_decoder:<package>, decode_error:<exception>, too_small."""
    head, _, rest = reason.partition(":")
    if head == "too_small" or not rest:
        return head
    return f"{head}:{rest.split(':', 1)[0].strip()}"


def plan_items(sources: list[str], n: int, per_tile: int, crops: int, root: Path | None,
               done: set[tuple[str, str, int]] | None = None,
               skip: set[tuple[str, str, int]] | None = None) -> dict[str, list[Item]]:
    """Unlabelled items per source for about n rows in total, the items already labelled (`done`, see
    labelled_items) counting towards n: spread over sources by availability, small sources first. Re-running the
    same command after the corpus grew, with a larger n, therefore labels only the new files. Items in `skip`
    (recorded as undecodable, see ledger_items) are left out without counting towards n."""
    done, skip = done or set(), skip or set()
    listed = {s: list_items(s, root) for s in sources}
    n_done = sum(item_key(it) in done for s in sources for it in listed[s])
    avail = {s: [it for it in listed[s] if item_key(it) not in done and item_key(it) not in skip] for s in sources}
    need = max(math.ceil(n / max(per_tile * crops, 1)) - n_done, 0)
    chosen: dict[str, list[Item]] = {}
    order = sorted(sources, key=lambda s: len(avail[s]))
    for i, s in enumerate(order):
        take = min(len(avail[s]), math.ceil(need / (len(order) - i)))
        chosen[s] = avail[s][:take]
        need -= take
    return {s: chosen[s] for s in sources if chosen[s]}


def _chunks(items: list[Item], size: int) -> list[list[Item]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


_DISTS: dict[str, Distributions] = {}


def _dists(root: str | None) -> Distributions:
    if root not in _DISTS:
        _DISTS[root] = Distributions.load(root)
    return _DISTS[root]


def run_chunk(job: dict) -> dict:
    """Worker: label every item of one chunk, one image in memory at a time. Pure in its inputs; the parent
    writes the shard. An item that cannot be decoded, or is smaller than a tile, goes to `skipped` with a
    reason instead of ending the chunk."""
    t0 = time.perf_counter()
    items = [Item(**d) for d in job["items"]]
    root, tile = Path(job["root"]), job["tile"]
    parquet = [it for it in items if it.row >= 0]
    errors: dict[Item, str] = {}
    imgs = load_items(parquet, root, errors) if parquet else {}
    rows, skipped = [], []
    for it in items:
        if it.row >= 0:
            if it not in imgs:
                reason = errors.get(it, f"decode_error:LookupError: row {it.row} is not in its shard")
                skipped.append({"file": it.file, "row": it.row, "reason": reason})
                continue
            dec = Decoded(imgs[it], "srgb8", False)
        else:
            try:
                dec = decode_item(it, root, tile)
            except MemoryError:
                raise           # never a property of the file: the chunk fails and the next run takes it again
            except Exception as ex:  # noqa: BLE001 - one unreadable file must not end the chunk
                skipped.append({"file": it.file, "row": it.row, "reason": skip_reason(ex)})
                continue
        h, w = dec.img.shape[:2]
        if h < tile or w < tile:
            skipped.append({"file": it.file, "row": it.row, "reason": f"too_small:{h}x{w}"})
            continue
        rows.extend(label_item(it, dec.img, _dists(job["derived_root"]), tile, job["crops"], job["per_tile"],
                               job["ss"], domain=dec.domain, is_hdr=dec.is_hdr, scale=dec.scale))
    return {"chunk": job["chunk"], "rows": rows, "skipped": skipped, "seconds": time.perf_counter() - t0}


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), str(path))
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                    for k, v in r.items()}) + "\n")


def read_ledger(out: Path) -> dict[str, dict]:
    p = out / LEDGER
    done = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                done[e["chunk"]] = e
    return done


def ledger_items(out: Path, config: dict, ledger: dict[str, dict] | None = None
                 ) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], str]]:
    """Under this config: the (source, file, row) keys whose rows are on disk, and the keys recorded as skipped
    with their reason. Labelled keys come from the ledger entries whose config matches and whose shard exists,
    read back by their source/file/row columns only (a deleted shard frees its items; a shard labelled under
    another tile size or draw count does not count). Skipped keys come from the entries themselves."""
    out = Path(out)
    ledger = read_ledger(out) if ledger is None else ledger
    keys: set[tuple[str, str, int]] = set()
    skipped: dict[tuple[str, str, int], str] = {}
    for e in ledger.values():
        if e.get("config") != config:
            continue
        for s in e.get("skipped", []):
            skipped[(e["source"], s["file"], s["row"])] = s["reason"]
        p = out / e["file"] if e.get("file") else None
        if p is None or not e.get("n_rows", 1) or not p.exists():
            continue
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(str(p), columns=["source", "file", "row"])
            keys.update(zip(t.column("source").to_pylist(), t.column("file").to_pylist(), t.column("row").to_pylist()))
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["source"], r["file"], r["row"]))
    for k in keys:
        skipped.pop(k, None)
    return keys, skipped


def labelled_items(out: Path, config: dict, ledger: dict[str, dict] | None = None) -> set[tuple[str, str, int]]:
    """(source, file, row) of every item whose rows are on disk under this config (ledger_items)."""
    return ledger_items(out, config, ledger)[0]


def read_labels(out: Path) -> list[dict]:
    """Every row of every shard under out, in shard then row order (NaN metrics read back as NaN)."""
    out = Path(out)
    rows = []
    for p in sorted(out.glob("labels-*.parquet")):
        import pyarrow.parquet as pq
        rows.extend(pq.read_table(str(p)).to_pylist())
    for p in sorted(out.glob("labels-*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({k: (math.nan if v is None and k.startswith("m_") else v) for k, v in r.items()})
    return rows


POOL_ATTEMPTS = 5   # process pools in a row that finish nothing before a run gives up on the remaining chunks


def _run_pool(todo: list[dict], jobs: int, finish, fail, log, attempts: int = POOL_ATTEMPTS) -> None:
    """Run the chunks in a process pool, handing each result to `finish` in chunk order and a chunk whose
    worker raised to `fail`. A worker that dies takes every unfinished future down with BrokenProcessPool
    (seen on Windows with Python 3.12: now and then a freshly spawned worker starts with an invalid queue
    handle, WinError 6, and dies on its first job; importing scipy in the child is the strongest trigger
    found). A chunk is only persisted once its result is in, so the unfinished ones are resubmitted to a
    fresh pool; the run gives up after `attempts` pools in a row that finished nothing. (OPENBLAS_NUM_THREADS=1
    in the workers cuts the bad spawns of an import-only pool but not the labeller's, so it is not set here.)"""
    pending, idle = list(todo), 0
    while pending:
        futures, lost = [], []
        with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as ex:
            try:
                for job in pending:
                    futures.append((job, ex.submit(run_chunk, job)))
            except BrokenProcessPool:
                pass
            for job, fut in futures:
                try:
                    finish(fut.result(), job)
                except BrokenProcessPool:
                    lost.append(job)
                except Exception as e:  # noqa: BLE001 - a failed chunk is left for the next run
                    fail(job, e)
        lost += pending[len(futures):]
        if not lost:
            return
        idle = idle + 1 if len(lost) == len(pending) else 0
        if idle >= attempts:
            raise BrokenProcessPool(f"{len(lost)} chunk(s) still unfinished after {attempts} process pools "
                                    "in a row finished nothing")
        log(f"label: the process pool broke after {len(pending) - len(lost)} of {len(pending)} chunk(s); "
            f"the rest go to a fresh pool")
        pending = lost


def run_label(sources: list[str], n: int, out: Path | None = None, per_tile: int = 4, crops: int = 4,
              tile: int = 256, jobs: int | None = None, fmt: str = "parquet", chunk: int = 8, ss: int = 1,
              root: Path | None = None, derived_root: Path | None = None, retry_skipped: bool = False,
              log=print) -> dict:
    """Label about n rows from the given sources into shards under out; resumable per item (a re-run after the
    corpus grew labels only the new files, into new chunks); returns a summary. Items recorded as skipped for
    a permanent reason are left alone unless retry_skipped; ones skipped for a missing decoder are retried, and
    so are the items of a chunk that failed (summary["failed"]) once its cause is fixed."""
    root = Path(root) if root else data_root()
    out = Path(out) if out else root / "labels"
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or os.cpu_count() or 1
    config = dict(zip(CONFIG_KEYS, (tile, per_tile, crops, ss, fmt)))
    ext = ".parquet" if fmt == "parquet" else ".jsonl"
    done = read_ledger(out)
    have, skipped_before = ledger_items(out, config, done)
    skip = set() if retry_skipped else {k for k, r in skipped_before.items() if not r.startswith(NO_DECODER)}
    kept = [e for e in done.values() if e.get("config") == config
            and (not e.get("file") or (out / e["file"]).exists())]
    skipped, rows_old = len(kept), sum(e["n_rows"] for e in kept)
    next_index: dict[str, int] = {}
    for cid, e in done.items():  # chunk ids never repeat, whether or not their shard survived
        next_index[e["source"]] = max(next_index.get(e["source"], 0), int(cid.rsplit("-", 1)[1]) + 1)
    todo = []
    for source, items in plan_items(sources, n, per_tile, crops, root, done=have, skip=skip).items():
        for i, part in enumerate(_chunks(items, chunk), start=next_index.get(source, 0)):
            cid = f"{source}-{i:05d}"
            todo.append({"chunk": cid, "source": source, "items": [asdict(it) for it in part], "tile": tile,
                         "crops": crops, "per_tile": per_tile, "ss": ss, "root": str(root),
                         "derived_root": str(derived_root) if derived_root else None})
    log(f"label: {len(todo)} chunks to run, {skipped} shards already done ({rows_old} rows, {len(have)} items, "
        f"{len(skip)} items skipped before), jobs={jobs}")
    t0 = time.perf_counter()
    rows_new, ms_total, files = 0, 0.0, []
    skips: dict[str, int] = {}
    items_skipped: list[dict] = []
    failed: list[dict] = []

    def finish(res: dict, job: dict) -> None:
        nonlocal rows_new, ms_total
        path = out / f"labels-{res['chunk']}{ext}"
        if res["rows"]:
            _write_shard(path, res["rows"])
            files.append(path.name)
        entry = {"chunk": res["chunk"], "source": job["source"], "file": path.name if res["rows"] else "",
                 "n_rows": len(res["rows"]), "n_items": len(job["items"]), "seconds": round(res["seconds"], 3),
                 "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": config}
        if res["skipped"]:
            entry["skipped"] = res["skipped"]
        with open(out / LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        rows_new += len(res["rows"])
        ms_total += sum(r["ms"] for r in res["rows"])
        note = ""
        if res["skipped"]:
            by = {}
            for s in res["skipped"]:
                by[skip_class(s["reason"])] = by.get(skip_class(s["reason"]), 0) + 1
                skips[skip_class(s["reason"])] = skips.get(skip_class(s["reason"]), 0) + 1
                items_skipped.append({"source": job["source"], **s})
            note = ", skipped " + ", ".join(f"{k} x{v}" for k, v in sorted(by.items()))
        log(f"  {res['chunk']}: {len(res['rows'])} rows in {res['seconds']:.1f} s{note}")

    def fail(job: dict, e: BaseException) -> None:
        failed.append({"chunk": job["chunk"], "source": job["source"], "n_items": len(job["items"]),
                       "error": f"{type(e).__name__}: {e}"})
        log(f"  {job['chunk']}: FAILED {failed[-1]['error']}")

    if jobs == 1 or len(todo) <= 1:
        for job in todo:
            try:
                finish(run_chunk(job), job)
            except Exception as e:  # noqa: BLE001 - a failed chunk is left for the next run
                fail(job, e)
    else:
        _run_pool(todo, jobs, finish, fail, log)
    seconds = time.perf_counter() - t0
    summary = {"rows": rows_old + rows_new, "rows_new": rows_new, "chunks": len(todo), "chunks_skipped": skipped,
               "items_done_before": len(have), "items_skipped_before": len(skip),
               "skipped": dict(sorted(skips.items())), "items_skipped": items_skipped,
               "chunks_failed": len(failed), "failed": failed,
               "seconds": round(seconds, 2), "rows_per_s_per_proc": (rows_new / (ms_total / 1000.0)) if ms_total else 0.0,
               "out": str(out), "files": files}
    log(f"label: {rows_new} new rows in {seconds:.1f} s, {summary['rows_per_s_per_proc']:.2f} rows/s per process"
        + (f", {len(items_skipped)} items skipped: " + ", ".join(f"{k} x{v}" for k, v in summary["skipped"].items())
           if items_skipped else "")
        + (f", {len(failed)} chunk(s) FAILED and left for the next run" if failed else ""))
    return summary
