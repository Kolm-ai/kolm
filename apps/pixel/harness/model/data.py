"""Training pairs rendered on the fly: state A -> state B of one clean tile, B's metric vector as conditioning.

No pair is stored; (item, crop, seeds) reproduce it. With probability P_RESTORE the target B is the prepared
clean tile (the restore direction); otherwise B is a second draw that keeps A's scene range, so the pair is
an edit toward a known state (exposure, white balance, noise, sharpening, ...). Tensors live in the log
domain `log1p(LOG_GAIN * linear)` with LOG_GAIN = 4096: one 12-bit LSB is where the log turns linear and the
clip encodes to 8.3, so above that floor an L1 error is a relative error. The network's output precision is
absolute in this domain (a flat field comes back with std 0.001 to 0.002 at mid-grey on run v6's checkpoints), and at the
earlier gain of 8 that floor was 20 to 50 % of the signal at 0.001 of the clip, where the deepest demo wedge
patches sit (runs v5 and v6: 4 to 12 dB against the camera's 26); at 4096 it is 100x finer there and unchanged
in the mid-tones and highlights (the slope at the clip is 1.0 against 0.9 before).
Training-set choices that are not the sampler's: lens residual blur above MAX_BLUR_PX at the pitch is
redrawn (the plan's open tail: a 46 px blur on a 256 px tile is not a restoration task), and clean LDR
references from parquet sources are exported once to PNG so loader workers read files, not shards."""
from __future__ import annotations

import fnmatch
import hashlib
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..data import data_root
from ..data.decodedstore import decode_clean, decoded_path, read_decoded, write_decoded
from ..data.degrade import DegradeParams, Distributions, degrade, prepare_tile, sample_params, tile_shading
from ..data.label import PARQUET_SOURCES, Item, clean_settings, list_items, load_items, scene_of, skip_reason
from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS
from ..io import save_png
from ..colorspace import srgb_to_linear
from .deadleaves import DEAD_LEAVES, dead_leaves

P_RESTORE = 0.7
P_IDENTITY = 0.15        # the pair is (clean, clean): without it the v2 model removed 30 % of the texture energy of a clean input
MAX_BLUR_PX = 8.0
MAX_REDRAWS = 32
LOG_GAIN = 4096.0        # one 12-bit LSB below the clip: the log domain's floor, see the module docstring
P_RESTORE_CAPTURE = 0.7  # restore pairs drawn with the sampler's raw "capture" profile (no IQA-ladder halos or
                         # posterise); the rest keep the "iqa" profile so phone JPEG inputs are still covered
HOLDOUT_MOD = 16          # scenes (scene_of) whose identity hash is 0 mod this are the validation split
TILES_PER_IMAGE = 16      # cap on consecutive draws from one decoded image in the stream, see TileSource.draws_on
TILE_AREAS_PER_DRAW = 8   # one draw per this many tile areas the image holds: a 13 MP picture gives the cap, a 2k HDRI 4,
                          # a KADID reference 1; at 4 draws on a 2k HDRI the stream measured no faster (212 vs 220 ms per
                          # sample, the sampler is the cost there), on a 4 s decode of a 50 MP JPEG the cap is what feeds the GPU
CACHE_BYTES = 1.5e9       # decoded images kept per loader worker: the 64 2k HDRIs of before, or ten 13 MP pictures
METRIC_COND_DIM = 2 * len(STILL_METRICS)
CAPTURE_DIM = 7
COND_DIM = METRIC_COND_DIM + CAPTURE_DIM
P_CAPTURE_KNOWN = 0.7
UNKNOWN_CAPTURE = np.zeros(CAPTURE_DIM, dtype=np.float32)
CLEAN_DIR = "clean_refs"  # <data_root>/derived/clean_refs/<source>/<row>.png


def encode(x: np.ndarray) -> np.ndarray:
    return np.log1p(LOG_GAIN * np.maximum(np.asarray(x, dtype=np.float32), 0.0)).astype(np.float32)


def decode(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.maximum(np.asarray(y, dtype=np.float64), 0.0)) / LOG_GAIN


def blur_px(p: DegradeParams) -> float:
    return p.residual_blur_um / p.pixel_pitch_um if p.camera_on and p.pixel_pitch_um > 0 else 0.0


GLOBAL_FIELDS = ("expose", "exposure_stops", "scene_peak_over_median",
                 "contrast_gain", "mean_shift", "saturation", "cast_r", "cast_b")
"""The fields no tile can identify and the metric vector does not carry: exposure, the sRGB tone affine,
saturation and the colour cast. Every target keeps the input's values; changing them is a closed-form edit
applied outside the model (a scalar multiply, a channel multiply, an sRGB affine)."""


def restore_target(a: DegradeParams) -> DegradeParams:
    """The identity state in A's rendering intent: degrade(tile, it) is A's prepared clean tile with A's
    global tone and colour (GLOBAL_FIELDS) and nothing else."""
    return DegradeParams(**{k: getattr(a, k) for k in GLOBAL_FIELDS})


def draw_state(rng: np.random.Generator, dists: Distributions, profile: str = "iqa") -> DegradeParams:
    p = sample_params(rng, dists, profile)
    for _ in range(MAX_REDRAWS):
        if blur_px(p) <= MAX_BLUR_PX:
            break
        p = sample_params(rng, dists, profile)
    return p


def draw_pair(rng: np.random.Generator, dists: Distributions) -> tuple[DegradeParams, DegradeParams, bool]:
    """(A, B, restore): identity with P_IDENTITY (A is the clean prepared tile itself), else restore with
    P_RESTORE (B is A's clean prepared tile), else an edit (B is a second drawn state on A's scene range).
    B always keeps A's exposure: the metric vector does not carry the target's brightness, so an edit that
    changed it would be unidentifiable and the model would regress every input toward the mean brightness
    (the third run brightened dark inputs 2x and darkened mid-grey ones 0.75x). An exposure change is a
    scalar multiply in linear light and is applied outside the model. A restore input is drawn with the
    sampler's raw "capture" profile with P_RESTORE_CAPTURE, else the "iqa" profile: the seventh bullet of
    section 7 of DEGRADATION_PLAN.md (a prior of IQA-ladder halos made the model darken every small bright
    patch of the demo wedge)."""
    if rng.random() < P_IDENTITY:
        t = restore_target(draw_state(rng, dists))
        return t, t, True
    if rng.random() < P_RESTORE:
        a = draw_state(rng, dists, "capture" if rng.random() < P_RESTORE_CAPTURE else "iqa")
        return a, restore_target(a), True
    a = draw_state(rng, dists)
    b = draw_state(rng, dists)
    b = DegradeParams(**{**b.to_dict(), **{k: getattr(a, k) for k in GLOBAL_FIELDS}})
    return a, b, False


class Normaliser:
    """Per-metric robust z-score of the conditioning vector; NaN -> 0 with a mask channel."""
    def __init__(self, mu: np.ndarray, sd: np.ndarray):
        self.mu, self.sd = np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32)

    @classmethod
    def fit(cls, vectors: np.ndarray) -> "Normaliser":
        v = np.asarray(vectors, dtype=np.float64)
        mu = np.nanmedian(v, axis=0)
        sd = 1.4826 * np.nanmedian(np.abs(v - mu), axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        return cls(np.nan_to_num(mu), sd)

    def __call__(self, metrics: dict) -> np.ndarray:
        raw = np.array([metrics[m] for m in STILL_METRICS], dtype=np.float64)
        ok = np.isfinite(raw)
        z = np.where(ok, (np.nan_to_num(raw) - self.mu) / self.sd, 0.0)
        return np.concatenate([np.clip(z, -6, 6), ok.astype(np.float64)]).astype(np.float32)

    def state(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "metrics": list(STILL_METRICS)}

    @classmethod
    def from_state(cls, d: dict) -> "Normaliser":
        return cls(np.array(d["mu"]), np.array(d["sd"]))


def input_shading(a: DegradeParams, h: int, w: int) -> np.ndarray | None:
    """A's lens shading as it stands in the input (None when the camera is off, corrected, or flat)."""
    if not a.camera_on or a.shading_corrected or a.shading_corner_stops <= 0:
        return None
    return tile_shading(h, w, a)


def render_pair(tile: np.ndarray, a: DegradeParams, b: DegradeParams, seed_a: int, seed_b: int,
                is_hdr: bool, ss: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    """(A tile, B tile, B's metric vector against the clean tile in B's rendering intent), all linear. Lens shading is
    treated as scene: the target carries A's uncorrected shading unchanged (a tile cannot tell shading from
    scene lighting, and real pipelines correct it from calibration tables, not by inference), and B's own
    shading draw is rendered corrected so the target carries no second map. The shading enters B's render as
    scene lighting, ahead of B's camera and ISP, where the input had it: multiplying it in after the ISP (runs
    four to seven) left a third of the restore targets up to 2.5x darker than their inputs at mid-tones whenever
    the tone affine was on, since the sRGB affine does not commute with a multiply (DEGRADATION_PLAN section 7).
    The label reports the shading."""
    xa = degrade(tile, a, seed_a, ss, is_hdr)
    b_render = b if b.shading_corrected else DegradeParams(**{**b.to_dict(), "shading_corrected": 1})
    shade = input_shading(a, tile.shape[0], tile.shape[1])
    xb = degrade(tile, b_render, seed_b, ss, is_hdr, scene_gain=shade)
    ref = xb if shade is None and b == restore_target(b) else degrade(tile, restore_target(b), seed_b, ss, is_hdr)
    return xa, xb, tile_metrics(xb, ref)


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def split_of(item: Item) -> str:
    """Hashed on the scene, not the file: the 1k and 2k copies of one Poly Haven HDRI land on the same side. A
    scene the source's manifest `clean.holdout` names is "test": never trained on, never validated on, kept for
    the real-pair evaluation (RawNIND's test reserve)."""
    scene = scene_of(item)
    if any(fnmatch.fnmatchcase(scene, pat) for pat in clean_settings(item.source).get("holdout", [])):
        return "test"
    return "val" if _h("split", item.source, scene, item.row) % HOLDOUT_MOD == 0 else "train"


def export_clean_refs(sources: list[str], root: Path | None = None, log=print) -> None:
    """Parquet-backed clean references -> PNG files under derived/clean_refs, once."""
    root = Path(root) if root else data_root()
    for s in sources:
        if s not in PARQUET_SOURCES:
            continue
        d = root / "derived" / CLEAN_DIR / s
        items = list_items(s, root)
        missing = [it for it in items if not (d / f"{it.row}.png").exists()]
        if not missing:
            continue
        d.mkdir(parents=True, exist_ok=True)
        log(f"clean refs: exporting {len(missing)} images of {s}")
        for it, img in load_items(missing, root).items():
            save_png(d / f"{it.row}.png", np.clip(img, 0.0, 1.0))


def row_image(it: Item, root: Path) -> np.ndarray:
    """A parquet row's clean image: its PNG export (made on first use), linear float32."""
    from PIL import Image
    p = Path(root) / "derived" / CLEAN_DIR / it.source / f"{it.row}.png"
    if not p.exists():
        export_clean_refs([it.source], root, log=lambda *_: None)
    with Image.open(p) as im:
        return srgb_to_linear(np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0).astype(np.float32)


class TileSource:
    """Clean tiles of one split: items listed once, decoded images cached per process (LRU under a byte budget,
    since a 2k HDRI is 25 MB and a 50 MP picture at clean scale 2 is 160 MB). With `store` a plain-file item comes
    from the decoded store (data.decodedstore) when it is there, memmapped, and is stored when it is not: the same
    array either way (float16 where float16 holds it), so the stream does not change, only its decode cost. With `audit` the items are
    the clean-target audit's (data.targetaudit.filter_targets: a source with a `clean.max_halo` keeps only its
    audited pictures at or below it), the per-source report in `audit_report`."""
    def __init__(self, sources: list[str], split: str, tile: int, root: Path | None = None,
                 cache_bytes: float = CACHE_BYTES, tiles_per_image: int = 1, weights: dict[str, float] | None = None,
                 store: bool = False, audit: bool = False):
        self.root = Path(root) if root else data_root()
        self.store, self._store_writes = bool(store), bool(store)
        self.store_hits = self.store_misses = 0
        self.tile, self.split, self.tiles_per_image = tile, split, tiles_per_image
        self._cur: Item | None = None
        self._left = 0
        self.dropped: dict[Item, str] = {}   # items whose file the decoder rejected, with the skip_reason
        self.items = [it for s in sources for it in list_items(s, self.root) if split_of(it) == split]
        self.audit_report: dict[str, dict] = {}
        if audit:
            from ..data.targetaudit import filter_targets
            self.items, self.audit_report = filter_targets(self.items, self.root)
        if not self.items:
            raise ValueError(f"no items for split {split!r} in {sources}")
        self._w = None                        # per-item draw weight, the item's source's (1 for a source not named)
        if weights:
            self._w = np.array([float(weights.get(it.source, 1.0)) for it in self.items])
            if (self._w < 0).any() or self._w.sum() <= 0:
                raise ValueError(f"source weights must be non-negative with a positive sum: {weights}")
        self._cache: OrderedDict[Item, np.ndarray] = OrderedDict()
        self._max_bytes, self._bytes = float(cache_bytes), 0

    def image(self, it: Item) -> np.ndarray:
        if it in self._cache:
            self._cache.move_to_end(it)
            return self._cache[it]
        if it.row >= 0:
            img = row_image(it, self.root)
        elif self.store:
            img = self._from_store(it)
        else:
            img = decode_clean(it, self.root)
        self._cache[it] = img
        self._bytes += img.nbytes
        while self._bytes > self._max_bytes and len(self._cache) > 1:
            self._bytes -= self._cache.popitem(last=False)[1].nbytes
        return img

    def _from_store(self, it: Item) -> np.ndarray:
        """The item's stored array, or its decode, stored for the next reader. A failed write (a full disk) turns
        writing off for this process with one warning; reading goes on."""
        p = decoded_path(it, self.root)
        img = read_decoded(p)
        if img is not None:
            self.store_hits += 1
            return img
        img = decode_clean(it, self.root)
        self.store_misses += 1
        if p is not None and self._store_writes and not write_decoded(p, img):
            self._store_writes = False
            warnings.warn(f"decoded store: writing {p} failed, this process decodes without storing from here on",
                          RuntimeWarning, stacklevel=2)
        return img

    def draws_on(self, img: np.ndarray) -> int:
        """Consecutive draws kept on one decoded image: one per TILE_AREAS_PER_DRAW tile areas it holds, at most
        tiles_per_image. Small references give one draw, a big picture amortises its decode over many."""
        h, w = img.shape[:2]
        return max(1, min(self.tiles_per_image, (h * w) // (self.tile * self.tile * TILE_AREAS_PER_DRAW)))

    def pick(self, rng: np.random.Generator) -> Item:
        """A random item: uniform, or with probability proportional to its source's weight."""
        if self._w is None:
            return self.items[int(rng.integers(len(self.items)))]
        return self.items[int(rng.choice(len(self.items), p=self._w / self._w.sum()))]

    def shares(self) -> dict[str, float]:
        """Each source's probability of supplying the next fresh draw (its items' weights over the total)."""
        w = self._w if self._w is not None else np.ones(len(self.items))
        out: dict[str, float] = {}
        for it, v in zip(self.items, w):
            out[it.source] = out.get(it.source, 0.0) + float(v)
        return {k: v / w.sum() for k, v in out.items()}

    def random_tile(self, rng: np.random.Generator) -> tuple[np.ndarray, Item, int, int]:
        """A random tile; with tiles_per_image > 1 consecutive draws stay on one decoded image (the stream's
        throughput is bound by decoding whole HDRIs, not by the sampler). An item whose file the decoder rejects
        is dropped and the draw moves on: one file never ends a training run."""
        for _ in range(64):
            if not self.items:
                raise ValueError("every item failed to decode")
            fresh = self._left <= 0 or self._cur is None
            it = self.pick(rng) if fresh else self._cur
            try:
                img = self.image(it)
            except Exception as e:  # noqa: BLE001 - recorded with its reason; a MemoryError here is the file's size
                self.drop(it, e)
                continue
            if fresh:
                self._cur, self._left = it, self.draws_on(img)
            self._left -= 1
            h, w = img.shape[:2]
            if h >= self.tile and w >= self.tile:
                y0, x0 = int(rng.integers(0, h - self.tile + 1)), int(rng.integers(0, w - self.tile + 1))
                return np.ascontiguousarray(img[y0:y0 + self.tile, x0:x0 + self.tile], dtype=np.float64), it, y0, x0
            self._left = 0
        raise ValueError("no item large enough for the tile")

    def drop(self, it: Item, e: BaseException) -> None:
        """Forget an item whose file the decoder rejects (the 188-megapixel featured picture past Pillow's pixel cap
        that ended the tenth run at step 4200 from a loader worker; a torn download): it leaves this process's list
        with its skip_reason in `dropped`, and a RuntimeWarning names it in the run's log."""
        self.dropped[it] = skip_reason(e)
        k = self.items.index(it)
        del self.items[k]
        if self._w is not None:
            self._w = np.delete(self._w, k)
        if self._cur == it:
            self._cur, self._left = None, 0
        warnings.warn(f"{it.source}/{it.file} dropped from the sampler: {self.dropped[it]}", RuntimeWarning, stacklevel=2)


def capture_descriptor(airy_first_zero_px: float, residual_sigma_px: float, electrons_at_clip: float,
                       read_noise_e: float, bit_depth: int, bayer: bool) -> np.ndarray:
    """What a calibrated pipeline knows about its own capture, in the model's units (the clip is 1.0, the pixel
    is 1 px): the Airy first zero at 550 nm in px, the residual Gaussian sigma in px, log10 of three noise
    scales relative to the clip (shot sigma at the clip = 1/sqrt(electrons), the read noise, one LSB), each
    floored and shifted so a clean render sits at -1 or -2 and a phone at ISO 800 near 0 or +1, the Bayer flag
    and known = 1. A JPEG from the web has none of this: UNKNOWN_CAPTURE (all zero, known = 0), and training
    withholds the descriptor with probability 1 - P_CAPTURE_KNOWN so one model serves both cases. The rule of
    section 7 of DEGRADATION_PLAN.md applied to the input: a blur or noise level a tile cannot identify is
    guessed at its conditional mean, and the known PSF is what a deconvolution needs."""
    e = float(electrons_at_clip)
    shot = max(math.log10(1.0 / math.sqrt(e)), -4.0) if e > 0 else -4.0
    read = max(math.log10(float(read_noise_e) / e), -5.0) if e > 0 and read_noise_e > 0 else -5.0
    lsb = -float(bit_depth) * math.log10(2.0) if bit_depth > 0 else -6.0
    return np.array([airy_first_zero_px / 2.0, residual_sigma_px / 4.0, shot + 3.0, read + 4.0, lsb + 4.0,
                     float(bool(bayer)), 1.0], dtype=np.float32)


def capture_of_params(p: DegradeParams) -> np.ndarray:
    """The descriptor of a sampler input: its camera stage's optics and noise, or a clean render when the camera
    is off. The sampler clips at radiance 1.0, where white_fill x full well electrons sit."""
    if not p.camera_on:
        return capture_descriptor(0.0, 0.0, 0.0, 0.0, 0, False)
    pitch = p.pixel_pitch_um or 1.0
    return capture_descriptor(1.22 * 0.55 * p.f_number / pitch, p.residual_blur_um / pitch,
                              min(p.white_fill, 1.0) * p.full_well_e, p.read_noise_e, p.bit_depth, bool(p.bayer))


def cond_vector(norm: Normaliser, s: dict, known: bool = True) -> np.ndarray:
    """A sample's full conditioning: the target's normalised metrics, then the input's capture descriptor (a real
    pair's measured one under "capture", a rendered pair's from its parameters; UNKNOWN_CAPTURE when withheld)."""
    cap = (s["capture"] if "capture" in s else capture_of_params(s["a"])) if known else UNKNOWN_CAPTURE
    return np.concatenate([norm(s["metrics"]), cap]).astype(np.float32)


def make_sample(src: TileSource, dists: Distributions, rng: np.random.Generator, ss: int = 1) -> dict:
    """One rendered pair with its bookkeeping; the conditioning is the raw metric dict."""
    tile, it, y0, x0 = src.random_tile(rng)
    return _rendered(tile, it, y0, x0, dists, rng, ss)


def dead_leaves_sample(tile: int, dists: Distributions, rng: np.random.Generator, ss: int = 1) -> dict:
    """One rendered pair on a procedural colour dead-leaves tile (deadleaves.py): its item is (DEAD_LEAVES, seed),
    the seed reproducing the tile."""
    seed = int(rng.integers(1 << 62))
    clean = dead_leaves(np.random.default_rng(seed), tile)
    return _rendered(clean, Item(DEAD_LEAVES, str(seed)), 0, 0, dists, rng, ss)


def _rendered(tile: np.ndarray, it: Item, y0: int, x0: int, dists: Distributions, rng: np.random.Generator,
              ss: int) -> dict:
    a, b, restore = draw_pair(rng, dists)
    sa, sb = int(rng.integers(1 << 62)), int(rng.integers(1 << 62))
    xa, xb, m = render_pair(tile, a, b, sa, sb, it.is_hdr, ss)
    return {"x": xa, "y": xb, "metrics": m, "restore": restore, "a": a, "b": b, "item": it, "y0": y0, "x0": x0,
            "seed_a": sa, "seed_b": sb}


def fixed_set(sources: list[str], split: str, n: int, tile: int, seed: int, root: Path | None = None,
              ss: int = 1, log=print, audit: bool = False) -> list[dict]:
    """n reproducible samples (seeded by split, n, tile and seed): the validation set and the calibration draw. With
    `audit` the clean items are the target audit's (TileSource)."""
    src, dists = TileSource(sources, split, tile, root, audit=audit), Distributions.load(root)
    rng = np.random.default_rng(_h("fixed", split, n, tile, seed))
    out = []
    for i in range(n):
        out.append(make_sample(src, dists, rng, ss))
        if log and (i + 1) % 32 == 0:
            log(f"  fixed set {split}: {i + 1}/{n}")
    return out


def real_sample(store, rng: np.random.Generator) -> dict:
    """One stored real pair's tile (data.realstore.RealStore) as a restore sample: the capture is the input, the
    gain-matched reference the target, the conditioning the reference's own metrics (as render_pair conditions a
    restore on its target), the measured noise law the capture descriptor in place of rendered parameters, and a
    random rotation and flip on both since a few hundred scenes are drawn from for hours."""
    s = store.random_tile(rng)
    x, y = s["x"], s["y"]
    k = int(rng.integers(4))
    if k:
        x, y = np.rot90(x, k), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    return {"x": x, "y": y, "metrics": tile_metrics(y, y), "restore": True, "capture": s["capture"], "real": True,
            "pair": s["pair"], "y0": s["y0"], "x0": s["x0"], "clipped": s["clipped"]}


def real_fixed_set(sources: list[str], split: str, n: int, tile: int, seed: int, root: Path | None = None,
                   subpix: bool = False) -> list[dict]:
    """n reproducible real-pair samples of a stored split: the real validation set. With `subpix` the targets are
    read at each pair's sub-pixel residual (RealStore's subpix); the tiles are the same either way."""
    from ..data.realstore import RealStore
    store = RealStore(list(sources), split, root, tile, subpix=subpix)
    rng = np.random.default_rng(_h("fixed-real", split, n, tile, seed))
    return [real_sample(store, rng) for _ in range(n)]


class PairStream:
    """Infinite iterator of (x, y, cond, restore, real) in the log domain, for one loader worker: rendered
    pairs of the clean sources (drawn by `weights`, uniform without), and with probability `p_real` a tile of the
    stored real pairs of `real` (data.realstore); of the rendered pairs a share `p_dead_leaves` is drawn on a
    procedural colour dead-leaves tile (deadleaves.py) in place of a source tile. `real` says the sample is a stored
    real pair (train's --shift-real searches its registration); `audit` draws the clean items the target audit
    keeps (TileSource); `subpix_real` reads each real pair's reference at its sub-pixel residual (RealStore)."""
    def __init__(self, sources: list[str], tile: int, normaliser: Normaliser, seed: int, root: Path | None = None,
                 ss: int = 1, real: tuple[str, ...] = (), p_real: float = 0.0, weights: dict[str, float] | None = None,
                 p_dead_leaves: float = 0.0, decoded_store: bool = True, audit: bool = False,
                 subpix_real: bool = False):
        self.audit, self.subpix_real = bool(audit), bool(subpix_real)
        self.sources, self.tile, self.norm, self.seed, self.root, self.ss = sources, tile, normaliser, seed, root, ss
        self.decoded_store = bool(decoded_store)
        self.real, self.p_real, self.weights = tuple(real), float(p_real), weights
        self.p_dead_leaves = float(p_dead_leaves)
        if self.p_real and not self.real:
            raise ValueError("p_real > 0 needs real sources")
        if not 0.0 <= self.p_dead_leaves <= 1.0:
            raise ValueError(f"p_dead_leaves is a probability, got {p_dead_leaves}")

    def __iter__(self):
        import torch
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi else 0
        src = TileSource(self.sources, "train", self.tile, self.root, tiles_per_image=TILES_PER_IMAGE,
                         weights=self.weights, store=self.decoded_store, audit=self.audit)
        store = None
        if self.real and self.p_real > 0:
            from ..data.realstore import RealStore
            store = RealStore(list(self.real), "train", self.root, self.tile, subpix=self.subpix_real)
        dists = Distributions.load(self.root)
        rng = np.random.default_rng(_h("stream", self.seed, wid))
        while True:
            if store is not None and rng.random() < self.p_real:
                s = real_sample(store, rng)
            elif self.p_dead_leaves > 0 and rng.random() < self.p_dead_leaves:
                s = dead_leaves_sample(self.tile, dists, rng, self.ss)
            else:
                s = make_sample(src, dists, rng, self.ss)
            x = torch.from_numpy(encode(s["x"])).permute(2, 0, 1).contiguous()
            y = torch.from_numpy(encode(s["y"])).permute(2, 0, 1).contiguous()
            c = torch.from_numpy(cond_vector(self.norm, s, rng.random() < P_CAPTURE_KNOWN))
            yield x, y, c, bool(s["restore"]), bool(s.get("real", False))
