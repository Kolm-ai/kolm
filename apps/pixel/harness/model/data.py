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

import hashlib
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..data import data_root
from ..data.degrade import DegradeParams, Distributions, degrade, prepare_tile, sample_params, tile_shading
from ..data.label import PARQUET_SOURCES, Item, list_items, load_items, scene_of
from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS
from ..io import save_png
from ..colorspace import srgb_to_linear

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
    """Hashed on the scene, not the file: the 1k and 2k copies of one Poly Haven HDRI land on the same side."""
    return "val" if _h("split", item.source, scene_of(item), item.row) % HOLDOUT_MOD == 0 else "train"


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


class TileSource:
    """Clean tiles of one split: items listed once, decoded images cached per process (LRU under a byte budget,
    since a 2k HDRI is 25 MB and a 50 MP picture at clean scale 2 is 160 MB)."""
    def __init__(self, sources: list[str], split: str, tile: int, root: Path | None = None,
                 cache_bytes: float = CACHE_BYTES, tiles_per_image: int = 1):
        self.root = Path(root) if root else data_root()
        self.tile, self.split, self.tiles_per_image = tile, split, tiles_per_image
        self._cur: Item | None = None
        self._left = 0
        self.items = [it for s in sources for it in list_items(s, self.root) if split_of(it) == split]
        if not self.items:
            raise ValueError(f"no items for split {split!r} in {sources}")
        self._cache: OrderedDict[Item, np.ndarray] = OrderedDict()
        self._max_bytes, self._bytes = float(cache_bytes), 0

    def image(self, it: Item) -> np.ndarray:
        if it in self._cache:
            self._cache.move_to_end(it)
            return self._cache[it]
        if it.row >= 0:
            from PIL import Image
            p = self.root / "derived" / CLEAN_DIR / it.source / f"{it.row}.png"
            if not p.exists():
                export_clean_refs([it.source], self.root, log=lambda *_: None)
            with Image.open(p) as im:
                img = srgb_to_linear(np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0)
        else:
            img = load_items([it], self.root)[it]
        img = img.astype(np.float32)
        self._cache[it] = img
        self._bytes += img.nbytes
        while self._bytes > self._max_bytes and len(self._cache) > 1:
            self._bytes -= self._cache.popitem(last=False)[1].nbytes
        return img

    def draws_on(self, img: np.ndarray) -> int:
        """Consecutive draws kept on one decoded image: one per TILE_AREAS_PER_DRAW tile areas it holds, at most
        tiles_per_image. Small references give one draw, a big picture amortises its decode over many."""
        h, w = img.shape[:2]
        return max(1, min(self.tiles_per_image, (h * w) // (self.tile * self.tile * TILE_AREAS_PER_DRAW)))

    def random_tile(self, rng: np.random.Generator) -> tuple[np.ndarray, Item, int, int]:
        """A random tile; with tiles_per_image > 1 consecutive draws stay on one decoded image (the stream's
        throughput is bound by decoding whole HDRIs, not by the sampler)."""
        for _ in range(64):
            if self._left > 0 and self._cur is not None:
                it = self._cur
                self._left -= 1
                img = self.image(it)
            else:
                it = self.items[int(rng.integers(len(self.items)))]
                img = self.image(it)
                self._cur, self._left = it, self.draws_on(img) - 1
            h, w = img.shape[:2]
            if h >= self.tile and w >= self.tile:
                y0, x0 = int(rng.integers(0, h - self.tile + 1)), int(rng.integers(0, w - self.tile + 1))
                return np.ascontiguousarray(img[y0:y0 + self.tile, x0:x0 + self.tile], dtype=np.float64), it, y0, x0
            self._left = 0
        raise ValueError("no item large enough for the tile")


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
    """A sample's full conditioning: the target's normalised metrics, then the input's capture descriptor."""
    cap = capture_of_params(s["a"]) if known else UNKNOWN_CAPTURE
    return np.concatenate([norm(s["metrics"]), cap]).astype(np.float32)


def make_sample(src: TileSource, dists: Distributions, rng: np.random.Generator, ss: int = 1) -> dict:
    """One rendered pair with its bookkeeping; the conditioning is the raw metric dict."""
    tile, it, y0, x0 = src.random_tile(rng)
    a, b, restore = draw_pair(rng, dists)
    sa, sb = int(rng.integers(1 << 62)), int(rng.integers(1 << 62))
    xa, xb, m = render_pair(tile, a, b, sa, sb, it.is_hdr, ss)
    return {"x": xa, "y": xb, "metrics": m, "restore": restore, "a": a, "b": b, "item": it, "y0": y0, "x0": x0,
            "seed_a": sa, "seed_b": sb}


def fixed_set(sources: list[str], split: str, n: int, tile: int, seed: int, root: Path | None = None,
              ss: int = 1, log=print) -> list[dict]:
    """n reproducible samples (seeded by split, n, tile and seed): the validation set and the calibration draw."""
    src, dists = TileSource(sources, split, tile, root), Distributions.load(root)
    rng = np.random.default_rng(_h("fixed", split, n, tile, seed))
    out = []
    for i in range(n):
        out.append(make_sample(src, dists, rng, ss))
        if log and (i + 1) % 32 == 0:
            log(f"  fixed set {split}: {i + 1}/{n}")
    return out


class PairStream:
    """Infinite iterator of (x, y, cond) float32 tensors in the log domain, for one loader worker."""
    def __init__(self, sources: list[str], tile: int, normaliser: Normaliser, seed: int, root: Path | None = None,
                 ss: int = 1):
        self.sources, self.tile, self.norm, self.seed, self.root, self.ss = sources, tile, normaliser, seed, root, ss

    def __iter__(self):
        import torch
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi else 0
        src = TileSource(self.sources, "train", self.tile, self.root, tiles_per_image=TILES_PER_IMAGE)
        dists = Distributions.load(self.root)
        rng = np.random.default_rng(_h("stream", self.seed, wid))
        while True:
            s = make_sample(src, dists, rng, self.ss)
            x = torch.from_numpy(encode(s["x"])).permute(2, 0, 1).contiguous()
            y = torch.from_numpy(encode(s["y"])).permute(2, 0, 1).contiguous()
            yield x, y, torch.from_numpy(cond_vector(self.norm, s, rng.random() < P_CAPTURE_KNOWN)), bool(s["restore"])
