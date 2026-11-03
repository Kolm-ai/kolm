"""Public restoration baselines through our gates and on SIDD validation: python -m harness model baseline.

MODEL_PLAN 11.3: a run is judged against public models, not only against the input and the camera. Three are
pulled through the collector (sources.json `baseline-*`, ledgered and hash-checked): NAFNet-SIDD-width64 and
Restormer's real denoiser (MIT, both trained on SIDD Medium's sRGB pairs) and SCUNet's real-noise PSNR model
(Apache-2.0, trained on a synthetic real-noise pipeline, not on SIDD). Their architectures are vendored under
baseline_archs/ with their licence notices; the weights load with torch.load(weights_only=True) and strict keys.

They restore sRGB, not linear captures. On our held-out sets each tile goes linear -> clipped at 1 -> the sRGB
curve -> the network -> the curve undone (colorspace), and is scored by heldout.evaluate_with exactly as a
checkpoint is, the reports under <models>/baseline_<name>/heldout_*_blind/: they take no capture descriptor, so
compare.py sets them beside every run's blind rows. Every cached held-out input lies inside [0, 1] (checked 10-29),
so the clip costs them nothing; the domain gap is theirs (raw noise through the sRGB curve, not a phone's ISP).
SIDD validation is the reverse: their home set, ours out of domain (sRGB through the curve undone, no descriptor).

SIDD validation is the 1,280 sRGB blocks of 256 px of ValidationNoisyBlocksSrgb and ValidationGtBlocksSrgb as
NAFNet's data preparation packs them (LMDB, key ValidationBlocksSrgb_<i>, listed in meta_info.txt), mirrored on
Hugging Face (hf-sidd-medium-srgb-mirror, val/). Per block, averaged over the blocks: PSNR of the 8-bit output
(clipped, rounded) and of the float output, both at peak 1; SSIM two ways on the 8-bit output, BasicSR's 2-D
(per channel, the 11 px Gaussian of sigma 1.5, the valid region, averaged over the channels) and NAFNet's default
3-D (one 11 x 11 x 11 Gaussian over the rows, columns and channels, replicate-padded, the whole map), the one its
README's 0.9614 is. Beside them the medians of mtf50 and texture fidelity (tilemetrics, on the linear blocks)
place each model on the perception-distortion plane: a PSNR bought by smoothing shows as texture far below the
truth's 1. Evaluation only: SIDD is never a training source."""
from __future__ import annotations

import importlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..colorspace import linear_to_srgb, srgb_to_linear
from ..data import data_root
from .heldout import HELDOUT_SOURCE, evaluate_with, report_name, set_path, write_report

BASELINE_DIR = "baseline_"   # <models>/baseline_<name>/heldout_*_blind/ and sidd_val/
SIDD_DIR = "sidd_val"        # sidd.json and sidd.md, beside a run's heldout_* reports
SIDD_SOURCE = "hf-sidd-medium-srgb-mirror"
SIDD_VAL = ("val/input_crops.lmdb", "val/gt_crops.lmdb")   # noisy, clean; under <root>/raw/<SIDD_SOURCE>/
SIDD_BLOCKS = 1280
SSIM_SIGMA = 1.5
SSIM_RADIUS = 5              # the 11 px window
SIDD_SCORES = ("psnr", "psnr_float", "ssim", "ssim3d")
SIDD_METRICS = ("mtf50_cpp", "texture_hf_fidelity")
# the blind held-out reports a baseline gets, the sets score_run scores: (source, real, subpix)
HELDOUT_SETS = ((HELDOUT_SOURCE, False, False), ("hdr-plus-burst", True, False), ("rawnind", True, False),
                ("si-hdr", True, False), ("rawnind", True, True), ("hdr-plus-burst", True, True))


@dataclass(frozen=True)
class Baseline:
    name: str
    source: str        # sources.json id: the weights at <root>/raw/<source>/<file>
    file: str
    arch: str          # baseline_archs.<module>:<class>
    kwargs: dict = field(default_factory=dict)
    multiple: int = 1  # the side the network needs a multiple of, reflect-padded up to it
    key: str | None = None   # where the state dict sits in the file (BasicSR saves {"params": ...})
    trained_on: str = ""
    licence: str = ""
    reported: str = ""       # the authors' own SIDD validation numbers


BASELINES = {
    "nafnet": Baseline(
        "nafnet", "baseline-nafnet-sidd", "NAFNet-SIDD-width64.pth", "nafnet:NAFNet",
        dict(img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]),
        multiple=16, key="params", trained_on="SIDD Medium sRGB", licence="MIT (with BasicSR Apache-2.0 parts)",
        reported="40.3045 dB, SSIM 0.9614 (README; float output, 3-D SSIM)"),
    "restormer": Baseline(
        "restormer", "baseline-restormer-real", "real_denoising.pth", "restormer:Restormer",
        dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
             heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree",
             dual_pixel_task=False),
        multiple=8, key="params", trained_on="SIDD Medium sRGB", licence="MIT",
        reported="40.02 dB, SSIM 0.960 (paper, Table 2)"),
    "scunet": Baseline(
        "scunet", "baseline-scunet-real", "scunet_color_real_psnr.pth", "scunet:SCUNet",
        dict(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64), multiple=64,
        trained_on="synthetic real-noise degradations (KAIR), not SIDD", licence="Apache-2.0",
        reported="none for this checkpoint"),
}


def weights_path(b: Baseline, root: Path | None = None) -> Path:
    return Path(root or data_root()) / "raw" / b.source / b.file


def load_net(name: str, device: str | None = None, root: Path | None = None):
    """The baseline's network with its released weights, in eval mode without gradients: (net, device)."""
    import torch
    b = BASELINES[name]
    p = weights_path(b, root)
    if not p.exists():
        raise FileNotFoundError(f"{p}: fetch it first (python -m harness data collect --ids {b.source})")
    mod, cls = b.arch.split(":")
    net = getattr(importlib.import_module(f".baseline_archs.{mod}", __package__), cls)(**b.kwargs)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    net.load_state_dict(sd[b.key] if b.key else sd, strict=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net.to(dev).eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net, dev


def srgb_restorer(net, multiple: int, device):
    """(H, W, 3) sRGB in [0, 1] -> the network's (H, W, 3) sRGB clamped to [0, 1]: fp32, reflect-padded to
    `multiple` and cropped back"""
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def run(img: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        h, w = t.shape[-2:]
        ph, pw = (-h) % multiple, (-w) % multiple
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        y = net(t)[..., :h, :w].clamp(0.0, 1.0)
        return y[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return run


def through_srgb(srgb_fn):
    """a restorer of sRGB as one of linear tiles: clipped at 1, the sRGB curve, the restorer, the curve undone"""
    return lambda x: srgb_to_linear(srgb_fn(linear_to_srgb(x)))


def through_linear(linear_fn):
    """a restorer of linear tiles as one of sRGB: the curve undone, the restorer, the curve (clipped at 1)"""
    return lambda img: linear_to_srgb(linear_fn(srgb_to_linear(img)))


def checkpoint_srgb(ckpt, device: str | None = None):
    """one of our checkpoints or releases as an sRGB restorer, the capture descriptor withheld: (restorer, step)"""
    from .apply import load_model, restore_frames
    b = load_model(ckpt, device)
    return through_linear(lambda x: restore_frames(b, x[None], white_level=1.0, capture=None)[0]), b.step


def set_built(source: str, real: bool, subpix: bool, n: int, tile: int, seed: int, root: Path) -> bool:
    """whether the held-out set is cached; a baseline never builds one (score_run's rule)"""
    if not real:
        return set_path(source, seed, n, tile, root).exists()
    from ..data.realpairs import REALPAIR_DIR
    d = Path(root) / "derived" / REALPAIR_DIR / source
    return any(p.name.endswith("_sp.npz") == subpix for p in d.glob(f"s{seed}_n{n}_t{tile}_k*.npz"))


def evaluate_heldout(name: str, models: Path | None = None, root: Path | None = None, device: str | None = None,
                     n: int = 64, tile: int = 256, seed: int = 0, save_tiles: int = 8, noise: str = "residual",
                     frontier: bool = True, log=print, net=None) -> dict[str, dict]:
    """The baseline on every cached set of HELDOUT_SETS, each report written to
    <models>/baseline_<name>/<report name>/: {report name: report}. `net` is (network, device) in place of the
    released weights (tests)."""
    root = Path(root) if root else data_root()
    b = BASELINES[name]
    net, dev = net or load_net(name, device, root)
    restore = through_srgb(srgb_restorer(net, b.multiple, dev))
    base = Path(models) if models else root / "models"
    fields = {"baseline": name, "ckpt": str(weights_path(b, root)), "blind": True, "trained_on": b.trained_on,
              "licence": b.licence}
    reps = {}
    for source, real, subpix in HELDOUT_SETS:
        rn = report_name(source, real, subpix, blind=True)
        if not set_built(source, real, subpix, n, tile, seed, root):
            if log:
                log(f"### {rn} skipped: its set is not built")
            continue
        out = base / f"{BASELINE_DIR}{name}" / rn
        rep = evaluate_with(lambda s: np.asarray(restore(s["x"]), dtype=np.float32), fields, source, n, tile, seed,
                            root=root, out=out, save_tiles=save_tiles, log=log, real=real, subpix=subpix,
                            noise=noise, frontier=frontier)
        write_report(out, rep)
        reps[rn] = rep
        if log:
            g = rep["gates"]
            log(f"{rn}: " + " | ".join(f"{k} {g[k]['overall']}" for k in ("capture", "camera", "frontier", "tone")
                                        if k in g))
    return reps


# ---- SIDD validation ---------------------------------------------------------------------------------------------
def sidd_val(root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(noisy, clean): uint8 (1280, 256, 256, 3) RGB, block i at index i, from the mirror's LMDB packs"""
    import lmdb
    from PIL import Image
    base = Path(root or data_root()) / "raw" / SIDD_SOURCE
    packs = []
    for rel in SIDD_VAL:
        d = base / rel
        names = [ln.split(" ")[0] for ln in (d / "meta_info.txt").read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        keys = sorted((nm.rsplit(".", 1)[0] for nm in names), key=lambda k: int(k.rsplit("_", 1)[1]))
        env = lmdb.open(str(d), readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                imgs = []
                for k in keys:
                    v = txn.get(k.encode("ascii"))
                    if v is None:
                        raise KeyError(f"{d}: {k} is in meta_info.txt but not in the pack")
                    imgs.append(np.asarray(Image.open(io.BytesIO(v)).convert("RGB")))
        finally:
            env.close()
        packs.append((keys, np.stack(imgs)))
    (ka, noisy), (kb, clean) = packs
    if ka != kb or len(ka) != SIDD_BLOCKS or noisy.shape != clean.shape:
        raise ValueError(f"SIDD validation: {len(ka)} noisy and {len(kb)} clean blocks, want {SIDD_BLOCKS} pairs")
    return noisy, clean


def to_u8(img: np.ndarray) -> np.ndarray:
    """[0, 1] -> uint8 as BasicSR's tensor2img: clipped, times 255, rounded"""
    return np.round(np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def _gauss() -> np.ndarray:
    x = np.arange(-SSIM_RADIUS, SSIM_RADIUS + 1, dtype=np.float64)
    g = np.exp(-x ** 2 / (2 * SSIM_SIGMA ** 2))
    return g / g.sum()


def _ssim_map(x: np.ndarray, y: np.ndarray, blur, peak: float) -> np.ndarray:
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mx, my = blur(x), blur(y)
    sx, sy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))


def ssim(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """BasicSR's _ssim on (H, W, 3): per channel the 11 px Gaussian window (separable, the same as its outer
    product), the valid region's mean, averaged over the channels"""
    from scipy.ndimage import correlate1d
    g, r = _gauss(), SSIM_RADIUS

    def blur(x):
        return correlate1d(correlate1d(x, g, axis=0), g, axis=1)[r:-r, r:-r]
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.mean([_ssim_map(a[..., c], b[..., c], blur, peak).mean() for c in range(a.shape[2])]))


def ssim3d(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """NAFNet's default _ssim_3d on (H, W, C): one 11 x 11 x 11 Gaussian over rows, columns and channels
    (separable), replicate-padded, the whole map's mean"""
    from scipy.ndimage import correlate1d
    g = _gauss()

    def blur(x):
        for ax in range(3):
            x = correlate1d(x, g, axis=ax, mode="nearest")
        return x
    return float(_ssim_map(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), blur, peak).mean())


def _block_scores(x: np.ndarray, y: np.ndarray, out_f: np.ndarray | None = None) -> dict:
    """one block's scores against the clean `y` (uint8); `x` the 8-bit image scored, `out_f` its float original"""
    from ..data.tilemetrics import tile_metrics
    m = tile_metrics(srgb_to_linear(x / 255.0), srgb_to_linear(y / 255.0))
    return {"psnr": psnr(x, y), "psnr_float": psnr(out_f if out_f is not None else x / 255.0, y / 255.0, 1.0),
            "ssim": ssim(x, y), "ssim3d": ssim3d(x, y), **{k: m[k] for k in SIDD_METRICS}}


def sidd_score(restore_srgb, noisy: np.ndarray, clean: np.ndarray, log=print) -> dict:
    """`restore_srgb((H, W, 3) sRGB in [0, 1]) -> sRGB` on every block: the means of SIDD_SCORES and the medians
    of SIDD_METRICS for the input and the output, each against the clean block"""
    rows = []
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(zip(noisy, clean)):
        out_f = np.clip(np.asarray(restore_srgb(x.astype(np.float64) / 255.0), dtype=np.float64), 0.0, 1.0)
        rows.append({"input": _block_scores(x, y), "output": _block_scores(to_u8(out_f), y, out_f)})
        if log and (i + 1) % 128 == 0:
            log(f"  SIDD validation: {i + 1}/{len(noisy)} blocks, {time.perf_counter() - t0:.0f} s")

    def agg(side: str) -> dict:
        d = {k: float(np.mean([r[side][k] for r in rows])) for k in SIDD_SCORES}
        for k in SIDD_METRICS:
            v = np.array([r[side][k] for r in rows], dtype=np.float64)
            d[k] = float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else None
        return d
    return {"n": len(rows), "input": agg("input"), "output": agg("output"),
            "seconds": round(time.perf_counter() - t0, 1), "blocks": rows}


def sidd_markdown(rep: dict) -> str:
    def f(v, p):
        return "-" if v is None else f"{v:.{p}f}"
    who = (f"Public baseline `{rep['baseline']}` ({rep['licence']}, trained on {rep['trained_on']}; its authors "
           f"report {rep['reported']})" if rep.get("baseline") else
           f"Checkpoint `{rep['ckpt']}` (step {rep['step']}), capture descriptor withheld, run on the sRGB block "
           f"through the curve undone")
    lines = [f"# SIDD validation: {rep['n']} sRGB blocks of 256 px", "", f"{who}; {rep['seconds']} s.", "",
             "| | PSNR 8-bit (dB) | PSNR float (dB) | SSIM 2-D | SSIM 3-D | mtf50 (median, truth 0.5) | "
             "texture fidelity (median, truth 1) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for side in ("input", "output"):
        s = rep[side]
        lines.append(f"| {side} | {f(s['psnr'], 2)} | {f(s['psnr_float'], 2)} | {f(s['ssim'], 4)} | "
                     f"{f(s['ssim3d'], 4)} | {f(s['mtf50_cpp'], 3)} | {f(s['texture_hf_fidelity'], 3)} |")
    lines += ["", "Per block, averaged: PSNR of the 8-bit and of the float output; SSIM on the 8-bit output, "
                  "BasicSR's per-channel 2-D and NAFNet's default 3-D. mtf50 and texture fidelity (tilemetrics) "
                  "on the linear blocks against the clean one: the perception side of the same restoration."]
    return "\n".join(lines) + "\n"


def evaluate_sidd(name: str | None = None, ckpt: str | None = None, out: Path | None = None,
                  root: Path | None = None, device: str | None = None, log=print, net=None) -> dict:
    """A baseline (`name`) or one of our checkpoints (`ckpt`) on SIDD validation, the report written to `out`
    (default <models>/baseline_<name>/sidd_val, or sidd_val beside the checkpoint)."""
    root = Path(root) if root else data_root()
    if (name is None) == (ckpt is None):
        raise ValueError("a baseline name or a checkpoint, one of them")
    if name is not None:
        b = BASELINES[name]
        net, dev = net or load_net(name, device, root)
        restore = srgb_restorer(net, b.multiple, dev)
        fields = {"baseline": name, "ckpt": str(weights_path(b, root)), "trained_on": b.trained_on,
                  "licence": b.licence, "reported": b.reported}
        out = Path(out) if out else root / "models" / f"{BASELINE_DIR}{name}" / SIDD_DIR
    else:
        restore, step = checkpoint_srgb(ckpt, device)
        fields = {"ckpt": str(ckpt), "step": step}
        out = Path(out) if out else Path(ckpt).resolve().parent / SIDD_DIR
    noisy, clean = sidd_val(root)
    rep = {**fields, **sidd_score(restore, noisy, clean, log)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sidd.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "sidd.md").write_text(sidd_markdown(rep), encoding="utf-8")
    return rep


def cmd_baseline_heldout(args) -> int:
    from .heldout import to_markdown
    reps = evaluate_heldout(args.name, device=args.device, n=args.n, tile=args.tile, seed=args.seed,
                            save_tiles=args.save_tiles, noise=args.noise, frontier=not args.no_frontier)
    for rep in reps.values():
        print(to_markdown(rep), end="")
    return 0 if reps else 1


def cmd_baseline_sidd(args) -> int:
    rep = evaluate_sidd(args.name, args.ckpt, Path(args.out) if args.out else None, device=args.device)
    print(sidd_markdown(rep), end="")
    return 0


def add_parser(ms) -> None:
    from .cli import _noise_args
    bp = ms.add_parser("baseline", help="public restoration baselines through the held-out gates and SIDD validation")
    bs = bp.add_subparsers(dest="bcmd", required=True)
    h = bs.add_parser("heldout", help="a baseline on every cached held-out set, blind reports beside the runs'")
    h.add_argument("--name", required=True, choices=sorted(BASELINES))
    h.add_argument("--n", type=int, default=64)
    h.add_argument("--tile", type=int, default=256)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--save-tiles", type=int, default=8)
    h.add_argument("--device")
    _noise_args(h)
    h.set_defaults(func=cmd_baseline_heldout)
    s = bs.add_parser("sidd", help="a baseline or one of our checkpoints on SIDD validation (1,280 sRGB blocks)")
    w = s.add_mutually_exclusive_group(required=True)
    w.add_argument("--name", choices=sorted(BASELINES))
    w.add_argument("--ckpt", help="one of our checkpoints or releases, run blind on the sRGB blocks")
    s.add_argument("--out", help="report directory (default: baseline_<name>/sidd_val, or beside the checkpoint)")
    s.add_argument("--device")
    s.set_defaults(func=cmd_baseline_sidd)
