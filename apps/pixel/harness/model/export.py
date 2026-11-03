"""A trained checkpoint as a release: the EMA weights alone (model.safetensors) and what inference needs beside
them (config.json), so the photo command loads the network's weights and not a training state with its optimiser.

    python -m harness model export CKPT OUT_DIR [--name NAME]

apply.load_model reads the directory (or its config.json) as it reads a checkpoint. config.json carries the
network's config, the metric normaliser and the default restore conditioning, the step and the training arguments
the checkpoint recorded, the weights' SHA-256, and the blind noise reading the photo command applies
(data/blindnoise: method and scale), so a release states what it was calibrated with."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT = "kolm-pixel-recovered/1"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def export(ckpt, out, name: str | None = None) -> dict:
    import torch
    from safetensors.torch import save_file

    from ..data import blindnoise
    ck = torch.load(Path(ckpt), map_location="cpu", weights_only=False, mmap=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().contiguous() for k, v in ck["ema"].items()}
    save_file(sd, out / WEIGHTS, metadata={"format": FORMAT, "step": str(int(ck["step"]))})
    cfg = {"format": FORMAT, "name": name or Path(ckpt).parent.name, "step": int(ck["step"]),
           "elapsed_min": ck.get("elapsed_min"), "parameters": int(sum(v.numel() for v in sd.values())),
           "net": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ck["config"].items()},
           "normaliser": ck["normaliser"], "cond_restore": [float(v) for v in ck["cond_restore"]],
           "weights": WEIGHTS, "sha256": sha256(out / WEIGHTS), "source_checkpoint": str(ckpt),
           "train": ck.get("train"),
           "blind_noise": {"method": blindnoise.METHOD, "var_scale": blindnoise.VAR_SCALE[blindnoise.METHOD],
                           "smooth_sigma": blindnoise.SMOOTH_SIGMA[blindnoise.METHOD], "block": blindnoise.BLOCK,
                           "quantile": blindnoise.QUANTILE, "calib_scale": blindnoise.CALIB_SCALE,
                           "scale_factor": {str(k[1]): v for k, v in blindnoise.SCALE_FACTOR.items()
                                            if k[0] == blindnoise.METHOD}}}
    (out / CONFIG).write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    return cfg


def is_release(path) -> bool:
    p = Path(path)
    return (p.is_dir() and (p / CONFIG).exists()) or p.name == CONFIG


def read_release(path) -> dict:
    """A release as load_model reads a checkpoint: config, ema, normaliser, cond_restore, step. The weights'
    SHA-256 is checked against config.json's."""
    from safetensors.torch import load_file
    p = Path(path)
    d = p if p.is_dir() else p.parent
    cfg = json.loads((d / CONFIG).read_text(encoding="utf-8"))
    if cfg.get("format") != FORMAT:
        raise ValueError(f"{d}: format {cfg.get('format')!r}, not {FORMAT!r}")
    w = d / cfg["weights"]
    if sha256(w) != cfg["sha256"]:
        raise ValueError(f"{w}: SHA-256 does not match {CONFIG}")
    net = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["net"].items()}
    return {"config": net, "ema": load_file(w), "normaliser": cfg["normaliser"], "cond_restore": cfg["cond_restore"],
            "step": cfg["step"]}


def cmd_export(args) -> int:
    cfg = export(args.ckpt, args.out, args.name)
    print(f"{cfg['name']}: step {cfg['step']}, {cfg['parameters'] / 1e6:.1f}M parameters -> {args.out} "
          f"({cfg['sha256'][:12]})")
    return 0


def add_parser(ms) -> None:
    e = ms.add_parser("export", help="a checkpoint's EMA weights and inference config as a release directory")
    e.add_argument("ckpt")
    e.add_argument("out")
    e.add_argument("--name", help="the release's name (default: the checkpoint's run directory)")
    e.set_defaults(func=cmd_export)
