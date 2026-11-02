"""SIDD's bursts as real pairs (harness.data.siddburst): a synthetic burst zip reduced to inputs and the mean of
the other frames (a moved frame registered and merged into the mean, an unsettled one and an unregistrable one
dropped, no input in the mean, the margins the template's), sub-pixel affine motion registered and undone, a
drifting burst with no inputs near one another, the kernel merge and the per-input pair gate, the stacks listed as
pairs keyed on SIDD's scene, the source's own decoder chosen for its pairs, and the batch run that deletes a zip
only once its stack pairs.
LibRaw is replaced by a reader of .npy members; the decoder's route through a template DNG runs on a real SIDD frame
when one is on disk."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from harness.data import siddburst as SB
from harness.data.label import Item, file_filter, scene_of
from harness.data.realpairs import SIDD, decode_raw, decoder_of, match_of, Pair
from harness.model.data import split_of

BLACK, WHITE, GAIN, READ_E = 63, 1023, 0.25, 2.0
TOP = 4                     # margin rows above the visible area, as the Pixel's raw_image has
H, W = 128, 192             # visible mosaic


def _scene() -> np.ndarray:
    """Electrons per pixel: a ramp across the columns with stripes, so the frames have edges and structure."""
    ramp = np.linspace(0.02, 0.9, W)[None, :] * np.ones((H, 1))
    stripes = (np.arange(H)[:, None] // 12 % 2 == 0) & (np.arange(W)[None, :] // 16 % 2 == 0)
    return np.where(stripes, ramp * 0.4 + 0.03, ramp) * (WHITE - BLACK) * GAIN


def _to_raw(e: np.ndarray, rng, gain: float = GAIN) -> np.ndarray:
    """Electrons on the visible area to a full raw_image (uint16, TOP margin rows of black plus noise above it)."""
    vis = rng.poisson(e) + rng.normal(0.0, READ_E, size=e.shape)
    full = np.vstack([rng.normal(0.0, READ_E, size=(TOP, W)), vis])
    return np.clip(np.round(full / gain + BLACK), 0, WHITE).astype(np.uint16)


def _frames(n: int, seed: int = 0, dark: tuple[int, ...] = (), moved: tuple[int, ...] = (), light: float = 1.0,
            gain: float = GAIN) -> list[np.ndarray]:
    """Frames of _scene(); frames in `dark` 7 % dark, frames in `moved` shifted four rows; `light` scales the
    scene's electrons, `gain` is e- per DN."""
    rng = np.random.default_rng(seed)
    e = _scene() * light
    out = []
    for i in range(n):
        ei = e * (0.93 if i in dark else 1.0)
        if i in moved:
            ei = np.pad(ei, ((4, 0), (0, 0)), mode="edge")[:-4]
        out.append(_to_raw(ei, rng, gain))
    return out


_B = np.random.default_rng(42)
BLOBS = (_B.uniform(-8, H + 8, 70), _B.uniform(-8, W + 8, 70), _B.uniform(3.0, 6.0, 70), _B.uniform(0.05, 0.35, 70))


WGAIN = 4.0                 # e- per DN of the smooth scene's frames: a low-ISO burst, so 24-quad patches register


def _world(y: np.ndarray, x: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Electrons at mosaic point (y, x) of a smooth scene, a gentle ramp across the columns and Gaussian blobs
    (band-limited on the CFA planes' lattice, so a cubic spline resamples it); `mirror` flips it left to right. A
    knock of 2.5 px moves the frame's mean by under 1 % (a steeper ramp would make it an outlier on 192 columns)."""
    if mirror:
        x = W - 1 - x
    v = 0.25 + 0.05 * x / W
    for cy, cx, s, a in zip(*BLOBS):
        v = v + a * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    return v * 0.6 * (WHITE - BLACK) * WGAIN


def _move(dy: float = 0.0, dx: float = 0.0, deg: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """A frame's map from its mosaic coordinates to the scene's: a turn and a zoom about the centre, then a shift."""
    c, s = np.cos(np.radians(deg)) * zoom, np.sin(np.radians(deg)) * zoom
    m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o = np.array([(H - 1) / 2, (W - 1) / 2])
    m[:2, 2] = o - m[:2, :2] @ o + (dy, dx)
    return m


def _rendered(maps: list[np.ndarray], seed: int = 0, mirror: tuple[int, ...] = (), flat: bool = False) -> list[np.ndarray]:
    """Frames of _world() under each map (frame(q) = world(map q)); frames in `mirror` of the mirrored world;
    `flat`, a scene of one level (no structure to register on)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    out = []
    for i, m in enumerate(maps):
        y = m[0, 0] * yy + m[0, 1] * xx + m[0, 2]
        x = m[1, 0] * yy + m[1, 1] * xx + m[1, 2]
        e = np.full((H, W), 0.4 * 0.6 * (WHITE - BLACK) * WGAIN) if flat else _world(y, x, mirror=i in mirror)
        out.append(_to_raw(e, rng, WGAIN))
    return out


def _zip(path: Path, inst: str, frames: list[np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for i, f in enumerate(frames):
            buf = io.BytesIO()
            np.save(buf, f)
            z.writestr(f"{inst}_DNG/{inst}_DSC_{i + 1:04d}.DNG", buf.getvalue())
    return path


def _fake_read(z: zipfile.ZipFile, member: str):
    data = z.read(member)
    full = np.load(io.BytesIO(data))
    return data, full, {"black": [BLACK] * 4, "white": WHITE, "pattern": [[0, 1], [3, 2]],
                        "raw_shape": list(full.shape), "visible": [TOP, 0, H, W]}


DRIFT = [_move(dx=0.4 * i) for i in range(20)]    # 0.4 px a frame: no two frames within INPUT_PX of each other
META = {"camera": "GP", "camera_full": "Google Pixel", "iso": 800, "shutter": "00100", "scene": "003",
        "brightness": "N", "illuminant_k": 4400, "scene_name": "0007_003_GP_00800_00100_4400_N"}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(SB, "_read", _fake_read)
    monkeypatch.setattr(SB, "INPUTS", 4)
    monkeypatch.setattr(SB, "MIN_INPUTS", 3)
    monkeypatch.setattr(SB, "MIN_MEAN", 8)
    monkeypatch.setattr(SB, "REG_PATCH", 24)         # the frames are 64 x 96 quads
    monkeypatch.setattr(SB, "REG_MARGIN", 6)
    monkeypatch.setattr(SB, "REG_GRID", (4, 6))
    monkeypatch.setattr(SB, "REG_MIN_PATCHES", 8)
    monkeypatch.setenv(SB.RESERVE_ENV, "0")


def _affine_of(rec: dict, name: str) -> np.ndarray:
    t = np.eye(3)
    t[:2] = rec["registration"]["frames"][name]["affine"]
    return t


def test_reduce_keeps_inputs_out_of_the_mean_registers_a_moved_frame_and_drops_an_unsettled_one(tmp_path, small):
    frames = _frames(30, dark=(0,), moved=(10,))
    zp = _zip(tmp_path / "0007_DNG.zip", "0007", frames)
    rec = SB.reduce_instance(zp, META, tmp_path / "stacks")
    d = tmp_path / "stacks" / "0007_s003"
    assert rec["ok"] and rec["stack"] == d.name and json.loads((d / "mean.json").read_text())["ok"]
    assert rec["dropped"] == {"0007_DNG/0007_DSC_0001.DNG": "plane mean off the burst's"}
    reg = rec["registration"]
    assert reg["registered"] and not reg["unregistered"] and reg["anchor"] == "0007_DSC_0016.DNG"
    moved = _affine_of(rec, "0007_DSC_0011.DNG")       # four rows down: lum(p + 2 quads) ~ ref(p)
    down = np.array([[1.0, 0, 2], [0, 1, 0], [0, 0, 1]])
    centre = np.array([[H / 4 - 0.5], [W / 4 - 0.5], [1.0]])
    assert SB.displacement_px(moved, down, centre) < 0.1   # 24-quad patches of 24 stripes: noise-limited at the corners
    assert SB.displacement_px(moved, down, SB.grid_corners((H // 2, W // 2))) < 0.4
    assert "0007_DSC_0011.DNG" in rec["mean_frames"] and reg["mean_px_max"] == pytest.approx(4, abs=0.4)
    assert len(rec["inputs"]) == 4 and len(rec["mean_frames"]) == 25
    assert not set(rec["inputs"]) & set(rec["mean_frames"]) and "0007_DSC_0001.DNG" not in rec["inputs"] + rec["mean_frames"]
    assert rec["inputs"] == ["0007_DSC_0002.DNG", "0007_DSC_0012.DNG", "0007_DSC_0021.DNG", "0007_DSC_0030.DNG"]
    assert max(reg["inputs_px"].values()) < SB.INPUT_PX
    with zipfile.ZipFile(zp) as z:             # the inputs are the members' bytes, as they were
        for f in rec["inputs"]:
            assert (d / "frames" / f).read_bytes() == z.read(f"0007_DNG/{f}")
    assert rec["template"] == f"frames/{rec['inputs'][0]}"
    mean = np.load(d / "mean.npy")
    assert mean.dtype == np.float32 and mean.shape == frames[0].shape
    template = np.load(d / rec["template"])
    assert np.array_equal(mean[:TOP], template[:TOP].astype(np.float32))
    truth = _scene() / GAIN + BLACK
    err = mean[TOP:] - truth
    sigma1 = np.sqrt((truth - BLACK) * GAIN + READ_E ** 2) / GAIN                  # one frame's noise in DN
    assert abs(float(err.mean())) < 0.2
    assert 20 < rec["n_eff"] <= 25.01                         # 25 frames, all but on the pixel: nearly their mean
    assert float(np.std(err / sigma1)) == pytest.approx(1 / np.sqrt(rec["n_eff"]), rel=0.25)
    assert reg["cover_min"] == 24                                                   # the moved frame's last rows
    assert not rec["inputs_dropped"] and set(rec["pairs"]) == set(rec["inputs"])
    assert rec["pair_excess"][2] <= SB.MAX_PAIR_EXCESS and abs(rec["pair_excess"][1]) < 0.3
    assert set(rec["planes"]) == {"00", "01", "10", "11"}
    assert rec["iso"] == 800 and rec["scene"] == "003" and rec["levels"]["white"] == WHITE


def test_sub_pixel_affine_motion_is_registered_and_undone_in_the_mean(tmp_path, small, monkeypatch):
    # SIDD's bursts drift by tenths of a pixel with a slow zoom, and the tripod is knocked now and then: 16 frames
    # within a few hundredths of a pixel of one another, the rest shifted, turned and zoomed
    rng = np.random.default_rng(7)
    maps = [_move(*rng.uniform(-0.015, 0.015, 2)) if 12 <= i < 28 else
            _move(*rng.uniform(-2.5, 2.5, 2), deg=rng.uniform(-0.25, 0.25), zoom=1 + rng.uniform(-0.002, 0.002))
            for i in range(40)]
    frames = _rendered(maps, seed=8)
    zp = _zip(tmp_path / "0020_DNG.zip", "0020", frames)
    rec = SB.reduce_instance(zp, META, tmp_path / "stacks")
    reg = rec["registration"]
    assert rec["ok"] and not rec["dropped"] and reg["registered"] and reg["near_anchor"] >= 16
    assert len(rec["inputs"]) == 4 and len(rec["mean_frames"]) == 36
    assert all(12 <= int(f[-8:-4]) - 1 < 28 for f in rec["inputs"])
    corners = SB.grid_corners((H // 2, W // 2))
    centre = np.array([[H / 4 - 0.5], [W / 4 - 0.5], [1.0]])
    mid = maps[20]
    for i, m in enumerate(maps):                        # frame i's map: S_i = A_i^-1 A_mid on the mosaic
        truth = np.linalg.inv(SB.QUAD) @ np.linalg.inv(m) @ mid @ SB.QUAD
        t = _affine_of(rec, f"0020_DSC_{i + 1:04d}.DNG")
        # px; the patches' own scatter (0.02 to 0.04 quads on 24-quad patches) is all there is, the most at the
        # corners, which the fit reaches past its patches, and on the knocked frames, whose patches run off the frame
        assert SB.displacement_px(t, truth, centre) < 0.05, i
        assert SB.displacement_px(t, truth, corners) < (0.11 if 12 <= i < 28 else 0.2), i
    t = np.eye(3)
    t[:2] = reg["target_affine_quads"]
    s_target = SB.QUAD @ t @ np.linalg.inv(SB.QUAD)     # the mean's geometry: mean(q) = world(A_mid S*^-1 q)
    q = np.linalg.inv(s_target)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    g = mid @ q
    truth = _world(g[0, 0] * yy + g[0, 1] * xx + g[0, 2], g[1, 0] * yy + g[1, 1] * xx + g[1, 2]) / WGAIN + BLACK
    mean = np.load(tmp_path / "stacks" / "0020_s003" / "mean.npy")[TOP:]
    b = 8                                               # mosaic px: past the knocked frames' reach
    err = (mean - truth)[b:H - b, b:W - b]
    sigma1 = (np.sqrt((truth - BLACK) * WGAIN + READ_E ** 2) / WGAIN)[b:H - b, b:W - b]
    assert abs(float(err.mean())) < 0.2
    assert float(np.std(err / sigma1)) < 1.2 / np.sqrt(rec["n_eff"])   # noise alone: the merge adds no error
    assert not rec["inputs_dropped"] and rec["pair_excess"][2] <= SB.MAX_PAIR_EXCESS
    monkeypatch.setattr(SB, "register", lambda *a, **k: None)   # the same frames taken as they lie: the mean blurs
    flat = SB.reduce_instance(zp, META, tmp_path / "unregistered")
    assert not flat["registration"]["registered"] and flat["pair_excess"][1] > 4 * max(rec["pair_excess"][1], 0.25)


def test_frames_of_another_scene_do_not_register_and_are_dropped(tmp_path, small):
    frames = _rendered([_move()] * 24, seed=9, mirror=(5, 17))
    rec = SB.reduce_instance(_zip(tmp_path / "0021_DNG.zip", "0021", frames), META, tmp_path / "stacks")
    assert rec["ok"] and rec["registration"]["registered"]
    assert rec["registration"]["unregistered"] == ["0021_DSC_0006.DNG", "0021_DSC_0018.DNG"]
    assert set(rec["dropped"]) == {"0021_DNG/0021_DSC_0006.DNG", "0021_DNG/0021_DSC_0018.DNG"}
    assert all(v.startswith("not registered") for v in rec["dropped"].values())
    assert len(rec["inputs"]) + len(rec["mean_frames"]) == 22


def test_a_burst_with_nothing_to_register_on_is_taken_as_it_lies(tmp_path, small):
    frames = _rendered([_move()] * 20, seed=10, flat=True)
    rec = SB.reduce_instance(_zip(tmp_path / "0022_DNG.zip", "0022", frames), META, tmp_path / "stacks")
    reg = rec["registration"]
    assert not reg["registered"] and "taken as they lie" in reg["why"] and len(reg["unregistered"]) == 19  # all but the reference
    assert not rec["dropped"] and len(reg["inputs_px"]) == 4 and len(rec["mean_frames"]) == 16
    assert not rec["ok"] and "no plane gave" in rec["why"] and rec["inputs"] == []   # one level: no law to judge by


def test_a_burst_that_moved_gives_no_pairs_and_keeps_no_frames(tmp_path, small, monkeypatch):
    monkeypatch.setattr(SB, "MAX_PAIR_EXCESS", -0.5)    # under any pair's: every input fails
    zp = _zip(tmp_path / "0008_DNG.zip", "0008", _frames(20, seed=1))
    rec = SB.reduce_instance(zp, dict(META, scene="004"), tmp_path / "stacks")
    d = tmp_path / "stacks" / "0008_s004"
    assert not rec["ok"] and "0 of 4 inputs within -0.5 noise variances" in rec["why"] and rec["inputs"] == []
    assert len(rec["inputs_dropped"]) == 4 and all("pair excess" in v for v in rec["inputs_dropped"].values())
    assert (d / "mean.json").exists() and not (d / "mean.npy").exists() and not (d / "frames").exists()
    monkeypatch.setattr(SB, "MAX_PAIR_EXCESS", 1.5)
    rec = SB.reduce_instance(_zip(tmp_path / "0009_DNG.zip", "0009", _rendered(DRIFT, seed=2)), dict(META, scene="005"),
                             tmp_path / "stacks")
    assert not rec["ok"] and "frames hold, 1 of them within" in rec["why"] and rec["registration"]["registered"]
    assert not (tmp_path / "stacks" / "0009_s005" / "frames").exists()
    with pytest.raises(ValueError, match="no metadata"):
        SB.reduce_instance(zp, None, tmp_path / "stacks")
    with pytest.raises(ValueError, match="frames"):
        SB.reduce_instance(_zip(tmp_path / "0010_DNG.zip", "0010", _frames(5)), META, tmp_path / "stacks")


def test_a_dim_burst_fits_its_noise_law_on_its_own_range(tmp_path, small, monkeypatch):
    # SIDD's 'low' bursts, as 0018 (Pixel, ISO 100): the planes average a few tens of DN above black of 960, all in
    # the first two of 48 bins over the full span; the fit window spans the burst's own range instead
    frames = _frames(30, seed=5, light=0.55, gain=4.0)
    rec = SB.reduce_instance(_zip(tmp_path / "0011_DNG.zip", "0011", frames), META, tmp_path / "stacks")
    assert rec["ok"] and rec["pair_excess"][2] <= SB.MAX_PAIR_EXCESS
    for p in rec["planes"].values():
        assert p["mean_dn_above_black"] < 30 and p["fit"]["ok"]
        assert p["fit"]["fit_window_dn"][1] < 0.1 * (WHITE - BLACK)
    monkeypatch.setattr(SB, "fit_span", lambda ps: ps.white - ps.black)       # the full span: no law fits
    rec = SB.reduce_instance(_zip(tmp_path / "0012_DNG.zip", "0012", frames), META, tmp_path / "full")
    assert not rec["ok"] and "no plane gave" in rec["why"]
    bright = SB.reduce_instance(_zip(tmp_path / "0013_DNG.zip", "0013", _frames(30, seed=6)), META, tmp_path / "b")
    assert bright["ok"] and all(p["fit"]["fit_window_dn"][1] > 0.5 * (WHITE - BLACK) for p in bright["planes"].values())


def test_the_split_keys_on_sidds_scene_and_only_means_are_clean_items():
    a = Item(SIDD, "stacks/0007_s003/mean.npy")
    assert scene_of(a) == "003" and split_of(a) == split_of(Item(SIDD, "stacks/0150_s003/mean.npy"))
    keep = file_filter(SIDD)
    assert keep("stacks/0007_s003/mean.npy")
    assert not keep("stacks/0007_s003/frames/0007_DSC_0002.DNG")
    assert not keep("data/train/0007_DNG.zip::0007_DNG/0007_DSC_0002.DNG")
    assert SB.spread(10, 4) == [0, 3, 6, 9] and SB.spread(3, 5) == [0, 1, 2] and SB.spread(5, 0) == []


def test_plane_maps_and_input_choice():
    m = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])   # 3 px down, 1 px left, on the mosaic
    assert np.allclose(SB.plane_map(m, 1, 0), [[1, 0, 1.5], [0, 1, -0.5], [0, 0, 1]])
    plane = np.arange(48, dtype=np.uint16).reshape(6, 8)
    val, dy, dx, valid = SB.nearest_samples(plane, SB.plane_map(np.array([[1.0, 0, 2], [0, 1, 0], [0, 0, 1]]), 0, 0))
    assert np.array_equal(val[:5], plane[1:]) and valid[:5].all() and not valid[5].any() and val.dtype == np.float64
    assert np.allclose(dy[:5], 0) and np.allclose(dy[5], -1) and np.allclose(dx, 0)   # off the plane: clamped
    val, dy, dx, valid = SB.nearest_samples(plane, SB.plane_map(np.array([[1.0, 0, 0.6], [0, 1, 0], [0, 0, 1]]), 0, 0))
    assert np.array_equal(val, plane) and np.allclose(dy, -0.3) and valid.all()    # 0.3 plane px: the same sample
    corners = SB.grid_corners((10, 20))
    shift = lambda dy: np.array([[1.0, 0, dy], [0, 1, 0], [0, 0, 1]])           # noqa: E731
    assert SB.displacement_px(shift(0.1), shift(0.0), corners) == pytest.approx(0.2)
    maps = {i: shift(0.05 * i) for i in range(8)}                              # 0.1 px a frame
    anchor, near = SB.choose_inputs(maps, corners, 4)
    assert anchor == 4 and near == [2, 3, 4, 5, 6]


def test_kernel_merge_fits_a_ramp_from_scattered_samples_and_counts_its_noise():
    # frames sampling a linear ramp at scattered sub-pixel phases: the order 1 fit gives the ramp at each pixel; with
    # the phases all on one side the plain weighted mean leans with their mean offset, the order 1 fit far less, and
    # its variance says it saw few samples near the pixel; at no offset it is the mean
    rng = np.random.default_rng(3)
    h, w = 16, 20
    u, v = np.mgrid[0:h, 0:w].astype(float)
    ramp = lambda y, x: 30.0 + 2.0 * y - 1.5 * x                               # noqa: E731
    inner = (slice(1, -1), slice(1, -1))
    err = {}
    for name, lo, hi in (("even", (-0.45, -0.45), (0.45, 0.45)), ("one side", (0.05, -0.45), (0.45, -0.05))):
        km, k0 = SB.KernelMerge((h, w)), [np.zeros((h, w)), np.zeros((h, w))]
        for dy, dx in rng.uniform(lo, hi, (40, 2)):
            frame = ramp(u + dy, v + dx)                # the frame's sample at q is the scene at q + d
            val, oy, ox, valid = SB.nearest_samples(frame, np.array([[1.0, 0, -dy], [0, 1, -dx], [0, 0, 1]]))
            km.add(val, oy, ox, valid)
            wt = np.exp(-(oy ** 2 + ox ** 2) / (2 * SB.MERGE_SIGMA ** 2)) * valid
            k0[0] += wt * val
            k0[1] += wt
        t, var = km.finish(rows=5)
        assert np.isfinite(var).all()
        err[name] = (float(np.abs(t - ramp(u, v))[inner].max()), float(np.abs(k0[0] / k0[1] - ramp(u, v))[inner].max()),
                     float(np.median(var)))
    assert err["even"][0] < 0.02 and 1 / 40 < err["even"][2] < 1 / 5
    assert err["one side"][1] > 0.3 and err["one side"][0] < err["one side"][1] / 3 and err["one side"][2] > 1 / 5
    km = SB.KernelMerge((4, 5))
    zero = np.zeros((4, 5))
    for i in range(10):
        valid = np.ones((4, 5), dtype=bool)
        valid[0, 0] = False                             # one pixel no frame covers
        km.add(np.full((4, 5), float(i)), zero, zero, valid)
    t, var = km.finish()
    assert np.allclose(t[valid], 4.5) and np.allclose(var[valid], 0.1, rtol=1e-5)
    assert t[0, 0] == 0 and np.isinf(var[0, 0]) and km.cover[0, 0] == 0 and km.cover[1, 1] == 10


def test_the_pair_excess_reads_noise_as_zero_and_a_moved_input_as_far_more():
    rng = np.random.default_rng(4)
    h, w = 128, 256
    u, v = np.mgrid[0:h, 0:w].astype(float)
    t = 300.0 + 200.0 * np.tanh((v - 100 - 0.3 * u) / 2.0) + 0.5 * u          # DN above black: an edge on a ramp
    fit = {"a_dn_per_dn": 0.5, "b_dn2": 4.0}
    v_mean = np.full((h, w), 0.02, dtype=np.float32)
    masks = SB.pair_masks(t, np.ones((h, w), dtype=bool), 1000.0)
    assert masks["edge"].sum() > 500 and not masks["base"][:SB.PAIR_BORDER].any()
    sd = np.sqrt(fit["a_dn_per_dn"] * 0.97 * t + fit["b_dn2"])
    x = 0.97 * t + rng.normal(0.0, 1.0, t.shape) * sd    # a dimmer exposure and noise: the mean itself noiseless
    p = SB.pair_excess(x, t, v_mean, fit, masks)
    assert p["gain"] == pytest.approx(0.97, abs=0.002)
    assert abs(p["edge"] + 0.02) < 0.25 and abs(p["flat"] + 0.02) < 0.1 and p["block_max"] < 0.5
    moved = 0.97 * (300.0 + 200.0 * np.tanh((v - 101 - 0.3 * u) / 2.0) + 0.5 * u) + rng.normal(0.0, 1.0, t.shape) * sd
    q = SB.pair_excess(moved, t, v_mean, fit, masks)     # one pixel across the edge
    assert q["edge"] > 20 and q["block_max"] > 1.5 and q["flat"] < 1.0


def test_stacks_list_as_pairs_and_decode_through_sidds_decoder(tmp_path, small, monkeypatch):
    raw = tmp_path / "raw" / SIDD
    SB.reduce_instance(_zip(raw / "data/train/0007_DNG.zip", "0007", _frames(20)), META, raw / "stacks")
    monkeypatch.setattr(SB, "MAX_PAIR_EXCESS", -0.5)
    SB.reduce_instance(_zip(raw / "data/train/0008_DNG.zip", "0008", _frames(20, seed=3)), dict(META, scene="004"),
                       raw / "stacks")
    logs = []
    pairs = SB.sidd_pairs(tmp_path, logs.append)
    assert len(pairs) == 4 and {p.scene for p in pairs} == {"0007_s003"}
    p = pairs[0]
    assert p.ref_file == "stacks/0007_s003/mean.npy" and p.in_file.startswith("stacks/0007_s003/frames/")
    assert p.split == split_of(Item(SIDD, p.ref_file)) and p.iso == 800.0 and p.meta["mean_frames"] == 16
    assert p.meta["n_eff"] > 12 and abs(p.meta["pair_excess"]) < 1.5
    assert "MIT" in p.licence and "0008_s004" in logs[0]
    assert decoder_of(SIDD) is SB.decode_sidd and decoder_of(SIDD, decode_raw) is SB.decode_sidd
    assert decoder_of("rawnind") is decode_raw and decoder_of("rawnind", decode_raw) is decode_raw
    mine = lambda path, scale: None                             # noqa: E731 - a caller's own decoder stays its own
    assert decoder_of(SIDD, mine) is mine
    seen = []
    rng = np.random.default_rng(0)
    clean = np.repeat(np.kron(rng.permutation(np.geomspace(0.01, 0.8, 192)).reshape(12, 16),
                              np.ones((16, 16)))[..., None], 3, axis=-1)
    noisy = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1)

    def fake(path, scale):
        seen.append(Path(path).name)
        return (clean if Path(path).suffix == ".npy" else noisy), 10
    monkeypatch.setattr(SB, "decode_sidd", fake)
    m, *_ = match_of(Pair(SIDD, "0007_s003", p.ref_file, p.in_file, p.split), raw, 1)
    assert seen == [Path(p.in_file).name, "mean.npy"] and m["align"]["aligned"]


def test_reduce_all_skips_done_stacks_and_deletes_only_paired_zips(tmp_path, small, monkeypatch):
    raw = tmp_path / "raw" / SIDD
    (raw / "metadata").mkdir(parents=True)
    with open(raw / SB.METADATA, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "scene_name", "scene_number", "camera", "camera_full", "iso", "shutter",
                    "illuminant_temp", "brightness", "num_images", "zip_file", "zip_bytes"])
        for inst, scene in (("0007", "003"), ("0008", "004")):
            w.writerow([inst, f"{inst}_{scene}_GP_00800_00100_4400_N", scene, "GP", "Google Pixel", 800, "00100",
                        4400, "N", 20, f"{inst}_DNG.zip", 0])
    _zip(raw / "data/train/0007_DNG.zip", "0007", _frames(20))
    _zip(raw / "data/train/0008_DNG.zip", "0008", _rendered(DRIFT, seed=4))
    (raw / "data/train/0009_DNG.zip.part").write_bytes(b"partial")
    logs = []
    tally = SB.reduce_all(tmp_path, workers=1, background=False, delete_zips=True, log=logs.append)
    assert tally["reduced"] == 2 and tally["pairs"] == 1 and tally["no_pairs"] == 1 and tally["failed"] == 0
    assert not (raw / "data/train/0007_DNG.zip").exists() and (raw / "data/train/0008_DNG.zip").exists()
    marker = json.loads((raw / "data/train/0007_DNG.zip.consumed").read_text(encoding="utf-8"))
    assert marker["instance"] == "0007" and not (raw / "data/train/0008_DNG.zip.consumed").exists()
    assert set(SB.done(tmp_path)) == {"0007", "0008"}
    again = SB.reduce_all(tmp_path, workers=1, background=False, delete_zips=True, log=logs.append)
    assert again["reduced"] == 0 and again["before"] == 2


def _sidd_frame() -> tuple[Path, str] | None:
    for root in ("C:/lens-harness-data/raw/sidd-dng/data/train", "D:/lens-harness-data/raw/hf-sidd-dng-mirror/data/train"):
        for zp in sorted(Path(root).glob("*_DNG.zip")) if Path(root).is_dir() else []:
            try:
                with zipfile.ZipFile(zp) as z:
                    names = sorted(n for n in z.namelist() if n.lower().endswith(".dng"))
                return (zp, names[0]) if names else None
            except (zipfile.BadZipFile, OSError):
                continue
    return None


@pytest.mark.skipif(_sidd_frame() is None, reason="no SIDD burst zip on disk")
def test_a_mean_decodes_through_its_template_as_the_frame_itself(tmp_path):
    """A mean equal to a frame's own raw_image decodes, through the template route, to the frame's own decode at
    the explicit levels: the route adds nothing but the scale K, which the levels undo."""
    rawpy = pytest.importorskip("rawpy")
    zp, member = _sidd_frame()
    d = tmp_path / "stacks" / "x_s001"
    (d / "frames").mkdir(parents=True)
    with zipfile.ZipFile(zp) as z:
        (d / "frames" / "t.DNG").write_bytes(z.read(member))
    with rawpy.imread(str(d / "frames" / "t.DNG")) as raw:
        np.save(d / "mean.npy", raw.raw_image.astype(np.float32))
    (d / "mean.json").write_text(json.dumps({"template": "frames/t.DNG"}), encoding="utf-8")
    a, ba = SB.decode_sidd(d / "mean.npy", 2)
    b, bb = SB.decode_sidd(d / "frames" / "t.DNG", 2)
    assert ba == bb and a.shape == b.shape and float(np.abs(a - b).max()) < 2.0 / 65535
    assert 0.2 < float(b.mean()) / max(float(decode_raw(d / "frames" / "t.DNG", 2)[0].mean()), 1e-9) < 1.0 + 1e-6
