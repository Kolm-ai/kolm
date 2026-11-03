"""The burst store (harness.data.burststore): the local shift grid, the flow reader, and the frames of a burst found
under the data root and a second frames root."""
import numpy as np
from scipy import ndimage

from harness.data import burststore
from harness.data.burststore import BLOCK, STRIDE, burst_frames, flow_at, frame_bases, local_grid

SRC = "hdr-plus-burst"


def _texture(side, seed=0):
    """A positive linear image with detail at the registration band: smoothed noise, grey in all three channels."""
    v = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(side, side)), 2.0)
    v = 0.25 + 0.08 * v / v.std()
    return np.repeat(v[..., None], 3, axis=2)


def _moved(y, s):
    """The frame whose pixel p + s holds the merge's pixel p: frame(p + s) ~ merge(p), the store's flow s."""
    return ndimage.shift(y, (s[0], s[1], 0.0), order=3, mode="reflect")


def test_local_grid_recovers_a_sub_pixel_shift():
    y = _texture(384)
    g = local_grid(_moved(y, (3.4, -5.3)), y, (0, 0))
    held = np.asarray(g["held"], dtype=bool)
    o = np.arange(0, 384 - BLOCK + 1, STRIDE)
    assert g["shape"] == [len(o)] * 2
    # a block holds where the frame reaches it at the integer shift (3, -5): not the last row, not the first column
    inside = ((o + 3 >= 0) & (o + 3 + BLOCK <= 384))[:, None] & ((o - 5 >= 0) & (o - 5 + BLOCK <= 384))[None, :]
    assert np.array_equal(held, inside)
    fy, fx = np.asarray(g["fy"]), np.asarray(g["fx"])
    assert np.abs(fy[held] - 3.4).max() < 0.2 and np.abs(fx[held] - -5.3).max() < 0.2
    py, px = flow_at(g, np.array([100.0, 250.0]), np.array([180.0, 60.0]))
    assert np.allclose(py, 3.4, atol=0.2) and np.allclose(px, -5.3, atol=0.2)


def test_local_grid_follows_two_motions():
    """Each half of the frame moves its own way: the blocks inside a half read that half's flow."""
    y = _texture(512, seed=1)
    a, b = (2.3, 1.6), (-3.4, -4.2)
    x = _moved(y, a)
    x[:, 256:] = _moved(y, b)[:, 256:]
    g = local_grid(x, y, (0, 0))
    fy, fx = np.asarray(g["fy"]), np.asarray(g["fx"])
    held = np.asarray(g["held"], dtype=bool)
    x0s = np.arange(g["shape"][1]) * STRIDE
    for cols, (sy, sx) in ((x0s + BLOCK <= 256 - 8, a), (x0s >= 256 + 8, b)):
        h = held[:, cols]
        assert cols.sum() == 2 and h.mean() > 0.6
        assert np.abs(fy[:, cols][h] - sy).max() < 0.25 and np.abs(fx[:, cols][h] - sx).max() < 0.25


def test_local_grid_falls_back_to_the_whole_frame_on_a_flat_frame():
    y = np.full((256, 256, 3), 0.2)
    g = local_grid(y, y, (2, -1))
    assert g["held_share"] == 0.0
    assert np.all(np.asarray(g["fy"]) == 2.0) and np.all(np.asarray(g["fx"]) == -1.0)


def _touch(base, rels):
    for r in rels:
        p = base / "raw" / SRC / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_burst_frames_reads_the_data_root_and_the_frames_root(tmp_path):
    root, extra = tmp_path / "d", tmp_path / "c"
    ref = "20171106/b1/payload_N002.dng"
    _touch(root, [ref, "20171106/b1/payload_N003.dng", "20171106/b1/merged.dng", "20171106/b2/payload_N000.dng"])
    _touch(extra, ["20171106/b1/payload_N000.dng", "20171106/b1/payload_N001.dng", "20171106/b1/payload_N003.dng"])
    assert burst_frames(SRC, ref, root) == ["20171106/b1/payload_N003.dng"]
    got = burst_frames(SRC, ref, root, extra)
    assert got == ["20171106/b1/payload_N003.dng", "20171106/b1/payload_N001.dng", "20171106/b1/payload_N000.dng"]
    assert frame_bases(SRC, root, extra) == [root / "raw" / SRC, extra / "raw" / SRC]
    assert frame_bases(SRC, root, root) == [root / "raw" / SRC]
    assert burst_frames(SRC, ref, root, root) == burst_frames(SRC, ref, root)


def test_write_burst_task_carries_the_frames_root():
    """The pool task is a 5-tuple with the frames root, so a spawned worker reads frames pulled to another drive."""
    import inspect
    assert "frames_root" in inspect.signature(burststore.write_burst).parameters
    assert "frames_root" in inspect.signature(burststore.build_store).parameters
