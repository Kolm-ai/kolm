"""DNG tag reader, exposure and noise-profile conversions, lens shading map, capture_stats run."""
import json
import math
import struct
from pathlib import Path

import numpy as np

from harness.data import dng


def _tiff(entries: list[tuple[int, int, list | bytes]], exif: list[tuple[int, int, list | bytes]] | None = None,
          strip: bytes = b"") -> bytes:
    """Little-endian TIFF with IFD0 and an optional Exif IFD; values that do not fit in 4 bytes go to a data area."""
    bo = "<"
    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 10: 8, 11: 4, 12: 8}
    fmt = {1: "B", 3: "H", 4: "I", 5: "I", 7: "B", 10: "i", 11: "f", 12: "d"}

    def pack(typ, val):
        if typ == 2:
            return val.encode("latin-1") + b"\0"
        if typ in (1, 7):
            return bytes(val)
        if typ in (5, 10):
            flat = []
            for v in val:
                den = 1_000_000
                flat += [int(round(v * den)), den]
            return struct.pack(bo + fmt[typ] * len(flat), *flat)
        return struct.pack(bo + fmt[typ] * len(val), *val)

    def build(items, ifd_off, data_off):
        blobs, out = b"", b""
        for tag, typ, val in items:
            payload = pack(typ, val)
            count = len(payload) if typ in (1, 2, 7) else len(val)
            if len(payload) <= 4:
                inline = payload + b"\0" * (4 - len(payload))
            else:
                inline = struct.pack(bo + "I", data_off + len(blobs))
                blobs += payload + (b"\0" if len(payload) % 2 else b"")
            out += struct.pack(bo + "HHI", tag, typ, count) + inline
        return struct.pack(bo + "H", len(items)) + out + struct.pack(bo + "I", 0), blobs

    entries = sorted(entries)
    head = 8
    n0 = len(entries) + (1 if exif else 0)
    ifd0_len = 2 + 12 * n0 + 4
    exif_off = head + ifd0_len
    exif_len = (2 + 12 * len(exif) + 4) if exif else 0
    data_off = exif_off + exif_len
    if exif:
        entries = sorted(entries + [(34665, 4, [exif_off])])
    ifd0_bytes, blobs0 = build(entries, head, data_off)
    exif_bytes, blobs1 = (build(exif, exif_off, data_off + len(blobs0)) if exif else (b"", b""))
    body = b"II" + struct.pack(bo + "HI", 42, head) + ifd0_bytes + exif_bytes + blobs0 + blobs1
    if strip:
        # patch StripOffsets (273) to point at the appended strip
        idx = body.find(struct.pack(bo + "HHI", 273, 4, 1))
        assert idx >= 0
        body = body[:idx + 8] + struct.pack(bo + "I", len(body)) + body[idx + 12:] + strip
    return body


def test_tag_reader_exposure_and_noise_profile(tmp_path):
    p = tmp_path / "frame.dng"
    p.write_bytes(_tiff(
        [(271, 2, "google"), (272, 2, "sailfish"), (256, 4, [4048]), (257, 4, [3036]), (258, 3, [16]),
         (262, 3, [32803]), (50714, 5, [64.0, 64.0, 64.0, 64.0]), (50717, 3, [1023]),
         (51041, 12, [1.0e-4, 4.0e-6, 1.0e-4, 4.0e-6])],
        exif=[(33434, 5, [0.01]), (33437, 5, [2.0]), (34855, 3, [100]), (37386, 5, [4.67])]))
    tags = dng.read_tags(p)
    assert tags["ifd0"]["Make"] == "google" and tags["ifd0"]["Model"] == "sailfish"
    ex = dng.exposure_summary(tags)
    assert abs(ex["exposure_s"] - 0.01) < 1e-9 and ex["iso"] == 100 and ex["f_number"] == 2.0
    assert abs(ex["ev100"] - math.log2(4 / 0.01)) < 1e-6  # 8.64
    assert abs(ex["luminance_cdm2"] - 2 ** ex["ev100"] * 12.5 / 100) < 1e-9  # 50 cd/m^2
    s = dng.sensor_summary(tags)
    assert s["white"] == 1023 and s["black"] == 64.0 and s["width_px"] == 4048
    nz = s["noise"]
    rng = 1023 - 64
    assert abs(nz["gain_e_per_dn"] - 1 / (rng * 1e-4)) < 1e-9  # 10.43 e-/DN
    assert abs(nz["read_noise_e"] - math.sqrt(4e-6) / 1e-4) < 1e-9  # 20 e-


def test_noise_profile_round_trip_from_electron_model():
    g, sigma_r, white, black = 4.0, 3.0, 4095, 256   # e-/DN, e-
    rng = white - black
    scale, offset = 1 / (rng * g), (sigma_r / g) ** 2 / rng ** 2   # var_DN = DN/g + sigma_r^2/g^2 in normalised units
    out = dng.noise_profile_electrons([scale, offset], white, black)
    assert abs(out["gain_e_per_dn"] - g) < 1e-9 and abs(out["read_noise_e"] - sigma_r) < 1e-9
    assert abs(out["full_well_e"] - rng * g) < 1e-9


def test_lens_shading_map_and_corner_falloff(tmp_path):
    h, w, s = 5, 7, 4
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h // 2) / (h // 2)) ** 2 + ((xx - w // 2) / (w // 2)) ** 2
    gain = 1.0 + 1.0 * r2 / 2  # corners at 2.0 relative to the centre 1.0 -> one stop
    arr = np.repeat(gain[..., None], s, axis=2).astype("<f4")
    p = tmp_path / "lens_shading_map_N000.tiff"
    p.write_bytes(_tiff([(256, 4, [w]), (257, 4, [h]), (258, 3, [32] * s), (259, 3, [1]), (277, 3, [s]),
                         (339, 3, [3] * s), (273, 4, [0]), (279, 4, [arr.nbytes]), (284, 3, [1])],
                        strip=arr.tobytes()))
    out = dng.lens_shading_summary(p)
    assert out["shape"] == [h, w, s]
    assert abs(out["corner_falloff_stops"] - 1.0) < 1e-6 and abs(out["range_stops"] - 1.0) < 1e-6


def test_run_summaries_and_sidd_cross_check(tmp_path):
    raw = tmp_path / "bursts"
    for i, (t, iso) in enumerate([(1 / 100, 50), (1 / 30, 400), (1 / 15, 1600)]):
        d = raw / f"{i:04d}_burst"
        d.mkdir(parents=True)
        scale = 1e-4 * iso / 100  # gain falls as 1/ISO
        (d / "payload_N000.dng").write_bytes(_tiff(
            [(271, 2, "google"), (272, 2, "sailfish"), (50714, 5, [64.0]), (50717, 3, [1023]),
             (51041, 12, [scale, 1e-6, scale, 1e-6])],
            exif=[(33434, 5, [t]), (33437, 5, [2.0]), (34855, 3, [iso])]))
    nz = {"iso_scaling": {"Google Pixel": {"iso": [100, 400, 1600], "gain_e_per_dn": [1 / (959 * 1e-4) / 1, 1 / (959 * 4e-4), 1 / (959 * 16e-4)]}}}
    out = dng.run(raw, noise_check=nz)
    s = out["summary"]
    assert s["n_ok"] == 3 and s["iso"]["median"] == 400 and abs(s["exposure_s"]["max"] - 1 / 15) < 1e-5
    assert s["fraction_exposure_ge_1_30"] == 2 / 3
    # L = N^2 / (t x ISO/100) x K/100, so with N fixed the two log slopes against luminance sum to exactly -1
    assert abs(s["log_exposure_vs_log_luminance_slope"] + s["log_iso_vs_log_luminance_slope"] + 1) < 1e-3
    m = out["models"]["Pixel"]
    assert m["codenames"] == ["sailfish"] and abs(m["noise_profile_log_gain_vs_log_iso_slope"] + 1) < 1e-6
    cc = out["sidd_cross_check"]["Pixel"]
    assert cc["n"] == 3 and abs(cc["profile_over_measured_gain"]["median"] - 1.0) < 1e-6
    json.dumps(out)  # serialisable
