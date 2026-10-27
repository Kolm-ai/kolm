import json
from pathlib import Path

import pytest

from harness.data.collect import Collector, sha256_of
from harness.data.manifest import FileSpec, Source, load_manifest, validate


def _local_source(tmp_path, n=3):
    srcdir = tmp_path / "srv"
    srcdir.mkdir()
    files = []
    for i in range(n):
        p = srcdir / f"f{i}.txt"
        p.write_text(f"payload {i}\n" * 100)
        files.append(FileSpec(url=p.as_uri(), path=f"sub/f{i}.txt", sha256=sha256_of(p)))
    return Source(id="local", name="local test", kind="reference", access="direct",
                  url="file://", files=files, size_bytes=1e3)


def test_manifest_loads_and_validates():
    sources = load_manifest()
    assert sources and all(s.tier in ("S", "M", "L", "XL") for s in sources)
    with pytest.raises(ValueError):
        validate([Source(id="x", name="x", kind="dataset", access="direct", url="u")])  # no files
    with pytest.raises(ValueError):
        validate([Source(id="x", name="x", kind="nope", access="manual", url="u")])


def test_collector_is_idempotent_and_verifies(tmp_path):
    src = _local_source(tmp_path)
    c = Collector(root=tmp_path / "data", jobs=2, log=lambda *_: None)
    s1 = c.fetch_source(src)
    assert s1["done"] == 3 and s1["failed"] == 0
    assert (tmp_path / "data" / "raw" / "local" / "sub" / "f0.txt").exists()
    c2 = Collector(root=tmp_path / "data", log=lambda *_: None)  # fresh process: reads the ledger
    s2 = c2.fetch_source(src)
    assert s2["already"] == 3 and s2["done"] == 0
    assert c2.status()["local"]["files"] == 3
    bad = Source(id="bad", name="bad", kind="reference", access="direct", url="file://",
                 files=[FileSpec(url=src.files[0].url, path="x.txt", sha256="0" * 64)])
    s3 = c2.fetch_source(bad)
    assert s3["failed"] == 1 and "sha256 mismatch" in s3["errors"][0]
    assert not (tmp_path / "data" / "raw" / "bad" / "x.txt").exists()
    lines = (tmp_path / "data" / "ledger.jsonl").read_text().splitlines()
    assert sum(json.loads(l)["status"] == "failed" for l in lines) == 1


def test_dry_run_touches_nothing(tmp_path):
    src = _local_source(tmp_path, 2)
    c = Collector(root=tmp_path / "data", dry_run=True, log=lambda *_: None)
    s = c.fetch_source(src)
    assert len(s["planned"]) == 2 and not (tmp_path / "data").exists()


SENSORS = """id,make,model,format,pixel_pitch_um,width_px,height_px,full_well_e,read_noise_e,qe_peak,bit_depth,condition,source_url,quote
s_big,Acme,Big,FF,5.9,6000,4000,90000,1.2,0.8,14,base ISO,https://example.org/big,"full well 90 ke-"
s_small,Acme,Small,phone,1.0,8000,6000,5000,2.5,0.7,12,base ISO,https://example.org/small,"full well 5 ke-"
"""
LENSES = """id,make,model,format,focal_mm,f_number,mtf50_lpmm_center,condition,source_url,quote
l_ff,Acme,50,FF,50,1.4,70,f/4 centre,https://example.org/l50,"MTF50 70 lp/mm"
l_ff2,Acme,85,FF,85,1.2,90,f/4 centre,https://example.org/l85,"MTF50 90 lp/mm"
"""
DISPLAYS = """id,make,model,class,diagonal_in,width_px,height_px,peak_nits,distance_m,condition,source_url,quote
d_8k,Acme,Mon8K,monitor,31.5,7680,4320,600,0.6,desk,https://example.org/d8k,"31.5 inch 8K"
d_tv,Acme,TV55,tv,55,3840,2160,1000,1.5,living room,https://example.org/tv,"55 inch 4K"
d_itu,ITU-R,UHDTV design distance,tv,65,3840,2160,,1.2141,1.5 H,https://example.org/itu,"1.5 H"
"""
VISION = """id,harness_param,value,unit,condition,source,source_url,quote
v1,gate.delta_e_mean,1.0,dE2000,JND at threshold,Std,https://example.org/v1,"JND ~1"
v2,display.distance_m,0.5,m,desktop,Std,https://example.org/v2,"0.5 m"
v3,tie.mtf50_cpp,0.02,cpp,n/a,Std,https://example.org/v3,"q"
"""


def test_reference_derivers_and_derived_presets(tmp_path, monkeypatch):
    from harness import camera, display
    from harness.data import derive
    ref = tmp_path / "reference"
    ref.mkdir()
    for name, body in (("sensors", SENSORS), ("lenses", LENSES), ("displays", DISPLAYS), ("vision_thresholds", VISION)):
        (ref / f"{name}.csv").write_text(body, encoding="utf-8")
    monkeypatch.setattr(derive, "REFERENCE_DIR", ref)
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path / "data"))
    rep = derive.run_derivers(["camera_presets", "displays", "vision_thresholds"])
    assert all(not v.startswith("FAILED") for v in rep.values()), rep
    cp = json.loads((tmp_path / "data" / "derived" / "camera_presets.json").read_text())
    assert cp["presets"]["s_big"]["lens"]["model"] == "85"  # sharpest FF lens wins
    assert cp["presets"]["s_small"]["lens"] is None  # no phone lens row -> harness default
    assert cp["presets"]["best_case_earth_sourced"]["alias_of"] == "s_big"  # 90000/1.2 beats 5000/2.5
    # simulated range: log2(FW / sqrt(rn^2 + LSB^2 / 12)); s_big 14-bit 15.47 stops, s_small 12-bit 10.95
    assert abs(cp["presets"]["s_big"]["dr_stops_simulated"] - 15.47) < 0.02
    assert abs(cp["presets"]["s_small"]["dr_stops_simulated"] - 10.95) < 0.02
    assert cp["frontier"]["dr_stops_simulated"]["best"] == "s_big"
    assert cp["presets"]["s_big"]["optics_source"] == "lenses.csv: Acme 85"
    assert cp["presets"]["s_small"]["optics_source"].startswith("harness default 50 mm f/1.4")
    assert 0 < cp["presets"]["s_big"]["residual_blur_um"] < 3
    cam, _ = camera.from_dict({"preset": "derived:s_big", "exposure_s": 0.01})
    assert cam.pixel_pitch_um == 5.9 and cam.full_well_e == 90000 and abs(cam.aperture_mm - 85 / 1.2) < 1e-9
    dp = json.loads((tmp_path / "data" / "derived" / "displays.json").read_text())
    assert abs(dp["displays"]["d_8k"]["ppd"] - 115) < 2 and dp["summary"]["above_threshold_low"] == 1
    itu = dp["displays"]["d_itu"]  # ITU design distance: defined as one pixel per arc-minute (60 ppd, 1.59 H), the
    assert abs(itu["ppd"] - 56.5) < 0.2 and abs(itu["ppd_at_design_distance"] - 56.5) < 0.2  # table's 1.5 H gives 56.5
    assert abs(itu["itu_design_distance_m"] - 1.2141) < 1e-3 and itu["peak_nits"] is None
    spec = display.from_dict({"preset": "derived:d_tv", "distance_m": 1.0})
    assert spec.diagonal_in == 55 and spec.distance_m == 1.0  # explicit key overrides the preset
    claim = json.loads((tmp_path / "data" / "derived" / "claim_sourced.json").read_text())
    tone = next(g for g in claim["gates"] if g["name"] == "tone_color")
    de = next(c for c in tone["checks"] if c["metric"] == "delta_e_mean")
    assert de["value"] == 1.0 and de["source"].startswith("https://")
    assert claim["display"]["distance_m"] == 0.5


NOISE_CHECK = {
    "iso_scaling": {
        "Google Pixel": {"iso": [100, 400], "gain_e_per_dn": [4.8, 1.2], "instances": ["a", "b"],
                         "log_gain_vs_log_iso_slope": -1.0, "unity_gain_iso": 495.0},
        "iPhone 7": {"iso": [200, 800], "gain_e_per_dn": [0.22, 0.22], "instances": ["c", "d"],
                     "log_gain_vs_log_iso_slope": 0.0, "unity_gain_iso": None}},
    "instances": [
        {"instance": "a", "ok": True, "camera_full": "Google Pixel", "iso": 100, "gain_e_per_dn": 4.8, "read_noise_e": 7.2,
         "full_well_e": 4608.0, "raw": {"white": 1023, "black": 63, "shape": [3024, 4032]}, "row_noise_ratio": 1.232},
        {"instance": "b", "ok": True, "camera_full": "Google Pixel", "iso": 400, "gain_e_per_dn": 1.2, "read_noise_e": 3.0,
         "full_well_e": 1152.0, "raw": {"white": 1023, "black": 63, "shape": [3024, 4032]}},
        {"instance": "c", "ok": True, "camera_full": "iPhone 7", "iso": 200, "gain_e_per_dn": 0.22, "read_noise_e": 1.9,
         "full_well_e": 785.0, "raw": {"white": 4095, "black": 528, "shape": [3024, 4032]}},
        {"instance": "d", "ok": True, "camera_full": "iPhone 7", "iso": 800, "gain_e_per_dn": 0.22, "read_noise_e": 1.7,
         "full_well_e": 785.0, "raw": {"white": 4095, "black": 528, "shape": [3024, 4032]}},
        {"instance": "e", "ok": False, "camera_full": "iPhone 7", "iso": 100}],
    "verdict": {}}


def test_measured_presets_one_per_camera(tmp_path, monkeypatch):
    from harness import camera
    from harness.data import derive
    ref = tmp_path / "reference"
    ref.mkdir()
    for name, body in (("sensors", SENSORS), ("lenses", LENSES)):
        (ref / f"{name}.csv").write_text(body, encoding="utf-8")
    monkeypatch.setattr(derive, "REFERENCE_DIR", ref)
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path / "data"))
    (tmp_path / "data" / "derived").mkdir(parents=True)
    (tmp_path / "data" / "derived" / "noise_check.json").write_text(json.dumps(NOISE_CHECK), encoding="utf-8")
    rep = derive.run_derivers(["camera_presets"])
    assert "measured_google_pixel" in rep["camera_presets"] and "measured_iphone_7" in rep["camera_presets"]
    cp = json.loads((tmp_path / "data" / "derived" / "camera_presets.json").read_text())
    px, ip = cp["presets"]["measured_google_pixel"], cp["presets"]["measured_iphone_7"]
    assert px["full_well_e"] == 4608.0 and px["read_noise_e"] == 7.2 and px["bit_depth"] == 10  # lowest ISO row
    assert px["iso_scaling_is_analog_gain"] is True and px["sensor"]["read_noise_e_by_iso"] == {"100": 7.2, "400": 3.0}
    assert ip["full_well_e"] == 785.0 and ip["bit_depth"] == 12 and ip["iso_scaling_is_analog_gain"] is False
    assert "not raw analog-gain steps" in ip["sensor"]["condition"]
    assert cp["presets"]["phone_measured"]["alias_of"] == "measured_google_pixel"
    assert px["row_noise_e"] == 0.055  # 7.2 * sqrt(0.232 / 4032), a lower bound
    assert ip["row_noise_e"] == 0.0 and "not measured" in ip["row_noise_source"]
    # pixel pitch per phone (hand-set, sourced), not the generic phone preset's 1.2 um; optics fall back to the
    # phone preset when the SIDD zip is not on disk (the fixture rows carry no zip)
    assert px["pixel_pitch_um"] == 1.55 and ip["pixel_pitch_um"] == 1.22
    assert "hand-set" in px["pixel_pitch_source"] and "hand-set" in ip["pixel_pitch_source"]
    assert px["optics_source"].startswith("optics from the phone preset")
    assert cp["measured_cameras"]["measured_google_pixel"]["row_noise_e"] == 0.055
    assert abs(px["dr_stops_simulated"] - 9.3) < 0.1  # 4608 e- over sqrt(7.2^2 + 4.5^2 / 12)
    assert set(cp["measured_cameras"]) == {"measured_google_pixel", "measured_iphone_7"}
    cam, _ = camera.from_dict({"preset": "derived:measured_iphone_7", "exposure_s": 0.01})
    assert cam.full_well_e == 785.0 and cam.read_noise_e == 1.9


def test_reference_rows_are_sourced():
    """Every checked-in reference row must cite a URL and quote the number (no source, no row)."""
    from harness.data import derive
    for name in derive.REQUIRED_COLUMNS:
        rows = derive.read_reference(name)  # raises on an unsourced row
        for r in rows:
            assert r["source_url"].startswith("http") and r["quote"].strip()


CGATS = """ORIGINATOR "ColorChecker24 - November2014 edition and newer"
"MeasurementCondition=M0\tFilter=no"
BEGIN_DATA_FORMAT
SAMPLE_NAME Lab_L   Lab_a   Lab_b
END_DATA_FORMAT
BEGIN_DATA
A1\t37,54\t14,37\t14,92
A4\t95,19\t-1,03\t2,93
END_DATA
"""
CAMSPEC = "Cam One\n" + "\n".join(" ".join(str(v) for v in row) for row in (
    [0.0] * 20 + [1.0] * 13,   # R: flat 600-720 -> effective 660
    [0.0] * 10 + [1.0] * 13 + [0.0] * 10,  # G: 500-620 -> 560
    [1.0] * 13 + [0.0] * 20)) + "\n"  # B: 400-520 -> 460


def test_colour_reference_derivers(tmp_path, monkeypatch):
    import zipfile
    from harness.data import derive
    import os
    if os.name == "nt" and not str(tmp_path).startswith('\\\\?\\'):
        tmp_path = Path('\\\\?\\' + str(tmp_path))  # basetemp under a deep cwd passes 260 chars
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path / "data"))
    raw = tmp_path / "data" / "raw"
    (raw / "zenodo-xrite-colorchecker").mkdir(parents=True)
    with zipfile.ZipFile(raw / "zenodo-xrite-colorchecker" / "ColorChecker24_After_Nov2014.zip", "w") as z:
        z.writestr("ColorChecker24_After_Nov2014.txt", CGATS)
    (raw / "zenodo-camspec-jiang2013").mkdir()
    (raw / "zenodo-camspec-jiang2013" / "camspec_database.txt").write_text(CAMSPEC)
    rep = derive.run_derivers(["colorchecker", "camspec"])
    assert all(not v.startswith("FAILED") for v in rep.values()), rep
    cc = json.loads((tmp_path / "data" / "derived" / "colorchecker.json").read_text())
    chart = cc["colorchecker24_after_nov2014"]
    assert chart["lab"]["A1"] == [37.54, 14.37, 14.92] and chart["measurement_condition"] == "M0"
    cs = json.loads((tmp_path / "data" / "derived" / "camspec.json").read_text())
    assert cs["cameras"]["Cam One"]["effective_wavelength_nm"] == [660.0, 560.0, 460.0]
    assert cs["typical_effective_wavelength_nm"] == [660.0, 560.0, 460.0]


def test_manifest_and_plan_render(tmp_path, monkeypatch):
    from harness.data.cli import render_plan
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path / "nodata"))
    sources = load_manifest()
    validate(sources)
    assert len(sources) >= 100 and sum(s.fetchable for s in sources) >= 60
    md = render_plan()
    assert md.startswith("**") and "### Tier S" in md and "`cie-colorimetry`" in md
    assert all(s.id in md for s in sources)


def test_rpu_index_lists_one_verified_raw_per_camera():
    from harness.data.apis import rpu_index
    cc0 = "<a href='https://creativecommons.org/publicdomain/zero/1.0/' class='cc'>co</a>"
    nc = "<a href='http://creativecommons.org/licenses/by-nc-sa/4.0/' class='cc'>nc</a>"
    sha = "ab" * 32

    def link(i, name, mb):
        return (f"<a href='https://raw.pixls.us/getfile.php/{i}/nice/{name}'>{name}</a>"
                f"<div class='checksumdata'><span title='sha256 Checksum'>{sha}</span>&nbsp;({mb}MB)</div>")

    rows = [
        ["Canon", "EOS 7D", "sRAW2 (3:2)", 17.9, "", cc0, "2016", link(1, "Canon - EOS 7D - sRAW2 (3:2).CR2", "10.55"), ""],
        ["Canon", "EOS 7D", "RAW", 17.9, "", cc0, "2016", link(2, "Canon - EOS 7D - RAW.CR2", "24.1"), ""],
        ["Nikon", "D850", None, 45.0, "", nc, "2017", link(3, "Nikon - D850.NEF", "50"), ""],
        ["Sony", "A7", "", None, "", cc0, "2014", link(4, "Sony - A7.ARW", "24"), ""],
    ]

    class Stub:
        def get_json(self, url, params=None, headers=None):
            assert params == {"set": "all"}
            return {"data": rows}

    src = Source(id="rpu", name="rpu", kind="dataset", access="api", url="u", expander="rpu_index", params={})
    specs = rpu_index(Stub(), src)
    assert [s.path for s in specs] == ["repository.json", "Canon/EOS_7D/sRAW2_3_2.cr2", "Sony/A7/default.arw"]
    assert specs[1].sha256 == sha and specs[1].size == 10_550_000
    assert "%20" in specs[1].url and " " not in specs[1].url
    src.params = {"one_per_model": "0", "license": "all", "limit": 3}
    assert [s.path for s in rpu_index(Stub(), src)] == ["repository.json", "Canon/EOS_7D/sRAW2_3_2.cr2", "Canon/EOS_7D/RAW.cr2"]


def test_dataverse_expander_lists_verified_files_by_pattern():
    from harness.data.apis import dataverse_dataset
    files = [
        {"directoryLabel": "iphone_11_pro_main", "dataFile": {"id": 1, "filename": "indoor.zip", "filesize": 379,
                                                              "md5": "aa" * 16, "checksum": {"type": "MD5", "value": "aa" * 16}}},
        {"dataFile": {"id": 2, "filename": "README.md", "filesize": 12, "checksum": {"type": "SHA-1", "value": "bb" * 20}}},
        {"dataFile": {"id": 3, "filename": "Bayer_plates_GT_ISO100.arw", "filesize": 49_088_000,
                      "checksum": {"type": "MD5", "value": "cc" * 16}}},
    ]

    class Stub:
        def get_json(self, url, params=None, headers=None):
            assert url == "https://dataverse.uclouvain.be/api/datasets/:persistentId/"
            assert params == {"persistentId": "doi:10.14428/DVN/DEQCIM"}
            return {"data": {"latestVersion": {"files": files}}}

    src = Source(id="dv", name="dv", kind="dataset", access="api", url="u", expander="dataverse_dataset",
                 params={"server": "https://dataverse.uclouvain.be/", "doi": "doi:10.14428/DVN/DEQCIM"})
    specs = dataverse_dataset(Stub(), src)
    assert [s.path for s in specs] == ["dataset.json", "Bayer_plates_GT_ISO100.arw", "README.md", "iphone_11_pro_main/indoor.zip"]
    assert specs[1].url.endswith("/api/access/datafile/3") and specs[1].md5 == "cc" * 16 and specs[1].size == 49_088_000
    assert specs[2].md5 is None                                   # SHA-1 only: nothing the collector can check
    assert specs[3].md5 == "aa" * 16
    src.params["pattern"] = "ISO100"
    assert [s.path for s in dataverse_dataset(Stub(), src)] == ["dataset.json", "Bayer_plates_GT_ISO100.arw"]


def test_gcs_bucket_expander_pages_and_converts_md5():
    import base64
    from harness.data.apis import gcs_bucket
    md5_hex = "0123456789abcdef" * 2
    b64 = base64.b64encode(bytes.fromhex(md5_hex)).decode()
    pages = {
        None: {"items": [{"name": "20171106_subset/bursts/", "size": "0"},
                         {"name": "20171106_subset/bursts/0006_a/payload_N000.dng", "size": "6356788", "md5Hash": b64},
                         {"name": "20171106_subset/bursts/0006_a/timing.txt", "size": "66", "md5Hash": b64}],
               "nextPageToken": "p2"},
        "p2": {"items": [{"name": "20171106_subset/results_20171023/0006_a/final.jpg", "size": "100", "md5Hash": b64}]},
    }
    calls = []

    class Stub:
        def get_json(self, url, params=None, headers=None):
            assert url == "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o"
            assert params["prefix"] == "20171106_subset/"
            calls.append(params.get("pageToken"))
            return pages[params.get("pageToken")]

    src = Source(id="g", name="g", kind="dataset", access="api", url="u", expander="gcs_bucket",
                 params={"bucket": "hdrplusdata", "prefix": "20171106_subset/"})
    specs = gcs_bucket(Stub(), src)
    assert [s.path for s in specs] == ["bursts/0006_a/payload_N000.dng", "bursts/0006_a/timing.txt",
                                       "results_20171023/0006_a/final.jpg"]
    assert specs[0].url == "https://storage.googleapis.com/hdrplusdata/20171106_subset/bursts/0006_a/payload_N000.dng"
    assert specs[0].md5 == md5_hex and specs[0].size == 6356788 and calls == [None, "p2"]
    calls.clear()
    src.params.update(pattern="payload", limit=1)
    assert [s.path for s in gcs_bucket(Stub(), src)] == ["bursts/0006_a/payload_N000.dng"] and calls == [None]


def test_expand_failures_are_ledgered_and_cleared(tmp_path):
    root = tmp_path / "data"
    c = Collector(root=root, log=lambda *_: None)
    src = Source(id="x", name="x", kind="dataset", access="api", url="u", expander="nope")
    c.note_expand(src, RuntimeError("404 gated"))
    assert c.status()["x"]["failed"] == 1 and "404 gated" in c.status()["x"]["error"]
    assert Collector(root=root, log=lambda *_: None).status()["x"]["failed"] == 1  # survives reload
    c.note_expand(src)  # a later successful listing clears it
    assert "x" not in c.status() and "x" not in Collector(root=root, log=lambda *_: None).status()
    Collector(root=root, dry_run=True, log=lambda *_: None).note_expand(src, RuntimeError("dry"))
    assert "x" not in Collector(root=root, log=lambda *_: None).status()

def test_wikimedia_category_keeps_the_licence_per_file_and_verifies_sha1():
    from harness.data.apis import wikimedia_category

    def page(pageid, title, lic, mime="image/jpeg", w=4000, h=3000, size=5_000_000):
        return {"pageid": pageid, "title": f"File:{title}", "imageinfo": [{"url": f"https://upload.wikimedia.org/{pageid}", "size": size,
                "width": w, "height": h, "sha1": f"{pageid:040x}", "mime": mime,
                "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": "A"}}}]}

    pages = [
        {"query": {"pages": [page(1, "Sunset over Dubrovnik.jpg", "CC BY-SA 4.0"), page(2, "Nc shot.jpg", "CC BY-NC-SA 2.0"),
                             page(3, "Map.svg", "CC0", mime="image/svg+xml"), page(4, "Tiny.jpg", "CC0", w=800, h=600),
                             page(5, "Huge scan.tif", "Public domain", mime="image/tiff", size=90_000_000)]},
         "continue": {"gcmcontinue": "page|2", "continue": "gcmcontinue||"}},
        {"query": {"pages": [page(6, "Old print.jpg", "Public domain"), page(7, "No-derivs.jpg", "CC BY-ND 3.0"), page(8, "Shot.png", "CC BY 2.0", mime="image/png")]}},
    ]

    class Stub:
        def __init__(self):
            self.calls = []

        def get_json(self, url, params=None, headers=None):
            assert url.startswith("https://commons.wikimedia.org/") and params["gcmtitle"] == "Category:T"
            self.calls.append(dict(params))
            return pages[len(self.calls) - 1]

        def log(self, msg):
            pass

    src = Source(id="wm", name="wm", kind="dataset", access="api", url="u", expander="wikimedia_category",
                 params={"category": "Category:T", "min_pixels": 4e6, "max_bytes_each": 40e6})
    stub = Stub()
    specs = wikimedia_category(stub, src)
    assert [s.path for s in specs] == ["index/0000.json", "cc-by-sa-4.0/1_Sunset_over_Dubrovnik.jpg", "index/0001.json",
                                       "public-domain/6_Old_print.jpg", "cc-by-2.0/8_Shot.png"]
    assert specs[1].sha1 == f"{1:040x}" and specs[1].size == 5_000_000 and specs[1].sha256 is None
    assert stub.calls[1]["gcmcontinue"] == "page|2" and "gcmcontinue" not in stub.calls[0]
    assert "gcmcontinue=page%7C2" in specs[2].url and specs[2].to_dict() == {"url": specs[2].url, "path": "index/0001.json"}
    assert specs[1].to_dict()["sha1"] == specs[1].sha1
    src.params = {"category": "Category:T", "limit": 1}
    assert [s.path for s in wikimedia_category(Stub(), src)] == ["index/0000.json", "cc-by-sa-4.0/1_Sunset_over_Dubrovnik.jpg"]
    src.params = {"category": "Category:T", "licences": "^cc0"}
    assert [s.path for s in wikimedia_category(Stub(), src) if not s.path.startswith("index/")] == ["cc0/4_Tiny.jpg"]


def test_listings_back_off_politely_and_honour_retry_after(tmp_path, monkeypatch):
    """A 429 with Retry-After waits that long (never less than the exponential floor), a 503 without one backs off
    exponentially, the walk then succeeds; hosts with an etiquette get their own interval; the agent names us."""
    import requests
    from harness.data import collect as C

    class R:
        def __init__(self, status, headers=None):
            self.status_code, self.headers, self.url = status, headers or {}, "https://commons.wikimedia.org/w/api.php"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code), response=self)

        def json(self):
            return {"ok": True}

        text = "ok"

    assert "github.com/Kolm-ai/kolm" in C.USER_AGENT and "python-requests" in C.USER_AGENT
    assert C._HostThrottle(0.05).per_host["commons.wikimedia.org"] >= 1.0
    assert C.retry_delay(0, R(429, {"Retry-After": "7"}), base=2.0) == 7.0
    assert C.retry_delay(3, R(429, {"Retry-After": "1"}), base=2.0) == 16.0          # floor wins over a short hint
    assert C.retry_delay(0, R(429, {"Retry-After": "9999"})) == C.MAX_BACKOFF_S      # cap
    assert C.retry_delay(0, R(503, {"Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT"})) == 1.5  # past date: floor
    assert C.retry_delay(2, None) == 6.0

    sleeps, calls = [], []
    monkeypatch.setattr(C.time, "sleep", lambda s: sleeps.append(s))
    c = C.Collector(root=tmp_path, jobs=1, dry_run=True, log=lambda *_: None)
    assert c._session.headers["User-Agent"] == C.USER_AGENT
    c._throttle = C._HostThrottle(0.0, {})
    responses = [R(429, {"Retry-After": "7"}), R(503), R(200)]
    c._session.get = lambda url, **kw: (calls.append((url, kw)), responses.pop(0))[1]
    assert c.get_json("https://commons.wikimedia.org/w/api.php", params={"a": "1"}) == {"ok": True}
    assert sleeps == [7.0, 4.0] and len(calls) == 3 and calls[0][1]["params"] == {"a": "1"}
    responses[:] = [R(429)] * C.LISTING_RETRIES
    sleeps.clear()
    with pytest.raises(requests.HTTPError):
        c.get_text("https://commons.wikimedia.org/w/api.php")
    assert len(sleeps) == C.LISTING_RETRIES - 1 and sleeps == sorted(sleeps) and max(sleeps) <= C.MAX_BACKOFF_S
