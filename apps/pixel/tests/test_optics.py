"""Traced optics (DATA_PLAN section 4 step 4): MTF50 reading, Cooke-triplet sanity bounds, the DeepLens
JSON builder against thick-lens formulas, failure rows and the summary, and the Zemax header reader."""
import json
import math

import numpy as np
import pytest

from harness.data import optics as O


def test_mtf50_first_crossing_interpolates():
    assert O.mtf50_lpmm([0, 10, 20, 30], [1.0, 0.8, 0.4, 0.1]) == pytest.approx(17.5)
    assert math.isnan(O.mtf50_lpmm([0, 10, 20], [1.0, 0.9, 0.6]))
    assert math.isnan(O.mtf50_lpmm([0, 10], [0.2, 0.1]))                       # below 0.5 at index 0: no crossing
    assert O.scalar(np.array([2.5])) == 2.5 and O.scalar(np.float64(3.0)) == 3.0
    assert O.diffraction_mtf50_lpmm(0.55, 5.0) == pytest.approx(146.9, abs=0.1)
    assert O._round(None, 2) is None and O._round(float("nan"), 2) is None and O._round(1.2345, 2) == 1.23


@pytest.fixture(scope="module")
def cooke():
    pytest.importorskip("optiland")
    from optiland.samples.objectives import CookeTriplet
    o = CookeTriplet()
    o.fields.set_type("angle")
    o.fields.fields.clear()
    for fr in O.FIELD_FRACTIONS:
        o.add_field(y=fr * 20.0)
    info = {"field_kind": "angle", "max_field": 20.0, "image_height_mm": 18.2}
    return O.analyse_optic(o, info)


def test_cooke_triplet_traces_between_its_paraxial_and_diffraction_limits(cooke):
    r = cooke
    assert r["focal_mm"] == pytest.approx(50.0, rel=0.02) and r["f_number"] == pytest.approx(5.0, rel=0.02)
    assert r["focus"]["best_rms_opd_waves"] < r["focus"]["paraxial_rms_opd_waves"]
    assert not r["focus"]["hit_scan_edge"]
    assert 0.3 < r["strehl_centre"] < 1.0
    assert 30.0 < r["mtf50_centre_lpmm"] < r["diffraction_mtf50_lpmm"] * 1.02
    assert r["mtf50_edge_lpmm"] < r["mtf50_centre_lpmm"]
    assert [f["fraction"] for f in r["fields"]] == list(O.FIELD_FRACTIONS)
    assert r["fields"][-1]["field"] == pytest.approx(20.0)
    assert r["residual_blur_um"] > 0 and r["n_surfaces"] == 8


def _singlet(R=50.0, t=4.0, nd=1.5168, vd=64.17, fnum=8.0, r_sensor=3.0):
    """Equiconvex thick singlet: 1/f = (n-1)[2/R - (n-1) t / (n R^2)], BFL = f (1 - (n-1) t / (n R))."""
    n = nd
    f = 1.0 / ((n - 1.0) * (2.0 / R - (n - 1.0) * t / (n * R * R)))
    bfl = f * (1.0 - (n - 1.0) * t / (n * R))
    d = {"info": "synthetic singlet", "foclen": f, "fnum": fnum, "r_sensor": r_sensor,
         "d_sensor": t + bfl, "sensor_size": [2 * r_sensor, 2 * r_sensor],
         "surfaces": [{"type": "Spheric", "r": 4.0, "c": 1.0 / R, "roc": R, "d": 0.0, "mat1": "air",
                       "mat2": f"{nd}/{vd}", "d_next": t},
                      {"type": "Spheric", "r": 4.0, "c": -1.0 / R, "roc": -R, "d": t, "mat1": f"{nd}/{vd}",
                       "mat2": "air", "d_next": bfl}]}
    return d, f, bfl


def test_deeplens_json_build_matches_thick_lens_focal_length():
    pytest.importorskip("optiland")
    d, f_expected, _ = _singlet()
    assert f_expected == pytest.approx(49.04, abs=0.05)
    o, info = O.build_deeplens(d)
    assert O.scalar(o.paraxial.f2()) == pytest.approx(f_expected, rel=2e-3)
    assert O.scalar(o.paraxial.FNO()) == pytest.approx(8.0, rel=1e-2)
    assert info["max_field"] == pytest.approx(math.degrees(math.atan(3.0 / f_expected)))
    assert any(s.is_stop for s in o.surfaces.surfaces)
    r = O.analyse_optic(o, info)
    assert 0.0 < r["strehl_centre"] <= 1.0 and math.isfinite(r["mtf50_centre_lpmm"])


def test_deeplens_aspheric_and_stop_rows(tmp_path):
    pytest.importorskip("optiland")
    d, _, _ = _singlet()
    d["surfaces"].insert(0, {"type": "Aperture", "r": 3.0, "d": -2.0, "d_next": 2.0, "mat1": "air", "mat2": "air"})
    lens = d["surfaces"][1]
    lens.update(type="Aspheric", k=0.0, ai=[0.0, -1e-6, 0.0])
    o, _ = O.build_deeplens(d)
    assert o.surfaces.surfaces[1].is_stop
    assert type(o.surfaces.surfaces[2].geometry).__name__ == "EvenAsphere"
    p = tmp_path / "singlet.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    r = O.trace_file(("deeplens", str(p)))
    assert r["ok"], r.get("error")
    assert r["source"] == "deeplens" and r["aperture"] == "imageFNO" and r["seconds"] >= 0


def test_repair_materials_replaces_abbe_zero_glass_with_constant_index():
    pytest.importorskip("optiland")
    from optiland.materials import AbbeMaterial
    d, f_expected, _ = _singlet()
    o, _ = O.build_deeplens(d)
    gap = [i for i, s in enumerate(o.surfaces.surfaces) if type(s.material_post).__name__ == "AbbeMaterial"]
    assert gap, "the singlet has one glass gap"
    bad = AbbeMaterial(1.5168, 0.0, model="buchdahl")
    o.surfaces.surfaces[gap[0]].material_post = bad
    assert not math.isfinite(O.scalar(o.paraxial.f2()))
    assert O.repair_materials(o) == 1 and O.repair_materials(o) == 0
    assert type(o.surfaces.surfaces[gap[0]].material_post).__name__ == "IdealMaterial"
    assert o.surfaces.surfaces[gap[0]].material_post is o.surfaces.surfaces[gap[0] + 1].material_pre
    assert O.scalar(o.paraxial.f2()) == pytest.approx(f_expected, rel=2e-3)


def _row(name, source, image_height, residual, fno, edge=True):
    return {"name": name, "source": source, "ok": True, "focal_mm": 50.0, "f_number": fno, "strehl_centre": 0.5,
            "mtf50_centre_lpmm": 60.0, "mtf50_edge_lpmm": 20.0, "diffraction_mtf50_lpmm": 100.0,
            "mtf50_centre_over_diffraction": 0.6, "residual_blur_um": residual, "image_height_mm": image_height,
            "focus": {"hit_scan_edge": edge}, "fields": [{"mtf50_t_lpmm": 60.0}, {"mtf50_t_lpmm": None}]}


def test_trace_file_records_failures_and_summary_counts_them(tmp_path):
    missing = O.trace_file(("claff", str(tmp_path / "missing.zmx")))
    assert missing["ok"] is False and missing["error"].startswith("FileNotFoundError")
    unknown = O.trace_file(("other", str(tmp_path / "x")))
    assert unknown["error"].startswith("ValueError: unknown prescription kind")
    rows = [_row("a", "claff", 21.6, 1.5, 2.0, edge=False), _row("b", "deeplens", 14.0, 1.0, 1.8, edge=False),
            _row("c", "claff", 21.6, 0.5, 4.0, edge=True), missing]
    s = O.summarise(rows)
    assert s["n_files"] == 4 and s["n_ok"] == 3 and s["n_failed"] == 1 and "missing.zmx" in s["failures"]
    assert s["n_hit_scan_edge"] == 1 and s["n_fields_untraceable"] == 3 and s["n_fields_total"] == 6
    assert s["n_full_frame"] == 2
    assert s["best_full_frame"]["name"] == "a" and s["n_stopped_down"] == 0       # c hit the scan edge
    assert s["best_full_frame_by_mtf50"][0]["name"] == "a" and s["best_full_frame_by_residual"][0]["name"] == "a"
    assert s["residual_blur_um"]["n"] == 2 and s["by_f_number"]["F1.6-2.2"]["n"] == 2
    assert s["best_by_mtf50_centre"][0]["mtf50_edge_lpmm"] == 20.0
    assert s["share_diffraction_limited_centre"] == 0.0
    assert O.summarise([missing])["best_full_frame"] is None


def test_zemax_meta_and_jobs(tmp_path):
    claff = tmp_path / "claff"
    claff.mkdir()
    text = "VERS 130\nNAME Example\nFTYP 3 0 3 3 0 0 0\nYFLN 0 15.12 21.6 0 0 0\nSURF 0\n"
    (claff / "A.zmx").write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
    (claff / "B.ZMX").write_text(text, encoding="latin-1")
    for p in (claff / "A.zmx", claff / "B.ZMX"):
        assert O.zemax_meta(O.read_text(p)) == {"ftyp": 3, "yfln": [0, 15.12, 21.6, 0, 0, 0], "name": "Example"}
    jobs = O.prescription_jobs(claff, tmp_path / "nowhere")
    assert [k for k, _ in jobs] == ["claff", "claff"]
    assert O.prescription_jobs(None, None) == []
