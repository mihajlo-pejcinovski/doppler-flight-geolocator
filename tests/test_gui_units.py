"""GUI building blocks that need no window: map GeoJSON, spectrogram pooling,
background jobs (cancel, questions answered on the UI thread)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dopplergeo.core.geo import haversine_m  # noqa: E402
from dopplergeo.gui import geojson as gj  # noqa: E402


def track(icao, lat0, lon0, n=50):
    t = 1000.0 + np.arange(n) * 2.0
    return SimpleNamespace(icao24=icao, callsign=icao.upper(), t=t, lat=lat0 + np.arange(n) * 1e-3,
                           lon=np.full(n, lon0))


def test_tracks_geojson_decimates_and_keeps_ends():
    tr = track("abc", 52.0, 4.0, n=1000)
    doc = gj.tracks_geojson([tr, track("short", 0, 0, n=1)], max_points=100)
    assert len(doc["features"]) == 1
    coords = doc["features"][0]["geometry"]["coordinates"]
    assert len(coords) <= 101
    assert coords[0] == [4.0, 52.0] and coords[-1][1] == pytest.approx(52.999)


def test_aircraft_position_and_heading():
    tr = track("abc", 52.0, 4.0)
    doc = gj.aircraft_geojson([tr], 1000.0 + 21.0)
    (f,) = doc["features"]
    lon, lat = f["geometry"]["coordinates"]
    assert lat == pytest.approx(52.0105, abs=1e-6)
    assert f["properties"]["heading"] == pytest.approx(0.0, abs=1e-6)  # flying north
    assert gj.aircraft_geojson([tr], 0.0)["features"] == []           # outside the track
    assert gj.aircraft_geojson([tr], 1010.0, colors={})["features"] == []  # not a candidate


def test_heatmap_keeps_best_cells_with_likelihood_weights():
    lat, lon = np.meshgrid(np.linspace(52, 52.1, 30), np.linspace(4, 4.1, 40), indexing="ij")
    chi2 = (lat - 52.05) ** 2 * 1e5 + (lon - 4.05) ** 2 * 1e5 + 10
    grid = SimpleNamespace(lat=lat, lon=lon, heatmap=lambda: (lat, lon, chi2 - chi2.min()))
    doc = gj.heatmap_geojson(grid, max_dchi2=20.0, max_cells=50)
    w = [f["properties"]["w"] for f in doc["features"]]
    assert 0 < len(w) <= 50 and max(w) == pytest.approx(1.0, abs=1e-3)
    assert all(f["properties"]["dchi2"] <= 20.0 for f in doc["features"])


def test_family_colors_are_stable():
    assert gj.family_color(0) == gj.family_color(len(gj.FAMILY_COLORS))
    assert gj.family_color(None) == "#888888"


def test_display_image_keeps_narrow_tones_when_pooling():
    from dopplergeo.gui.panels.spectrogram import display_image

    times = np.arange(10_000) * 0.1
    freqs = np.arange(4000) * 0.5
    v = np.zeros((freqs.size, times.size), np.float32)
    v[1234, :] = 50.0  # one-bin tone at 617 Hz
    img, rect = display_image(times, freqs, v, fmax=1500.0, max_rows=300, max_cols=1000)
    assert img.shape[0] <= 300 and img.shape[1] <= 1000
    assert img.max() == 50.0  # max-pooling keeps the line
    row = int(np.argmax(img[:, 0]))
    f_row = rect.top() + (row + 0.5) * rect.height() / img.shape[0]
    assert f_row == pytest.approx(617.0, abs=rect.height() / img.shape[0])


@pytest.fixture
def runner(qtbot):
    from dopplergeo.gui.workers import JobRunner

    r = JobRunner()
    yield r
    assert r.shutdown(10_000)


def test_job_result_progress_and_question(qtbot, runner):
    got, logs, asked = [], [], []

    def job(ctx):
        ctx.log("working")
        ctx.progress("download", 1, 2)
        ok = ctx.ask("download 3 GB?")
        return threading.current_thread() is threading.main_thread(), ok

    def answer(q):
        asked.append((q, threading.current_thread() is threading.main_thread()))
        return True

    runner.start("j", job, on_result=got.append, on_error=lambda m, tb: got.append(m), on_log=logs.append,
                 on_question=answer)
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    assert got == [(False, True)]                 # ran off the UI thread; question answered yes
    assert asked == [("download 3 GB?", True)]    # asked on the UI thread
    assert logs == ["working"]
    qtbot.waitUntil(lambda: not runner.busy(), timeout=2000)


def test_cancel_drops_the_result_and_errors_are_reported(qtbot, runner):
    got, errors = [], []

    def slow(ctx):
        for _ in range(200):
            time.sleep(0.01)
            ctx.check()
        return "late"

    runner.start("j", slow, on_result=got.append, on_error=lambda m, tb: errors.append(m))
    runner.cancel("j")
    qtbot.waitUntil(lambda: not runner.busy(), timeout=5000)
    assert got == [] and errors == []

    runner.start("k", lambda ctx: 1 / 0, on_result=got.append, on_error=lambda m, tb: errors.append(m))
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    assert "ZeroDivisionError" in errors[0] and got == []


def test_restarting_a_job_supersedes_the_old_one(qtbot, runner):
    got = []
    release = threading.Event()

    def first(ctx):
        release.wait(5)
        return "old"

    runner.start("j", first, on_result=got.append, on_error=print)
    runner.start("j", lambda ctx: "new", on_result=got.append, on_error=print)
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    release.set()
    qtbot.waitUntil(lambda: not runner.busy(), timeout=5000)
    runner.wait(5000)
    assert got == ["new"]


def test_airline_names_from_callsigns():
    from dopplergeo.flights.airlines import airline_name

    assert airline_name("EZY32NC") == "easyJet"
    assert airline_name("ryr45jl ") == "Ryanair"
    assert airline_name("TOM5TC") == "TUI Airways"
    assert airline_name("GABCD") is None      # registration-style callsign
    assert airline_name("ZZZ123") is None     # unknown designator
    assert airline_name(None) is None and airline_name("") is None


def test_aircraft_colors_are_stable_and_per_aircraft():
    assert gj.aircraft_color("40756f") == gj.aircraft_color("40756F")
    colors = {gj.aircraft_color(f"{i:06x}") for i in range(200)}
    assert len(colors) == len(gj.AIRCRAFT_COLORS)  # the hash spreads over the palette


def test_tracks_layer_carries_each_aircraft_colour():
    class Tr:
        def __init__(self, icao):
            self.icao24, self.callsign = icao, "EZY1"
            self.lon, self.lat = np.array([0.0, 0.1]), np.array([50.0, 50.1])

    feats = gj.tracks_geojson([Tr("40756f"), Tr("4d2288")])["features"]
    assert [f["properties"]["color"] for f in feats] == [gj.aircraft_color("40756f"), gj.aircraft_color("4d2288")]


def test_true_position_layer_kinds():
    tag = gj.gps_geojson((53.4, -2.1, None))["features"][0]
    assert tag["properties"]["kind"] == "gps_tag" and tag["geometry"]["coordinates"] == [-2.1, 53.4]
    ref = gj.gps_geojson((53.44054, -2.099867), "reference", "recordist geotag")["features"][0]
    assert ref["properties"] == {"kind": "reference", "note": "recordist geotag"}
    assert gj.gps_geojson(None)["features"] == []


# --- map distance tool -----------------------------------------------------------

def test_format_distance_thresholds():
    from dopplergeo.gui.map.bridge import format_distance

    assert format_distance(0) == "0 m"
    assert format_distance(842.3) == "842 m"
    assert format_distance(999) == "999 m"
    assert format_distance(999.4) == "999 m"
    assert format_distance(999.6) == "1.00 km"     # would round to "1000 m"
    assert format_distance(1000) == "1.00 km"
    assert format_distance(1270) == "1.27 km"
    assert format_distance(9994) == "9.99 km"
    assert format_distance(9999) == "10.0 km"      # never "10.00 km"
    assert format_distance(12_412) == "12.4 km"
    assert format_distance(float("nan")) == "" and format_distance(-1) == ""


@pytest.fixture
def map_view(qtbot):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from dopplergeo.gui.map.bridge import MapView

    v = MapView(style_url=None)  # plain offline background: no network
    v.resize(800, 600)
    errors: list[str] = []
    v.bridge.js_error.connect(errors.append)
    v.show()  # a hidden QWebEngineView never runs its page
    qtbot.waitUntil(lambda: v.bridge.is_ready, timeout=30_000)
    if not run_js(qtbot, v, "dg.webgl()"):
        v.close(); v.deleteLater(); qtbot.wait(200)
        pytest.skip("no WebGL in this environment")
    qtbot.waitUntil(lambda: run_js(qtbot, v, "dg.ready()") is True, timeout=30_000)
    yield v
    assert errors == []
    # Destroy the page while the QApplication still runs (else a segfault at exit).
    v.close()
    v.deleteLater()
    qtbot.wait(200)


def run_js(qtbot, view, script):
    out = []
    view.eval_js(script, out.append)
    qtbot.waitUntil(lambda: bool(out), timeout=10_000)
    return out[0]


def test_map_distance_formatting_matches_python(qtbot, map_view):
    from dopplergeo.gui.map.bridge import format_distance

    values = [0, 1, 842.3, 998.5, 999, 999.4, 999.6, 1000, 1005, 1270, 9994, 9995.1, 9999, 10_000, 12_412,
              123_456.7]
    got = json.loads(run_js(qtbot, map_view, f"JSON.stringify({json.dumps(values)}.map(dg.fmtDistance))"))
    assert got == [format_distance(v) for v in values]
    assert got[values.index(999)] == "999 m" and got[values.index(1000)] == "1.00 km"
    # Same geodesic as core.geo.haversine_m (same formula and radius).
    pairs = [(53.44054, -2.099867, 53.47, -2.24), (51.47, -0.45, 40.64, -73.78), (0, 0, 0, 179.9)]
    for a, b, c, d in pairs:
        js_m = run_js(qtbot, map_view, f"dg.geodesicM({a}, {b}, {c}, {d})")
        assert js_m == pytest.approx(float(haversine_m(a, b, c, d)), rel=1e-9)


def test_map_measure_tool(qtbot, map_view):
    from dopplergeo.gui.map.bridge import format_distance

    v, br = map_view, map_view.bridge

    def labels_now():
        return json.loads(run_js(qtbot, v, "JSON.stringify(dg.labels())"))
    clicks, toggles, measured = [], [], []
    br.point_clicked.connect(lambda lat, lon: clicks.append((lat, lon)))
    br.measure_toggled.connect(toggles.append)
    br.distance_measured.connect(measured.append)

    # Without the tool a map click goes to Python.
    run_js(qtbot, v, "dg.clickAt(53.44, -2.10); true")
    qtbot.waitUntil(lambda: bool(clicks), timeout=5000)
    clicks.clear()

    br.setMeasureMode(True)
    qtbot.waitUntil(lambda: toggles == [True], timeout=5000)
    assert run_js(qtbot, v, "dg.measuring()") is True
    assert run_js(qtbot, v, "dg.cursor()") == "crosshair"

    a, b, c = (53.44054, -2.099867), (53.4520, -2.0800), (53.4700, -2.2400)
    run_js(qtbot, v, "dg.jumpTo(53.455, -2.15, 11)")  # points several pixels apart
    run_js(qtbot, v, f"dg.clickAt({a[0]}, {a[1]}); dg.clickAt({b[0]}, {b[1]}); true")
    ab = float(haversine_m(*a, *b))
    total = run_js(qtbot, v, "dg.measureTotal()")
    assert total == pytest.approx(ab, rel=0.005)
    labels = labels_now()
    assert format_distance(ab) in labels, labels
    # Rubber band to the mouse shows the would-be total.
    run_js(qtbot, v, f"dg.hoverAt({c[0]}, {c[1]}); true")
    bc = float(haversine_m(*b, *c))
    live = run_js(qtbot, v, "document.getElementById('measureLive').textContent")
    assert live == f"+{format_distance(bc)} = {format_distance(ab + bc)}"
    # Third point: per-segment labels plus the total.
    run_js(qtbot, v, f"dg.clickAt({c[0]}, {c[1]}); true")
    labels = labels_now()
    for want in (format_distance(ab), format_distance(bc), format_distance(ab + bc)):
        assert want in labels, labels
    assert run_js(qtbot, v, "document.getElementById('measureTotal').textContent") == format_distance(ab + bc)
    assert clicks == []  # the tool swallowed every click

    # Backspace removes the last point; Enter finishes and reports the total to Python.
    run_js(qtbot, v, "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Backspace'})); true")
    assert run_js(qtbot, v, "dg.measurePoints()") == 2
    run_js(qtbot, v, "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'})); true")
    qtbot.waitUntil(lambda: bool(measured), timeout=5000)
    assert measured[0] == pytest.approx(ab, rel=0.005)
    assert labels_now() == [format_distance(ab)]  # the finished line stays

    # Escape clears; toggling off (here from the map button) clears and tells Python.
    run_js(qtbot, v, "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'})); true")
    assert run_js(qtbot, v, "dg.measurePoints()") == 0
    assert labels_now() == []
    run_js(qtbot, v, f"dg.clickAt({a[0]}, {a[1]}); document.getElementById('measureBtn').click(); true")
    qtbot.waitUntil(lambda: toggles == [True, False], timeout=5000)
    assert run_js(qtbot, v, "dg.measurePoints()") == 0
    assert labels_now() == []
    assert run_js(qtbot, v, "dg.cursor()") == ""
    run_js(qtbot, v, "dg.clickAt(53.44, -2.10); true")
    qtbot.waitUntil(lambda: bool(clicks), timeout=5000)  # normal clicks are back


def test_view_menu_measure_action_and_log(qtbot, tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    # A modal message box never returns headless: answer any with Discard / Ok.
    B = QMessageBox.StandardButton
    for name, ans in (("question", B.Discard), ("warning", B.Ok), ("information", B.Ok), ("critical", B.Ok)):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, _ans=ans, **k: _ans))

    from dopplergeo.gui.main_window import MainWindow

    w = MainWindow(style_url=None, cache_dir=tmp_path,
                   settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    try:
        w.show()
        qtbot.waitUntil(lambda: w.map.bridge.is_ready, timeout=30_000)
        w.a_measure.setChecked(True)
        qtbot.waitUntil(lambda: run_js(qtbot, w.map, "dg.measuring()") is True, timeout=5000)
        run_js(qtbot, w.map, "document.getElementById('measureBtn').click(); true")  # off from the map
        qtbot.waitUntil(lambda: not w.a_measure.isChecked(), timeout=5000)
        w.map.bridge.measured(1270.0)  # what JS calls when a line is finished
        assert "measured distance 1.27 km (1270 m)" in w.log_view.toPlainText()
    finally:
        w.dirty = False
        w.jobs.shutdown(30_000)
        w.close()
        w.deleteLater()
        qtbot.wait(200)
