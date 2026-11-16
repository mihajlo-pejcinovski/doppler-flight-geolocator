"""The main window driven end to end (headless): Load > Track > Flights >
Locate on the shared two-aircraft scenario, the map receiving every layer,
candidate exclusion, track deletion, and project save/reopen."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt  # noqa: E402

from dopplergeo.core.geo import haversine_m  # noqa: E402
from scenario import H_OBS, LAT0, LON0, OBS_LAT, OBS_LON  # noqa: E402

SLOW = 180_000  # ms: decode + ridge tracking + solve on a 150 s clip


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal QMessageBox never returns in a headless test: on window close the
    'Save the current project first?' question blocked teardown forever (results of
    the last jobs arrive during shutdown and mark the window dirty again). Answer
    every question with Discard / No and record what was asked."""
    from PySide6.QtWidgets import QMessageBox

    asked: list[tuple[str, str]] = []

    def answer(kind, default):
        def f(parent, title, text, *a, **k):
            asked.append((kind, title))
            return default
        return f

    B = QMessageBox.StandardButton
    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer("question", B.Discard)))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(answer("warning", B.Ok)))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(answer("information", B.Ok)))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(answer("critical", B.Ok)))
    return asked


def dispose(qtbot, w) -> None:
    """Stop the window's jobs and destroy it while the QApplication still runs.
    qtbot.addWidget only closes and deleteLater()s, and nothing processes that
    after the last test: the windows' QtWebEngine pages then die at interpreter
    exit, after the QApplication, which segfaulted the run."""
    w.dirty = False
    w.jobs.shutdown(30_000)
    w.close()
    w.deleteLater()
    qtbot.wait(200)  # runs the deferred delete


@pytest.fixture
def window(qtbot, recording, tmp_path):
    from PySide6.QtCore import QSettings

    from dopplergeo.gui.main_window import MainWindow

    _, cache_dir, _ = recording
    w = MainWindow(style_url=None, cache_dir=cache_dir,
                   settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    w.show()  # a hidden QWebEngineView never runs its page
    w.props.p_opensky.setChecked(False)
    yield w
    dispose(qtbot, w)


def js(qtbot, w, script):
    out = []
    w.map.eval_js(script, out.append)
    qtbot.waitUntil(lambda: bool(out), timeout=10_000)
    return out[0]


def located(qtbot, w, media):
    w.open_media(media)
    qtbot.waitUntil(lambda: w.analysis is not None, timeout=SLOW)
    w.props.radius.setValue(15)
    w.props.set_near(LAT0, LON0)
    w.props.use_height.setChecked(True)
    w.props.height.setValue(H_OBS)
    assert w.step_actions["Flights"].isEnabled()
    w.run_flights()
    qtbot.waitUntil(lambda: w.flights is not None, timeout=SLOW)
    assert w.step_actions["Locate"].isEnabled()
    w.run_locate()
    qtbot.waitUntil(lambda: w.result is not None, timeout=SLOW)
    return w.result


def test_full_workflow_locates_and_draws_everything(qtbot, window, recording):
    w = window
    media, _, _ = recording
    r = located(qtbot, w, media)
    sol = r.solution
    assert sol is not None, r.error
    assert haversine_m(sol.lat, sol.lon, OBS_LAT, OBS_LON) < 50.0

    # Fit table, flights table, predicted curves on the spectrogram.
    assert w.fits.table.rowCount() == len(w.analysis.families) >= 2
    assert w.flights_panel.table.rowCount() >= 2
    assert len(w.spectro._pred_items) >= 2
    assert "Position" in w.flights_panel.summary.text()
    # Every map layer got its data (queued until the page was ready, then replayed).
    last = w.map.bridge._last
    assert len(json.loads(last["candidatesChanged"])["features"]) >= 2
    kinds = {f["properties"]["kind"] for f in json.loads(last["solutionChanged"])["features"]}
    assert {"solution", "ellipse_95"} <= kinds
    assert json.loads(last["heatmapChanged"])["features"]
    assert json.loads(last["gpsChanged"])["features"]
    assert json.loads(last["regionChanged"]) is not None
    # The playhead moves the aircraft icons.
    w.set_cursor_time(75.0)
    qtbot.waitUntil(lambda: "aircraftChanged" in last and bool(json.loads(last["aircraftChanged"])["features"]),
                    timeout=5000)
    icaos = [f["properties"]["icao24"] for f in json.loads(last["aircraftChanged"])["features"]]
    assert len(icaos) == len(set(icaos))  # one icon per aircraft, even if gated for two families
    # ... and the page itself shows them.
    qtbot.waitUntil(lambda: w.map.bridge.is_ready, timeout=30_000)
    qtbot.waitUntil(lambda: js(qtbot, w, "dg.ready()") is True, timeout=30_000)
    counts = js(qtbot, w, "JSON.stringify(['candidates','solution','heatmap','gps','aircraft'].map(dg.count))")
    assert all(n > 0 for n in json.loads(counts)), counts
    # The legend names what is on the map (solution vs mirror vs the file's GPS tag).
    legend = js(qtbot, w, "document.getElementById('legend').innerText")
    for want in ("Estimated recording position", "95% area", "GPS tag", "Aircraft tracks"):
        assert want in legend, legend

    # The Track toolbar action re-analyses (regression: triggered(checked=False) used to
    # arrive as restore_tracks and fail with "'bool' object is not iterable").
    errors = []
    monkey_err = w._error
    w._error = lambda title: (lambda msg, tb: errors.append((title, msg)))
    old = w.analysis
    w.step_actions["Track"].trigger()
    qtbot.waitUntil(lambda: (w.analysis is not None and w.analysis is not old) or bool(errors), timeout=SLOW)
    w._error = monkey_err
    assert not errors, errors
    assert len(w.analysis.families) == len(old.families)
    w.run_flights()
    qtbot.waitUntil(lambda: w.flights is not None, timeout=SLOW)
    w.run_locate()
    qtbot.waitUntil(lambda: w.result is not None, timeout=SLOW)
    sol = w.result.solution

    # Clicking a candidate overlays its predicted curve (one more dashed line).
    n_pred = len(w.spectro._pred_items)
    other = next(c for c in sol.candidates if sol.assignment.get(c.family_id) != c.icao24) \
        if any(sol.assignment.get(c.family_id) != c.icao24 for c in sol.candidates) else None
    if other is not None:
        w._candidate_selected((other.family_id, other.icao24))
        assert len(w.spectro._pred_items) > n_pred


def test_run_all_labels_and_maximised_spectrogram(qtbot, window, recording):
    """'Run all' chains Track > Flights > Locate by itself; the fitted curves get
    labels (with the aircraft once located); the spectrogram can fill the window
    and the layout comes back."""
    w = window
    media, _, _ = recording
    w.props.radius.setValue(15)
    w.props.set_near(LAT0, LON0)
    w.props.use_height.setChecked(True)
    w.props.height.setValue(H_OBS)
    stages, labels_seen = [], set()
    orig_progress = w._progress

    def spy(stage, done, total):
        orig_progress(stage, done, total)
        stages.append(stage)
        labels_seen.add(w.step_label.text().split(":")[0])
    w._progress = spy
    w.open_media(media)  # starts Track; Run all picks the chain up from there
    w.run_all()
    qtbot.waitUntil(lambda: w.result is not None and not w._auto, timeout=SLOW)
    w._progress = orig_progress
    # The status bar said which step and stage was running, not just a busy bar.
    for want in ("decoding audio", "reading the ADS-B cache", "selecting candidate aircraft",
                 "grid search", "refining position 1/"):
        assert any(st.startswith(want) for st in stages), (want, stages)
    # tracked now, or (an earlier test analysed the same clip) loaded from the analysis cache
    assert any(st.startswith("tracking tones") or "analysis from cache" in st for st in stages), stages
    assert {"Track", "Flights", "Locate"} <= labels_seen, labels_seen
    assert not w.step_label.isVisible()
    sol = w.result.solution
    assert sol is not None and haversine_m(sol.lat, sol.lon, OBS_LAT, OBS_LON) < 50.0

    labels = [it.toPlainText() for it in w.spectro._label_items]
    assert len(labels) == sum(1 for fr in w.analysis.families if fr.fit is not None) >= 2
    calls = {(c.callsign or c.icao24).strip() for c in sol.candidates if sol.assignment.get(c.family_id) == c.icao24}
    assert any(any(cs in t for cs in calls) for t in labels), (labels, calls)
    assert w.flights_panel.table.horizontalHeaderItem(3).text() == "Airline"

    # Zoomed onto a stretch of a curve away from its CPA, its label stays on screen.
    sp = w.spectro
    fid, it = next(iter(sp._label_fid.items()))
    t, f = sp._curves[fid][0]
    far = np.argmax(np.abs(t - it._home[0]))          # the curve end farthest from the CPA
    sp.plot.setXRange(t[far] - 5, t[far] + 5, padding=0)
    sp.plot.setYRange(f[far] - 15, f[far] + 15, padding=0)
    qtbot.wait(50)
    (x0, x1), (y0, y1) = sp.plot.getPlotItem().getViewBox().viewRange()
    assert it.isVisible() and x0 <= it.pos().x() <= x1 and y0 <= it.pos().y() <= y1
    # "Go to curve": one entry per curve, named by the aircraft; picking one zooms to it.
    entries = [sp.goto.itemText(i) for i in range(sp.goto.count())]
    assert any(any(cs in e for cs in calls) for e in entries), entries
    idx = next(i for i in range(sp.goto.count()) if sp.goto.itemData(i) == fid)
    sp.goto.activated.emit(idx)
    (x0, x1), _ = sp.plot.getPlotItem().getViewBox().viewRange()
    assert x0 < t.min() and x1 > t.max()

    others = [w.d_video, w.d_props, w.d_log]
    w.spectro.max_btn.setChecked(True)
    qtbot.waitUntil(lambda: not w.centralWidget().isVisible(), timeout=5000)
    assert w.d_spectro.isVisible() and not any(d.isVisible() for d in others)
    w.spectro.max_btn.setChecked(False)
    qtbot.waitUntil(lambda: w.centralWidget().isVisible(), timeout=5000)
    assert w.d_video.isVisible() and w.d_log.isVisible()


def test_track_editing_drawing_locks_and_video_mute(qtbot, window, recording):
    """Merge/split a track, set its harmonic number, draw a track by hand, lock a
    fit parameter, and mark a mute range from the video panel."""
    w = window
    media, _, _ = recording
    located(qtbot, w, media)
    res = w.analysis

    def refitted(before):
        qtbot.waitUntil(lambda: w.analysis.families is not before and not w.jobs.busy(), timeout=SLOW)

    # Split a track off into a new family, through the panel's controls.
    tr = next(t for t in res.tracks if t.family_id is not None)
    n_fam = len({t.family_id for t in res.tracks if t.family_id is not None})
    w.spectro.select_track(tr.id)
    w.spectro.fam_box.setCurrentIndex(w.spectro.fam_box.count() - 1)   # "new family"
    w.spectro.k_box.setValue(2.0)
    before = res.families
    w.spectro.apply_btn.click()
    refitted(before)
    assert tr.k == 2.0 and len({t.family_id for t in w.analysis.tracks if t.family_id is not None}) == n_fam + 1

    # Draw a track by hand along another one's points: a new, manual family.
    other = max((t for t in res.tracks if t.id != tr.id), key=lambda t: t.t.size)
    w.spectro.draw_btn.setChecked(True)
    for i in (0, other.t.size // 2, other.t.size - 1):
        w.spectro.add_draw_point(other.t[i], other.f[i])
    before = w.analysis.families
    w.spectro.finish_btn.click()
    refitted(before)
    drawn = [t for t in w.analysis.tracks if t.manual]
    assert len(drawn) == 1 and drawn[0].t.size > 10 and not w.spectro.draw_btn.isChecked()

    # Lock v of a fitted family: the re-fit holds it.
    fr = next(x for x in w.analysis.families if x.fit is not None and not any(t.manual for t in x.family.tracks))
    row = next(r for r in range(w.fits.table.rowCount())
               if w.fits.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == fr.family.id)
    w.fits.table.selectRow(row)
    w.fits.lock_v_val.setValue(95.0)
    w.fits.lock_v.setChecked(True)
    assert w.project.fit_locks[fr.family.id] == {"v": 95.0}
    before = w.analysis.families
    w.fits.refit_btn.click()
    refitted(before)
    got = next(x for x in w.analysis.families if x.family.id == fr.family.id)
    assert got.fit.v == pytest.approx(95.0, rel=1e-5)

    # Mute range from the video panel (playhead positions faked).
    pos = iter([20.0, 31.5])
    w.video.position_s = lambda: next(pos)
    w.video.mute_btn.click()
    assert "End mute" in w.video.mute_btn.text()
    w.video.mute_btn.click()
    assert (20.0, 31.5) in [tuple(m) for m in w.project.mutes]


def test_settings_dialog_uses_the_keychain(qtbot, tmp_path, monkeypatch):
    """Keys go to the (here: fake) keychain and the providers' environment variables;
    the real keychain is never touched by the tests."""
    from PySide6.QtCore import QSettings

    from dopplergeo.gui import settings as S

    store = {}

    class FakeKeyring:
        def get_keyring(self):
            return "fake"

        def get_password(self, svc, name):
            return store.get((svc, name))

        def set_password(self, svc, name, value):
            store[(svc, name)] = value

        def delete_password(self, svc, name):
            store.pop((svc, name), None)
    monkeypatch.setattr(S, "_keyring", lambda: FakeKeyring())
    env = {}
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    dlg = S.SettingsDialog(qs, "https://tiles.example/style.json")
    qtbot.addWidget(dlg)
    dlg.adsblol_key.setText("sk-lol-test")
    dlg.os_id.setText("me")
    dlg.cache.setText(str(tmp_path / "flights"))
    changed = dlg.save(environ=env)
    assert store[("dopplergeo", "adsblol_api_key")] == "sk-lol-test"
    assert env["ADSBLOL_API_KEY"] == "sk-lol-test" and env["OPENSKY_CLIENT_ID"] == "me"
    assert "OPENSKY_CLIENT_SECRET" not in env and changed["cache"]
    assert qs.value("cache/dir") == str(tmp_path / "flights")
    # At start-up the keychain fills unset variables; one the user set wins.
    env2 = {"OPENSKY_CLIENT_ID": "from-shell"}
    assert S.apply_saved_credentials(env2) == ["ADSBLOL_API_KEY"]
    assert env2 == {"OPENSKY_CLIENT_ID": "from-shell", "ADSBLOL_API_KEY": "sk-lol-test"}


def test_exclusion_track_deletion_and_project_roundtrip(qtbot, window, recording, tmp_path):
    from PySide6.QtCore import QSettings

    from dopplergeo.gui.main_window import MainWindow

    w = window
    media, cache_dir, _ = recording
    sol = located(qtbot, w, media).solution
    assert sol is not None

    # Tick one aircraft out and locate again: it is listed but never assigned.
    out = sorted(set(sol.assignment.values()))[0]
    w._inclusion_changed(out, False)
    w.run_locate()
    qtbot.waitUntil(lambda: w.result is not None and out not in w.result.solution.assignment.values()
                    if w.result and w.result.solution else False, timeout=SLOW)
    assert any(c.icao24 == out and not c.included for c in w.result.solution.candidates)

    # Delete a track: the families are re-fitted and the old solution is cleared.
    n_tracks = len(w.analysis.tracks)
    tid = w.analysis.tracks[-1].id
    w.spectro.select_track(tid)
    w.spectro.delete_btn.click()
    qtbot.waitUntil(lambda: len(w.analysis.tracks) == n_tracks - 1 and w.result is None, timeout=SLOW)
    assert tid not in [t.id for t in w.analysis.tracks]

    # Save, reopen in a fresh window: inputs, edits and tracks come back.
    w.project_dir = tmp_path / "proj"
    assert w.save()
    w2 = MainWindow(style_url=None, cache_dir=cache_dir,
                    settings=QSettings(str(tmp_path / "s2.ini"), QSettings.Format.IniFormat))
    w2.show()
    try:
        w2.open_project(tmp_path / "proj")
        qtbot.waitUntil(lambda: w2.analysis is not None, timeout=SLOW)
        assert [t.id for t in w2.analysis.tracks] == [t.id for t in w.analysis.tracks]
        for a, b in zip(w.analysis.tracks, w2.analysis.tracks):
            np.testing.assert_array_equal(a.f, b.f)
        assert w2.excluded == {out}
        r = w2.props.region()
        assert r.kind == "near" and r.lat == pytest.approx(LAT0) and r.radius_km == pytest.approx(15)
        assert w2.props.height_m() == pytest.approx(H_OBS)
        assert w2.project.media.sha256 == w.project.media.sha256
        assert not w2.dirty
    finally:
        dispose(qtbot, w2)
