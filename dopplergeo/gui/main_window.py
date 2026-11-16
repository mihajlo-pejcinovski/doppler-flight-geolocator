"""Main window (spec "GUI"): five-step workflow bar Load > Track > Fit >
Flights > Locate over dockable panels. Every step runs in a worker; editing an
earlier step clears the results that depend on it, so the user re-runs from
there."""

from __future__ import annotations

import html
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QScrollArea,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolBar,
    QWidget,
)

from ..core.pipeline import AnalysisResult, analyze_file, analyze_file_cached, refit_families
from ..core.solver import SolverSettings
from ..flights import FlightCache
from ..core.geo import haversine_m
from ..flights.airlines import airline_name
from ..locate import (
    LocateResult,
    family_observations,
    fetch_flights,
    predicted_curve,
    resolve_region,
    resolve_start,
    solve_position,
    write_outputs,
)
from ..project import (
    FlightQueryModel,
    MediaSource,
    Project,
    ProjectError,
    SpectroModel,
    candidate_models,
    check_media,
    fit_models,
    load_project,
    save_project,
    sha256_file,
)
from . import geojson as gj
from .map.bridge import OPENFREEMAP_STYLE, MapView, format_distance
from .panels.fits import FitPanel
from .panels.flights import FlightsPanel
from .panels.properties import PropertiesPanel
from .panels.spectrogram import SpectrogramPanel
from .panels.video import VideoPanel
from .workers import JobContext, JobRunner

FROM_SETTINGS = object()  # MainWindow(style_url=...) default: the saved map style
STEPS = ["Load", "Track", "Fit", "Flights", "Locate"]
MEDIA_FILTER = "Media (*.mp4 *.mov *.mkv *.m4a *.wav *.flac *.mp3 *.ogg *.webm);;All files (*)"
PRIVACY_NOTE = ("Locating where someone else's video was recorded can reveal where they live or were. "
                "Only analyse recordings you made or have permission to analyse.")


class MainWindow(QMainWindow):
    def __init__(self, *, style_url: str | None | object = FROM_SETTINGS, cache_dir: str | Path | None = None,
                 settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("dopplergeo")
        self.resize(1500, 950)
        self.qsettings = settings or QSettings("dopplergeo", "dopplergeo")
        saved_cache = str(self.qsettings.value("cache/dir", "") or "")
        self.cache_dir = Path(cache_dir) if cache_dir else (Path(saved_cache) if saved_cache else None)
        self._style_arg = style_url
        self.jobs = JobRunner(parent=self)

        # session state
        self.project = Project()
        self.project_dir: Path | None = None
        self.media_path: Path | None = None
        self.analysis: AnalysisResult | None = None
        self.flights = None
        self.result: LocateResult | None = None
        self.excluded: set[str] = set()
        self.highlight: tuple[int, str] | None = None
        self.dirty = False
        self._cursor_t = 0.0
        self._auto = False          # "Run all": chain Track > Flights > Locate

        self._build_panels()
        self._build_toolbar()
        self._build_menus()
        self._wire()
        self._update_steps()
        self._cursor_timer = QTimer(self, interval=200, singleShot=True)
        self._cursor_timer.timeout.connect(self._update_map_cursor)

    # ------------------------------------------------------------------ UI
    def _build_panels(self) -> None:
        self.video = VideoPanel()
        self.map = MapView(self._style_url() if self._style_arg is FROM_SETTINGS else self._style_arg)
        self.props = PropertiesPanel()
        self.fits = FitPanel()
        self.flights_panel = FlightsPanel()
        self.spectro = SpectrogramPanel()
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(5000)

        self.setCentralWidget(self.map)
        self.setDockOptions(QMainWindow.DockOption.AllowTabbedDocks | QMainWindow.DockOption.AnimatedDocks)

        def dock(title: str, w: QWidget, area, name: str) -> QDockWidget:
            d = QDockWidget(title, self)
            d.setObjectName(name)
            d.setWidget(w)
            self.addDockWidget(area, d)
            return d

        L, B = Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea
        self.d_video = dock("Video", self.video, L, "video")
        # Scrollable: laid out at full size the properties form needs ~590 px, which
        # stacked with the video and log made the window's minimum height 1035 px, too
        # tall for a 1080p screen below GNOME's top bar, so it could not maximise.
        props_scroll = QScrollArea()
        props_scroll.setWidget(self.props)
        props_scroll.setWidgetResizable(True)
        props_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.d_props = dock("Properties", props_scroll, L, "properties")
        self.d_fits = dock("Fit", self.fits, L, "fit")
        self.d_flights = dock("Flights", self.flights_panel, L, "flights")
        self.tabifyDockWidget(self.d_props, self.d_fits)
        self.tabifyDockWidget(self.d_fits, self.d_flights)
        self.d_props.raise_()
        self.d_spectro = dock("Spectrogram", self.spectro, B, "spectrogram")
        self.d_log = dock("Log", self.log_view, B, "log")
        self.resizeDocks([self.d_video, self.d_props], [520, 520], Qt.Orientation.Horizontal)
        self.resizeDocks([self.d_spectro, self.d_log], [380, 90], Qt.Orientation.Vertical)

        self.progress = QProgressBar(maximumWidth=220, visible=False)
        self.progress.setTextVisible(True)
        self.cancel_btn = QPushButton("Cancel", visible=False)
        self.cancel_btn.clicked.connect(lambda: self.jobs.cancel())
        self.status_utc = QLabel("")
        # What is running: "Track: tracking window 3/10 · 1:42" (step, stage, elapsed).
        self.step_label = QLabel("", visible=False)
        self._job_name, self._job_t0, self._stage = "", 0.0, ""
        self._elapsed_timer = QTimer(self, interval=1000)
        self._elapsed_timer.timeout.connect(self._refresh_step_label)
        self.statusBar().addPermanentWidget(self.status_utc)
        self.statusBar().addPermanentWidget(self.step_label)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

    def _style_url(self) -> str:
        return str(self.qsettings.value("map/style_url", OPENFREEMAP_STYLE) or "")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Workflow")
        tb.setObjectName("workflow")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.step_actions: dict[str, QAction] = {}
        tips = {
            "Load": "Open a video or audio file",
            "Track": "Find tone tracks and fit their Doppler curves (honours mute ranges)",
            "Fit": "Re-fit the families from the tracks as edited",
            "Flights": "Fetch ADS-B tracks for the recording window and search region",
            "Locate": "Solve for the recording position",
        }
        for i, name in enumerate(STEPS):
            if i:
                sep = QLabel(" › ")
                tb.addWidget(sep)
            a = QAction(name, self)
            a.setToolTip(tips[name])
            tb.addAction(a)
            self.step_actions[name] = a
        tb.addSeparator()
        self.a_run_all = QAction("▶ Run all", self)
        self.a_run_all.setToolTip("Track, fetch flights and locate in one go (opens a file first if none is loaded)")
        self.a_run_all.setShortcut("Ctrl+R")
        tb.addAction(self.a_run_all)
        tb.addSeparator()
        self.project_label = QLabel("")
        tb.addWidget(self.project_label)

    def _build_menus(self) -> None:
        m = self.menuBar().addMenu("&File")
        self.a_open_media = m.addAction("Open media…", self.open_media_dialog, QKeySequence.StandardKey.Open)
        self.a_open_proj = m.addAction("Open project…", self.open_project_dialog)
        self.a_save = m.addAction("Save project", self.save, QKeySequence.StandardKey.Save)
        self.a_save_as = m.addAction("Save project as…", self.save_as_dialog, QKeySequence.StandardKey.SaveAs)
        m.addSeparator()
        m.addAction("Settings…", self.settings_dialog, QKeySequence.StandardKey.Preferences)
        m.addSeparator()
        self.a_export = m.addAction("Export results…", self.export_dialog)
        self.a_export_png = m.addAction("Export map PNG…", self.export_png_dialog)
        m.addSeparator()
        m.addAction("Quit", self.close, QKeySequence.StandardKey.Quit)
        v = self.menuBar().addMenu("&View")
        for d in (self.d_video, self.d_props, self.d_fits, self.d_flights, self.d_spectro, self.d_log):
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        v.addAction("Zoom map to data", self.map.bridge.zoomToData)
        self.a_measure = v.addAction("Measure distance on map")
        self.a_measure.setCheckable(True)
        self.a_measure.toggled.connect(self.map.bridge.setMeasureMode)
        h = self.menuBar().addMenu("&Help")
        h.addAction("Privacy", lambda: QMessageBox.information(self, "Privacy", PRIVACY_NOTE))

    def _wire(self) -> None:
        a = self.step_actions
        a["Load"].triggered.connect(self.open_media_dialog)
        # Not connect(self.run_track): Qt would pass triggered(checked=False) as restore_tracks.
        a["Track"].triggered.connect(lambda: self.run_track())
        a["Fit"].triggered.connect(self.run_fit)
        a["Flights"].triggered.connect(self.run_flights)
        a["Locate"].triggered.connect(self.run_locate)
        self.a_run_all.triggered.connect(lambda: self.run_all())
        self.spectro.maximize_toggled.connect(self._spectro_maximized)
        self.video.position_changed.connect(self.set_cursor_time)
        self.spectro.seek_requested.connect(self.video.seek)
        self.video.mute_marked.connect(self._video_mute)
        self.spectro.seek_requested.connect(self.set_cursor_time)
        self.spectro.on_delete(self.delete_track)
        self.spectro.track_edited.connect(self.edit_track)
        self.spectro.track_drawn.connect(self.add_drawn_track)
        self.spectro.mutes_changed.connect(self._mutes_changed)
        self.fits.refit_requested.connect(self.run_fit)
        self.fits.locks_changed.connect(self._locks_changed)
        self.flights_panel.inclusion_changed.connect(self._inclusion_changed)
        self.flights_panel.candidate_selected.connect(self._candidate_selected)
        self.props.changed.connect(self._mark_dirty)
        self.props.region_changed.connect(self._region_changed)
        br = self.map.bridge
        br.region_drawn.connect(lambda s, w, n, e: self.props.set_bbox((s, w, n, e)))
        br.point_clicked.connect(self._map_clicked)
        br.aircraft_clicked.connect(self.flights_panel.select_aircraft)
        br.js_error.connect(lambda m: self.log(f"map: {m}"))
        br.distance_measured.connect(lambda m: self.log(f"map: measured distance {format_distance(m)} ({m:.0f} m)"))
        br.measure_toggled.connect(self._measure_toggled)
        self.jobs.started.connect(self._jobs_changed)
        self.jobs.stopped.connect(self._jobs_changed)

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp}  {msg}")

    def _error(self, title: str):
        def show(message: str, tb: str) -> None:
            if self._auto:
                self._auto = False
                self.log("Run all stopped")
            self.log(f"{title} failed: {message}")
            if tb:
                self.log_view.appendPlainText(tb.rstrip())
            QMessageBox.warning(self, title, html.escape(message))
        return show

    def _progress(self, stage: str, done: int, total: int) -> None:
        """A job's stage: shown next to the bar; a count makes the bar determinate
        (byte counts of downloads are shown in MB)."""
        self.progress.setVisible(True)
        if total:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(100 * min(done, total) / total))
            self.progress.setFormat("%p%")
            if stage.startswith("download"):  # adsb.lol archive parts: byte counts
                stage = f"{stage}  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB"
        else:
            self.progress.setRange(0, 0)  # busy: Qt draws no text in this mode, the label says what
        self._stage = stage
        self._refresh_step_label()

    def _refresh_step_label(self) -> None:
        if not self._job_name:
            return
        m, sec = divmod(int(time.monotonic() - self._job_t0), 60)
        stage = f": {self._stage}" if self._stage else ": starting"
        self.step_label.setText(f"{self._job_name}{stage} · {m}:{sec:02d}")

    def _jobs_changed(self, *_):
        busy = self.jobs.busy()
        self.cancel_btn.setVisible(busy)
        self.progress.setVisible(busy)
        self.step_label.setVisible(busy)
        if busy:
            name = next(iter(self.jobs.active), "")
            if name.capitalize() != self._job_name:
                self._job_name, self._job_t0, self._stage = name.capitalize(), time.monotonic(), ""
                self.progress.setRange(0, 0)
            self._refresh_step_label()
            self._elapsed_timer.start()
        else:
            self._job_name, self._stage = "", ""
            self._elapsed_timer.stop()
        self._update_steps()

    def _ask(self, question: str) -> bool:
        return QMessageBox.question(self, "dopplergeo", question) == QMessageBox.StandardButton.Yes

    def _cache(self) -> FlightCache:
        return FlightCache(self.cache_dir)

    def _mark_dirty(self, *_):
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.project_dir.name if self.project_dir else (self.media_path.name if self.media_path else "")
        self.project_label.setText(f"  {name}{' *' if self.dirty else ''}")
        self.setWindowTitle(f"dopplergeo - {name}{' *' if self.dirty else ''}" if name else "dopplergeo")

    def _update_steps(self) -> None:
        busy = self.jobs.busy()
        have_media = self.media_path is not None
        have_fit = self.analysis is not None and bool(family_observations(self.analysis))
        a = self.step_actions
        a["Load"].setEnabled(not busy)
        a["Track"].setEnabled(have_media and not busy)
        a["Fit"].setEnabled(self.analysis is not None and not busy)
        a["Flights"].setEnabled(have_fit and not busy)
        a["Locate"].setEnabled(have_fit and self.flights is not None and not busy)
        self.a_run_all.setEnabled(not busy)
        for x in (self.a_save, self.a_save_as):
            x.setEnabled(have_media)
        self.a_export.setEnabled(self.result is not None)

    def epoch(self) -> float | None:
        if self.analysis is None:
            return None
        try:
            return resolve_start(self.analysis, self.props.start_override())[0]
        except Exception:
            return None

    # ------------------------------------------------------------------ cursor
    def set_cursor_time(self, t: float) -> None:
        self._cursor_t = float(t)
        self.spectro.set_cursor_time(t)
        ep = self.epoch()
        if ep is not None:
            utc = datetime.fromtimestamp(ep + t, tz=timezone.utc)
            self.status_utc.setText(f"UTC {utc:%Y-%m-%d %H:%M:%S}")
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _update_map_cursor(self) -> None:
        ep = self.epoch()
        if ep is None or self.flights is None:
            return
        dt = self.result.solution.dt_clock if self.result and self.result.solution else 0.0
        if self.result and self.result.solution:
            colors = {c.icao24: gj.aircraft_color(c.icao24) for c in self.result.solution.candidates}
            for fid, icao in self.result.solution.assignment.items():
                colors[icao] = gj.family_color(fid)
            tracks = list({c.icao24: c.traj.track for c in self.result.solution.candidates}.values())
        else:
            colors, tracks = None, self._flight_tracks()
        self.map.bridge.setAircraft(gj.aircraft_geojson(tracks, ep + self._cursor_t + dt, colors))

    def _flight_tracks(self):
        if self.flights is None:
            return []
        if not hasattr(self, "_tracks_cache") or self._tracks_cache[0] is not self.flights:
            self._tracks_cache = (self.flights, self.flights.tracks())
        return self._tracks_cache[1]

    # ------------------------------------------------------------------ Load / Track
    # ------------------------------------------------------------------ Run all
    def run_all(self) -> None:
        """Track > Flights > Locate in one go, each step started when the previous
        one has delivered (an error or a step that cannot run stops the chain).
        Reuses an analysis and flights that are already there."""
        if self.media_path is None:
            self.open_media_dialog()  # open_media starts Track
            if self.media_path is None:
                return
            self._auto = True
            self.log("Run all: Track > Flights > Locate")
            return
        self._auto = True
        self.log("Run all: Track > Flights > Locate")
        if self.jobs.busy("track"):
            return  # the running Track continues the chain when it finishes
        if self.analysis is None:
            self.run_track()
        elif self.flights is None:
            self._auto_next("Flights", self.run_flights)
        else:
            self._auto_next("Locate", self.run_locate)

    def _auto_next(self, name: str, run) -> None:
        # Deferred, and retried while any job is still busy: the finishing job's
        # 'done' signal can arrive after its result.
        def go() -> None:
            if not self._auto:
                return
            if self.jobs.busy():
                QTimer.singleShot(50, go)
                return
            self._update_steps()
            if not self.step_actions[name].isEnabled():
                self._auto = False
                self.log(f"Run all stopped: {name} cannot run yet (see the steps above)")
                return
            run()
        QTimer.singleShot(0, go)

    def open_media_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open media", str(self.qsettings.value("dirs/media", "")),
                                              MEDIA_FILTER)
        if path:
            self.qsettings.setValue("dirs/media", str(Path(path).parent))
            self.open_media(path)

    def open_media(self, path: str | Path) -> None:
        if not self._confirm_discard():
            return
        self._reset()
        self.media_path = Path(path)
        self.project = Project(name=self.media_path.stem)
        self.video.load(str(self.media_path))
        self.log(f"opened {self.media_path}")
        self._update_title()
        self.run_track()

    def _reset(self) -> None:
        self.jobs.cancel()
        self.analysis = self.flights = self.result = None
        self.excluded = set()
        self.highlight = None
        self.project_dir = None
        self.dirty = False
        self.spectro.set_spectrogram(None)
        self.spectro.set_tracks([])
        self.spectro.set_fit_curves({})
        self.spectro.set_predicted_curves([])
        self.spectro.set_mutes([])
        self.fits.set_results([], None)
        self.flights_panel.set_candidates(None, set())
        self.flights_panel.set_summary("No flight data yet.")
        self.map.bridge.clear()

    def run_track(self, restore_tracks=None) -> None:
        if self.media_path is None:
            return
        path = self.media_path
        mutes = self.spectro.mutes()
        spectro = self.project.spectro.to_settings()
        locks = {int(k): dict(v) for k, v in self.project.fit_locks.items()}
        from ..flights.cache import default_cache_dir
        analysis_dir = Path(self.cache_dir or default_cache_dir()).parent / "analysis"

        def job(ctx: JobContext):
            ctx.log(f"decoding and tracking {path.name}")
            ctx.progress("checksum", 0, 0)
            sha = sha256_file(path)
            ctx.check()
            # Tracks and fits are cached by file content + settings + code: reopening the
            # same file skips the tracking (minutes for a long recording).
            res, _cached = analyze_file_cached(path, analysis_dir, sha, mutes=mutes, spectro=spectro,
                                               log=ctx.log, progress=ctx.progress)
            ctx.check()
            if restore_tracks is not None:
                res.tracks = restore_tracks
                res.families = refit_families(restore_tracks, locks=locks)
            return res, sha, restore_tracks is not None

        self.jobs.start("track", job, on_result=self._track_done, on_error=self._error("Track"),
                        on_log=self.log, on_progress=self._progress)

    def _track_done(self, out) -> None:
        res, sha, restoring = out
        self.analysis = res
        clip = res.clip
        self.project.media = MediaSource(path=str(self.media_path), sha256=sha, duration_s=clip.duration,
                                         sample_rate=clip.native_sample_rate, start_utc=clip.start_utc,
                                         start_utc_source=clip.start_utc_source, gps_tag=clip.gps)
        self.props.set_media(clip, f"sha256 {sha[:10]}")
        if self.project.start_utc_override is not None:
            self.props.set_start_override(self.project.start_utc_override)
        self.spectro.set_spectrogram(res.spectrogram)
        self._show_true_position()
        self._show_analysis()
        # A new analysis invalidates the solution (flights stay valid for the same window).
        self._clear_solution()
        n_good = len(res.reliable)
        self.log(f"{len(res.tracks)} tone tracks, {len(res.families)} families, {n_good} graded A/B")
        if not self.spectro.max_btn.isChecked():
            self.d_fits.raise_()
        if self._auto:
            self._auto_next("Flights", self.run_flights)
        if restoring:
            # Re-analysing an opened project is not an edit. It finishes after
            # open_project returned, so it must leave the project clean itself
            # (and set_media above may have marked it dirty through props.changed).
            self.dirty = False
            self._update_title()
        else:
            self._mark_dirty()
        self._update_steps()

    def _show_analysis(self) -> None:
        res = self.analysis
        if res is None:
            return
        self.spectro.set_tracks(res.tracks)
        curves = {}
        for fr in res.families:
            if fr.fit is None:
                continue
            t0, t1 = fr.family.t_range
            t = np.linspace(t0, t1, 400)
            curves[fr.family.id] = [(t, fr.fit.predict(t, k)) for k in sorted(set(fr.family.ks))]
        self.spectro.set_fit_curves(curves)
        self._update_curve_labels()
        self.fits.set_results(res.families, self.epoch())
        self.fits.set_locks(self.project.fit_locks)

    def _show_true_position(self) -> None:
        """The file's GPS tag on the map, else the project's known true position."""
        clip = self.analysis.clip if self.analysis else None
        ref = self.project.reference_position
        if clip is not None and clip.gps is not None:
            self.map.bridge.setGps(gj.gps_geojson(clip.gps))
        elif ref is not None:
            self.map.bridge.setGps(gj.gps_geojson(ref, "reference", self.project.reference_note))
        else:
            self.map.bridge.setGps(gj.fc([]))

    def _update_curve_labels(self) -> None:
        """Label each fitted curve at its CPA on the lowest harmonic: family, grade,
        speed and distance, and the assigned aircraft once located."""
        res = self.analysis
        if res is None:
            self.spectro.set_curve_labels({})
            return
        sol = self.result.solution if self.result else None
        names = {}
        if sol is not None:
            calls = {c.icao24: (c.callsign or "").strip() for c in sol.candidates}
            for fid, icao in sol.assignment.items():
                cs = calls.get(icao) or icao
                airline = airline_name(cs)
                names[fid] = f"{cs} ({airline})" if airline else cs
        labels = {}
        for fr in res.families:
            f = fr.fit
            if f is None:
                continue
            t0, t1 = fr.family.t_range
            t = float(np.clip(f.t_cpa, t0, t1))
            k = min(fr.family.ks)
            text = f"#{fr.family.id} {f.grade}  {f.v:.0f} m/s  {f.d:.0f} m"
            if fr.family.id in names:
                text = f"{names[fr.family.id]}  " + text
            labels[fr.family.id] = (t, float(f.predict([t], k)[0]), text)
        self.spectro.set_curve_labels(labels)

    def _spectro_maximized(self, on: bool) -> None:
        """Let the spectrogram fill the window: hide the map and every other dock,
        then put the exact previous layout back on restore."""
        docks = [self.d_video, self.d_props, self.d_fits, self.d_flights, self.d_log]
        if on:
            self._layout_before_max = self.saveState()
            for d in docks:
                d.hide()
            self.centralWidget().hide()
            self.d_spectro.show()
            self.d_spectro.raise_()
        else:
            self.centralWidget().show()
            state = getattr(self, "_layout_before_max", None)
            if state is not None:
                self.restoreState(state)
            else:
                for d in docks:
                    d.show()

    # ------------------------------------------------------------------ Fit / edits
    def edit_track(self, tid: int, family_id, k: float) -> None:
        """Move a track to another family (merge) or a new one (None: split off),
        and/or set its harmonic number; then re-fit."""
        if self.analysis is None:
            return
        tr = next((t for t in self.analysis.tracks if t.id == tid), None)
        if tr is None:
            return
        if family_id is None:
            family_id = 1 + max((t.family_id for t in self.analysis.tracks if t.family_id is not None), default=-1)
        tr.family_id, tr.k = int(family_id), float(k)
        self.log(f"track {tid}: family #{tr.family_id}, k = {tr.k:g}")
        self.run_fit()

    def add_drawn_track(self, points) -> None:
        """A hand-drawn track (manual=True, never cut by the fit's trimming) as a new family."""
        if self.analysis is None or len(points) < 2:
            return
        from ..core.tracking import ToneTrack
        a = np.asarray(points, dtype=float)
        spec = self.analysis.spectrogram
        res_hz = spec.bin_hz if spec is not None else 0.5
        frames = np.searchsorted(spec.times, a[:, 0]) if spec is not None else np.arange(len(a))
        tid = 1 + max((t.id for t in self.analysis.tracks), default=-1)
        fid = 1 + max((t.family_id for t in self.analysis.tracks if t.family_id is not None), default=-1)
        sigma = np.maximum(0.5 * res_hz, 0.002 * a[:, 1])
        tr = ToneTrack(tid, frames.astype(np.intp), a[:, 0], a[:, 1], np.zeros(len(a)), sigma, manual=True,
                       family_id=fid, k=1.0)
        self.analysis.tracks.append(tr)
        self.log(f"drew track {tid} ({a[0, 0]:.0f}-{a[-1, 0]:.0f} s, {len(a)} points) as family #{fid}")
        self.run_fit()

    def delete_track(self, tid: int) -> None:
        if self.analysis is None:
            return
        self.analysis.tracks = [t for t in self.analysis.tracks if t.id != tid]
        self.log(f"deleted track {tid}")
        self.run_fit()

    def run_fit(self) -> None:
        if self.analysis is None:
            return
        tracks = list(self.analysis.tracks)
        locks = {int(k): dict(v) for k, v in self.project.fit_locks.items()}

        def job(ctx: JobContext):
            return refit_families(tracks, locks=locks)

        def done(results):
            self.analysis.families = results
            self._show_analysis()
            self._clear_solution()
            self._mark_dirty()
            self.log(f"re-fitted {len(results)} families")
            self._update_steps()

        self.jobs.start("fit", job, on_result=done, on_error=self._error("Fit"), on_log=self.log)

    def _video_mute(self, a: float, b: float) -> None:
        """A mute range marked in the video panel: drawn on the spectrogram like one
        added there, and stored with the project."""
        mutes = self.spectro.mutes() + [(a, b)]
        self.spectro.set_mutes(mutes)
        self._mutes_changed(self.spectro.mutes())
        self.log(f"mute range {a:.1f}-{b:.1f} s marked in the video")

    def settings_dialog(self) -> None:
        from .settings import SettingsDialog
        dlg = SettingsDialog(self.qsettings, OPENFREEMAP_STYLE, self)
        if dlg.exec() != SettingsDialog.DialogCode.Accepted:
            return
        changed = dlg.save()
        if changed.get("style"):
            self.map.bridge.setStyle(self._style_url())
        if changed.get("cache"):
            d = str(self.qsettings.value("cache/dir", "") or "")
            self.cache_dir = Path(d) if d else None
            self.flights = None
            self._update_steps()
            self.log(f"flight cache folder: {self.cache_dir or 'default'} (run Flights again)")
        self.log("settings saved")

    def _locks_changed(self, fid: int, lk: dict) -> None:
        if lk:
            self.project.fit_locks[int(fid)] = dict(lk)
        else:
            self.project.fit_locks.pop(int(fid), None)
        self._mark_dirty()
        self.statusBar().showMessage("Locked parameters change: press Re-fit to apply", 6000)

    def _mutes_changed(self, mutes) -> None:
        self.project.mutes = [tuple(m) for m in mutes]
        self._mark_dirty()
        self.statusBar().showMessage("Mute ranges changed: run Track again to apply them", 6000)

    # ------------------------------------------------------------------ Flights
    def _region(self):
        r = self.props.region()
        if r.kind == "bbox":
            from ..flights import BBox
            return resolve_region(bbox=BBox(*r.bbox))
        if r.kind == "near":
            return resolve_region(near=(r.lat, r.lon), radius_km=r.radius_km)
        return resolve_region()

    def _region_changed(self) -> None:
        region, src = self._region()
        self.map.bridge.setSearchRegion(None if src == "worldwide" else region.as_tuple())

    def _measure_toggled(self, on: bool) -> None:
        """Keep View > Measure distance in step with the map's own button."""
        self.a_measure.blockSignals(True)
        self.a_measure.setChecked(on)
        self.a_measure.blockSignals(False)

    def _map_clicked(self, lat: float, lon: float) -> None:
        if not self.props.r_bbox.isChecked():
            self.props.set_near(lat, lon)

    def _providers(self, ctx: JobContext):
        from ..flights.adsblol import AdsbLolProvider
        from ..flights.opensky_rest import OpenSkyCredentials, OpenSkyRestProvider
        out = []
        for n in self.props.providers():
            if n == "adsblol-api":
                from ..flights.adsblol_api import AdsbLolApiProvider
                out.append(AdsbLolApiProvider(progress=ctx.progress, log=ctx.log))
            elif n == "adsblol":
                out.append(AdsbLolProvider(
                    confirm=lambda plan: ctx.ask(f"Download the adsb.lol archive for this day?\n\n{plan.describe()}"
                                                 "\n\nThe day is indexed once and kept in the cache."),
                    progress=ctx.progress))
            elif n == "opensky":
                creds = OpenSkyCredentials.from_env()
                if creds is not None:
                    out.append(OpenSkyRestProvider(credentials=creds))
        return out

    def run_flights(self) -> None:
        if self.analysis is None:
            return
        try:
            epoch, _ = resolve_start(self.analysis, self.props.start_override())
        except Exception as e:
            self._error("Flights")(str(e), "")
            return
        region, src = self._region()
        fams = family_observations(self.analysis)
        clock = float(self.props.clock.value())

        def job(ctx: JobContext):
            ctx.progress("reading the ADS-B cache", 0, 0)
            with self._cache() as cache:
                return fetch_flights(fams, epoch, region, clock_bound_s=clock, providers=self._providers(ctx),
                                     cache=cache, log=ctx.log)

        self.jobs.start("flights", job, on_result=self._flights_done, on_error=self._error("Flights"),
                        on_log=self.log, on_progress=self._progress, on_question=self._ask)

    def _flights_done(self, data) -> None:
        self.flights = data
        from ..flights.adsblol_api import load_aircraft_info
        from ..flights.cache import default_cache_dir
        # The folder only: opening the cache database here would lock it for other processes.
        self.flights_panel.aircraft_info = load_aircraft_info(self.cache_dir or default_cache_dir())
        tracks = self._flight_tracks()
        self.map.bridge.setTracks(gj.tracks_geojson(tracks))
        self.map.bridge.zoomToData()
        q = data.query
        self.project.flight_query = FlightQueryModel(
            provider=data.provider, t0=q.t0, t1=q.t1, bbox=q.bbox.as_tuple(), fetched_at=datetime.now(timezone.utc),
            n_aircraft=len(tracks), from_cache=data.from_cache)
        self.flights_panel.set_summary(
            f"{len(tracks)} aircraft from {data.provider}{' (cache)' if data.from_cache else ''}. "
            f"Run Locate to gate candidates. {data.attribution}")
        if not self.spectro.max_btn.isChecked():
            self.d_flights.raise_()
        self._clear_solution()
        self._mark_dirty()
        self._update_steps()
        if self._auto:
            self._auto_next("Locate", self.run_locate)

    # ------------------------------------------------------------------ Locate
    def run_locate(self) -> None:
        if self.analysis is None or self.flights is None:
            return
        try:
            epoch, esrc = resolve_start(self.analysis, self.props.start_override())
        except Exception as e:
            self._error("Locate")(str(e), "")
            return
        region, rsrc = self._region()
        res, data = self.analysis, self.flights
        clock = float(self.props.clock.value())
        elevation, height = self.props.elevation.currentText(), self.props.height_m()
        settings = SolverSettings(exclude_aircraft=frozenset(self.excluded))

        def job(ctx: JobContext):
            cache = self._cache()
            cache.close()  # the solve only needs the cache folder (elevation tiles), not the database
            return solve_position(res, epoch, esrc, region, rsrc, data, clock_bound_s=clock, cache=cache,
                                  elevation=elevation, height_m=height, settings=settings, log=ctx.log,
                                  progress=ctx.progress)

        self.jobs.start("locate", job, on_result=self._locate_done, on_error=self._error("Locate"),
                        on_log=self.log, on_progress=self._progress)

    def _locate_done(self, r: LocateResult) -> None:
        self.result = r
        if self._auto:
            self._auto = False
            self.log("Run all finished")
        sol = r.solution
        br = self.map.bridge
        if sol is None:
            self.log(f"no position: {r.error}")
            self.flights_panel.set_summary(f"No position: {r.error}")
            br.setSolution(gj.fc([]))
            br.setHeatmap(gj.fc([]))
            br.setCandidates(gj.fc([]))
        else:
            a, b, ang = sol.ellipse_95
            msg = (f"Position {sol.lat:.6f}, {sol.lon:.6f}  grade {sol.grade}  95% {a:.0f} x {b:.0f} m  "
                   f"clock {sol.dt_clock:+.2f} s")
            if sol.mirror:
                msg += f"  mirror Δχ² {sol.mirror['delta_chi2']:.1f}" + ("" if sol.resolved else " (UNRESOLVED)")
            gerr = r.gps_tag_error_m()
            ref = self.project.reference_position
            if gerr is not None:
                msg += f"  GPS tag {gerr:.0f} m away"
            elif ref is not None:
                msg += f"  known true position {haversine_m(sol.lat, sol.lon, ref[0], ref[1]):.0f} m away"
            self.log(msg)
            self.flights_panel.set_summary(msg)
            br.setSolution(gj.solution_geojson(sol))
            br.setHeatmap(gj.heatmap_geojson(sol.grid))
            br.setCandidates(gj.candidates_geojson(sol))
            br.zoomToData()
        self.flights_panel.set_candidates(sol, self.excluded)
        self.project.candidates = candidate_models(sol)
        self.project.solution = sol.as_dict() if sol else None
        self._draw_predicted()
        self._update_curve_labels()
        self._update_map_cursor()
        if not self.spectro.max_btn.isChecked():
            self.d_flights.raise_()
        self._mark_dirty()
        self._update_steps()

    def _clear_solution(self) -> None:
        if self.result is None:
            return
        self.result = None
        self.project.candidates, self.project.solution = [], None
        self.flights_panel.set_candidates(None, self.excluded)
        self.spectro.set_predicted_curves([])
        self._update_curve_labels()
        br = self.map.bridge
        for f in (br.setSolution, br.setHeatmap, br.setCandidates):
            f(gj.fc([]))

    def _inclusion_changed(self, icao: str, included: bool) -> None:
        (self.excluded.discard if included else self.excluded.add)(icao)
        self.project.excluded_aircraft = sorted(self.excluded)
        self._mark_dirty()
        self.statusBar().showMessage("Candidate selection changed: run Locate again", 6000)

    def _candidate_selected(self, key) -> None:
        self.highlight = tuple(key) if key else None
        sol = self.result.solution if self.result else None
        if sol is not None:
            self.map.bridge.setCandidates(gj.candidates_geojson(sol, highlight=key[1] if key else None))
        self._draw_predicted()

    def _draw_predicted(self) -> None:
        """Predicted curves at the current best position: the assigned aircraft
        of each family, plus the highlighted candidate (spec "Key UX rule")."""
        sol = self.result.solution if self.result else None
        if sol is None or sol.best is None:
            self.spectro.set_predicted_curves([])
            return
        pairs = [(fid, icao) for fid, icao in sol.assignment.items()]
        if self.highlight and self.highlight not in pairs:
            pairs.append(self.highlight)
        curves = []
        fams = {f.id: f for f in self.result.families}
        for fid, icao in pairs:
            fam = fams.get(fid)
            if fam is None:
                continue
            t = np.linspace(float(fam.t.min()) - 20, float(fam.t.max()) + 20, 600)
            color = "#ffffff" if self.highlight == (fid, icao) and sol.assignment.get(fid) != icao \
                else gj.family_color(fid)
            for k in sorted(set(np.asarray(fam.k).tolist())):
                try:
                    curves.append((t, predicted_curve(sol, fam, icao, t, k), color))
                except (KeyError, ValueError):
                    pass
        self.spectro.set_predicted_curves(curves)

    # ------------------------------------------------------------------ project files
    def _collect_project(self) -> Project:
        p = self.project
        return p.model_copy(update=dict(
            start_utc_override=self.props.start_override(), clock_bound_s=float(self.props.clock.value()),
            region=self.props.region(), mutes=self.spectro.mutes(), providers=self.props.providers(),
            elevation=self.props.elevation.currentText(), height_m=self.props.height_m(),
            excluded_aircraft=sorted(self.excluded),
            fits=fit_models(self.analysis.families) if self.analysis else [],
            spectro=p.spectro if isinstance(p.spectro, SpectroModel) else SpectroModel()))

    def save(self) -> bool:
        if self.project_dir is None:
            return self.save_as_dialog()
        self.project = self._collect_project()
        save_project(self.project_dir, self.project, self.analysis.tracks if self.analysis else None)
        self.dirty = False
        self._update_title()
        self.log(f"saved {self.project_dir}")
        return True

    def save_as_dialog(self) -> bool:
        d = QFileDialog.getExistingDirectory(self, "Save project into folder",
                                             str(self.qsettings.value("dirs/projects", "")))
        if not d:
            return False
        self.qsettings.setValue("dirs/projects", d)
        self.project_dir = Path(d)
        return self.save()

    def open_project_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open project folder",
                                             str(self.qsettings.value("dirs/projects", "")))
        if d:
            self.open_project(d)

    def open_project(self, folder: str | Path) -> None:
        if not self._confirm_discard():
            return
        try:
            project, tracks = load_project(folder)
            media = check_media(project, Path(folder))
        except ProjectError as e:
            self._error("Open project")(str(e), "")
            return
        self._reset()
        self.project, self.project_dir, self.media_path = project, Path(folder), media
        self.excluded = set(project.excluded_aircraft)
        self.props.set_region(project.region)
        self.props.clock.setValue(project.clock_bound_s)
        self.props.elevation.setCurrentText(project.elevation)
        if project.height_m is not None:
            self.props.use_height.setChecked(True)
            self.props.height.setValue(project.height_m)
        self.props.p_adsblol.setChecked("adsblol" in project.providers)
        self.props.p_adsblol_api.setChecked("adsblol-api" in project.providers)
        self.props.p_opensky.setChecked("opensky" in project.providers)
        self.spectro.set_mutes(project.mutes)
        self.video.load(str(media))
        self.log(f"opened project {folder}")
        self.run_track(restore_tracks=tracks)
        self.dirty = False
        self._update_title()

    def _confirm_discard(self) -> bool:
        if not self.dirty or self.media_path is None:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current project first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------ export
    def export_dialog(self) -> None:
        if self.result is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Export results into folder")
        if d:
            paths = write_outputs(self.result, Path(d))
            self.map.snapshot_png(Path(d) / "map.png")
            self.log(f"exported {', '.join(p.name for p in paths)}, map.png")

    def export_png_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export map", "map.png", "PNG (*.png)")
        if path:
            self.map.snapshot_png(path)

    def closeEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        if not self._confirm_discard():
            ev.ignore()
            return
        self.jobs.shutdown(10_000)
        super().closeEvent(ev)
