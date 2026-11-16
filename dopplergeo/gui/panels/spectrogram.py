"""Spectrogram panel: pyqtgraph image + tone tracks + fitted and predicted
curves + mute ranges + a playhead synced to the video (spec "GUI / Panels")."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..geojson import family_color

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

MAX_ROWS = 1200       # display rows (frequency) after max-pooling
MAX_COLS = 6000       # display columns (time)


def display_image(times: np.ndarray, freqs: np.ndarray, values: np.ndarray, fmax: float,
                  max_rows: int = MAX_ROWS, max_cols: int = MAX_COLS) -> tuple[np.ndarray, QRectF]:
    """Crop to ``fmax`` and max-pool down to a displayable size, keeping
    narrow tonal lines visible. Returns (image[rows=freq, cols=time], rect)."""
    keep = freqs <= fmax
    v = values[keep]
    f = freqs[keep]
    ry = max(1, int(np.ceil(v.shape[0] / max_rows)))
    rx = max(1, int(np.ceil(v.shape[1] / max_cols)))
    ny, nx = v.shape[0] // ry, v.shape[1] // rx
    if ny == 0 or nx == 0:
        return np.zeros((1, 1), np.float32), QRectF(0, 0, 1, 1)
    v = v[: ny * ry, : nx * rx].reshape(ny, ry, nx, rx).max(axis=(1, 3)).astype(np.float32)
    df = (f[1] - f[0]) if f.size > 1 else 1.0
    dt = (times[1] - times[0]) if times.size > 1 else 1.0
    rect = QRectF(times[0] - dt / 2, f[0] - df / 2, nx * rx * dt, ny * ry * df)
    return v, rect


def robust_levels(img: np.ndarray, lo: float = 5.0, hi: float = 99.7) -> tuple[float, float]:
    ok = np.isfinite(img)
    if not ok.any():
        return 0.0, 1.0
    a, b = np.percentile(img[ok], [lo, hi])
    return float(a), float(max(b, a + 1.0))


class SpectrogramPanel(QWidget):
    seek_requested = Signal(float)              # media time (s)
    track_selected = Signal(object)             # track id or None
    mutes_changed = Signal(list)                # [(t0, t1), ...]
    maximize_toggled = Signal(bool)             # True: fill the window, False: restore the layout
    track_edited = Signal(int, object, float)   # track id, new family id (None: a new family), harmonic k
    track_drawn = Signal(object)                # [(t, f), ...] of a hand-drawn track
    fmax_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._spec = None
        self._whiten = False
        self._tracks: list = []
        self._selected: int | None = None
        self._track_items: dict[int, pg.PlotDataItem] = {}
        self._fit_items: list[pg.PlotDataItem] = []
        self._pred_items: list[pg.PlotDataItem] = []
        self._label_items: list[pg.TextItem] = []
        self._label_fid: dict[int, pg.TextItem] = {}         # family id -> its label
        self._curves: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}  # fitted curves per family
        self._placing = False
        self._mute_items: list[pg.LinearRegionItem] = []

        # toolbar
        self.fmax = QDoubleSpinBox(suffix=" Hz", decimals=0, minimum=50, maximum=8000, singleStep=100, value=2000)
        self.fmax.setToolTip("Highest frequency shown")
        self.whiten = QCheckBox("Whitened")
        self.whiten.setToolTip("Show dB above the running median across frequency (what the tracker sees)")
        self.mute_btn = QPushButton("Add mute range")
        self.mute_btn.setToolTip("Exclude a time range (speech, cars) from tracking; drag its edges, "
                                 "double-click to remove")
        self.delete_btn = QPushButton("Delete track")
        self.delete_btn.setEnabled(False)
        self.info = QLabel("")
        self.labels_box = QCheckBox("Labels")
        self.labels_box.setChecked(True)
        self.labels_box.setToolTip("Label each fitted Doppler curve (family, grade, speed, distance, aircraft)")
        self.goto = QComboBox()
        self.goto.setToolTip("Zoom to one Doppler curve (named by its aircraft once located)")
        self.goto.setMinimumContentsLength(28)
        self.goto.addItem("Go to curve…", None)
        self.max_btn = QPushButton("⤢ Maximise")
        self.max_btn.setCheckable(True)
        self.max_btn.setToolTip("Let the spectrogram fill the window (click again to restore the layout)")
        # Editing row: the selected track's family and harmonic number; hand drawing.
        self.sel_label = QLabel("No track selected")
        self.fam_box = QComboBox()
        self.fam_box.setToolTip("Move the selected track to another family (merge), or to a new one (split)")
        self.k_box = QDoubleSpinBox(decimals=2, minimum=0.25, maximum=20.0, singleStep=1.0, value=1.0)
        self.k_box.setPrefix("k = ")
        self.k_box.setToolTip("Harmonic number of the selected track relative to its family's rest frequency")
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Move / relabel the selected track and re-fit")
        self.draw_btn = QPushButton("✎ Draw track")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip("Click along a tone the tracker missed; each click snaps to the strongest "
                                 "frequency nearby. Finish (or double-click) adds it as a new family.")
        self.finish_btn = QPushButton("Finish")
        self.cancel_draw_btn = QPushButton("Cancel")
        for w in (self.fam_box, self.k_box, self.apply_btn, self.finish_btn, self.cancel_draw_btn):
            w.setEnabled(False)
        edit = QHBoxLayout()
        for w in (self.sel_label, self.fam_box, self.k_box, self.apply_btn, self.delete_btn):
            edit.addWidget(w)
        edit.addSpacing(16)
        for w in (self.draw_btn, self.finish_btn, self.cancel_draw_btn):
            edit.addWidget(w)
        edit.addStretch(1)
        self._draw_pts: list[tuple[float, float]] = []
        self._draw_item = pg.PlotDataItem([], [], pen=pg.mkPen("#4fd1c5", width=2, style=Qt.PenStyle.DotLine),
                                          symbol="+", symbolSize=10, symbolPen=pg.mkPen("#4fd1c5", width=2))
        self._draw_item.setZValue(60)

        bar = QHBoxLayout()
        for w in (QLabel("Max"), self.fmax, self.whiten, self.labels_box, self.goto, self.mute_btn):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.info)
        bar.addWidget(self.max_btn)

        # plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "media time", units="s")
        self.plot.setLabel("left", "frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.image = pg.ImageItem()
        self.image.setColorMap(pg.colormap.get("inferno"))
        self.plot.addItem(self.image)
        self.hist = pg.HistogramLUTWidget(image=self.image)  # dB range slider
        self.hist.setMaximumWidth(110)
        self.hist.gradient.setColorMap(pg.colormap.get("inferno"))
        self.playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=True, pen=pg.mkPen("#4fd1c5", width=2))
        self.playhead.setZValue(50)
        self.plot.addItem(self.playhead)
        self.playhead.sigDragged.connect(lambda line: self.seek_requested.emit(float(line.value())))
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)

        row = QHBoxLayout()
        row.addWidget(self.plot, 1)
        row.addWidget(self.hist)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(bar)
        lay.addLayout(edit)
        lay.addLayout(row, 1)
        self.plot.addItem(self._draw_item)

        self.fmax.valueChanged.connect(self._fmax_changed)
        self.whiten.toggled.connect(self._whiten_toggled)
        self.mute_btn.clicked.connect(self._add_mute_at_view)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.labels_box.toggled.connect(lambda on: self._place_labels())
        self.goto.activated.connect(self._goto_selected)
        self.plot.getPlotItem().getViewBox().sigRangeChanged.connect(lambda *_: self._place_labels())
        self.max_btn.toggled.connect(self._max_toggled)
        self.apply_btn.clicked.connect(self._apply_edit)
        self.draw_btn.toggled.connect(self._draw_toggled)
        self.finish_btn.clicked.connect(self._finish_draw)
        self.cancel_draw_btn.clicked.connect(lambda: self.draw_btn.setChecked(False))
        self._on_delete = None

    # -- data -------------------------------------------------------------
    def set_spectrogram(self, spec) -> None:
        self._spec = spec
        self._redraw_image(autorange=True)

    def _redraw_image(self, autorange: bool = False) -> None:
        spec = self._spec
        if spec is None:
            self.image.clear()
            return
        values = spec.whitened_db() if self._whiten else spec.db
        img, rect = display_image(spec.times, spec.freqs, values, float(self.fmax.value()))
        self.image.setImage(img, autoLevels=False)
        self.image.setRect(rect)
        lo, hi = robust_levels(img, 20.0 if self._whiten else 5.0, 99.7)
        self.image.setLevels((lo, hi))
        self.hist.setLevels(lo, hi)
        if autorange:
            self.plot.setXRange(rect.left(), rect.right(), padding=0)
            self.plot.setYRange(0, float(self.fmax.value()), padding=0)

    def set_tracks(self, tracks: Sequence) -> None:
        """Tone tracks as points, coloured by family; click one to select it."""
        for it in self._track_items.values():
            self.plot.removeItem(it)
        self._track_items.clear()
        self._tracks = list(tracks)
        for tr in self._tracks:
            color = family_color(tr.family_id)
            it = pg.PlotDataItem(tr.t, tr.f, pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                 symbolBrush=pg.mkBrush(color))
            it.setZValue(20)
            it.setCurveClickable(False)
            it.sigPointsClicked.connect(lambda _item, _pts, ev=None, tid=tr.id: self.select_track(tid))
            self.plot.addItem(it)
            self._track_items[tr.id] = it
        if self._selected not in self._track_items:
            self.select_track(None)
        self.info.setText(f"{len(self._tracks)} tracks")

    def select_track(self, tid: int | None) -> None:
        self._selected = tid
        for i, it in self._track_items.items():
            tr = next(t for t in self._tracks if t.id == i)
            color = family_color(tr.family_id)
            sel = i == tid
            it.setSymbolSize(8 if sel else 4)
            it.setSymbolPen(pg.mkPen("w", width=1.5) if sel else None)
            it.setSymbolBrush(pg.mkBrush(color))
        self.delete_btn.setEnabled(tid is not None)
        tr = next((t for t in self._tracks if t.id == tid), None)
        for w in (self.fam_box, self.k_box, self.apply_btn):
            w.setEnabled(tr is not None)
        if tr is not None:
            fams = sorted({t.family_id for t in self._tracks if t.family_id is not None})
            self.fam_box.clear()
            for fid in fams:
                self.fam_box.addItem(f"family #{fid}", fid)
            self.fam_box.addItem("new family", None)
            self.fam_box.setCurrentIndex(fams.index(tr.family_id) if tr.family_id in fams else len(fams))
            self.k_box.setValue(float(tr.k))
            self.sel_label.setText(f"Track {tr.id} ({tr.t[0]:.0f}-{tr.t[-1]:.0f} s):")
        else:
            self.sel_label.setText("No track selected")
        self.track_selected.emit(tid)

    def _apply_edit(self) -> None:
        if self._selected is not None:
            self.track_edited.emit(int(self._selected), self.fam_box.currentData(), float(self.k_box.value()))

    # -- hand-drawn tracks -----------------------------------------------------
    def _draw_toggled(self, on: bool) -> None:
        self._draw_pts = []
        self._draw_item.setData([], [])
        self.finish_btn.setEnabled(False)
        self.cancel_draw_btn.setEnabled(on)
        self.draw_btn.setText("✎ Drawing… click along the tone" if on else "✎ Draw track")
        self.plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode if on else pg.ViewBox.RectMode)

    def _snap(self, t: float, f: float, half_hz: float = 4.0) -> float:
        """The strongest frequency within +-half_hz of f in the frame nearest t."""
        spec = self._spec
        if spec is None:
            return f
        j = int(np.clip(np.searchsorted(spec.times, t), 0, spec.times.size - 1))
        m = np.flatnonzero(np.abs(spec.freqs - f) <= max(half_hz, spec.bin_hz))
        if m.size == 0:
            return f
        col = spec.power[m, j]
        return float(spec.freqs[m[int(np.argmax(col))]])

    def add_draw_point(self, t: float, f: float) -> None:
        self._draw_pts.append((float(t), self._snap(t, f)))
        self._draw_pts.sort()
        a = np.array(self._draw_pts)
        self._draw_item.setData(a[:, 0], a[:, 1])
        self.finish_btn.setEnabled(len(self._draw_pts) >= 2)

    def _finish_draw(self) -> None:
        """Fill the clicked points every 0.2 s (linear in f, snapped to the nearest
        peak again) and hand the track over."""
        pts = sorted(self._draw_pts)
        if len(pts) < 2:
            return
        a = np.array(pts)
        t = np.arange(a[0, 0], a[-1, 0] + 1e-9, 0.2)
        f = np.interp(t, a[:, 0], a[:, 1])
        f = np.array([self._snap(ti, fi, 2.0) for ti, fi in zip(t, f)])
        self.draw_btn.setChecked(False)
        self.track_drawn.emit(list(zip(t.tolist(), f.tolist())))

    def selected_track(self) -> int | None:
        return self._selected

    def on_delete(self, callback) -> None:
        """``callback(track_id)`` runs when the user deletes the selected track."""
        self._on_delete = callback

    def _delete_selected(self) -> None:
        if self._selected is not None and self._on_delete:
            self._on_delete(self._selected)

    def set_fit_curves(self, curves: dict[int, list[tuple[np.ndarray, np.ndarray]]]) -> None:
        """Fitted Doppler curves per family: {family_id: [(t, f) per harmonic]}."""
        for it in self._fit_items:
            self.plot.removeItem(it)
        self._fit_items = []
        self._curves = {fid: [(np.asarray(t), np.asarray(f)) for t, f in lines] for fid, lines in curves.items()}
        for fid, lines in curves.items():
            for t, f in lines:
                it = pg.PlotDataItem(t, f, pen=pg.mkPen(family_color(fid), width=1.5))
                it.setZValue(15)
                self.plot.addItem(it)
                self._fit_items.append(it)

    def set_curve_labels(self, labels: dict[int, tuple[float, float, str]]) -> None:
        """A text label per family, {family_id: (t, f, text)}, in the family colour
        just above its curve. (t, f) is the preferred spot (the CPA); when that is out
        of view the label moves to the visible part of the curve (``_place_labels``),
        so zooming in never loses it. Also fills the "Go to curve" list."""
        for it in self._label_items:
            self.plot.removeItem(it)
        self._label_items, self._label_fid = [], {}
        self.goto.clear()
        self.goto.addItem("Go to curve…", None)
        for fid, (t, f, text) in sorted(labels.items()):
            it = pg.TextItem(text, color=family_color(fid), anchor=(0.0, 1.1), fill=pg.mkBrush(0, 0, 0, 150))
            it.setPos(float(t), float(f))
            it._home = (float(t), float(f))
            it.setZValue(40)
            self.plot.addItem(it)
            self._label_items.append(it)
            self._label_fid[fid] = it
            self.goto.addItem(text.replace("  ", " · "), fid)
        if labels:
            self.goto.addItem("Show all", -1)
        self._place_labels()

    def _place_labels(self) -> None:
        """Keep each label on the visible part of its curve: at its CPA when that is
        in view, else at the visible curve point nearest the middle of the view;
        hidden when none of the curve is visible."""
        if self._placing:
            return
        self._placing = True
        try:
            (x0, x1), (y0, y1) = self.plot.getPlotItem().getViewBox().viewRange()
            on = self.labels_box.isChecked()
            for fid, it in self._label_fid.items():
                th, fh = it._home
                if not on:
                    it.setVisible(False)
                    continue
                if x0 <= th <= x1 and y0 <= fh <= y1:
                    it.setPos(th, fh)
                    it.setVisible(True)
                    continue
                best = None
                for t, f in self._curves.get(fid, []):
                    m = (t >= x0) & (t <= x1) & (f >= y0) & (f <= y1)
                    if m.any():
                        i = np.flatnonzero(m)
                        j = i[np.argmin(np.abs(t[i] - 0.5 * (x0 + x1)))]
                        if best is None or abs(t[j] - 0.5 * (x0 + x1)) < abs(best[0] - 0.5 * (x0 + x1)):
                            best = (float(t[j]), float(f[j]))
                it.setVisible(best is not None)
                if best is not None:
                    it.setPos(*best)
        finally:
            self._placing = False

    def zoom_to_family(self, fid: int) -> None:
        """Fit the view to one family's curves (all harmonics), with a margin."""
        lines = self._curves.get(fid)
        if not lines:
            return
        t = np.concatenate([a for a, _ in lines])
        f = np.concatenate([b for _, b in lines])
        dt = max(10.0, 0.15 * (t.max() - t.min()))
        df = max(20.0, 0.25 * (f.max() - f.min()))
        self.plot.setXRange(t.min() - dt, t.max() + dt, padding=0)
        self.plot.setYRange(max(0.0, f.min() - df), f.max() + df, padding=0)

    def _goto_selected(self, index: int) -> None:
        fid = self.goto.itemData(index)
        if fid is None:
            return
        if fid == -1:
            if self._spec is not None:
                self._redraw_image(autorange=True)
        else:
            self.zoom_to_family(int(fid))
        self.goto.setCurrentIndex(0)

    def _max_toggled(self, on: bool) -> None:
        self.max_btn.setText("⤡ Restore" if on else "⤢ Maximise")
        self.maximize_toggled.emit(on)

    def set_predicted_curves(self, curves: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
        """Predicted curves from ADS-B at the current position: [(t, f, color)].
        Dashed white-edged lines on top: the user's main trust signal."""
        for it in self._pred_items:
            self.plot.removeItem(it)
        self._pred_items = []
        for t, f, color in curves:
            it = pg.PlotDataItem(t, f, pen=pg.mkPen(color, width=2.5, style=Qt.PenStyle.DashLine),
                                 connect="finite")
            it.setZValue(30)
            self.plot.addItem(it)
            self._pred_items.append(it)

    # -- mutes -------------------------------------------------------------
    def set_mutes(self, mutes: Sequence[tuple[float, float]]) -> None:
        for it in self._mute_items:
            self.plot.removeItem(it)
        self._mute_items = []
        for a, b in mutes:
            self._add_mute_item(a, b)

    def mutes(self) -> list[tuple[float, float]]:
        return [tuple(float(x) for x in sorted(it.getRegion())) for it in self._mute_items]

    def _add_mute_item(self, a: float, b: float) -> None:
        it = pg.LinearRegionItem(values=(a, b), brush=pg.mkBrush(120, 120, 120, 90), movable=True)
        it.setZValue(10)
        it.sigRegionChangeFinished.connect(lambda _: self.mutes_changed.emit(self.mutes()))
        it.mouseDoubleClickEvent = lambda ev, it=it: self._remove_mute(it)
        self.plot.addItem(it)
        self._mute_items.append(it)

    def _remove_mute(self, it) -> None:
        if it in self._mute_items:
            self._mute_items.remove(it)
            self.plot.removeItem(it)
            self.mutes_changed.emit(self.mutes())

    def _add_mute_at_view(self) -> None:
        (x0, x1), _ = self.plot.getPlotItem().getViewBox().viewRange()
        c, w = (x0 + x1) / 2, max((x1 - x0) / 10, 1.0)
        self._add_mute_item(c - w / 2, c + w / 2)
        self.mutes_changed.emit(self.mutes())

    # -- playhead -----------------------------------------------------------
    def set_cursor_time(self, t: float) -> None:
        if not self.playhead.moving:
            self.playhead.setValue(float(t))

    def _scene_clicked(self, ev) -> None:
        vb = self.plot.getPlotItem().getViewBox()
        if self.draw_btn.isChecked() and ev.button() == Qt.MouseButton.LeftButton \
                and vb.sceneBoundingRect().contains(ev.scenePos()):
            if ev.double():
                self._finish_draw()
            else:
                p = vb.mapSceneToView(ev.scenePos())
                self.add_draw_point(p.x(), p.y())
            return
        if ev.double() or ev.button() != Qt.MouseButton.LeftButton:
            return
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        items = self.plot.scene().items(ev.scenePos())
        if any(isinstance(i, (pg.ScatterPlotItem, pg.LinearRegionItem, pg.InfiniteLine)) for i in items):
            return
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            t = vb.mapSceneToView(ev.scenePos()).x()
            self.set_cursor_time(t)
            self.seek_requested.emit(float(t))
        else:
            self.select_track(None)

    # -- toolbar -------------------------------------------------------------
    def _fmax_changed(self, v: float) -> None:
        self._redraw_image()
        self.plot.setYRange(0, v, padding=0)
        self.fmax_changed.emit(float(v))

    def _whiten_toggled(self, on: bool) -> None:
        self._whiten = bool(on)
        self._redraw_image()
