"""Properties panel: media facts and the Locate inputs (start time, clock
bound, search region, data sources, observer height)."""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QDateTime, QTimeZone, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...project import RegionModel

UTC = QTimeZone.utc()


def _qdt(dt: datetime) -> QDateTime:
    q = QDateTime.fromSecsSinceEpoch(int(dt.timestamp()), UTC)
    return q.addMSecs(int(round((dt.timestamp() % 1) * 1000)))


class PropertiesPanel(QWidget):
    changed = Signal()
    region_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # media
        self.media_info = QLabel("No file loaded.")
        self.media_info.setWordWrap(True)
        self.media_info.setTextInteractionFlags(self.media_info.textInteractionFlags())
        media = QGroupBox("Media")
        QVBoxLayout(media).addWidget(self.media_info)

        # time
        self.use_file_time = QCheckBox("Use the file's creation time")
        self.use_file_time.setChecked(True)
        self.start = QDateTimeEdit()
        self.start.setTimeZone(UTC)
        self.start.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz 'UTC'")
        self.start.setCalendarPopup(True)
        self.start.setEnabled(False)
        self.clock = QDoubleSpinBox(suffix=" s", decimals=0, minimum=0, maximum=3600, singleStep=10)
        self.clock.setToolTip("How far the file clock may be from true UTC (solved as a free parameter). "
                              "Use 120 s or more for re-encoded or messenger videos.")
        timebox = QGroupBox("Recording time")
        tf = QFormLayout(timebox)
        tf.addRow(self.use_file_time)
        tf.addRow("Start", self.start)
        tf.addRow("Clock bound", self.clock)

        # region
        self.r_world = QRadioButton("Worldwide")
        self.r_near = QRadioButton("Near")
        self.r_bbox = QRadioButton("Drawn box")
        self.r_bbox.setEnabled(False)
        self.r_world.setChecked(True)
        grp = QButtonGroup(self)
        for b in (self.r_world, self.r_near, self.r_bbox):
            grp.addButton(b)
        self.lat = QDoubleSpinBox(decimals=5, minimum=-90, maximum=90)
        self.lon = QDoubleSpinBox(decimals=5, minimum=-180, maximum=180)
        self.radius = QDoubleSpinBox(suffix=" km", decimals=1, minimum=1, maximum=500, value=50)
        self.bbox_label = QLabel("Use 'Draw region' on the map, or click the map to set 'Near'.")
        self.bbox_label.setWordWrap(True)
        self._bbox: tuple[float, float, float, float] | None = None
        regionbox = QGroupBox("Search region")
        rf = QFormLayout(regionbox)
        radios = QHBoxLayout()
        for b in (self.r_world, self.r_near, self.r_bbox):
            radios.addWidget(b)
        rf.addRow(radios)
        rf.addRow("Lat", self.lat)
        rf.addRow("Lon", self.lon)
        rf.addRow("Radius", self.radius)
        rf.addRow(self.bbox_label)

        # sources and height
        self.p_adsblol_api = QCheckBox("adsb.lol API (last 2 h, fast)")
        self.p_adsblol_api.setChecked(True)
        self.p_adsblol_api.setToolTip("Recent recordings without the daily archive download; uses ADSBLOL_API_KEY "
                                      "if set (500 requests/day)")
        self.p_adsblol = QCheckBox("adsb.lol archive (past days)")
        self.p_adsblol.setChecked(True)
        self.p_opensky = QCheckBox("OpenSky REST (needs credentials)")
        self.p_opensky.setChecked(True)
        self.elevation = QComboBox()
        self.elevation.addItems(["auto", "dem", "adsb", "constant"])
        self.elevation.setToolTip("Observer height source: auto = terrain model, else airport ground from ADS-B")
        self.height = QDoubleSpinBox(suffix=" m", decimals=1, minimum=-500, maximum=9000)
        self.use_height = QCheckBox("Fixed height")
        src = QGroupBox("Flight data and height")
        sf = QFormLayout(src)
        sf.addRow(self.p_adsblol_api)
        sf.addRow(self.p_adsblol)
        sf.addRow(self.p_opensky)
        sf.addRow("Height source", self.elevation)
        hrow = QHBoxLayout()
        hrow.addWidget(self.use_height)
        hrow.addWidget(self.height)
        sf.addRow(hrow)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        for w in (media, timebox, regionbox, src):
            lay.addWidget(w)
        lay.addStretch(1)

        self.use_file_time.toggled.connect(lambda on: self.start.setEnabled(not on))
        for w in (self.use_file_time, self.p_adsblol_api, self.p_adsblol, self.p_opensky, self.use_height):
            w.toggled.connect(self.changed)
        for w in (self.clock, self.height):
            w.valueChanged.connect(self.changed)
        self.start.dateTimeChanged.connect(self.changed)
        self.elevation.currentTextChanged.connect(self.changed)
        for b in (self.r_world, self.r_near, self.r_bbox):
            b.toggled.connect(self._region_toggled)
        for w in (self.lat, self.lon, self.radius):
            w.valueChanged.connect(self._region_toggled)

    def _region_toggled(self, *_):
        self.region_changed.emit()
        self.changed.emit()

    # -- media ----------------------------------------------------------------
    def set_media(self, clip, sha_short: str = "") -> None:
        lines = [f"<b>{clip.path.split('/')[-1]}</b>",
                 f"{clip.duration:.1f} s, {clip.native_sample_rate} Hz {sha_short}"]
        if clip.start_utc is not None:
            lines.append(f"Start {clip.start_utc.isoformat(timespec='milliseconds')} ({clip.start_utc_source})")
            self.start.setDateTime(_qdt(clip.start_utc))
            self.use_file_time.setChecked(True)
        else:
            lines.append("<span style='color:#c53030'>No creation time: enter the start below.</span>")
            self.use_file_time.setChecked(False)
        if clip.gps is not None:
            lines.append(f"GPS tag {clip.gps[0]:.5f}, {clip.gps[1]:.5f} (shown on the map; never used as input)")
        self.media_info.setText("<br>".join(lines))

    # -- values ---------------------------------------------------------------
    def start_override(self) -> datetime | None:
        if self.use_file_time.isChecked():
            return None
        return datetime.fromtimestamp(self.start.dateTime().toMSecsSinceEpoch() / 1000, tz=timezone.utc)

    def set_start_override(self, dt: datetime | None) -> None:
        self.use_file_time.setChecked(dt is None)
        if dt is not None:
            self.start.setDateTime(_qdt(dt))

    def providers(self) -> list[str]:
        return [n for n, b in (("adsblol-api", self.p_adsblol_api), ("opensky", self.p_opensky),
                               ("adsblol", self.p_adsblol)) if b.isChecked()]

    def height_m(self) -> float | None:
        return float(self.height.value()) if self.use_height.isChecked() else None

    def set_near(self, lat: float, lon: float) -> None:
        self.lat.setValue(lat)
        self.lon.setValue(lon)
        self.r_near.setChecked(True)

    def set_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        self._bbox = tuple(float(x) for x in bbox)
        s, w, n, e = self._bbox
        self.bbox_label.setText(f"Box {s:.4f}, {w:.4f} to {n:.4f}, {e:.4f}")
        self.r_bbox.setEnabled(True)
        self.r_bbox.setChecked(True)
        self._region_toggled()

    def region(self) -> RegionModel:
        if self.r_bbox.isChecked() and self._bbox is not None:
            return RegionModel(kind="bbox", bbox=self._bbox)
        if self.r_near.isChecked():
            return RegionModel(kind="near", lat=self.lat.value(), lon=self.lon.value(),
                               radius_km=self.radius.value())
        return RegionModel(kind="worldwide")

    def set_region(self, r: RegionModel) -> None:
        if r.kind == "bbox" and r.bbox is not None:
            self.set_bbox(r.bbox)
        elif r.kind == "near" and r.lat is not None and r.lon is not None:
            self.radius.setValue(r.radius_km)
            self.set_near(r.lat, r.lon)
        else:
            self.r_world.setChecked(True)
