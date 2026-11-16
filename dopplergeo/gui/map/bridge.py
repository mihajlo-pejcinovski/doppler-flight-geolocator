"""Python <-> MapLibre bridge (spec "Map / Integration design").

Python -> JS: the ``*Changed`` signals carry GeoJSON as JSON strings.
JS -> Python: ``regionDrawn``, ``aircraftClicked``, ``pointClicked``,
``measured``, ``measureToggled`` slots re-emit as typed Qt signals. Updates made before the page is ready are kept
and replayed once JS calls ``ready()`` (creating the map before
``loadFinished`` would leave the page blank).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from .. import geojson as gj

MAP_DIR = Path(__file__).resolve().parent
INDEX_HTML = MAP_DIR / "index.html"
OPENFREEMAP_STYLE = "https://tiles.openfreemap.org/styles/liberty"


def format_distance(m: float) -> str:
    """Metric distance text, as the map's distance tool shows it (``fmtDistance``
    in index.html): "842 m" below 1 km, "1.27 km" below 10 km, "12.4 km" above."""
    if not 0 <= m < math.inf:
        return ""
    whole = math.floor(m + 0.5)  # JS Math.round (Python's round() rounds halves to even)
    if whole < 1000:
        return f"{whole} m"
    km = m / 1000.0
    return f"{km:.2f} km" if float(f"{km:.2f}") < 10 else f"{km:.1f} km"


class MapBridge(QObject):
    # Python -> JS
    tracksChanged = Signal(str)
    candidatesChanged = Signal(str)
    heatmapChanged = Signal(str)
    solutionChanged = Signal(str)
    gpsChanged = Signal(str)
    aircraftChanged = Signal(str)
    regionChanged = Signal(str)
    zoomRequested = Signal()
    styleChanged = Signal(str)
    measureRequested = Signal(bool)
    # JS -> Python, re-emitted for the app
    region_drawn = Signal(float, float, float, float)   # lat_min, lon_min, lat_max, lon_max
    aircraft_clicked = Signal(str)
    point_clicked = Signal(float, float)
    distance_measured = Signal(float)                  # total length in metres (line finished)
    measure_toggled = Signal(bool)                     # distance tool on/off (map button or Python)
    map_ready = Signal()
    js_error = Signal(str)

    def __init__(self, style_url: str | None = OPENFREEMAP_STYLE, parent: QObject | None = None):
        super().__init__(parent)
        self._style_url = style_url or ""
        self.is_ready = False
        self._last: dict[str, str] = {}   # latest payload per signal, replayed on ready
        self._measure_on: bool | None = None  # distance tool request made before ready

    # -- JS reads this when it creates the map ---------------------------
    def _get_style(self) -> str:
        return self._style_url

    styleUrl = Property(str, _get_style, constant=True)

    # -- Python API ------------------------------------------------------
    def _send(self, name: str, payload: object) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
        self._last[name] = text
        if self.is_ready:
            getattr(self, name).emit(text)

    def setTracks(self, geojson: dict) -> None:
        self._send("tracksChanged", geojson)

    def setCandidates(self, geojson: dict) -> None:
        self._send("candidatesChanged", geojson)

    def setHeatmap(self, geojson: dict) -> None:
        self._send("heatmapChanged", geojson)

    def setSolution(self, geojson: dict) -> None:
        self._send("solutionChanged", geojson)

    def setGps(self, geojson: dict) -> None:
        self._send("gpsChanged", geojson)

    def setAircraft(self, geojson: dict) -> None:
        self._send("aircraftChanged", geojson)

    def setSearchRegion(self, bbox: tuple[float, float, float, float] | None) -> None:
        self._send("regionChanged", json.dumps(None if bbox is None else [float(x) for x in bbox]))

    def setStyle(self, url: str | None) -> None:
        self._style_url = url or ""
        if self.is_ready:
            self.styleChanged.emit(self._style_url)

    def setMeasureMode(self, on: bool) -> None:
        """Turn the map's distance-measuring tool on or off (off clears the line)."""
        if self.is_ready:
            self.measureRequested.emit(bool(on))
        else:
            self._measure_on = bool(on)

    def zoomToData(self) -> None:
        if self.is_ready:
            self.zoomRequested.emit()
        else:
            self._last["zoomRequested"] = ""

    def clear(self) -> None:
        empty = gj.fc([])
        for name in ("setTracks", "setCandidates", "setHeatmap", "setSolution", "setGps", "setAircraft"):
            getattr(self, name)(empty)
        self.setSearchRegion(None)

    # -- called from JS ----------------------------------------------------
    @Slot()
    def ready(self) -> None:
        self.is_ready = True
        for name, text in list(self._last.items()):
            if name == "zoomRequested":
                continue
            getattr(self, name).emit(text)
        if self._last.pop("zoomRequested", None) is not None:
            self.zoomRequested.emit()
        if self._measure_on is not None:
            self.measureRequested.emit(self._measure_on)
            self._measure_on = None
        self.map_ready.emit()

    @Slot(str)
    def regionDrawn(self, text: str) -> None:
        b = json.loads(text)
        self.region_drawn.emit(*(float(x) for x in b))

    @Slot(str)
    def aircraftClicked(self, icao24: str) -> None:
        self.aircraft_clicked.emit(icao24)

    @Slot(float, float)
    def pointClicked(self, lat: float, lon: float) -> None:
        self.point_clicked.emit(lat, lon)

    @Slot(float)
    def measured(self, total_m: float) -> None:
        self.distance_measured.emit(total_m)

    @Slot(bool)
    def measureToggled(self, on: bool) -> None:
        self.measure_toggled.emit(on)

    @Slot(str)
    def jsError(self, text: str) -> None:
        self.js_error.emit(text)


class _Page(QWebEnginePage):
    def __init__(self, bridge: MapBridge, parent=None):
        super().__init__(parent)
        self._bridge = bridge

    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802 (Qt API)
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self._bridge.js_error.emit(f"{message} ({source}:{line})")


class MapView(QWebEngineView):
    """The map widget: loads the bundled page (local MapLibre copy, no CDN at
    runtime) and wires the bridge through QWebChannel."""

    def __init__(self, style_url: str | None = OPENFREEMAP_STYLE, parent=None):
        super().__init__(parent)
        self.bridge = MapBridge(style_url, self)
        page = _Page(self.bridge, self)
        s = page.settings()
        # The page is a local file that fetches remote tiles. Enabling remote access
        # alone stops local <script src> loading in QtWebEngine 6.12, so set both.
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        self.channel = QWebChannel(page)
        self.channel.registerObject("bridge", self.bridge)
        page.setWebChannel(self.channel)
        self.setPage(page)
        self.load(QUrl.fromLocalFile(str(INDEX_HTML)))

    def eval_js(self, script: str, callback=None) -> None:
        if callback is None:
            self.page().runJavaScript(script)
        else:
            self.page().runJavaScript(script, 0, callback)

    def snapshot_png(self, path: str | Path) -> bool:
        """Save the current map view (Qt grab; the page also keeps its drawing
        buffer so ``map.getCanvas().toDataURL()`` works for the report)."""
        return self.grab().save(str(path), "PNG")
