"""Qt-free builders for the map's GeoJSON layers (WGS84 lon/lat)."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

# Tone-family colours, shared by the spectrogram and the map (colour-blind-safe order).
FAMILY_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#56b4e9", "#f0e442", "#000000"]


def family_color(fid: int | None) -> str:
    return "#888888" if fid is None else FAMILY_COLORS[int(fid) % len(FAMILY_COLORS)]


# One colour per aircraft on the map (tracks, candidates, icons): 12 well-separated
# hues, chosen by a stable hash of icao24 so a plane keeps its colour across runs.
AIRCRAFT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
                   "#f032e6", "#9a6324", "#469990", "#800000", "#808000", "#000075"]


def aircraft_color(icao24: str) -> str:
    return AIRCRAFT_COLORS[zlib.crc32(icao24.lower().encode()) % len(AIRCRAFT_COLORS)]


def fc(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _line(lon: np.ndarray, lat: np.ndarray, max_points: int) -> list[list[float]]:
    n = lon.size
    step = max(1, int(np.ceil(n / max_points)))
    idx = np.r_[np.arange(0, n, step), n - 1] if n else np.arange(0)
    idx = np.unique(idx)
    return np.c_[lon[idx], lat[idx]].round(6).tolist()


def tracks_geojson(tracks: Sequence, max_points: int = 400) -> dict:
    """All aircraft tracks (``FlightTrack``-like: icao24, callsign, lon, lat)."""
    feats = []
    for tr in tracks:
        if np.size(tr.lon) < 2:
            continue
        feats.append({"type": "Feature", "properties": {"icao24": tr.icao24, "callsign": tr.callsign or "",
                                                        "color": aircraft_color(tr.icao24)},
                      "geometry": {"type": "LineString",
                                   "coordinates": _line(np.asarray(tr.lon), np.asarray(tr.lat), max_points)}})
    return fc(feats)


def candidates_geojson(solution, highlight: str | None = None, max_points: int = 600) -> dict:
    """Gated candidates in their aircraft's colour (as on the tracks layer); the
    assigned aircraft drawn thicker, excluded ones faded."""
    if solution is None:
        return fc([])
    feats, seen = [], set()
    used = {v: k for k, v in solution.assignment.items()}
    for c in sorted(solution.candidates, key=lambda c: (c.icao24 not in used, c.family_id)):
        if c.icao24 in seen:
            continue
        seen.add(c.icao24)
        tr = c.traj.track
        fid = used.get(c.icao24, c.family_id)
        feats.append({"type": "Feature", "properties": {
            "icao24": c.icao24, "callsign": c.callsign or "", "family_id": int(fid),
            "color": aircraft_color(c.icao24), "family_color": family_color(fid), "used": c.icao24 in used, "included": bool(c.included),
            "highlight": c.icao24 == highlight},
            "geometry": {"type": "LineString",
                         "coordinates": _line(np.asarray(tr.lon), np.asarray(tr.lat), max_points)}})
    return fc(feats)


def aircraft_geojson(tracks: Sequence, t_utc: float, colors: dict[str, str] | None = None) -> dict:
    """Each aircraft's position and heading at ``t_utc`` (linear between fixes)."""
    feats = []
    for tr in tracks:
        t = np.asarray(tr.t)
        if t.size < 2 or not (t[0] <= t_utc <= t[-1]):
            continue
        lat = float(np.interp(t_utc, t, tr.lat))
        lon = float(np.interp(t_utc, t, tr.lon))
        i = int(np.clip(np.searchsorted(t, t_utc), 1, t.size - 1))
        dlat = tr.lat[i] - tr.lat[i - 1]
        dlon = (tr.lon[i] - tr.lon[i - 1]) * np.cos(np.radians(lat))
        heading = float(np.degrees(np.arctan2(dlon, dlat)) % 360)
        color = (colors or {}).get(tr.icao24)
        if colors is not None and color is None:
            continue  # only candidates get an icon
        feats.append({"type": "Feature", "properties": {"icao24": tr.icao24, "heading": heading,
                                                        "color": color or "#333333"},
                      "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}})
    return fc(feats)


def heatmap_geojson(grid, max_dchi2: float = 50.0, max_cells: int = 20_000) -> dict:
    """Grid cells with delta-chi2 below ``max_dchi2``, weighted exp(-dchi2/2)
    (relative likelihood), thinned to at most ``max_cells`` (best kept)."""
    if grid is None or np.size(grid.lat) == 0:
        return fc([])
    lat, lon, d = grid.heatmap()
    lat, lon = np.broadcast_arrays(lat, lon) if np.ndim(lat) == np.ndim(d) else np.meshgrid(lat, lon, indexing="ij")
    lat, lon, d = np.ravel(lat), np.ravel(lon), np.ravel(d)
    ok = np.isfinite(d) & (d <= max_dchi2)
    lat, lon, d = lat[ok], lon[ok], d[ok]
    if d.size > max_cells:
        keep = np.argsort(d)[:max_cells]
        lat, lon, d = lat[keep], lon[keep], d[keep]
    w = np.exp(-d / 2.0)
    return fc([{"type": "Feature", "properties": {"w": round(float(wi), 4), "dchi2": round(float(di), 2)},
                "geometry": {"type": "Point", "coordinates": [round(float(lo), 6), round(float(la), 6)]}}
               for la, lo, di, wi in zip(lat, lon, d, w)])


def solution_geojson(solution) -> dict:
    return fc([]) if solution is None or solution.best is None else solution.to_geojson()


def gps_geojson(gps: tuple | None, kind: str = "gps_tag", note: str = "") -> dict:
    """The true position: the file's own GPS tag (``gps_tag``) or a known position
    stored in the project (``reference``, e.g. a recordist's geotag)."""
    if gps is None:
        return fc([])
    return fc([{"type": "Feature", "properties": {"kind": kind, "note": note},
                "geometry": {"type": "Point", "coordinates": [gps[1], gps[0]]}}])
