"""End to end: media file -> tone families -> ADS-B tracks -> observer position.

Qt-free orchestration shared by ``dopplergeo locate`` and (later) the GUI's
Locate step: ``locate()`` returns everything a report or map needs, and
``write_outputs()`` writes it to a folder.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core.atmosphere import Atmosphere
from .core.geo import haversine_m
from .core.pipeline import AnalysisResult, analyze_file
from .core.solver import FamilyObs, Solution, SolverError, SolverSettings, solve
from .core.spectral import SpectroSettings
from .core.tracking import TrackerSettings
from .flights import BBox, FlightCache, FlightData, FlightProvider, FlightService, query_window
from .flights.base import ProviderError
from .flights.ground import observer_elevation


class LocateError(RuntimeError):
    pass


@dataclass
class LocateResult:
    analysis: AnalysisResult
    epoch: float                     # file UTC at media time 0
    epoch_source: str
    region: BBox
    region_source: str
    flights: FlightData
    families: list[FamilyObs]
    solution: Solution | None
    error: str | None = None
    elevation_notes: list[str] = field(default_factory=list)
    attribution: list[str] = field(default_factory=list)
    global_result: object | None = None   # GlobalResult when no region was given

    @property
    def gps_tag(self) -> tuple[float, float, float | None] | None:
        return self.analysis.clip.gps if self.analysis.clip else None

    def gps_tag_error_m(self) -> float | None:
        """Distance from the solution to the file's own GPS tag (ground truth), if both exist."""
        g, s = self.gps_tag, self.solution
        if g is None or s is None:
            return None
        return float(haversine_m(s.lat, s.lon, g[0], g[1]))


def resolve_start(res: AnalysisResult, start_utc: datetime | None) -> tuple[float, str]:
    """File UTC at media time 0 and where it came from (user override first)."""
    if start_utc is not None:
        return start_utc.timestamp(), "user"
    clip = res.clip
    if clip is not None and clip.start_utc is not None:
        return clip.start_utc.timestamp(), clip.start_utc_source
    raise LocateError("the file has no creation time; pass the recording start (UTC)")


def resolve_region(near: tuple[float, float] | None = None, radius_km: float = 50.0,
                   bbox: BBox | None = None) -> tuple[BBox, str]:
    """Search region. No hint -> worldwide search; the file's own GPS tag is never
    used as input, only compared against at the end."""
    if bbox is not None:
        return bbox, "bbox"
    if near is not None:
        return BBox.around(near[0], near[1], radius_km), "near"
    return BBox.world(), "worldwide"


def family_observations(res: AnalysisResult) -> list[FamilyObs]:
    return [FamilyObs.from_result(fr) for fr in res.families if fr.fit is not None]


def fetch_flights(fams: Sequence[FamilyObs], epoch: float, region: BBox, *, clock_bound_s: float = 0.0,
                  providers: Sequence[FlightProvider] = (), cache: FlightCache,
                  log: Callable[[str], None] = lambda m: None) -> FlightData:
    """ADS-B tracks for the time span of the fitted families (cache first)."""
    if not fams:
        raise LocateError("no tone family could be fitted: no usable aircraft sound in the recording")
    t_lo = min(float(f.t.min()) for f in fams)
    t_hi = max(float(f.t.max()) for f in fams)
    query = query_window(epoch + t_lo, t_hi - t_lo, region, clock_bound_s=clock_bound_s)
    log("fetching ADS-B tracks")
    data = FlightService(cache, list(providers)).get(query)  # NoFlightData propagates
    log(f"{len(data.tracks())} airborne tracks from {data.provider}{' (cache)' if data.from_cache else ''}")
    return data


# Start times stamped by the phone itself (NTP-synced clock, file not re-encoded).
PHONE_CLOCK_SOURCES = ("com.apple.quicktime.creationdate", "android")
PHONE_CLOCK_SIGMA_S = 2.0


def auto_clock_sigma(epoch_src: str) -> float:
    """Clock prior sigma (s) for a start time source: a phone's own tag gets
    ``PHONE_CLOCK_SIGMA_S``; anything else (plain creation_time, which editors and
    messengers rewrite, or a user-typed start) gets none (flat within the bound)."""
    return PHONE_CLOCK_SIGMA_S if any(k in epoch_src for k in PHONE_CLOCK_SOURCES) else 0.0


def air_for(region: BBox, t_utc: float, weather: str = "auto", temp_c: float | None = None,
            wind: tuple[float, float] | None = None, cache_dir: Path | None = None,
            log: Callable[[str], None] = lambda m: None) -> tuple[Atmosphere | None, list[str]]:
    """The atmosphere for the solver and its attribution lines.

    ``weather``: "auto" (Open-Meteo for the region centre and hour; standard air if
    that fails), "standard" (15 deg C, ISA lapse, calm) or "none" (the old still-air
    model at a constant 15 deg C). ``temp_c`` and ``wind`` ((speed m/s, from deg))
    override the fetched values."""
    from .core.atmosphere import wind_from_met

    if weather == "none" and temp_c is None and wind is None:
        return None, []
    atmos, attr = Atmosphere(source="standard"), []
    if weather == "auto":
        from .weather import ATTRIBUTION, WeatherError, atmosphere_at
        lat0, lon0, lat1, lon1 = region.as_tuple()
        try:
            atmos = atmosphere_at((lat0 + lat1) / 2, (lon0 + lon1) / 2, t_utc, cache_dir)
            attr = [ATTRIBUTION]
        except (WeatherError, ProviderError, OSError, KeyError, ValueError) as e:
            log(f"weather unavailable ({e}); using standard air (15 °C, calm)")
    if temp_c is not None:
        atmos = replace(atmos, temp_c=float(temp_c), source=atmos.source + " + user temperature")
    if wind is not None:
        we, wn = wind_from_met(*wind)
        atmos = replace(atmos, wind_e=we, wind_n=wn, source=atmos.source + " + user wind")
    log(f"air: {atmos.describe()}")
    return atmos, attr


def solve_position(res: AnalysisResult, epoch: float, epoch_src: str, region: BBox, region_src: str,
                   data: FlightData, *, clock_bound_s: float = 0.0, cache: FlightCache,
                   clock_sigma_s: float | None = None,
                   weather: str = "none", temp_c: float | None = None,
                   wind: tuple[float, float] | None = None, fit_wind: bool = False,
                   elevation: str = "auto", height_m: float | None = None,
                   settings: SolverSettings | None = None,
                   log: Callable[[str], None] = lambda m: None,
                   progress: Callable[[str, int, int], None] | None = None) -> LocateResult:
    """Observer position from an analysis and its flight data. A solver failure
    is returned in ``error`` so the analysis and flights can still be inspected."""
    from .core.elevation import ConstantElevation

    fams = family_observations(res)
    if not fams:
        raise LocateError("no tone family could be fitted: no usable aircraft sound in the recording")
    tracks = data.tracks()
    if region_src == "worldwide":
        return _locate_global(res, epoch, epoch_src, region, data, tracks, fams, clock_bound_s, elevation,
                              height_m, settings, cache, log)
    if progress:
        progress("observer height (terrain model)", 0, 0)
    if height_m is not None:
        elev, notes = ConstantElevation(height_m, 5.0, "user"), []
    else:
        elev, notes = observer_elevation(elevation, region, data.table, cache_dir=cache.root.parent)
    log(f"observer height source: {elev.source}")

    s = replace(settings or SolverSettings(), clock_bound_s=clock_bound_s, elevation=elev)
    t_mid = epoch + 0.5 * (min(float(f.t.min()) for f in fams) + max(float(f.t.max()) for f in fams))
    atmos, wx_attr = air_for(region, t_mid, weather, temp_c, wind, cache.root.parent / "weather", log)
    if atmos is not None:
        s = replace(s, atmos=atmos, fit_wind=fit_wind)
    if clock_sigma_s is None:
        clock_sigma_s = auto_clock_sigma(epoch_src)
    if clock_sigma_s > 0 and clock_bound_s > 0:
        s = replace(s, clock_sigma_s=clock_sigma_s)
        log(f"clock prior: +-{clock_sigma_s:g} s (1 sigma) on the file time ({epoch_src})")
    attribution = [data.attribution] + ([getattr(elev, "attribution")] if getattr(elev, "attribution", "") else []) \
        + wx_attr
    try:
        sol, err = solve(fams, tracks, epoch, region, s, progress=progress), None
    except SolverError as e:
        sol, err = None, str(e)
    return LocateResult(res, epoch, epoch_src, region, region_src, data, fams, sol, err, notes, attribution)


def locate(media: str | Path, *, start_utc: datetime | None = None, near: tuple[float, float] | None = None,
           radius_km: float = 50.0, bbox: BBox | None = None, clock_bound_s: float = 0.0,
           providers: Sequence[FlightProvider] = (), cache: FlightCache | None = None,
           elevation: str = "auto", height_m: float | None = None,
           settings: SolverSettings | None = None, mutes: Sequence[tuple[float, float]] = (),
           noise_range: tuple[float, float] | None = None, spectro: SpectroSettings = SpectroSettings(),
           tracker: TrackerSettings = TrackerSettings(),
           log: Callable[[str], None] = lambda m: None) -> LocateResult:
    """Run the whole chain. Raises ``LocateError`` for missing inputs (no start
    time, no region, no flight data); a solver failure is returned in ``error``
    so the analysis and flights can still be inspected. The GUI runs the same
    steps one at a time: ``analyze_file``, ``fetch_flights``, ``solve_position``."""
    log(f"analysing {media}")
    res = analyze_file(media, mutes=mutes, noise_range=noise_range, spectro=spectro, tracker=tracker, log=log)
    epoch, epoch_src = resolve_start(res, start_utc)
    region, region_src = resolve_region(near, radius_km, bbox)
    fams = family_observations(res)
    log(f"{len(res.tracks)} tone tracks, {len(res.families)} families, {len(fams)} with a Doppler fit")
    if cache is None:
        cache = FlightCache()
    data = fetch_flights(fams, epoch, region, clock_bound_s=clock_bound_s, providers=providers, cache=cache,
                         log=log)
    return solve_position(res, epoch, epoch_src, region, region_src, data, clock_bound_s=clock_bound_s,
                          cache=cache, elevation=elevation, height_m=height_m, settings=settings, log=log)


def _locate_global(res, epoch, epoch_src, region, data, tracks, fams, clock_bound_s, elevation, height_m,
                   settings, cache, log) -> LocateResult:
    from .core.elevation import ConstantElevation
    from .flights.ground import adsb_ground_elevation
    from .global_search import GlobalSearchError, solve_global

    if height_m is not None:
        ground = ConstantElevation(height_m, 5.0, "user")
        elevation_for = None
    else:
        ground = adsb_ground_elevation(data.table) or ConstantElevation(0.0, 1500.0, "unknown")
        if elevation == "adsb" or elevation == "constant":
            elevation_for = None
        else:
            def elevation_for(r: BBox):
                return observer_elevation(elevation, r, data.table, cache_dir=cache.root.parent)[0]
    s = replace(settings or SolverSettings(), clock_bound_s=clock_bound_s)
    log(f"worldwide search over {len(tracks)} aircraft")
    try:
        g = solve_global(fams, tracks, epoch, s, ground=ground, elevation_for=elevation_for, log=log)
    except GlobalSearchError as e:
        return LocateResult(res, epoch, epoch_src, region, "worldwide", data, fams, None, str(e),
                            attribution=[data.attribution])
    sol = g.solution
    err = None if sol else "; ".join(g.notes) or "no position found"
    if sol is not None:
        region = BBox.around(sol.lat, sol.lon, 6.0)
    attribution = [data.attribution]
    return LocateResult(res, epoch, epoch_src, region, "worldwide", data, fams, sol, err, g.notes,
                        attribution, g)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds")


def solution_geojson(r: LocateResult) -> dict:
    """Solution + ellipse + mirror, candidate tracks (lines), GPS tag; WGS84 lon/lat."""
    feats: list[dict] = []
    sol = r.solution
    if sol is not None:
        feats += sol.to_geojson()["features"]
        used = set(sol.aircraft_used)
        seen = set()
        for c in sol.candidates:
            if c.icao24 in seen:
                continue
            seen.add(c.icao24)
            tr = c.traj.track
            feats.append({"type": "Feature", "properties": {
                "kind": "candidate_track", "icao24": c.icao24, "callsign": c.callsign, "used": c.icao24 in used},
                "geometry": {"type": "LineString", "coordinates": np.c_[tr.lon, tr.lat].round(6).tolist()}})
    if r.gps_tag is not None:
        feats.append({"type": "Feature", "properties": {"kind": "gps_tag"},
                      "geometry": {"type": "Point", "coordinates": [r.gps_tag[1], r.gps_tag[0]]}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"attribution": r.attribution}}


def write_outputs(r: LocateResult, out: Path) -> list[Path]:
    from .cli import write_fits_csv, write_tracks_csv

    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "tracks.csv", out / "fits.csv", out / "candidates.csv", out / "solution.json",
             out / "solution.geojson"]
    write_tracks_csv(paths[0], r.analysis, r.epoch)
    write_fits_csv(paths[1], r.analysis, r.epoch)
    with paths[2].open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_id", "icao24", "callsign", "used", "cpa_time_utc", "cpa_range_m", "speed_match", "chi2"])
        if r.solution is not None:
            for c in sorted(r.solution.candidates, key=lambda c: (c.family_id, c.chi2)):
                w.writerow([c.family_id, c.icao24, c.callsign or "",
                            int(r.solution.assignment.get(c.family_id) == c.icao24), _utc(c.t_cpa_utc),
                            f"{c.cpa_range_m:.0f}", f"{c.speed_match:.3f}", f"{c.chi2:.1f}"])
    doc = {
        "media": str(r.analysis.clip.path) if r.analysis.clip else None,
        "start_utc": _utc(r.epoch), "start_utc_source": r.epoch_source,
        "region": dict(zip(("lat_min", "lon_min", "lat_max", "lon_max"), r.region.as_tuple())),
        "region_source": r.region_source,
        "flight_source": r.flights.provider, "attribution": r.attribution,
        "elevation_notes": r.elevation_notes,
        "gps_tag": r.gps_tag, "gps_tag_error_m": r.gps_tag_error_m(),
        "solution": r.solution.as_dict() if r.solution else None, "error": r.error,
        "worldwide": None if r.global_result is None else {
            "ambiguous": r.global_result.ambiguous, "places": r.global_result.places(),
            "aircraft_searched": r.global_result.n_tracks,
            "candidates_per_family": {str(k): v for k, v in r.global_result.candidates_per_family.items()}},
    }
    paths[3].write_text(json.dumps(doc, indent=2, default=float))
    paths[4].write_text(json.dumps(solution_geojson(r)))
    return paths


# ---------------------------------------------------------------------------
# Predicted curves (spec "Key UX rule")
# ---------------------------------------------------------------------------


def predicted_curve(sol: Solution, fam: FamilyObs, icao24: str, t_media: np.ndarray,
                    k: float = 1.0) -> np.ndarray:
    """Frequency of harmonic ``k`` that aircraft ``icao24`` would produce at the
    solution position, for any gated candidate of the family (not only the
    assigned one). The rest frequency is re-fitted to the family's points
    (closed form in log f), so the curve shows the shape match. NaN where the
    track does not cover the reception time."""
    from .core.solver import log_doppler

    if sol.best is None or sol.frame is None:
        raise ValueError("solution has no refined position")
    cand = next((c for c in sol.candidates if c.icao24 == icao24 and c.family_id == fam.id), None)
    if cand is None:
        cand = next((c for c in sol.candidates if c.icao24 == icao24), None)
    if cand is None:
        raise KeyError(f"{icao24} is not a candidate")
    obs = sol.frame.observer(sol.best.e, sol.best.n, sol.h)
    shift, atm = sol.best.time_shift(icao24), sol.best.atmos
    lg, ok, _ = log_doppler(cand.traj, obs, np.asarray(fam.t, float) + shift, sol.c, atmos=atm)
    w = 1.0 / np.maximum(np.asarray(fam.sigma, float) / np.asarray(fam.f, float), 1e-9) ** 2
    r = np.log(np.asarray(fam.f, float) / np.asarray(fam.k, float)) - lg
    if not ok.any():
        return np.full(np.shape(t_media), np.nan)
    f0 = float(np.exp(np.sum(w[ok] * r[ok]) / np.sum(w[ok])))
    t = np.asarray(t_media, float)
    lg_t, ok_t, _ = log_doppler(cand.traj, obs, t + shift, sol.c, atmos=atm)
    return np.where(ok_t, k * f0 * np.exp(lg_t), np.nan)
