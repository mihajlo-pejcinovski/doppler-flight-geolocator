"""Worldwide search: locate the recorder with no hint about the region.

A 250 m grid cannot cover the planet, so the search starts from the aircraft:

1. **Screen** every track for each reliable tone family: low enough to be heard
   (altitude above a guessed ground ≤ d + margin), ground speed matching the
   fitted v, and flying at the fitted emission CPA time (± clock bound).
2. **Seed**: at that time the observer sits at slant distance d, perpendicular
   to the flight path, on the ground: two spots (left/right of the track),
   more along the track when the clock is uncertain.
3. **Score** each seed with a small local grid (±1 km); keep the best per family.
4. **Combine**: with several families, the spots of different aircraft must
   coincide (clusters within a few km). One family alone gives many
   near-equal places worldwide, so the result is a ranked list.
5. **Solve** the top clusters with the regional solver (full grid, refinement,
   mirror check, terrain from ``elevation_for``) and rank the solutions. A
   competing place that fits about as well makes the result ambiguous (grade C).

Ground height while seeding comes from airport surfaces seen in the same ADS-B
data (aircraft taxiing), since low aircraft are almost always near an airport.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .core.elevation import ConstantElevation, ElevationModel
from .core.geo import enu_rotation, enu_to_geodetic, haversine_m
from .core.solver import (
    Candidate,
    FamilyObs,
    Frame,
    SampledTrajectory,
    Solution,
    SolverError,
    SolverSettings,
    grid_search,
    solve,
)
from .flights.base import BBox, FlightTrack


class GlobalSearchError(RuntimeError):
    pass


@dataclass
class GlobalSettings:
    max_ground_m: float = 1500.0        # highest terrain assumed when no airport is nearby
    alt_margin_m: float = 300.0
    speed_tol: float = 0.15             # |gs/v - 1| for well-fitted (grade A/B) families
    speed_tol_poor: float = 0.30        # grade C fits (partial or curved passes) misjudge v
    seed_box_km: float = 1.0            # local grid half-size around each seed
    seed_grid_m: float = 200.0
    keep_per_family: int = 60           # after the coarse local grid
    refine_per_family: int = 40         # best coarse seeds refined continuously
    seed_chi2_red_factor: float = 3.0   # keep refined seeds within this factor of the family's best
    max_chi2_per_bin: float = 25.0      # seeds worse than this are dropped
    cluster_km: float = 3.0
    top_clusters: int = 8
    solve_radius_km: float = 6.0
    track_margin_km: float = 40.0
    ambiguity_dchi2: float = 9.0
    ambiguity_km: float = 2.0


@dataclass
class Seed:
    family_id: int
    track: FlightTrack = field(repr=False)
    lat: float
    lon: float
    chi2_per_bin: float = float("inf")


@dataclass
class GlobalResult:
    solution: Solution | None
    alternatives: list[Solution]        # other places, best first
    ambiguous: bool
    n_tracks: int
    candidates_per_family: dict[int, int]
    seeds_per_family: dict[int, int]
    clusters: list[dict]
    notes: list[str] = field(default_factory=list)

    def places(self) -> list[dict]:
        """Ranked candidate places (the solution first) for display."""
        out = []
        for s in ([self.solution] if self.solution else []) + self.alternatives:
            out.append({"lat": s.lat, "lon": s.lon, "chi2_red": s.chi2_red, "grade": s.grade,
                        "families": len(s.assignment), "aircraft": s.aircraft_used, "kind": "solution"})
            if s.mirror and not s.resolved:  # an unresolved mirror is an equally valid place
                out.append({"lat": s.mirror["lat"], "lon": s.mirror["lon"], "chi2_red": s.chi2_red,
                            "grade": "C", "families": len(s.assignment), "aircraft": s.aircraft_used,
                            "kind": "mirror"})
        return out


# ---------------------------------------------------------------------------
# 1-2: screening and seeding
# ---------------------------------------------------------------------------


def _screen(fam: FamilyObs, tracks: Sequence[FlightTrack], epoch: float, s: SolverSettings,
            g: GlobalSettings, ground: ElevationModel) -> list[tuple[FlightTrack, float]]:
    """Tracks that could have made this family's pass; with the CPA emission time (UTC)."""
    fit = fam.fit
    tau_c = epoch + fit.t_cpa - fit.d / s.c
    B = s.clock_bound_s
    out = []
    for tr in tracks:
        if tr.t_end < tau_c - B - 5 or tr.t_start > tau_c + B + 5:
            continue
        m = (tr.t >= tau_c - B - 30) & (tr.t <= tau_c + B + 30)
        if not m.any():
            i = int(np.clip(np.searchsorted(tr.t, tau_c), 1, tr.t.size - 1))
            m = np.zeros(tr.t.size, bool)
            m[i - 1:i + 1] = True
        gs = tr.gs_mps[m]
        gs = gs[np.isfinite(gs)]
        tol = g.speed_tol if fit.grade in ("A", "B") else g.speed_tol_poor
        if gs.size and abs(float(np.median(gs)) / fit.v - 1.0) > tol:
            continue
        h_ground = float(ground.height(tr.lat[m].mean(), tr.lon[m].mean())[0])
        if float(np.min(tr.alt_m[m])) - h_ground > fit.d + g.alt_margin_m:
            continue
        out.append((tr, tau_c))
    return out


def _seeds_for(fam: FamilyObs, tr: FlightTrack, tau_c: float, s: SolverSettings, g: GlobalSettings,
               ground: ElevationModel) -> list[Seed]:
    fit = fam.fit
    B = s.clock_bound_s
    v = max(fit.v, 30.0)
    step = 1000.0 * g.seed_box_km / v  # along-track spacing covered by one seed's local grid
    taus = tau_c + (np.arange(-B, B + 1e-9, step) if B > 0 else np.array([0.0]))
    taus = taus[(taus >= tr.t_start) & (taus <= tr.t_end)]
    out = []
    for tau in taus:
        lat, lon, alt = (float(x[0]) for x in tr.geodetic([tau]))
        if not np.isfinite(lat):
            continue
        hg = float(ground.height(lat, lon)[0])
        dz = alt - hg
        r_h = math.sqrt(max(fit.d**2 - dz**2, 0.0))
        vel = tr.velocity([tau])[0]
        ve, vn, _ = enu_rotation(lat, lon) @ vel
        sp = math.hypot(ve, vn) or 1.0
        nrm = np.array([-vn / sp, ve / sp])  # left of the track
        for side in ((1.0, -1.0) if r_h > 50 else (0.0,)):
            e, n = side * r_h * nrm
            la, lo, _ = enu_to_geodetic([e, n, 0.0], lat, lon)
            out.append(Seed(fam.id, tr, float(la), float(lo)))
    return out


def _score_seed(seed: Seed, fam: FamilyObs, epoch: float, s: SolverSettings, g: GlobalSettings,
                ground: ElevationModel) -> Seed:
    frame = Frame(seed.lat, seed.lon, epoch)
    traj = SampledTrajectory(seed.track, seed.lat, seed.lon, epoch, s.node_dt_s)
    cand = Candidate(fam.id, seed.track.icao24, seed.track.callsign, traj, 0.0, 0.0, 1.0)
    region = BBox.around(seed.lat, seed.lon, g.seed_box_km)
    sl = replace(s, grid_step_m=g.seed_grid_m, elevation=ground)
    grid = grid_search([fam], [cand], frame, region, sl)
    fc = grid.family_chi2[0]
    if not np.isfinite(fc).any():
        return seed
    i = int(np.nanargmin(np.where(np.isfinite(fc), fc, np.nan)))
    nbins = len(fam.binned(s.grid_points)[0])
    return Seed(seed.family_id, seed.track, float(grid.lat.ravel()[i]), float(grid.lon.ravel()[i]),
                float(fc.ravel()[i]) / max(nbins, 1))


def _refine_seed(seed: Seed, fam: FamilyObs, epoch: float, s: SolverSettings, ground: ElevationModel) -> Seed:
    """Continuous single-family fit at the seed: the true aircraft reaches chi2_red ~1
    at the exact spot, which the coarse grid cannot show."""
    from .core.solver import refine

    frame = Frame(seed.lat, seed.lon, epoch)
    traj = SampledTrajectory(seed.track, seed.lat, seed.lon, epoch, s.node_dt_s)
    cand = Candidate(fam.id, seed.track.icao24, seed.track.callsign, traj, 0.0, 0.0, 1.0)
    try:
        r = refine([fam], {fam.id: cand}, frame, 0.0, 0.0, 0.0, replace(s, elevation=ground))
    except (SolverError, ValueError, OverflowError, np.linalg.LinAlgError):
        return Seed(seed.family_id, seed.track, seed.lat, seed.lon, float("inf"))
    if math.hypot(r.e, r.n) > 3000.0:  # wandered off: not this seed's basin
        return Seed(seed.family_id, seed.track, seed.lat, seed.lon, float("inf"))
    return Seed(seed.family_id, seed.track, r.lat, r.lon, r.chi2_red)


# ---------------------------------------------------------------------------
# 4: clusters across families
# ---------------------------------------------------------------------------


def _clusters(seeds: dict[int, list[Seed]], g: GlobalSettings) -> list[dict]:
    fids = [f for f in seeds if seeds[f]]
    allseeds = [sd for f in fids for sd in seeds[f]]
    if not allseeds:
        return []
    lat = np.array([sd.lat for sd in allseeds])
    lon = np.array([sd.lon for sd in allseeds])
    fam = np.array([sd.family_id for sd in allseeds])
    score = np.array([sd.chi2_per_bin for sd in allseeds])
    out = []
    for i, anchor in enumerate(allseeds):
        d = haversine_m(anchor.lat, anchor.lon, lat, lon) / 1000.0
        near = d <= g.cluster_km
        members = {}
        for f in fids:
            m = near & (fam == f)
            if m.any():
                j = np.flatnonzero(m)[np.argmin(score[m])]
                members[f] = int(j)
        total = sum(score[j] for j in members.values()) + 100.0 * (len(fids) - len(members))
        pts = list(members.values())
        out.append({"lat": float(lat[pts].mean()), "lon": float(lon[pts].mean()), "families": len(members),
                    "score": float(total), "aircraft": sorted({allseeds[j].track.icao24 for j in pts})})
    out.sort(key=lambda c: (-c["families"], c["score"]))
    kept: list[dict] = []
    for c in out:
        if all(haversine_m(c["lat"], c["lon"], k["lat"], k["lon"]) > 2000.0 * g.cluster_km for k in kept):
            kept.append(c)
        if len(kept) >= g.top_clusters:
            break
    return kept


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def solve_global(families: Sequence[FamilyObs], tracks: Sequence[FlightTrack], epoch: float,
                 settings: SolverSettings | None = None, gsettings: GlobalSettings | None = None,
                 ground: ElevationModel | None = None,
                 elevation_for: Callable[[BBox], ElevationModel] | None = None,
                 log: Callable[[str], None] = lambda m: None) -> GlobalResult:
    """Locate the observer anywhere: ``tracks`` should cover the whole world
    (or whatever area the recording could come from) for the recording window."""
    s = settings or SolverSettings()
    g = gsettings or GlobalSettings()
    ground = ground or ConstantElevation(0.0, g.max_ground_m, "unknown")
    notes: list[str] = []
    seedable = [f for f in families if not f.loose]
    if not seedable:
        raise GlobalSearchError("no tone family has a reliable Doppler fit (v, d, CPA time) to search with")
    if len(seedable) < len(families):
        notes.append(f"families {[f.id for f in families if f.loose]} have unreliable fits: "
                     "not used for seeding, only in the final local solve")

    cand_count, seeds_by_fam = {}, {}
    for fam in seedable:
        screened = _screen(fam, tracks, epoch, s, g, ground)
        cand_count[fam.id] = len(screened)
        raw = [sd for tr, tau in screened for sd in _seeds_for(fam, tr, tau, s, g, ground)]
        scored = [_score_seed(sd, fam, epoch, s, g, ground) for sd in raw]
        coarse = sorted((sd for sd in scored if sd.chi2_per_bin <= g.max_chi2_per_bin),
                        key=lambda sd: sd.chi2_per_bin)[:g.refine_per_family]
        fine = [_refine_seed(sd, fam, epoch, s, ground) for sd in coarse]
        fine = [sd for sd in fine if np.isfinite(sd.chi2_per_bin)]  # now refined chi2_red
        best = min((sd.chi2_per_bin for sd in fine), default=np.inf)
        good = sorted((sd for sd in fine if sd.chi2_per_bin <= g.seed_chi2_red_factor * max(best, 1.0)),
                      key=lambda sd: sd.chi2_per_bin)[:g.keep_per_family]
        seeds_by_fam[fam.id] = good
        log(f"family {fam.id}: {len(screened)} aircraft pass the screen, {len(raw)} seeds, "
            f"{len(coarse)} refined, {len(good)} fit well (best chi2_red {best:.2f})")
    clusters = _clusters(seeds_by_fam, g)
    if not clusters:
        return GlobalResult(None, [], True, len(tracks), cand_count,
                            {k: len(v) for k, v in seeds_by_fam.items()}, [],
                            notes + ["no aircraft anywhere matches the recorded passes"])

    d_max = max(f.fit.d for f in seedable)
    radius = max(g.solve_radius_km, 2 * d_max / 1000.0 + 2.0)
    sols: list[Solution] = []
    for c in clusters:
        region = BBox.around(c["lat"], c["lon"], radius)
        near = region.expanded(g.track_margin_km)
        local = [tr for tr in tracks if near.contains(tr.lat, tr.lon).any()]
        elev = elevation_for(region) if elevation_for else ground
        log(f"solving near {c['lat']:.4f}, {c['lon']:.4f} ({c['families']} families, {c['aircraft']})")
        try:
            sols.append(solve(families, local, epoch, region, replace(s, elevation=elev)))
        except SolverError as e:
            notes.append(f"cluster at {c['lat']:.3f}, {c['lon']:.3f}: {e}")
    if not sols:
        return GlobalResult(None, [], True, len(tracks), cand_count,
                            {k: len(v) for k, v in seeds_by_fam.items()}, clusters, notes)

    floor = min(x.chi2_red for x in sols)
    limit = max(s.max_chi2_red, 3.0 * floor)
    sols.sort(key=lambda x: (x.chi2_red > limit, -len(x.assignment), x.chi2))
    best, rest = sols[0], sols[1:]
    rest = [x for x in rest if haversine_m(x.lat, x.lon, best.lat, best.lon) > 1000.0 * g.ambiguity_km]
    ambiguous = any(len(x.assignment) >= len(best.assignment)
                    and (x.chi2 - best.chi2) / max(best.chi2_red, 1.0) < g.ambiguity_dchi2
                    and x.chi2_red <= limit for x in rest)
    if ambiguous:
        best.grade = "C"
        notes.append("another place fits about as well: record more aircraft, or give a search region")
    return GlobalResult(best, rest, ambiguous, len(tracks), cand_count,
                        {k: len(v) for k, v in seeds_by_fam.items()}, clusters, notes)
