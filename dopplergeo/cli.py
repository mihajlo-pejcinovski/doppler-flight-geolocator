"""Command line for dopplergeo.

``dopplergeo analyze VIDEO --out DIR``
    Track tones and fit Doppler curves; writes ``tracks.csv`` and ``fits.csv``.
``dopplergeo flights --start UTC --center LAT,LON``
    Fetch ADS-B tracks for the recording window (cached) and list the aircraft.
``dopplergeo locate VIDEO --near LAT,LON``
    The whole chain: tones -> ADS-B -> the recorder's GPS position (+ GeoJSON).
``dopplergeo adsblol-index DATE``
    Download (streaming) or index local parts of one adsb.lol day into the cache.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core.media import parse_timestamp
from .core.pipeline import AnalysisResult, analyze_file
from .core.solver import SolverSettings
from .core.spectral import SpectroSettings
from .core.tracking import TrackerSettings


def _parse_range(text: str) -> tuple[float, float]:
    a, _, b = text.partition(":")
    try:
        lo, hi = float(a), float(b)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected START:END in seconds, got {text!r}") from e
    if hi <= lo:
        raise argparse.ArgumentTypeError(f"END must be after START in {text!r}")
    return lo, hi


def _utc(epoch: float | None, t: float) -> str:
    if epoch is None:
        return ""
    return datetime.fromtimestamp(epoch + t, tz=timezone.utc).isoformat(timespec="milliseconds")


def write_tracks_csv(path: Path, res: AnalysisResult, epoch: float | None) -> int:
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "family_id", "k", "manual", "t_media_s", "t_utc", "f_hz", "snr_db", "sigma_hz"])
        for tr in res.tracks:
            for t, f, s, sg in zip(tr.t, tr.f, tr.snr_db, tr.sigma_f, strict=True):
                w.writerow([tr.id, tr.family_id, f"{tr.k:g}", int(tr.manual), f"{t:.3f}", _utc(epoch, t),
                            f"{f:.4f}", f"{s:.1f}", f"{sg:.4f}"])
                n += 1
    return n


FIT_COLUMNS = ["family_id", "n_tracks", "ks", "grade", "f0_hz", "f0_err", "v_mps", "v_err", "d_m", "d_err",
               "t_cpa_media_s", "t_cpa_err", "t_cpa_utc", "chi2_red", "r2", "cpa_covered",
               "asymptotes_visible", "jumps", "error"]


def write_fits_csv(path: Path, res: AnalysisResult, epoch: float | None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIT_COLUMNS)
        for fr in res.families:
            ks = " ".join(f"{k:g}" for k in fr.family.ks)
            flags = fr.flags
            if fr.fit is None:
                w.writerow([fr.family.id, len(fr.family.tracks), ks, "C"] + [""] * 13
                           + [flags["jumps"], fr.error])
                continue
            f = fr.fit
            w.writerow([
                fr.family.id, len(fr.family.tracks), ks, f.grade,
                f"{f.f0:.3f}", f"{f.stderr('f0'):.3f}", f"{f.v:.2f}", f"{f.stderr('v'):.2f}",
                f"{f.d:.1f}", f"{f.stderr('d'):.1f}", f"{f.t_cpa:.3f}", f"{f.stderr('t_cpa'):.3f}",
                _utc(epoch, f.t_cpa), f"{f.chi2_red:.2f}", f"{f.r2:.5f}",
                int(flags["cpa_covered"]), int(flags["asymptotes_visible"]), flags["jumps"], "",
            ])


def cmd_analyze(args: argparse.Namespace) -> int:
    spectro = SpectroSettings(window_s=args.window, hop_s=args.hop, fmin=args.fmin, fmax=args.fmax)
    tracker = TrackerSettings(threshold_db=args.threshold, min_snr_db=args.min_snr,
                              min_duration_s=args.min_duration)
    res = analyze_file(args.media, sample_rate=args.rate, mutes=args.mute or (),
                       noise_range=args.noise, spectro=spectro, tracker=tracker)
    clip = res.clip
    start = clip.start_utc
    source = clip.start_utc_source
    if args.start_utc:
        start = parse_timestamp(args.start_utc)
        if start is None:
            print(f"error: cannot parse --start-utc {args.start_utc!r}", file=sys.stderr)
            return 2
        source = "user"
    epoch = start.timestamp() if start else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_pts = write_tracks_csv(out / "tracks.csv", res, epoch)
    write_fits_csv(out / "fits.csv", res, epoch)

    print(f"file      {clip.path}")
    print(f"duration  {clip.duration:.1f} s @ {clip.sample_rate} Hz (native {clip.native_sample_rate} Hz)")
    print(f"start UTC {start.isoformat() if start else 'unknown'} ({source})")
    print(f"GPS tag   {clip.gps if clip.gps else 'none'}")
    print(f"tracks    {len(res.tracks)} ({n_pts} points) -> {out / 'tracks.csv'}")
    print(f"families  {len(res.families)} -> {out / 'fits.csv'}")
    print()
    print(f"{'fam':>3} {'grade':5} {'ks':12} {'f0 Hz':>9} {'v m/s':>12} {'d m':>14} {'t_cpa s':>12}")
    for fr in res.families:
        ks = ",".join(f"{k:g}" for k in fr.family.ks)
        if fr.fit is None:
            print(f"{fr.family.id:>3} {'C':5} {ks:12} fit failed: {fr.error}")
            continue
        f = fr.fit
        print(f"{fr.family.id:>3} {f.grade:5} {ks:12} {f.f0:9.2f} "
              f"{f.v:6.1f}±{f.stderr('v'):<5.1f} {f.d:7.0f}±{f.stderr('d'):<6.0f} "
              f"{f.t_cpa:6.2f}±{f.stderr('t_cpa'):<5.2f}")
    return 0


# ---------------------------------------------------------------------------
# Flight data
# ---------------------------------------------------------------------------


def _latlon(text: str) -> tuple[float, float]:
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected LAT,LON, got {text!r}") from e
    return lat, lon


def _bbox(text: str):
    from .flights import BBox
    try:
        return BBox.parse(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _confirm(assume_yes: bool):
    def ask(plan) -> bool:
        print(f"adsb.lol: {plan.describe()}", file=sys.stderr)
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            print("refusing to download without --yes (stdin is not a terminal)", file=sys.stderr)
            return False
        return input("download? [y/N] ").strip().lower() in ("y", "yes")
    return ask


def _progress():
    state = {"stage": None, "pct": -1, "t": time.monotonic()}

    def show(stage: str, done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 100
        if stage != state["stage"] or pct >= state["pct"] + 2:
            el = time.monotonic() - state["t"]
            rate = f" {done / el / 1e6:5.1f} MB/s" if stage.startswith("download") and el > 0 else ""
            print(f"  {stage:10} {pct:3d}%{rate}", file=sys.stderr)
            state.update(stage=stage, pct=pct)
    return show


def _wind(text: str) -> tuple[float, float]:
    try:
        spd, frm = text.split("@")
        return float(spd), float(frm)
    except ValueError:
        raise argparse.ArgumentTypeError("expected SPEED@FROM_DEG, e.g. 6@240") from None


def _providers(names: list[str], args: argparse.Namespace):
    from .flights.adsblol import AdsbLolProvider
    from .flights.opensky_rest import OpenSkyCredentials, OpenSkyRestProvider
    out = []
    for n in names:
        if n == "adsblol":
            out.append(AdsbLolProvider(confirm=_confirm(args.yes), progress=_progress(),
                                       keep_bbox=getattr(args, "keep_bbox", None)))
        elif n == "opensky":
            out.append(OpenSkyRestProvider(credentials=OpenSkyCredentials.from_env()))
        elif n == "adsblol-api":
            from .flights.adsblol_api import AdsbLolApiProvider
            out.append(AdsbLolApiProvider(progress=_progress(), log=lambda m: print(f"  {m}", file=sys.stderr)))
        else:
            raise SystemExit(f"error: unknown provider {n!r} (choose from adsblol-api, adsblol, opensky)")
    return out


def cmd_flights(args: argparse.Namespace) -> int:
    import numpy as np

    from .core.geo import haversine_m
    from .flights import BBox, FlightCache, FlightService, NoFlightData, query_window

    start = parse_timestamp(args.start)
    if start is None:
        print(f"error: cannot parse --start {args.start!r}", file=sys.stderr)
        return 2
    if args.bbox is not None:
        region = args.bbox
    elif args.center is not None:
        region = BBox.around(*args.center, args.radius)
    else:
        print("error: give --center LAT,LON (with --radius) or --bbox", file=sys.stderr)
        return 2
    query = query_window(start.timestamp(), args.duration, region, clock_bound_s=args.clock_bound)
    with FlightCache(args.cache) as cache:
        svc = FlightService(cache, _providers(args.providers.split(","), args))
        t = time.perf_counter()
        try:
            data = svc.get(query)
        except NoFlightData as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        dt = time.perf_counter() - t
    tracks = data.tracks()
    lat0, lon0 = region.center
    print(f"window    {datetime.fromtimestamp(query.t0, tz=timezone.utc):%Y-%m-%d %H:%M:%S} .. "
          f"{datetime.fromtimestamp(query.t1, tz=timezone.utc):%H:%M:%S} UTC")
    print(f"area      {', '.join(f'{x:.4f}' for x in query.bbox.as_tuple())}")
    print(f"source    {data.provider} ({'cache' if data.from_cache else 'fetched'}, {dt:.2f} s) "
          f"{data.table.num_rows} points, {len(tracks)} airborne tracks")
    print(f"credit    {data.attribution}")
    print()
    print(f"{'icao24':8} {'callsign':9} {'pts':>4} {'span s':>7} {'alt m':>11} {'gs m/s':>7} "
          f"{'min dist km':>11} {'reliable':>8}")
    rows = []
    for tr in tracks:
        dist = haversine_m(tr.lat, tr.lon, lat0, lon0) / 1000
        grid = tr.resample(1.0)
        gs = np.nanmedian(tr.gs_mps) if np.isfinite(tr.gs_mps).any() else float("nan")
        rows.append((dist.min(), tr, gs, grid["reliable"].mean()))
    for dmin, tr, gs, rel in sorted(rows, key=lambda r: r[0]):
        print(f"{tr.icao24:8} {tr.callsign or '-':9} {len(tr):4d} {tr.t_end - tr.t_start:7.0f} "
              f"{tr.alt_m.min():5.0f}-{tr.alt_m.max():<5.0f} {gs:7.1f} {dmin:11.1f} {rel:8.0%}")
    if args.out:
        import pyarrow.csv as pacsv
        pacsv.write_csv(data.table, args.out)
        print(f"\npoints -> {args.out}")
    return 0


def cmd_adsblol_index(args: argparse.Namespace) -> int:
    from .flights import FlightCache
    from .flights.adsblol import AdsbLolProvider

    day = parse_timestamp(args.date + "T00:00:00Z") if len(args.date) == 10 else None
    if day is None:
        print(f"error: expected DATE as YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
        return 2
    prov = AdsbLolProvider(confirm=_confirm(args.yes), progress=_progress(), keep_bbox=args.keep_bbox,
                           variant=args.variant)
    t = time.perf_counter()
    with FlightCache(args.cache) as cache:
        if args.archive:
            prov.index_archive(day, cache, args.archive)
        else:
            prov.index_day(day, cache)
        rec = [r for r in cache.records("adsblol") if r["t0"] == day.timestamp()][-1]
    from pathlib import Path as _P
    size = _P(rec["path"]).stat().st_size
    print(f"indexed {args.date}: {rec['n_points']:,} points, {rec['meta']} -> {rec['path']} "
          f"({size / 1e6:.0f} MB) in {time.perf_counter() - t:.0f} s")
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    from .flights import FlightCache, NoFlightData
    from .locate import LocateError, write_outputs

    start = None
    if args.start:
        start = parse_timestamp(args.start)
        if start is None:
            print(f"error: cannot parse --start {args.start!r}", file=sys.stderr)
            return 2
    spectro = SpectroSettings(window_s=args.window, hop_s=args.hop)
    from .core.pipeline import analyze_file
    from .locate import family_observations, fetch_flights, resolve_region, resolve_start, solve_position

    log = lambda m: print(f"  {m}", file=sys.stderr)  # noqa: E731
    try:
        log(f"analysing {args.media}")
        res = analyze_file(args.media, mutes=args.mute or (), noise_range=args.noise, spectro=spectro, log=log)
        epoch, epoch_src = resolve_start(res, start)
        region, region_src = resolve_region(args.near, args.radius, args.bbox)
        fams = family_observations(res)
        log(f"{len(res.tracks)} tone tracks, {len(res.families)} families, {len(fams)} with a Doppler fit")
        # The cache database is held only while fetching: the solve can take minutes and
        # another process (the GUI, a second run) must be able to use the cache meanwhile.
        with FlightCache(args.cache) as cache:
            data = fetch_flights(fams, epoch, region, clock_bound_s=args.clock_bound,
                                 providers=_providers(args.providers.split(","), args), cache=cache, log=log)
        settings = SolverSettings(adsb_lag_sigma_s=args.adsb_lag_sigma)
        r = solve_position(res, epoch, epoch_src, region, region_src, data, clock_bound_s=args.clock_bound,
                           clock_sigma_s=args.clock_sigma, settings=settings,
                           weather=args.weather, temp_c=args.temp, wind=args.wind, fit_wind=args.fit_wind,
                           cache=cache, elevation=args.elevation, height_m=args.height, log=log)
    except (LocateError, NoFlightData) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    paths = write_outputs(r, Path(args.out))
    print(f"start UTC {datetime.fromtimestamp(r.epoch, tz=timezone.utc).isoformat()} ({r.epoch_source})")
    print(f"region    {', '.join(f'{x:.4f}' for x in r.region.as_tuple())} ({r.region_source})")
    print(f"flights   {r.flights.provider}: {r.flights.table.num_rows} points")
    for fam in r.families:
        f = fam.fit
        print(f"family {fam.id}: f0 {f.f0:7.2f} Hz  v {f.v:5.1f} m/s  d {f.d:6.0f} m  "
              f"t_cpa {f.t_cpa:6.1f} s  fit grade {f.grade}{'  (loose gating)' if fam.loose else ''}")
    sol = r.solution
    if sol is None:
        print(f"\nNO POSITION: {r.error}")
        print(f"outputs -> {Path(args.out)}")
        return 3
    a, b, ang = sol.ellipse_95
    print()
    print(f"POSITION  {sol.lat:.6f}, {sol.lon:.6f}   grade {sol.grade}")
    print(f"          https://www.openstreetmap.org/?mlat={sol.lat:.6f}&mlon={sol.lon:.6f}#map=16/{sol.lat:.5f}/{sol.lon:.5f}")
    print(f"95%       {a:.1f} x {b:.1f} m, major axis {ang:.0f} deg")
    print(f"height    {sol.h:.1f} m above WGS84 ({sol.h_source}, ±{sol.h_sigma:.1f} m)")
    print(f"clock     {sol.dt_clock:+.2f} s correction to the file time")
    if sol.best is not None and sol.best.atmos is not None:
        print(f"air       {sol.best.atmos.describe()}")
    if sol.best is not None and sol.best.lags:
        print("adsb lag  " + ", ".join(f"{a} {v:+.2f} s" for a, v in sorted(sol.best.lags.items())))
    print(f"fit       chi2_red {sol.chi2_red:.2f} ({sol.chi2_excess:.1f}x the passes' own fits)")
    calls = {c.icao24: c.callsign for c in sol.candidates}
    for fid, icao in sorted(sol.assignment.items()):
        print(f"  family {fid} <- {icao} {calls.get(icao) or ''}")
    if sol.unexplained:
        print(f"  unexplained families: {sol.unexplained} (no matching ADS-B aircraft)")
    if sol.mirror:
        m = sol.mirror
        state = "rejected" if sol.resolved else "NOT RESOLVED: the position may be this one"
        print(f"mirror    {m['lat']:.6f}, {m['lon']:.6f}  delta chi2 {m['delta_chi2']:.1f} ({state})")
    if r.global_result is not None:
        g = r.global_result
        print(f"worldwide {g.n_tracks} aircraft searched; "
              f"{'AMBIGUOUS: several places fit' if g.ambiguous else 'unique place'}")
        for i, p in enumerate(g.places()[:8]):
            print(f"  #{i + 1} {p['lat']:.6f}, {p['lon']:.6f}  {p['kind']:8} chi2_red {p['chi2_red']:.2f}  "
                  f"{p['families']} families  {', '.join(p['aircraft'])}")
    if r.gps_tag_error_m() is not None:
        print(f"GPS tag   {r.gps_tag[0]:.6f}, {r.gps_tag[1]:.6f}  -> solution is {r.gps_tag_error_m():.0f} m away")
    for note in r.elevation_notes:
        print(f"note      {note}")
    print(f"credit    {'; '.join(r.attribution)}")
    print(f"outputs -> {', '.join(str(p) for p in paths)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dopplergeo", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="track tones and fit Doppler curves in a video/audio file")
    a.add_argument("media", help="video or audio file")
    a.add_argument("--out", default="dopplergeo_out", help="output folder (default: %(default)s)")
    a.add_argument("--start-utc", help="override recording start time (ISO 8601)")
    a.add_argument("--mute", type=_parse_range, action="append", metavar="START:END",
                   help="exclude a time range in seconds (repeatable)")
    a.add_argument("--noise", type=_parse_range, metavar="START:END",
                   help="noise-only range for spectral subtraction")
    a.add_argument("--rate", type=int, default=16_000, help="analysis sample rate (default: %(default)s)")
    a.add_argument("--window", type=float, default=1.0, help="STFT window s (default: %(default)s)")
    a.add_argument("--hop", type=float, default=0.1, help="STFT hop s (default: %(default)s)")
    a.add_argument("--fmin", type=float, default=20.0)
    a.add_argument("--fmax", type=float, default=5000.0)
    a.add_argument("--threshold", type=float, default=6.0, help="peak threshold dB (default: %(default)s)")
    a.add_argument("--min-snr", type=float, default=8.0, help="track median SNR dB (default: %(default)s)")
    a.add_argument("--min-duration", type=float, default=5.0, help="track min length s (default: %(default)s)")
    a.set_defaults(func=cmd_analyze)

    from .flights.cache import default_cache_dir
    f = sub.add_parser("flights", help="fetch and list ADS-B tracks around a recording")
    f.add_argument("--start", required=True, help="recording start, ISO 8601 UTC")
    f.add_argument("--duration", type=float, default=300.0, help="recording length s (default: %(default)s)")
    f.add_argument("--center", type=_latlon, metavar="LAT,LON", help="search region centre")
    f.add_argument("--radius", type=float, default=50.0, help="search radius km (default: %(default)s)")
    f.add_argument("--bbox", type=_bbox, metavar="LATMIN,LONMIN,LATMAX,LONMAX", help="search region box")
    f.add_argument("--clock-bound", type=float, default=0.0, help="max file clock error s")
    f.add_argument("--providers", default="adsblol-api,opensky,adsblol",
                   help="provider order (default: %(default)s)")
    f.add_argument("--cache", default=str(default_cache_dir()), help="cache folder (default: %(default)s)")
    f.add_argument("--yes", action="store_true", help="approve large archive downloads")
    f.add_argument("--out", help="write the points to this CSV")
    f.set_defaults(func=cmd_flights)

    lo = sub.add_parser("locate", help="find where a video was recorded (GPS coordinates)")
    lo.add_argument("media", help="video or audio file")
    lo.add_argument("--start", help="recording start, ISO 8601 UTC (default: file creation_time)")
    lo.add_argument("--near", type=_latlon, metavar="LAT,LON",
                    help="search region centre (omit for a worldwide search: needs >= 2 aircraft in the clip)")
    lo.add_argument("--radius", type=float, default=50.0, help="search radius km (default: %(default)s)")
    lo.add_argument("--bbox", type=_bbox, metavar="LATMIN,LONMIN,LATMAX,LONMAX", help="search region box")
    lo.add_argument("--clock-bound", type=float, default=0.0,
                    help="max error of the file clock, s (0 = trust it; re-encoded videos: try 120)")
    lo.add_argument("--clock-sigma", type=float, default=None,
                    help="Gaussian prior on the clock error, s (default: 2 for a phone's own time tag, "
                         "else none; 0 = none). Needs --clock-bound > 0")
    lo.add_argument("--adsb-lag-sigma", type=float, default=SolverSettings.adsb_lag_sigma_s,
                    help="per-aircraft ADS-B timing slack, s (default: %(default)s; 0 = trust ADS-B times)")
    lo.add_argument("--weather", default="none", choices=["auto", "standard", "none"],
                    help="air model: still air at a constant 15 °C (none, default), Open-Meteo temperature + "
                         "100 m wind for the hour with the ISA lapse (auto), or 15 °C calm with the lapse "
                         "(standard). auto/standard are experimental: on Manchester the lapse moves the "
                         "solution ~340 m across the approach")
    lo.add_argument("--temp", type=float, metavar="DEG_C", help="air temperature at the observer (overrides)")
    lo.add_argument("--wind", type=_wind, metavar="SPEED@FROM", help="wind, m/s and the direction it blows from "
                                                                     "in degrees, e.g. 6@240 (overrides)")
    lo.add_argument("--fit-wind", action="store_true", help="fit the wind (needs >= 2 headings), weather as prior")
    lo.add_argument("--providers", default="adsblol-api,opensky,adsblol", help="flight data order (default: %(default)s)")
    lo.add_argument("--cache", default=str(default_cache_dir()), help="flight cache (default: %(default)s)")
    lo.add_argument("--yes", action="store_true", help="approve large archive downloads")
    lo.add_argument("--elevation", default="auto", choices=["auto", "dem", "adsb", "constant"],
                    help="observer height source (default: %(default)s)")
    lo.add_argument("--height", type=float, help="known observer height above WGS84, m (overrides --elevation)")
    lo.add_argument("--mute", type=_parse_range, action="append", metavar="START:END",
                    help="exclude a time range in seconds (repeatable)")
    lo.add_argument("--noise", type=_parse_range, metavar="START:END", help="noise-only range")
    lo.add_argument("--window", type=float, default=1.0, help="STFT window s (default: %(default)s)")
    lo.add_argument("--hop", type=float, default=0.1, help="STFT hop s (default: %(default)s)")
    lo.add_argument("--out", default="dopplergeo_out", help="output folder (default: %(default)s)")
    lo.set_defaults(func=cmd_locate)

    x = sub.add_parser("adsblol-index", help="download/index one adsb.lol day into the cache")
    x.add_argument("date", help="UTC date YYYY-MM-DD")
    x.add_argument("--archive", nargs="+", help="index these local .tar / .tar.aa .tar.ab parts instead")
    x.add_argument("--keep-bbox", type=_bbox, metavar="LATMIN,LONMIN,LATMAX,LONMAX",
                   help="only keep points inside this box (smaller index)")
    x.add_argument("--variant", default="prod", help="release variant (default: %(default)s)")
    x.add_argument("--cache", default=str(default_cache_dir()), help="cache folder (default: %(default)s)")
    x.add_argument("--yes", action="store_true", help="do not ask before downloading")
    x.set_defaults(func=cmd_adsblol_index)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
