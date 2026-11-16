# dopplergeo

Locate where a video was recorded from the Doppler curve of aircraft passing
overhead, matched against ADS-B flight tracks. Full design: [SPEC.md](SPEC.md).

## Status

| Milestone | State |
| --- | --- |
| M1 Core physics (model, synthetic generator, single-pass fit) | done |
| M2 Audio pipeline | done |
| M3 Flight data (adsb.lol, OpenSky REST, cache, trajectories) | done |
| M4 Solver (gating, grid, refine, mirror, grading, observer height) | done |
| M5 Validation | in progress: Manchester real end-to-end test at 211 m from the geotag, grade A (6 aircraft, joint solve over the full 75-min file); tracker finds 14 of 29 passes, 6 with right v/d; GPS-tagged set pending |
| M6 GUI | done: `dopplergeo-gui` (install `.[gui]`): Run all, track editing and hand drawing, fit locks, video mutes, settings with keychain, cached re-analysis; end-to-end GUI tests pass |
| M7 Polish (report/KML/PDF, settings parity, OpenSky Trino, packaging) | specified in SPEC.md, not started |

## Layout

```
dopplergeo/core/
  atmosphere.py   speed of sound, ISA profile, path-mean c
  doppler.py      closed-form + trajectory Doppler model, fit, bootstrap
  synthetic.py    straight-line flights, frequency tracks, rendered audio
  geo.py          WGS84 geodetic/ECEF/ENU, ENU velocity rotation
  solver.py       gating, grid search, joint refine (position, clock, height), mirror, grade
  elevation.py    observer height: constant, grid, Copernicus GLO-30 + EGM2008
dopplergeo/locate.py  media -> families -> flights -> solution; outputs (shared by the CLI and the GUI)
dopplergeo/flights/
  base.py         TrackPoint schema, BBox, query window, FlightProvider, FlightTrack (ECEF Hermite spline)
  cache.py        Parquet parts + DuckDB coverage index
  adsblol.py      daily archive: tag lookup, header walk, streamed traces/ indexing
  opensky_rest.py OAuth2, /states/all (1 h), /flights/all + /tracks/all (30 days), credits
  service.py      cache first, then providers in order
  ground.py       airport ground height from on-ground ADS-B; elevation source choice
tests/
  test_doppler_model.py   closed form vs numeric, asymptotes, symmetry, slope
  test_fit.py             recovery, harmonics, drift, outliers, 1-sigma calibration
  test_synthetic.py       rendered audio -> ridge -> fit recovers geometry
examples/m1_demo.py       synthetic flyover -> fit -> truth vs estimate table
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -e '.[test,dem]'   # dem = rasterio for the terrain model
.venv/bin/pytest
.venv/bin/python examples/m1_demo.py
```

## Start the GUI

```
python3 -m venv .venv
.venv/bin/pip install -e '.[gui,dem]'
.venv/bin/dopplergeo-gui                       # empty window
.venv/bin/dopplergeo-gui flyover.mp4           # open a media file
.venv/bin/dopplergeo-gui path/to/project        # open a saved project folder
```

Options: `--cache DIR` (flight cache, default `~/.cache/dopplergeo/flights`) and
`--no-basemap` (plain background instead of online OpenFreeMap tiles, e.g. offline).

Work left to right through the step bar: **Load → Track → Fit → Flights → Locate**
(or *Run all*). API keys (adsb.lol, OpenSky) go in *Settings* and are stored in the
OS keychain; a variable already set in the shell wins.

The map needs WebGL (QtWebEngine). On a machine without a usable GPU, start with
`QTWEBENGINE_CHROMIUM_FLAGS=--enable-unsafe-swiftshader`; if QtWebEngine fails to
start on Ubuntu 24.04 because of the sandbox, use `QTWEBENGINE_DISABLE_SANDBOX=1`.

The audio is not in the repo (Freesound clips; their licences vary). The
Manchester test clip is [Freesound 868926](https://freesound.org/s/868926/): save
its HQ preview as `data/freesound/868926_5287430-hq.mp3` and open it with
`dopplergeo-gui data/freesound/868926_5287430-hq.mp3`.

## Locate a video

```
dopplergeo locate flyover.mp4 --near 52.31,4.76 --radius 30 --yes
```

Runs the whole chain: tone tracking and Doppler fits, ADS-B tracks for the
recording window (cache first; adsb.lol for past days, OpenSky for the last hour
with credentials), observer height (terrain model, else airport ground from ADS-B),
then the solver. Prints the GPS position with its 95% ellipse, grade (A/B/C),
which aircraft explained which tone, the clock correction and any mirror
solution, and writes `solution.json`, `solution.geojson` (point, ellipse,
mirror, candidate tracks, the file's GPS tag), `candidates.csv`, `tracks.csv`
and `fits.csv` to `--out`. Use `--start` when the file has no creation time
and `--clock-bound 120` for re-encoded videos whose clock may be off.

## Flight data

```
# Index one adsb.lol day (~3 GB streamed; the heatmap prefix is skipped)
dopplergeo adsblol-index 2026-11-12 --yes
# ... or from parts you downloaded yourself
dopplergeo adsblol-index 2026-11-12 --archive v2026.11.12-planes-readsb-prod-0.tar.a?

# Aircraft around a recording (±2 min, region + 20 km)
dopplergeo flights --start 2026-11-12T14:02:31Z --duration 240 --center 52.31,4.76 --radius 30
```

Recordings from the last 2 hours can use the adsb.lol API instead of the daily archive
(`--providers adsblol-api,...`, on by default): nearby aircraft are listed and their traces fetched
one request each. It works without a key (500 requests/day per IP, throttled after ~10 quick requests);
set `ADSBLOL_API_KEY` for a key's higher limits.

OpenSky needs OAuth2 client credentials in `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`
(keychain storage comes with the GUI). Set `GITHUB_TOKEN` to lift GitHub's 60 requests/hour
limit for release lookups. The cache lives in `~/.cache/dopplergeo/flights` unless `--cache` is given.
