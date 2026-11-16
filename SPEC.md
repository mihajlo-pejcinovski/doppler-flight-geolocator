# Doppler Flight Geolocator — Specification

Version 0.1 · 2026-11-13 · Living copy: https://claude.ai/artifact/1wePkrtdfpJvvgC7U5i8bz

## Overview

The app takes a video that has an aircraft flyover in its audio, plus an approximate date/time. From the audio it pulls out the Doppler curve of the engine tones. It then gets ADS-B tracks for that time window and finds the observer position whose predicted Doppler curve best matches what was heard. Result: a position estimate with a confidence ellipse, plotted on an OpenStreetMap-based map.

**Goals**

- Load a video or audio file (MP4, MOV, MKV, WAV, FLAC) and show the waveform and spectrogram.
- Detect tonal components (fan/blade-pass tones, harmonics) and track them over time as frequency curves.
- Fit a moving-source Doppler model to each curve to get the closest-approach time t0, the closest-approach distance (as a ratio to speed), and the rest frequency f0.
- Query free ADS-B services for aircraft in a time window and bounding box. Match the fitted curves to candidate flights.
- Solve for the observer latitude/longitude (plus optional altitude and clock offset) and show it with its uncertainty.
- Export a report: candidate flights, fitted parameters, the position estimate, residual plots.

**Non-goals (v1)**

- Identifying aircraft type from sound alone (a v2 classifier could add this).
- Real-time or live-stream processing.
- Helicopters and propeller-only GA aircraft. They are often missing from ADS-B coverage and have very different tonal structure. They can be supported later, but nothing is tuned for them.
- Locating the observer with no flight data (the Doppler curve on its own gives only relative geometry).

**Key inputs**

| Input | Required | Notes |
| --- | --- | --- |
| Media file | yes | Audio track ≥ 16 kHz sample rate; mono is enough |
| Recording start time (UTC) | yes, ±10 min | From container metadata (`creation_time`), EXIF/QuickTime tags, or user input |
| Search region | yes | A bounding box or a centre plus radius. Defaults to 50 km around a user-given hint, or a GPS tag if one exists |
| Clock offset bound | optional | How far the file timestamp can be from true UTC; solved as a free parameter |

## Physics background

A jet or turboprop flying past emits steady tones (fan blade-pass, propeller blade rate, and their harmonics). The ground observer hears them shifted by the Doppler factor. The shift depends only on the component of aircraft velocity along the line of sight, at the time the sound was **emitted**. That gives an S-shaped curve: a high frequency on approach, a low frequency on departure, and the steepest slope at the closest point of approach (CPA).

**Doppler for a moving source and a stationary observer** (still air):

```latex
f_{obs}(t) = \frac{f_0}{1 + \dfrac{\mathbf{v}\cdot\hat{\mathbf{r}}(\tau)}{c}}, \qquad t = \tau + \frac{\lVert \mathbf{r}(\tau) \rVert}{c}
```

Per-peak σf = √(σ_CRB² + (0.03/T)²): the Cramér–Rao term plus a floor for interpolation bias and the tone sweeping inside the window (measured 0.027 Hz at > 35 dB with T = 1 s, where the CRB alone predicts 0.007 Hz).

Here τ is emission time, t is reception time, **r**(τ) is the aircraft-minus-observer vector, and the unit vector points from observer to aircraft (so v · r̂ is negative while the aircraft approaches). c is the speed of sound, about 340 m/s at 15 °C; it is computed from the air temperature.

**Straight-line, constant-velocity model** (the standard model for a single microphone). Take speed v, CPA slant distance d and emission CPA time τ₀:

```latex
\mathbf{v}\cdot\hat{\mathbf{r}}(\tau) = \frac{v^2(\tau-\tau_0)}{\sqrt{d^2 + v^2(\tau-\tau_0)^2}}
```

The unknowns per tone are f₀, v, d and τ₀. They pin the observed curve down as follows:

- f₀ sets the curve's mid-level. Approach and departure asymptotes are f₀/(1 − v/c) and f₀/(1 + v/c), so their ratio gives v/c directly.
- The slope at CPA is about −f₀ v²/(c d). Together with v this gives d.
- Harmonics share v, d and τ₀ with integer multiples of f₀. Fitting them together makes the fit much more robust.

**What the audio alone cannot tell you.** One microphone gives speed, CPA distance and CPA time. It does not give which side the aircraft passed on, its heading, or altitude separately from horizontal offset. The observer is constrained to a cylinder of radius d around the flight track. ADS-B supplies the track. Two or more flyovers, or one flyover plus altitude from ADS-B, collapse that cylinder to a point or a pair of points mirrored about the track.

**Effects to model or bound**

| Effect | Size | Handling |
| --- | --- | --- |
| Propagation delay ‖r‖/c | 3–30 s at 1–10 km | Exact retarded-time solve (above) |
| Wind along the line of sight | v_wind/c, e.g. 10 m/s gives about 3% | Optional wind vector from ERA5/Open-Meteo; otherwise widen the error bound |
| Temperature gradient (c changes with altitude) | 1–3% in c | Use the mean c along the path from the ISA lapse rate |
| Aircraft not at constant velocity | climb, turn, decel | Use ADS-B-interpolated positions in the forward model instead of a straight line |
| Ground reflection (Lloyd's mirror) | interference fringes in the spectrogram | Ignore for the tone tracks; optional altitude cue later |
| Video audio processing | AGC, resampling, codec smearing | Use phase-vocoder IF refinement; never trust absolute level |
| Engine power change | f₀ drifts | Allow a linear f₀(t) term when the residuals show a trend |

## Processing pipeline

```
 AUDIO LANE
 +--------------+    +---------------+    +------------------+    +-----------------+
 | Video file   |--->| Spectrogram   |--->| Tone tracks      |--->| Doppler fit     |
 | PyAV, 16 kHz |    | STFT, whiten  |    | DP ridge+harmonics |    | f0, v, d, t_CPA |
 +--------------+    +---------------+    +------------------+    +-----------------+
                                                                          | v, t_CPA
 FLIGHT DATA LANE                                                         v
 +--------------+    +---------------+    +------------------+    +-----------------+
 | Time+region  |--->| ADS-B         |--->| Cache + splines  |--->| Candidate       |
 | meta or user |    | adsb.lol,     |    | DuckDB, 1 s ECEF |    | gating          |
 +--------------+    | OpenSky       |    +------------------+    | speed, CPA time |
                     +---------------+                            +-----------------+
                                                                          | candidates
                                                                          v
                                          +------------------+    +-----------------+
                                          | Map + report     |<---| Position solver |
                                          | MapLibre,ellipse |    | grid, refine,   |
                                          +------------------+    | mirror check    |
                                                                  +-----------------+
```

The audio lane alone gives speed, CPA distance and CPA time. Only the ADS-B lane turns that into a place, through gating and the joint solver.

## Audio ingest and tone tracking

The output of this stage is a set of tone tracks. Each track is a list of (t, f, SNR, σf) points following one spectral line through the flyover, grouped into harmonic families.

**1. Ingest**

- Demux the audio with **PyAV** (FFmpeg bindings). Read `creation_time`, QuickTime `com.apple.quicktime.location.ISO6709` and `com.apple.quicktime.creationdate`, and Android `location` tags. A GPS tag, if present, is shown as a ground-truth marker. It does not seed the solver.
- Convert to mono float32. Resample to 16 kHz with `soxr` (aircraft tones of interest sit between 20 Hz and 5 kHz). Keep the native rate for an optional high-band view.
- Record the stream start PTS offset so audio time maps exactly onto video time and onto the container's UTC timestamp.

**2. Pre-processing**

- High-pass at 15 Hz to remove DC and handling rumble.
- Optional spectral-subtraction denoise using a user-selected noise-only segment.
- The user can mute time ranges (speech, car pass-bys) in the GUI. Muted ranges are excluded from tracking and fitting.

**3. Time–frequency representation**

| Parameter | Default | Reason |
| --- | --- | --- |
| Window | Hann, 1.0 s (16384 samples at 16 kHz) | ~1 Hz bins resolve slow-moving low tones |
| Hop | 0.1 s | 10 points/s is enough for a curve that changes over 10–60 s |
| Zero-pad | ×2 | Smoother peak interpolation |
| Short window (toggle) | 0.25 s | For fast CPA passes where the frequency moves >2 Hz/s |
| Reassignment / synchrosqueezing | optional (`librosa.reassigned_spectrogram`) | Sharper ridges at CPA |

The spectrogram is computed with `scipy.signal.ShortTimeFFT` and shown in dB with a percentile-clipped colour scale.

**4. Ridge extraction**

1. Per frame: whiten with a running median across frequency (61 bins), then pick peaks above a threshold (default 6 dB) with `scipy.signal.find_peaks`.
2. Refine each peak by quadratic interpolation over three bins. Optionally refine with phase-vocoder instantaneous frequency across two hops.
3. Link tones across frames by **dynamic-programming ridge extraction** (default; `TrackerSettings.method = "ridge"`). Cell score = min(whitened dB, 25) − 9 dB, so noise sums negative and a tone (even a weak, wandering one) sums positive along its path. A local Viterbi pass (paths start and stop anywhere) finds the best-scoring ridge; a step of k bins between frames costs 9·(k/K)² dB. K is the rate limit: falling at up to 25%·f per second (10–120 Hz/s), **rising at only 3%·f per second**, because a passing aircraft's tone only falls and rises only with RPM changes. This stops a ridge climbing onto another tone. The ridge and its whole spectral skirt (descending from the peak to the valley, up to 5%·f, widened by |df/dt|·window for sweeping tones) are suppressed, the DP is recomputed from the ridge's first frame on, and extraction repeats while a ridge scores ≥ 15 dB·s. Points weaker than 3 dB are gaps. Each point's σ is the Cramér–Rao error combined in quadrature with 0.2%·f (the slow source wander measured on real flyovers).
   The M2 peak-to-peak Kalman/Hungarian linker (gate 3σ, coasting through crossings) remains as `method = "kalman"`. On real recordings it fragmented every tone (the gate trusts the per-peak error, which real tones exceed) and strung broadband-noise peaks into hundreds of false tracks. On a close, fast pass it returned a confident grade A at v = 44.8 m/s against a true 60 m/s.
4. Drop tracks shorter than 5 s, with median SNR below 8 dB, or with peaks in under 50% of their span (noise chains measured at 17–42%, real tones at 77–99%).
5. Group tones by source: all tones of one aircraft share the same Doppler factor, so their frequency ratio is constant while they sweep (harmonics of one source never cross). Two tracks are a family when, over ≥ 3 s of overlap, ≥ 60% of the common frames have f_a/f_b within ±0.5% of its median and the tone sweeps by ≥ 1%. The fraction (not a percentile spread) lets a pair survive seconds where one tone is captured by another aircraft's tone within the resolution. Those disagreeing frames are then dropped from the family. k is the ratio to the lowest member, snapped to integers when within 1%, allowing a missing fundamental (2:3 gives k = 2, 3). Pieces with the same k are merged. Two more passes join pieces that never overlap: families whose fits agree on v, d, t_CPA and rest-frequency ratio are merged, and low-grade families are absorbed when every track keeps a constant ratio to a good family's fitted curve.
6. Manual mode: the user draws a rough polyline on the spectrogram. The tracker searches ±N Hz around it. This is the fallback when automatic tracking fails in noise.

**5. Quality flags shown in the GUI**

- The track covers CPA (the inflection point lies inside the track, not at its edge). Without CPA the fit is badly conditioned.
- Approach/departure asymptotes are visible (needed for a good v/c).
- Sudden jumps in the track (power change, track switching), flagged for user review.

## Doppler curve fitting

Each harmonic family is fitted on its own, with no flight data, to a straight-line constant-velocity source. The output is θ = (f₀, v, d, τ₀) with a covariance matrix. These parameters are used twice: to pre-filter ADS-B candidates, and as the initial guess for the joint solver.

**Model.** For each observation time t, solve the retarded time τ from t = τ + √(d² + v²(τ−τ₀)²)/c. The equation is a quadratic in (τ−τ₀) and has a closed form. Then:

```latex
f_{k}(t) = \frac{k\, f_0}{1 + \dfrac{v^2(\tau-\tau_0)}{c\sqrt{d^2+v^2(\tau-\tau_0)^2}}}
```

Here k is the harmonic number. Harmonics share v, d and τ₀. f₀ can optionally drift linearly: f₀ + ḟ₀(t−τ₀).

**Estimator**

1. Initial guess: τ₀ from the time of steepest slope. f₀ is the harmonic mean of the two asymptotes, 2·f_hi·f_lo/(f_hi + f_lo), and v = c·(f_hi − f_lo)/(f_hi + f_lo). d comes from the CPA slope: d ≈ f₀ v²/(c·|slope|).
2. Weighted nonlinear least squares with `scipy.optimize.least_squares` (method `trf`, `soft_l1` loss to down-weight stray points). Weights are 1/σf² from peak SNR.
3. Bounds: v between 30 and 320 m/s, d between 100 m and 20 km, f₀ between 10 Hz and 5 kHz.
4. Uncertainty: covariance from (JᵀWJ)⁻¹ scaled by the reduced χ². Confirm with a 200-sample residual bootstrap. Show 1σ bands on the plot.
5. Optional: MCMC with `emcee` for the full posterior when d and v correlate strongly (distant passes with a shallow S curve).

**Outputs per family**

| Field | Unit | Use |
| --- | --- | --- |
| f₀ | Hz | Engine/fan signature; harmonic count |
| v | m/s | Gate ADS-B candidates by ground speed ± wind |
| d | m | Radius of the observer cylinder around the track |
| τ₀, t_CPA | s, UTC | Gate candidates by CPA time; t_CPA = τ₀ + d/c |
| Covariance, χ²_red (inliers), outlier fraction, R² | — | Quality grade: A = σv/v < 3%, σd/d < 10%, CPA covered, χ²_red < 3, outliers < 5%; B = 10%, 30%, χ²_red < 10, outliers < 20%; else C |

**Sanity checks** (warn, never block): v/c above 0.9 is impossible for a subsonic tone model. Also warn when d is under 300 m while v is above 200 m/s (implausible outside an airport), and when the residuals show an S-shaped pattern (wrong model, e.g. a turning aircraft).

## Flight data sources

The recording date decides the source. For the last hour, a registered OpenSky account is enough. For one day or more in the past, the free option is the adsb.lol daily archive (ODbL). OpenSky Trino has the best history (since 2013) but is open only to approved research, government and aviation-authority users. All sources sit behind one `FlightProvider` interface; the app tries them in a user-set order and caches every response.

| Service | Free history | Access | Limits / terms | Role in app |
| --- | --- | --- | --- | --- |
| [OpenSky REST](https://openskynetwork.github.io/opensky-api/rest.html) | `/states/all` up to 1 h back (registered); `/tracks` up to 30 days (marked experimental) | OAuth2 client credentials, 30 min tokens | 4,000 credits/day (standard user), 8,000 for active feeders; area ≤ 25 sq° costs 1 credit | Recent recordings; per-aircraft tracks within 30 days |
| [OpenSky Trino](https://openskynetwork.github.io/opensky-api/trino.html) | `state_vectors_data4` since 2013, unlimited retention | Application review; research, government, aviation authorities only | Must filter on the `hour` partition | Best source when the user is eligible |
| [adsb.lol globe_history](https://www.adsb.lol/docs/open-data/historical/) | One release per day of the whole sky, on GitHub | None | ODbL 1.0; split `.tar.aa/.tar.ab` archives, large | Default free source for any past date |
| [adsb.lol API](https://api.adsb.lol/docs) | `/1/trace/full/{hex}`: one aircraft's last ~24 h only (tested 2026-11-14: EZY32NC returned only that day). No date parameter, no area+time query. `/v2/point/{lat}/{lon}/{radius}` (≤ 250 nm): aircraft there now, with registration `r`, type `t` and model `desc` | Header `x-api-key` | 500 requests/day (anonymous key or none), 1,000 adhoc, 7,200 feeder; over quota: `429` + `Retry-After`, resets 00:00 UTC; no quota headers | Recordings from the last ~2 h, without the 3 GB archive download; aircraft model names |
| [ADS-B Exchange samples](https://www.adsbexchange.com/data-products/sample-data/) | The 1st of each month only; `readsb-hist` every 5 s since May 2020 | None for samples | Other dates are paid | Test data and validation sets |
| [adsb.fi open data](https://github.com/adsbfi/opendata/blob/main/README.md) | None (live only) | None | 1 req/s; personal, non-commercial; must cite adsb.fi with a link | Live mode / demo |
| [airplanes.live](https://airplanes.live/api-guide) | Not confirmed | None | 1 req/s, non-commercial, no SLA (from search listing; page refused automated fetch) | Live fallback |

**Access strategy**

1. Time window = [t_start − 2 min, t_end + 2 min] plus the clock-offset bound. Area = the search region expanded by 20 km.
2. **adsb.lol path:** look up the release tag for that UTC date through the GitHub REST API (listing releases as a fallback). The tar starts with ~0.9 GB of `heatmap/` blobs; walk the tar headers with Range requests and stream only from the first `traces/` header (~2.9 GB of 3.75 GB for a 2026 day), resuming on dropped connections. Index the traces by time and bounding box into a local DuckDB/Parquet cache, so a day is downloaded once (measured: 104 M points, 1.8 GB sorted Parquet, bbox queries 0.1–0.15 s; the DuckDB sort runs with a 2 GB memory cap and spills to disk). Warn the user about download size before starting.
3. **OpenSky Trino path:** one SQL query filtered on `hour` and lat/lon/time. Use `pyopensky` or the `trino` client.
4. **OpenSky REST path:** `/states/all` with a bbox at 5 s steps (recordings under 1 h old), or `/flights/all` (≤ 2 h interval) then `/tracks/all` per icao24.
4b. **adsb.lol API path** (`flights/adsblol_api.py`, windows up to 2 h old): `/v2/point` around the region now (radius grown by 320 m/s × the window's age, ≤ 250 nm); keep aircraft that can have been over the region in the window (current position, 1.3 × ground speed, ≥ 80 m/s), airborne first; `/1/trace/full/{hex}` per aircraft (≤ 150), cut to the window and region; registration/type/model saved to `aircraft_info.json` for the Flights table. A rejected key falls back to keyless access; 429 with a short `Retry-After` (28–42 s seen) is waited out, a long one is the daily quota. Live, keyless (2026-11-14): 16 requests, 44 s of which 42 s throttled.
5. Normalise everything to one `TrackPoint(t, icao24, callsign, lat, lon, alt_geo_m, alt_baro_m, gs_mps, track_deg, vrate_mps)`. Interpolate with a cubic spline in ECEF at 1 s steps; mark gaps over 30 s as unreliable.
6. Enrichment (optional): aircraft type and registration from the readsb trace metadata or the OpenSky aircraft database.

**adsb.lol API key.** REVOKED: adsb.lol has answered `401 api key revoked` since about 13:45 UTC on 2026-11-14, about an hour after first use; a new key is needed. The current key is kept locally in `lolsdb_key.txt` (git-ignored), not in this spec. The app reads it from the `ADSBLOL_API_KEY` environment variable (the settings dialog will store it with `keyring`).

**Licensing note.** adsb.fi and airplanes.live are non-commercial. ODbL needs attribution and share-alike for derived databases. The exported report carries attribution for every source used.

Open question: which sources the user is eligible for (OpenSky Trino?) and whether any paid API (FlightAware AeroAPI, Flightradar24 API) is acceptable for dates the free archives miss.

## Geolocation solver

The solver finds the observer position **p** = (lat, lon, h) and clock offset Δt. Using the real ADS-B trajectory, these predict Doppler curves that best match all measured tone tracks. Up to four steps run in order: gate the candidates, coarse grid, refine, assess ambiguity.

**Unknowns**

| Parameter | Prior | Notes |
| --- | --- | --- |
| lat, lon | Uniform over the search region | Main output |
| h (observer altitude) | Ellipsoidal ground height + 1.5 m from an elevation source: Copernicus GLO-30 + EGM2008 geoid (σ 6 m, surface model: rooftops in cities), else ADS-B airport ground (on-ground GNSS altitude − 4.5 m antenna; σ grows 1 m/km from the airport), else a constant (σ 100 m) | Always fitted as a correction dh with the source's σ as prior; ≥ 2 aircraft pin it to ~1 m. The DEM grid is capped at 1.5 M cells (`regional_dem`; uncapped, a box over Europe needed > 16 GB): larger regions get a coarser grid (σ grows with its spacing) for the grid search, and above 250 m spacing (~350 km box) none at all (ADS-B airport ground or unknown, as in the worldwide search); 100 m patches (0.1°, on disk) load on demand where the solver refines (`PatchedDem`). Europe: 2 s, 215 MB |
| Δt (clock offset) | Uniform within the user bound; N(0, 2 s) inside it when the start time is the phone's own tag (`com.apple.quicktime.creationdate`, Android-corrected `creation_time`), `--clock-sigma` to override | Phones are usually within 1 s of UTC; edited or re-encoded video can be off by minutes. Along a straight track position trades against time (1 s ≈ 80 m on an approach), so the prior pins the along-track position of a single pass. It also narrows the grid's clock search to ±4σ |
| ADS-B lag per aircraft | N(0, 0.5 s) (`adsb_lag_sigma_s`, `--adsb-lag-sigma`) | Each track may be early or late by its own lag: transponder position latency, feeder clock, engine vs GNSS antenna. Without it one shared clock must explain every aircraft's timing exactly; synthetic test with one track 1 s late: the stiff fit pruned a good pass and its ellipse excluded the truth by ~470σ |
| f₀ per family | from the curve fit | Nuisance parameter |
| Air (temperature, wind) | Open-Meteo for the region centre (rounded to 0.1°) and hour: 2 m temperature with the ISA lapse, 100 m wind (`--weather auto`; experimental, the default is `none` until a GPS-tagged clip settles the lapse question); `standard` = 15 °C calm with the lapse; `none` = the old still air at 15 °C; `--temp`, `--wind SPEED@FROM` override; `--fit-wind` fits the wind with the weather as prior (σ 3 m/s) | Forward model exact for a uniform wind: the wavefront drifts with the air, so it reaches the observer when ‖q‖ = c(t − τ), q = obs − p(τ) − w(t − τ), and f/f₀ = (c + q̂·w)/(c + q̂·(w − v)). c is the travel-time mean between observer and aircraft height (closed form for the ISA lapse). Checked against an independent wavefront solution to 1e-7. Synthetic: 12.6 m/s wind, still-air solver 21.5 m off, windy model 0.0 m, fitted wind exact |

**Step 1: candidate gating.** For every aircraft in the time and area window, keep it only when:

- the ground speed from ADS-B is within 25% of the fitted v, and
- some point of its track is within 1.5·d + 5 km of the search region, with CPA time inside the window ± the clock bound, and
- its altitude is at least as large as the fitted d × 0.3 (rules out ground traffic at long range).

Typically this leaves 1–5 candidates per tone family, out of hundreds of aircraft.

**Step 2: coarse grid search.** For each tone family and candidate aircraft, evaluate the forward model on a 250 m grid over the search region (a 50 km radius is ~125k cells, vectorised in NumPy and fast enough). The forward model: for each grid point and audio time t, solve the retarded emission time τ against the interpolated track, compute the line-of-sight velocity, and get the predicted f(t). f₀ is fitted analytically per cell (a 1-D linear least squares in log f). The score is the weighted χ². The assignment of families to aircraft is solved by trying every candidate per family and keeping the lowest total χ². The resulting score map is drawn on the map as a heatmap. Each (family, candidate) pair scores only the grid block around its track's bounding box plus its reach (cost follows the track, not the region), and the pairs run on one thread per CPU core (`grid_workers`, serial numba kernel per thread) and merge in pair order, so the result equals a one-thread scan. Progress is reported per pair, so a GUI cancel stops within seconds. Manchester 25 km: 33 s → 5.1 s (13.8 s on one thread); 150 km (262 candidates): 13 s, against > 35 min before.

**Step 3: refinement.** Start from the best 3 grid minima. Run `scipy.optimize.least_squares` jointly over (lat, lon, Δt, an ADS-B lag per aircraft, f₀…, optionally h and c). The 95 % ellipse is then measured, not taken from the Jacobian (`measure_ellipse`, `SolverSettings.measure_ellipse`): in 8 directions aimed along the Jacobian ellipse's own axes (whitened, so a thin valley is not missed), the observer is held at increasing distances with everything else refitted until the correlation-weighted χ² has risen by 5.99 in the covariance's units; the crossing is interpolated as a power law (the valley flattens with distance) and an ellipse is fitted through the 8 crossings. A probe that fits better re-solves from there (once); a direction still inside at 3 km makes the ellipse unbounded (grade C). Manchester: Jacobian 116 × 42 m → measured 228 × 85 m, 17 refits, +6 s. The clock and lag priors are weighted by k = (points / effective independent points) × the passes' own χ²_red, so each counts as 1/σ² against the serially correlated audio residuals, and their residuals are transformed so the soft-L1 loss keeps them quadratic. Covariance gives a 2×2 position error ellipse (95%), drawn on the map in metres.

**Step 4: ambiguity.** One straight pass leaves a mirror solution on the other side of the track. The GUI shows both. It reports a single position only when χ² separates them by Δχ² > 9, a curved track breaks the symmetry, or a second aircraft is used. Each extra flyover at a different heading shrinks the ellipse. The report lists how many independent passes were used.

**Confidence grade**

- **A:** ≥ 2 aircraft, single minimum, 95% ellipse semi-major axis < 500 m.
- **B:** 1 aircraft with a broken mirror, or an ellipse < 2 km.
- **C:** Mirror ambiguity unresolved, or the ellipse is larger than 2 km. Shown as a band, not a point.

**Optional extras (v2)**: Lloyd's-mirror fringe spacing as an extra altitude constraint; wind from ERA5; using a second recorder (two videos of the same event) for TDOA.

## Map

Recommendation: **MapLibre GL JS** (BSD-3) with OpenStreetMap data, embedded in a `QWebEngineView` and driven from Python over `QWebChannel`. Tiles come from **OpenFreeMap** (hosted, no API key) online, and from a **Protomaps PMTiles** regional extract offline. MapLibre draws thousands of track points and a solver heatmap smoothly in WebGL, and its GeoJSON sources map one-to-one onto the app's data.

| Option | Licence | Offline | Fit for this app |
| --- | --- | --- | --- |
| **MapLibre GL JS + OpenFreeMap / PMTiles** | BSD-3; OSM data ODbL | Yes (PMTiles file) | **Chosen.** Vector styling, smooth heatmaps and many tracks, terrain hillshade |
| Leaflet + OSM raster tiles | BSD-2 | Only with a pre-seeded tile cache | Fallback for machines without WebGL. Simple and light, but slower with many layers. The OSM tile servers' usage policy forbids heavy or bulk use |
| Qt Location (QML `Map`, OSM plugin) | LGPL/GPL | Partial | Native Qt, but the default OSM provider's tile sources have been unreliable and styling is limited |
| folium / ipyleaflet | MIT | No | Built for notebooks; one-way HTML output, no live Python ↔ map calls |

**Integration design**

- `map/index.html` + a pinned local copy of `maplibre-gl.js`, `pmtiles.js` and `qwebchannel.js`, shipped as Qt resources (no CDN at runtime).
- The Python `MapBridge(QObject)` exposes slots/signals. Python to JS: `setTracks(geojson)`, `setCandidates(geojson)`, `setHeatmap(geojson)`, `setSolution(lat, lon, ellipse)`, `setSearchRegion(bbox)`, `setCursorTime(t)`, `setMeasureMode(on)`. JS to Python: `regionDrawn(bbox)`, `aircraftClicked(icao24)`, `pointClicked(lat, lon)`, `measured(total_m)`, `measureToggled(on)`.
- Distance tool: a Measure toggle on the map (also View > Measure distance on map). Click to add points; each segment and the running total are labelled, and a rubber band to the cursor shows the would-be total. Double-click/Enter finishes, Backspace removes the last point, Esc clears. Metric only: "842 m" below 1 km, "1.27 km" below 10 km, "12.4 km" above. Distances are great-circle haversine on R = 6,371,008.8 m, the same as `core.geo.haversine_m`. Uses its own GeoJSON layers only, so it works with any tile source and offline. A finished line's total is logged. Labels are DOM markers and are not in the PNG snapshot.
- Create the map only after `loadFinished`; otherwise the page loads blank.
- Layers: search region (polygon), all aircraft tracks (thin, grey), gated candidates (coloured, matching the tone-family colours on the spectrogram), aircraft position at the current playhead (an icon rotated to its heading), χ² heatmap, solution point + 95% ellipse + mirror solution, and the GPS tag if any.
- Tile source setting: OpenFreeMap (default), local `.pmtiles` file, or a custom style URL. Attribution “© OpenStreetMap contributors” is always visible.
- Export: PNG snapshot through `map.getCanvas().toDataURL()`; GeoJSON/KML of the solution and tracks.

## GUI

The GUI is a single PySide6 main window with dockable panels. The user moves left to right through a five-step workflow bar (Load → Track → Fit → Flights → Locate), but every panel stays live, so going back to edit a track re-runs the later steps.

```
+--------------------------------------------------------------------------------+
| [Load] > [Track] > [Fit] > [Flights] > [Locate]     project.dfg   UTC 14:02:31 |
+---------------------------+----------------------------------------------------+
| Video (QVideoWidget)      | Map (QWebEngineView + MapLibre)                    |
|  play/pause, scrub        |  region box, tracks, heatmap, ellipse              |
+---------------------------+                                                    |
| Properties / Results      |                                                    |
|  start UTC, clock bound   |                                                    |
|  fit table per family     |                                                    |
|  candidate flights table  |                                                    |
+---------------------------+----------------------------------------------------+
| Spectrogram (pyqtgraph ImageItem) + tone tracks + fitted curves + residuals    |
|  <------------------- shared time axis, playhead synced to video ------------>  |
+--------------------------------------------------------------------------------+
| Log / progress                                                                 |
+--------------------------------------------------------------------------------+
```

**Panels**

| Panel | Widget | Main interactions |
| --- | --- | --- |
| Video | `QMediaPlayer` + `QVideoWidget` | Playhead drives every other panel's cursor; mark mute ranges |
| Spectrogram | `pyqtgraph.ImageItem` + `PlotDataItem` overlays | Zoom/pan, dB range slider, window toggle, draw a manual track, select/merge/delete tracks, assign the harmonic number |
| Fit | `QTableView` + small pyqtgraph residual plot | Per family: f₀, v, d, t_CPA ± 1σ, grade; re-fit; lock parameters |
| Flights | `QTableView` (icao24, callsign, type, CPA time, CPA range, speed match, χ²) | Tick candidates in or out; click to highlight on the map; overlay the predicted Doppler curve on the spectrogram |
| Map | `QWebEngineView` | Draw the search region, click a candidate, show the heatmap / solution / mirror, measure distances |
| Settings | `QDialog` | Data sources and credentials (stored with `keyring`), tile source, cache folder, physical constants |

**Threading.** Heavy work (decode, STFT, archive download, grid search) runs in `QThreadPool` workers that post progress signals. The UI never blocks. A long job can be cancelled. Results are cached by content hash, so re-opening a project is instant.

**Key UX rule.** The predicted curve for any candidate aircraft, at the current best position, is always drawable on the spectrogram. Being able to see that the predicted curve sits on the measured one is the main trust signal for the user.

## Architecture and stack

The core is a pure-Python package with no Qt imports, so tracking, fitting and solving can be tested headless and run from a CLI. The GUI is a thin layer on top that calls the core through worker threads.

```
dopplergeo/
  core/
    media.py        # PyAV demux, metadata (creation_time, ISO6709), resample
    spectral.py     # STFT, whitening, reassignment
    tracking.py     # DP ridge extraction (+ legacy Kalman linking), harmonic grouping
    doppler.py      # retarded-time model, single-pass fit, bootstrap
    geo.py          # WGS84/ECEF/ENU (NumPy closed form)
    elevation.py    # observer height sources: constant, grid, Copernicus GLO-30 + EGM2008 (remote COG)
    solver.py       # gating, grid search, joint refine, ambiguity, grading
    atmosphere.py   # speed of sound, ISA lapse, Atmosphere (temperature + uniform wind)
    forward.py      # numba kernel: retarded-time Newton solve + Doppler factor per point (still or moving air)
  flights/
    base.py         # FlightProvider ABC, TrackPoint, spline interpolation
    opensky_rest.py # OAuth2 client credentials, credit accounting
    opensky_trino.py
    adsblol.py      # GitHub release listing, partial download, trace parsing
    adsbx_samples.py
    adsbfi.py       # live only
    cache.py        # DuckDB + Parquet store keyed by (source, date, bbox)
    ground.py       # airport ground height from on-ground ADS-B; picks the elevation source
  gui/
    app.py, main_window.py, panels/*.py, workers.py
    map/index.html, map/bridge.py, map/vendor/ (maplibre, pmtiles, qwebchannel)
  weather.py        # Open-Meteo archive/forecast: hourly temperature and wind, cached per place and day
  cli.py            # dopplergeo locate video.mp4 --start ... --region ...
  report.py         # HTML/PDF report, GeoJSON/KML export
tests/
  synthetic/, fixtures/, test_*.py
```

| Dependency | Version (min) | Reason |
| --- | --- | --- |
| Python | 3.11 | `tomllib`, speed, PySide6 wheels |
| PySide6 (incl. QtWebEngine, QtMultimedia) | 6.7 | LGPL Qt bindings; video and web map |
| pyqtgraph | 0.13 | Fast spectrogram and plot rendering in Qt |
| numpy, scipy | 1.26, 1.13 | STFT (`ShortTimeFFT`), `least_squares`, `find_peaks`, assignment |
| PyAV | 12 | Audio demux and container metadata without a subprocess |
| soxr | 0.3 | High-quality resampling |
| librosa (optional) | 0.10 | Reassigned spectrogram |
| emcee (optional) | 3.1 | Posterior sampling for weak fits |
| pyproj | 3.6 | Geodesy, ECEF/ENU |
| rasterio (optional) | 1.3 | DEM lookup for observer altitude |
| httpx | 0.27 | HTTP/2 client with retries for all APIs |
| pyopensky / trino | latest | OpenSky Trino access |
| duckdb, pyarrow | 1.0, 16 | Local flight cache and fast spatial/time filtering |
| keyring | 25 | Store API credentials in the OS keychain |
| pydantic | 2 | Project file and settings schema |
| pytest, hypothesis | — | Tests and property-based tests on the Doppler model |

Packaging: `pyproject.toml` (hatch); `briefcase` or `pyinstaller` for Windows/Linux/macOS bundles. Linux needs the QtWebEngine system libraries and a GPU or SwiftShader for WebGL.

## Data model and files

A project is a folder with one `project.json` (pydantic schema, versioned) plus derived artifacts. The media file is referenced by path and SHA-256, never copied.

| Entity | Key fields | Stored in |
| --- | --- | --- |
| `MediaSource` | path, sha256, duration_s, sample_rate, start_utc, start_utc_source (metadata / user), gps_tag | project.json |
| `SpectroSettings` | window_s, hop_s, zero_pad, fmin, fmax | project.json |
| `ToneTrack` | id, family_id, harmonic_k, t[], f[], snr[], sigma_f[], manual | `tracks.parquet` |
| `DopplerFit` | family_id, f0, v, d, tau0, t_cpa, cov[4×4], chi2_red, grade | project.json |
| `FlightQuery` | provider, t0, t1, bbox, fetched_at, n_aircraft | project.json |
| `TrackPoint` | t, icao24, callsign, lat, lon, alt_geo_m, alt_baro_m, gs_mps, track_deg, vrate_mps | `cache/*.parquet` (shared across projects) |
| `Candidate` | icao24, family_id, cpa_time, cpa_range_m, speed_match, chi2, included | project.json |
| `Solution` | lat, lon, h, dt_clock, cov[2×2], ellipse_95, mirror{lat, lon, chi2}, grade, families_used, aircraft_used | project.json + `solution.geojson` |

**Caching**

- STFT magnitude: `cache/stft_<sha>_<settings-hash>.npy` (memory-mapped).
- Flight data: DuckDB database over Parquet partitions by provider/date/hour. Any query whose box and window are covered is served locally.
- adsb.lol raw archives: kept only if the user opts in (they are large); the extracted per-day index is always kept.

**Exports**: an HTML report with embedded plots and the map snapshot, GeoJSON and KML (solution, ellipse, candidate tracks), and CSV of tone tracks and fits. Every export carries data-source attribution.

## Testing and validation

Accuracy is proven in three layers: synthetic audio with a known answer, then real recordings with GPS tags, then blind tests. The targets below are design goals to confirm in milestone M5, not measured results.

**1. Synthetic generator** (`tests/synthetic/`). Takes a real or straight-line ADS-B track and an observer position. Renders the tone stack via the retarded-time model, with 1/r spreading, simple air absorption, a ground reflection, wind noise and AAC re-encoding. Every pipeline stage is tested against the known truth.

**2. Unit and property tests**

- Doppler model: symmetry, asymptotes f₀/(1 ∓ v/c), the closed-form retarded-time solve matches a numeric root-finder to < 1 µs.
- Fit recovers (f₀, v, d, τ₀) within 1σ in ≥ 68% of 500 noisy Monte Carlo runs (calibration check on the covariance).
- Providers: recorded HTTP fixtures (`respx`) for every API, including 429 and token-expiry handling.
- Geodesy: ENU round-trip error < 1 mm.

**3. Real-world validation set**

- 20+ phone videos recorded by the team with a GPS tag, near airports and under cruise traffic. Include days that fall on the 1st of a month, so ADS-B Exchange free samples can cross-check adsb.lol.
- Metric: horizontal error between the solution and the GPS tag; whether the true position lies inside the 95% ellipse (coverage).

**Design targets**

| Scenario | Target horizontal error (median) | Ellipse coverage |
| --- | --- | --- |
| Approach/departure pass, d < 2 km, ≥ 2 aircraft | < 300 m | 90–98% |
| Single low pass with a curved track | < 1 km | 90–98% |
| Cruise altitude (d ≈ 10 km), single aircraft | Line or mirror band only, < 3 km across | 90–98% |

**4. Performance budgets**: a 5-minute 48 kHz clip decodes and builds its STFT in under 5 s. The grid search over a 50 km radius at 250 m, with 3 candidates, runs in under 10 s on a 4-core laptop. The UI stays responsive (< 50 ms frame) throughout.

## M7 polish

M7 turns a working tool into one that can be handed to someone else: a report they can read, settings that work the same from the GUI and the CLI, the one flight source with full history, and an app that installs without Python. Each part below starts with what already exists (checked in the code on 2026-11-15), then what is missing.

### Reports and exports

**Already done.** `locate.write_outputs(r, out)` writes five files, and both `dopplergeo locate --out` and the GUI's "Export results…" (`MainWindow.export_dialog`) call it:

| File | Writer | Content |
| --- | --- | --- |
| `tracks.csv` | `cli.write_tracks_csv` | One row per tone-track point: track, family, k, manual, media and UTC time, f, SNR, σ |
| `fits.csv` | `cli.write_fits_csv` | One row per family: `FIT_COLUMNS` (grade, f₀, v, d, t_CPA ± 1σ, χ²_red, flags) |
| `candidates.csv` | `write_outputs` | Gated candidates: family, icao24, callsign, used, CPA time, CPA range, speed match, χ² |
| `solution.json` | `write_outputs` | Inputs, sources, `Solution.as_dict()`, error, GPS-tag error, worldwide places, `attribution` |
| `solution.geojson` | `locate.solution_geojson` | Solution point, 95% ellipse polygon, mirror point (`Solution.to_geojson`), candidate tracks, GPS tag; attribution as a FeatureCollection foreign member |

The GUI also writes `map.png` (`MapView.snapshot_png`, a widget grab) into the export folder, and has "Export map PNG…". Attribution strings already exist: `flights.adsblol.ATTRIBUTION` (adsb.lol, ODbL 1.0; also used by the adsb.lol API provider), `flights.opensky_rest.ATTRIBUTION` (OpenSky, Schäfer et al. 2014), `core.elevation.ATTRIBUTION_GLO30` (Copernicus DEM) and `weather.ATTRIBUTION` (Open-Meteo, CC BY 4.0). `solve_position` collects the ones a run used in `LocateResult.attribution`.

**Missing:** the HTML report, PDF, KML, and attribution on the CSV files. There is no `report.py` yet.

**`dopplergeo/report.py`** (Qt-free):

- `report_html(r: LocateResult, *, map_png: bytes | None = None, project: Project | None = None) -> str`
- `solution_kml(r: LocateResult) -> str`
- `attribution_lines(r, *, with_basemap: bool) -> list[str]`

`write_outputs(r, out, map_png=None)` gains three files: `report.html`, `solution.kml` and `ATTRIBUTION.txt`. CSV has no place for a credit line, so `ATTRIBUTION.txt` sits beside it in every export folder.

**HTML report.** One self-contained file: inline CSS, plots as inline SVG, the map as a base64 PNG. No JavaScript and no external `src`/`href` resources (plain links to the sources are fine), so it opens offline and prints as is. Sections, in order:

| Section | Content | Source |
| --- | --- | --- |
| Header | Title, media file name, app version, generated (UTC), the privacy note (`gui.main_window.PRIVACY_NOTE`, moved to a Qt-free module) | — |
| Result | Position (6 decimals) with an OpenStreetMap link, grade, 95% ellipse (a × b m, major-axis bearing), height and its source, clock correction, χ²_red and χ² excess, mirror (lat/lon, Δχ², rejected or not), unexplained families, distance to the GPS tag or reference position. With no solution: "NO POSITION" and `r.error`. Worldwide runs: the `GlobalResult.places()` table and the ambiguity flag | `Solution`, `LocateResult` |
| Inputs | Media path, SHA-256 (from the project), duration, sample rate, start UTC and its source, clock bound and clock σ, search region and its source, providers tried and the one used (cache or fetched), air model (`Atmosphere.describe()`), fitted ADS-B lags, observer-height source and `elevation_notes`, mute ranges, fit locks | `LocateResult`, `Project` |
| Tone tracks and fits | SVG plot of every track (time vs frequency, coloured by family with `gui.geojson.family_color`, which imports no Qt) with the fitted Doppler curves; the fit table (the `fits.csv` columns, without the flags) | `AnalysisResult` |
| Candidate flights | The `candidates.csv` table; assigned rows in bold; aircraft model from `aircraft_info.json` when the adsb.lol API provider saved it | `Solution.candidates`, `assignment` |
| Residuals | Per assigned family: measured − predicted frequency (Hz) against time, the predicted curve from `locate.predicted_curve` at the solution, with the ±1σ band from the track σ; one small SVG per family | `predicted_curve`, `FamilyObs` |
| Map | The GUI's map PNG when given. From the CLI: an SVG sketch in local metres around the solution (candidate tracks, solution, ellipse, mirror, GPS tag, scale bar, north arrow), labelled "no basemap". The CLI fetches no tiles: the OSM tile policy forbids that kind of use, and OpenFreeMap vector tiles need a renderer | `solution_geojson` |
| Data sources | Every line of `r.attribution`; "© OpenStreetMap contributors" and "Map tiles: OpenFreeMap" when the map PNG has a basemap; the ODbL notice below | `attribution_lines` |

SVG plots are written by a small helper in `report.py` (axes, ticks, polylines, points). matplotlib is not a dependency and stays out; pyqtgraph's exporters need Qt.

ODbL notice: when adsb.lol data was used, the report, `ATTRIBUTION.txt` and the KML say that the track and candidate data are derived from adsb.lol under ODbL 1.0, and that a redistributed database made from them must stay under ODbL.

**PDF.** Printed from the same HTML, not built separately. `gui/pdf.py: html_to_pdf(html_path, pdf_path) -> bool` loads the file in a `QWebEnginePage` and calls `printToPdf` (A4, portrait, print CSS: tables do not split across pages, SVGs scale to the page width). The GUI adds "Export report PDF…". The CLI gets `locate --pdf`, which needs the `gui` extra and runs QtWebEngine with `QT_QPA_PLATFORM=offscreen`; without PySide6 it says so and writes the HTML only. Unverified: that `printToPdf` works under the offscreen platform; test it before relying on it, and fall back to "PDF from the GUI only" if not.

**KML** (`solution.kml`, written with `xml.etree.ElementTree`, no new dependency). One `Document` with: the solution `Placemark` (point; description = grade, ellipse, clock), the 95% ellipse (polygon, the same ring as `Solution.to_geojson`), the mirror point when it exists, the GPS tag or reference position, one `LineString` per candidate track styled with `gui.geojson.aircraft_color` (KML colours are `aabbggrr`), used aircraft drawn wider. Tracks are `clampToGround`: ADS-B geometric altitude is above the WGS84 ellipsoid and KML's `absolute` is above mean sea level, so absolute altitudes would be off by the geoid height (tens of metres). The `Document` description carries the attribution lines.

**GUI changes.** `export_dialog` grabs the map first, passes the PNG bytes to `write_outputs`, still writes `map.png`, and offers to open `report.html`. "Export report PDF…" writes the folder and then the PDF. Export is enabled only after a Locate result exists (as now).

### Settings and keychain

**Already done** (M6, checked in `gui/settings.py`): the Settings dialog stores the adsb.lol key and the OpenSky client id and secret in the OS keychain (`keyring`, service `dopplergeo`), and the map style URL and flight-cache folder in `QSettings`. `apply_saved_credentials()` copies stored keys into `ADSBLOL_API_KEY`, `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` at GUI start; a variable already set in the shell wins. Without a usable keychain the dialog says the keys apply to this session only. The Properties panel holds the per-project Locate inputs (start override, clock bound, region, provider checkboxes, height source, fixed height), saved in `project.json`.

**Missing:**

1. **The CLI cannot read stored keys.** The keyring helpers live in `gui/settings.py`, which imports PySide6 at module level. Move `SECRETS`, `load_secret`, `save_secret` and `apply_saved_credentials` to a Qt-free `dopplergeo/credentials.py`; `gui/settings.py` imports them from there. `cli.main()` calls `apply_saved_credentials()` before dispatch (shell variables still win). New subcommand:

   ```
   dopplergeo credentials list            # name, environment variable, stored yes/no, active source (env / keychain / none); never values
   dopplergeo credentials set NAME        # value read with getpass, never from argv (shell history)
   dopplergeo credentials clear NAME
   ```

   Add `github_token` → `GITHUB_TOKEN` (adsb.lol release lookups; already read by `AdsbLolProvider`) to `SECRETS` and the dialog, and the OpenSky Trino entries (next section).

2. **The GUI Locate step lacks the CLI's air-model and timing inputs.** These are per-recording inputs, so they go in the Properties panel and `Project`, not in the Settings dialog:

   | Input | CLI | GUI today | M7 |
   | --- | --- | --- | --- |
   | Air model | `--weather auto\|standard\|none`, default `auto` | Not passed: `solve_position`'s default `weather="none"` applies, so the GUI solves in still air at 15 °C while the CLI uses Open-Meteo | Combo, default `auto`; `Project.weather` |
   | Temperature / wind override | `--temp`, `--wind SPEED@FROM` | Missing | Optional fields; `Project.temp_c`, `Project.wind` |
   | Fit wind | `--fit-wind` | Missing | Checkbox; `Project.fit_wind` |
   | Clock prior σ | `--clock-sigma`, default auto (`auto_clock_sigma`: 2 s for a phone's own time tag) | Auto, not editable | Spin box with an "auto" special value; `Project.clock_sigma_s: float \| None` |
   | ADS-B lag σ | `--adsb-lag-sigma`, default `SolverSettings.adsb_lag_sigma_s` (0.5 s) | Default, not editable | Spin box; `Project.adsb_lag_sigma_s` |
   | Excluded aircraft | Missing | Flights table checkboxes (`exclude_aircraft`) | CLI `--exclude ICAO24` (repeatable) |
   | Provider order | `--providers` list | Fixed-order checkboxes | Add `opensky-trino` |

   New `Project` fields have defaults, so older `project.json` files load unchanged; their saved solution stays on screen until the next Locate (which then uses `weather=auto`). `MainWindow.run_locate` passes all of them to `solve_position`, as `cmd_locate` does.

3. **Cache location.** The GUI's cache folder is a `QSettings` value; the CLI keeps `--cache` and does not read `QSettings` (no shared settings file in M7). `flights.cache.default_cache_dir()` becomes platform-aware: `$XDG_CACHE_HOME` or `~/.cache` on Linux (unchanged), `%LOCALAPPDATA%\dopplergeo\flights` on Windows. Weather, DEM tiles and the analysis cache already live beside it (`cache.root.parent`).

Not planned: the "physical constants" row in the GUI Settings table. The speed of sound comes from the air model (`Atmosphere`), which item 2 exposes per project.

### OpenSky Trino provider

**Interface it plugs into** (`flights/base.py`). A provider subclasses `FlightProvider`, sets `name` and `attribution`, and implements `can_serve(query, now) -> str | None` (None = can try, else the reason) and `fetch(query, cache) -> list[Coverage]`. `fetch` writes rows in the `SCHEMA` columns (`TrackPoint` fields) with `cache.write_part(table, [Coverage(name, t0, t1, bbox)], meta=...)`. `FlightService.get` serves any query that one provider's registered coverage already answers from the DuckDB/Parquet cache, else tries providers in order, skipping those whose `can_serve` gives a reason and those that raise `ProviderError`, and collects the reasons for `NoFlightData`. The existing providers follow this: `OpenSkyRestProvider` (OAuth2 client credentials, credit budget, `states`/`tracks` modes by window age), `AdsbLolProvider` (one whole day per part, confirmation before download), `AdsbLolApiProvider` (window ≤ 2 h old). `opensky_trino.py` is already listed in the architecture tree but does not exist.

**`flights/opensky_trino.py`: `OpenSkyTrinoProvider`**

- `name = "opensky-trino"`, `attribution = opensky_rest.ATTRIBUTION` (the same OpenSky citation).
- `can_serve`: a reason when no Trino credentials are configured ("OpenSky Trino needs an approved account"); a reason when the window ends in the future or ends less than about an hour ago (state vectors reach the historical database with a delay; the exact delay is unverified). Otherwise None: the history goes back to 2013 with no retention limit.
- `fetch`: one SQL query per call, rows mapped to `TrackPoint`s, one cache part with `Coverage(name, query.t0, query.t1, query.bbox)` and `meta={"rows": n, "hours": [...], "query_s": ...}`. An empty result is still cached: it answers the query (no aircraft there then).

**Query** (`state_vectors_data4`; the table name and the `hour` partition rule are confirmed in the Flight data table; the column list below is from pyopensky's usage and unverified against the current Trino docs):

```sql
SELECT lastposupdate, icao24, callsign, lat, lon, geoaltitude, baroaltitude,
       velocity, heading, vertrate, onground
FROM state_vectors_data4
WHERE hour >= {h0} AND hour <= {h1}            -- partition column: epoch seconds at the start of each hour
  AND time >= {t0} AND time <= {t1}
  AND lat BETWEEN {lat_min} AND {lat_max}
  AND lon BETWEEN {lon_min} AND {lon_max}
  AND lat IS NOT NULL AND lastposupdate IS NOT NULL
```

`h0 = floor(t0 / 3600) * 3600`, `h1 = floor(t1 / 3600) * 3600`, so a window crossing an hour boundary reads both partitions. Every value in the SQL is a number formatted by the provider (`int` for times, 6-decimal floats for coordinates); no user text is ever put in the SQL. A test asserts that every generated statement contains the `hour` bounds.

**Mapping to `TrackPoint`.** As in `opensky_rest.parse_states`: `t = lastposupdate` (the time of the position, not of the state vector), `icao24` lower case, callsign stripped (empty → None), `alt_geo_m = geoaltitude`, `alt_baro_m = baroaltitude`, `gs_mps = velocity`, `track_deg = heading`, `vrate_mps = vertrate`, `on_ground = onground`; units are already SI (unverified for the Trino table; the REST state vectors are m and m/s). State vectors repeat a position until the next update, so rows are deduplicated on `(icao24, lastposupdate)` before writing.

**Client and auth.** Use the `trino` Python client (DB-API) with an injectable `connect` factory, so tests never open a connection. Host `trino.opensky-network.org`, port 443, HTTPS, catalog `minio`, schema `osky`: these come from pyopensky's defaults and are **unverified**. Auth is OAuth through OpenSky's Keycloak realm (`auth.opensky-network.org`, as for REST), with the token passed to `trino.auth.JWTAuthentication`; which grant Trino accepts is **unverified**. pyopensky has used a password grant (username and password, client id `trino-client`); whether the REST API's client-credentials pair also works for Trino is unknown. M7 therefore stores `opensky_username` and `opensky_password` (`OPENSKY_USERNAME`, `OPENSKY_PASSWORD`) in the keychain next to the client id and secret, and the token function picks the grant from what is configured. Read pyopensky's current source and the Trino page before writing this code, and correct this paragraph. pyopensky itself is not a dependency (it pulls in pandas); it is the reference. New optional extra: `trino = ["trino>=0.330"]`.

**Errors.** No credentials → `can_serve` reason. 401/403 at token or query time → `ProviderUnavailable("OpenSky Trino refused the account: access is limited to approved research, government and aviation-authority users")`. Queued or slow queries: a 300 s timeout, then `ProviderError`; a cancelled GUI job calls `cursor.cancel()`. The token is refreshed before expiry, as in `OpenSkyRestProvider._bearer`.

**Eligibility caveat.** Trino access needs an application that OpenSky reviews, and is open only to research, government and aviation-authority users. The user's eligibility is still an open question (Flight data section). The provider is therefore built and tested offline, is in the default order only after the adsb.lol API (`adsblol-api,opensky,opensky-trino,adsblol`), and is skipped silently without credentials. When it is configured, it runs before the 3 GB adsb.lol day download.

**Tests** (`tests/test_opensky_trino.py`, offline):

- Recorded rows: `tests/fixtures/opensky_trino_rows.json`, the column list plus rows from one real query (recorded once by an eligible account, with the query text and date), or synthetic rows from `core.synthetic` in the same shape until such an account exists; the fixture says which it is.
- SQL builder: hour bounds for windows inside one hour and across a boundary; bbox and time bounds; no non-numeric interpolation.
- Mapping: duplicate positions removed, nulls kept as None, callsign stripped, `t = lastposupdate`.
- Cache: a fetch writes one part; the same query is then served from the cache with no `connect` call.
- Auth: token request and refresh with `respx` (httpx); 401/403 become `ProviderUnavailable`; no credentials → `can_serve` reason and `FlightService` moves to the next provider.

### Packaging for Linux and Windows

**Already in place.** `pyproject.toml` (hatchling) defines `dopplergeo = "dopplergeo.cli:main"` and the GUI script `dopplergeo-gui = "dopplergeo.gui.app:main"`; extras `gui` (PySide6, pyqtgraph, keyring), `dem` (rasterio), `test`. The map page loads from files beside `gui/map/bridge.py` (`MAP_DIR`, `index.html`, `vendor/`: MapLibre GL JS 4.7.1, pmtiles.js, qwebchannel.js), not from Qt resources. Nothing else for packaging exists.

**Tool choice.**

| | PyInstaller | Briefcase |
| --- | --- | --- |
| How | Analyses imports, freezes into a folder or one file | Installs the wheels into its own Python in an app template |
| PySide6 / QtWebEngine | Hooks ship with PyInstaller (QtWebEngineProcess, `.pak` resources, locales) | Works as installed wheels; no hooks needed |
| numba / llvmlite, duckdb, pyarrow, PyAV | Hooks in `pyinstaller-hooks-contrib`; data and metadata may need listing | Installed normally |
| Output | Folder or single exe; installers are a separate step | MSI (Windows); Flatpak or system packages (Linux; AppImage support status unverified) |
| Fit | Widely used for PySide6 + WebEngine apps; one config for both OSes | Native installers, fewer surprises with compiled packages; less used with QtWebEngine |

Recommendation: **PyInstaller, one-folder mode**, with `packaging/dopplergeo.spec` producing both executables (CLI and GUI) in one folder. One-file mode would unpack about 300 MB (QtWebEngine) to a temp folder at every start. Ship v1 as a `.tar.gz` (Linux) and a `.zip` (Windows); an installer (Inno Setup, AppImage) is a later step. Each OS builds its own bundle (PyInstaller does not cross-compile); the Linux build runs on the oldest distribution to support (Ubuntu 22.04), since the bundle needs a glibc at least as new as the build machine's. Revisit Briefcase if a signed MSI or a Flatpak becomes a requirement.

**What the bundle must contain**

| Item | How |
| --- | --- |
| QtWebEngine (process, resources, locales), QtWebChannel, QtMultimedia with its FFmpeg plugin (video panel) | PySide6 hooks; check the video plays in the smoke test |
| `dopplergeo/gui/map/index.html` and `gui/map/vendor/*` (MapLibre, pmtiles, qwebchannel, licence) | `datas`, at the same relative path so `MAP_DIR` resolves |
| numba + llvmlite | contrib hooks; the jitted modules also as `.py` source files (see below) |
| duckdb, pyarrow, PyAV (bundled FFmpeg), soxr, scipy | Hooks; DuckDB needs no extensions (`read_parquet` is built in), so nothing is downloaded at run time |
| keyring and its backends (SecretService via jeepney on Linux, Windows Credential Locker) | `copy_metadata("keyring")` (backends are found through entry points) and hidden imports of the backend modules |
| rasterio (DEM heights) with GDAL and PROJ data | Bundled so `--elevation auto` works; check a remote COG read in the smoke test |
| Licences | `LICENSES/` with Qt/PySide6 (LGPL; one-folder mode keeps the Qt libraries replaceable), MapLibre (BSD-3), FFmpeg via PyAV, and the Python packages (generated at build) |

**numba cache in a frozen app.** `tracking._ridge_dp_frames` and the `core/forward.py` kernels use `@numba.njit(cache=True)`. numba writes its cache next to the source file (`__pycache__`) and needs that file to exist; in a frozen app the source is missing and the install folder may be read-only (Program Files). So:

- Both entry points set `NUMBA_CACHE_DIR` to `<user cache>/dopplergeo/numba` before anything imports numba, unless the variable is already set (`dopplergeo/_runtime.py`, called first in `cli.main` and `gui.app.main`; harmless when not frozen).
- `dopplergeo.core.tracking` and `dopplergeo.core.forward` are collected as source as well as bytecode (PyInstaller `module_collection_mode` `"pyz+py"`), so numba can find the file it keys the cache on. To verify: that numba's user-provided cache locator accepts these paths in a frozen app.
- The first run compiles the kernels (time not yet measured on a clean machine); the GUI shows "Preparing the analysis engine (first run only)" during the first Track step. A prebuilt cache is not shipped: numba's cache is specific to the CPU and the source file's timestamp.
- `parallel=True` uses numba's built-in `workqueue` threading layer; TBB and OpenMP are not bundled.

**Multiprocessing.** `pipeline.analyze_windowed` can use worker processes with the `forkserver` start method, which does not exist on Windows. On Windows use `spawn`, and call `multiprocessing.freeze_support()` first in both entry points (needed by frozen apps that start processes). The default stays 1 process.

**Caches and user data.** Nothing is written inside the bundle. The flight cache (`index.duckdb` and Parquet parts), the analysis, weather and DEM caches and the numba cache all live under `default_cache_dir().parent` (Linux `~/.cache/dopplergeo/`, Windows `%LOCALAPPDATA%\dopplergeo\`, see Settings item 3). DuckDB allows one process per database file: the GUI and a CLI run on the same cache take turns (both already release it before solving).

**WebGL on Linux.** MapLibre needs WebGL. On machines without a usable GPU (VMs, some remote desktops), Chromium falls back to SwiftShader; recent Chromium versions only do so with `--enable-unsafe-swiftshader` (whether the Chromium in Qt 6.12's WebEngine needs that flag is unverified). `--disable-gpu` turns WebGL off completely (seen in the GUI tests). M7 adds:

- a WebGL check in `index.html` at start; if no context can be created, the map area shows "The map needs WebGL" and the log names the fix;
- `dopplergeo-gui --software-gl`, which sets `QTWEBENGINE_CHROMIUM_FLAGS=--enable-unsafe-swiftshader` (appended to any flags already set) before the `QApplication` is created.

Also on Linux: Chromium's sandbox needs unprivileged user namespaces, which Ubuntu 24.04's AppArmor restricts for unpackaged binaries. If QtWebEngine fails to start for that reason, the documented workaround is `QTWEBENGINE_DISABLE_SANDBOX=1` (check on a clean 24.04). Windows uses ANGLE (Direct3D) for WebGL and needs no flags.

**Smoke test, per platform.** Two built-in commands, run from the unpacked bundle on a clean VM with no Python and no development packages:

- `dopplergeo selftest` (no network): imports every module; runs both numba kernels on a small synthetic input and checks the cache folder filled, then runs them again and checks the second run is much faster; runs a synthetic end to end (`core.synthetic` audio + a straight-line track, solve) and checks the error is < 100 m; writes and queries a `FlightCache` in a temp folder; writes `report.html` and `solution.kml` and parses the KML; prints the keyring backend (it must not be `fail.Keyring` on Windows). Exit code 0 or 1.
- `dopplergeo-gui --selftest`: opens the main window, waits for the map page's `loadFinished` and a JS check that MapLibre created a WebGL context, plays one second of a short WAV rendered with `core.synthetic.render_audio`, writes a map PNG and a report PDF to a temp folder, quits with 0 or 1 within 60 s.

Then by hand, once per release: open the Manchester project, run all steps with network access, export the report, and compare the solution with the last recorded result.

## Milestones and risks

The build order is set by risk: prove the physics on synthetic data before any GUI or API work. Each milestone ends with a checklist that must pass.

**Bring-up checklist**

- [x] **M1 Core physics:** Doppler model, synthetic generator, single-pass fit. Fit recovers synthetic parameters within 1σ; property tests green.
- [x] **M2 Audio pipeline:** PyAV ingest, STFT, ridge tracking, harmonic grouping; a CLI that dumps tracks and fits to CSV. Works on 5 real flyover clips. (Only true since M5: the M2 Kalman linker failed on all real clips; see M5.)
- [x] **M3 Flight data:** OpenSky REST + adsb.lol providers, cache, normalised `TrackPoint`s. A one-day adsb.lol download is indexed and a bbox query returns in < 1 s.
- [x] **M4 Solver:** gating, grid, refine, ambiguity, grading. Synthetic end-to-end error < 100 m with no noise.
- [ ] **M5 Validation:** real GPS-tagged set; tune the defaults; check the design targets.
  - [x] Real flyover audio (Freesound): tracker rebuilt as DP ridge extraction; fit gets realistic σ (0.2%·f), covariance inflated for serially correlated residuals (1 + 2Σρ), multi-start over CPA time/distance chosen by median |residual|, d lower bound 20 m, fits pinned at a bound graded C. LAX turboprop (threshold): grade A, v 59.3 m/s, d 62 m, two independent tone families agree. ATR 72 landing: grade B, d 437 ± 42 m vs recordist's ~424 m. Jets (B777, IL-76) are mostly broadband: no A/B fit (honest).
  - [x] A recording with known time and place, run end to end through `locate`. Test clip: Freesound 868926 (Haughton Green, 2026-09-03 ~11:00 UTC, 75 min, geotag "not exact", 128 kbps preview), under the Manchester 23R/05L approach: ADS-B shows an airliner every 2–3 min at ~900 m slant, 77–91 m/s. First run (excerpt 300–780 s, region 25 km around MAN, clock ±600 s): **failed**, solution 34 km off, grade C (honest). The spectrogram shows a clean approach (three tones falling 500→410, 340→260, 170→115 Hz) masked at the CPA by broadband jet noise. Tracker gaps: (1) steady local hum lines (130/330/390 Hz) become no-Doppler families pinned at v = 30 m/s; drop tracks that do not sweep; (2) harmonics broken by the broadband CPA are not grouped, so each piece fits alone without its CPA. Also found: grade B allowed CPAs far outside the data and fits at d = 21–22 m (now C), and the solver did thousands of refinements with 14 weak families (now capped: `max_families`, `max_readmit_candidates`; 447 s).
    - Other clips tried: Whitehorse 737 take-off (820320) and Gergueil (802703) have no trackable tone.
    - Tracker fixes since: steady lines removed before ridge extraction (each bin's running 150 s time median subtracted, `TrackerSettings.stationary_window_s`) and tracks sweeping < 1% dropped (`min_sweep_rel`); excerpt 2 (700–1100 s) went from a dozen families to 3, one grade B (v 84.0 m/s, d 734 m) that matches EZY32NC passing the geotag at 82.6 m/s, 906 m. Solver fixes: each strictly gated family seeds the refinement at its own best grid cell; hypotheses rank by strict families explained before total families; a solution leaving a strict family with candidates unexplained is graded C; refinements outside the search region (+2 km) are rejected; the CLI and GUI release the cache database before solving.
    - Results (region 25 km around MAN, geotag withheld): excerpt 2 with clock ±600 s: right aircraft (EZY32NC), **6.6 km off**, clock −84 s, grade C. With clock ±30 s (diagnostic only; the true offset, about −5 s, is known from matching ADS-B): **956 m off**, grade C. The 6.6 km is the single-pass clock/along-track degeneracy (78 s × 83 m/s along the approach line), expected for one straight pass with an unknown clock. Excerpt 1: 19.5 km off, grade C; its harmonic family is still corrupted (the k = 2 ridge slides onto the k = 1 tone in the broadband roar at the CPA; a gap/jump split broke the close-pass regression, so it is open).
    - Joint multi-pass solve (2026-11-14), **accepted as done at 543 m**: the full 75-min file is analysed in overlapping windows (`analyze_windowed`: 600 s windows, 150 s overlap; a pass belongs to the window whose core holds its fitted CPA, so a pass cut by an edge is fitted once, whole; optional worker processes). 52 families, 3 graded B. Solver changes: (1) the grid rescores every family at one clock offset per cell (median of the strict families' own offsets), `common_clock`; (2) when ≥ 2 families are graded A/B and not loose, only those are solved (`reliable_only_min`); (3) the position covariance weights each family by 1/`corr_factor` of its single-pass fit (effective independent points); (4) the grade judges fit quality as `chi2_excess` = joint χ²_red / the families' own single-pass χ²_red (limit 3) instead of a fixed χ²_red 6: good real passes fit only to χ²_red 4–8 on their own.
    - Result (fair: region 25 km around MAN, clock ±600 s, geotag withheld): **543 m** from the geotag, all three aircraft right (RYR60SF, EZY32NC, TOM5TC), clock +1.9 s, χ² excess 1.2, mirror 5.7 km off and rejected (Δχ² 917). Diagnostics: all 52 families before change (2): 6.1 km, 4 of 6 aircraft right; without (1): 10.1 km, all wrong; the 4 correctly matched families alone: 492 m. Run through the same functions as `locate` on a saved analysis (made before the window-ownership fix), not the CLI itself.
    - Speed (2026-11-14): the ridge tracker's DP (~80% of analysis time) is a numba kernel (`tracking._ridge_dp_frames`, threads over bins), bit-identical to the former numpy version, which was memory-bandwidth bound (a 49 × 10k temporary per frame; 4 worker processes each ran ~4× slower, so windowed analysis defaults to 1 process). Ridge tracking on a 400 s excerpt 20.1 s → 3.8 s; the full 75-min Manchester file 897 s → 223 s. Re-run on that fresh analysis with all changes: 543 m, same three aircraft.
    - Tracker recall (2026-11-14), benchmark `examples/manchester_recall.py` against the 29 aircraft ADS-B shows within 3 km (`data/freesound/manchester_passes.json`). Baseline: 15 found, 5 with v and d right, 2 right and graded B. The steady ~345 Hz line is mostly the airliner stream's approach asymptote (an aircraft every 2-3 min), not local hum; still, switching the hum filter off scored worse (3 right, 2 false B), as did 60 ridges per window (no change) and a more sensitive DP (21 found but 0 good; with trimming 36 km off). Fix kept: when a family fit is not grade A, tracks are cut at sudden steps (> 2.5 % in 1.5 s between flat parts: ridges jumping onto another tone after the CPA) if the trimmed fit still covers its CPA and grades better or halves chi2_red (`pipeline._trim_steps`; real close passes keep their CPA drop because cutting it loses the CPA). Result: 6 right, 3 good (RYR506G now v 79.4 / d 747 vs 79.0 / 713). Also: refine drops a family with < 10 points in range of its aircraft at the seed (its zero Jacobian column made the covariance singular). Manchester end to end: **211 m, grade A**, ellipse 176 x 35 m, 6 aircraft, mirror rejected (Δχ² 384). Analysis 223 s -> 267 s.
    - Open: graded A, but the 95% ellipse (372 × 42 m, major axis 142°, across the approach line) does not contain the geotag: still overconfident along track. Every matched pass is heard ~7.6 s later than the geotag predicts (5 passes, ±0.3 s): either the clock offset is ~−13 s or the geotag ("not exact") is ~600 m along the approach line, so this clip cannot confirm the 300 m target; that needs the GPS-tagged set. The tracker yields usable tones for only ~8 of the 38 approaches ADS-B shows overhead. The hum filter (150 s running median) removes a pass's long flat asymptote when a window holds ~100 s of it. Harmonic slide at broadband CPAs still open.
    - Timing model (2026-11-15): per-aircraft ADS-B lag (prior N(0, 0.5 s)) and a clock prior for phone-stamped files. Manchester, fair run on a fresh analysis (`examples/manchester_timing.py cache/analysis/manchester_full.pkl`): no lag 213 m (ellipse 136 × 35 m, geotag m² 193), σ 0.5 s 200 m (m² 136), σ 1 s 190 m (m² 117); all grade A, 6 aircraft, clock −0.2 s, fitted lags −0.6…+0.8 s. The geotag stays ~11σ outside. The miss is along the approach line (solution − geotag: +194 m along, −51 m across; the ellipse is narrow in that direction, 42 m), matching the clip's known ~7.6 s late passes (clock or "not exact" geotag). (A first reading of the ellipse's major-axis bearing as the error direction was wrong.) The clock prior is tested only on synthetic data (no clip here has a phone clock).
    - Air model and ellipse (2026-11-15): moving-air forward model (exact uniform wind, ISA-lapse path-mean c, Open-Meteo weather; numba kernel `core/forward.py`, grid 22–25× and refine calls ~3× faster, Manchester solve 452 s → 87 s, same result). Manchester: still air 200 m; lapse only 489 m; Open-Meteo weather (20.2 °C, 6.6 m/s from 254°) 424 m; wind fitted 7.7 m/s from 260° (agrees with Open-Meteo); c fitted ±1.7 m/s changes nothing. The lapse moves the solution ~340 m across the approach while the temperature level does not, and the audio fits the lapse slightly better (χ²_red 4.76 → 4.57). Profile (`examples/manchester_profile.py`: observer held along each ellipse axis, everything else refitted): along the track the data rise is 5–30× flatter than the ellipse claims (−200 m: 14 vs 139; the clock trades ~1.2 s per 100 m), across it is lopsided; the geotag sits ~4σ from either model's minimum by the data, not 11σ. The 95% ellipse from the Jacobian at the minimum is overconfident; now measured (see Step 3): 228 × 85 m; the geotag is still outside (m² 33), because the data themselves disfavour its along-track position (profile rise 14 at −200 m), i.e. the clip's 7.6 s systematic, not the ellipse. Weather stays opt-in (`--weather none` default) until a GPS-tagged clip decides the lapse.
  - [x] Android `creation_time` is the stop time, not the start (seen on a real phone video): corrected by the media duration when `com.android.version` is present.
- [x] **M6 GUI:** main window, spectrogram, video sync, MapLibre map with bridge, workers, project save/load.
  - Remaining M6 items done (2026-11-14): spectrogram editing row (move the selected track to another family or a new one, set its harmonic k, hand-draw a track: clicks snap to the strongest bin within ±4 Hz, filled every 0.2 s, manual tracks never trimmed); Fit panel locks for v / d (`fit_doppler(fixed=...)`, exempt from the at-bound grade rule, saved as `Project.fit_locks`); "Mute from here / End mute here" in the video panel; Settings dialog (adsb.lol key and OpenSky client id/secret in the OS keychain via `keyring`, map tile style, cache folder; keys reach providers through their environment variables, a variable set in the shell wins); analysis cache (`pipeline.analyze_file_cached`: tracks and fits stored in `<cache>/../analysis/` keyed by media SHA-256 + settings + mutes + a hash of the analysis source files; a reopen decodes and rebuilds the display spectrogram only). Re-fits after user edits fit exactly as labelled (no step trimming).
  - End-to-end GUI tests (`tests/test_gui_workflow.py`) pass headless (2026-11-14). The hang was a modal "Save the current project first?" `QMessageBox` on window close: the last jobs' results arrive during shutdown and mark the window dirty again (tests now answer every message box). Also fixed: reopening a project left it marked dirty (the background re-analysis marked it after `open_project` returned), and a segfault at exit (windows whose QtWebEngine pages outlived the QApplication; tests destroy their windows, the app sets `WA_DeleteOnClose`).
- [ ] **M7 Polish:** reports and exports, settings and keychain, OpenSky Trino provider, packaging for Linux and Windows. Details in "M7 polish".
  - [ ] **M7.1 Report and KML:** `report.py` writes a self-contained `report.html` and `solution.kml` beside the existing CSV/JSON/GeoJSON; `ATTRIBUTION.txt` in every export folder. Check: on the `test_locate` synthetic scenario the report has every section and every attribution line, the HTML references no `http(s)` resource, the KML parses and holds the solution, ellipse and mirror placemarks.
  - [ ] **M7.2 GUI export and PDF:** "Export results…" embeds the map PNG in the report; "Export report PDF…" prints it through QtWebEngine; `locate --pdf` does the same headless. Check: a GUI workflow test exports a folder whose report shows the map image, and a PDF of ≥ 1 page.
  - [ ] **M7.3 Settings parity:** credentials move to a Qt-free `credentials.py` that the CLI also reads; `dopplergeo credentials` subcommand; the GUI Locate inputs gain weather, temperature/wind override, fit wind, clock σ and ADS-B lag σ. Check: a key saved with `dopplergeo credentials set` reaches `locate` with no environment variable; GUI and CLI with the same inputs give the same solution (same lat/lon to 1 m).
  - [ ] **M7.4 OpenSky Trino provider:** `flights/opensky_trino.py`, `name = "opensky-trino"`, cached like the others. Check: offline tests on recorded rows pass; every generated SQL filters on `hour`; with no access the provider is skipped with a clear reason; one live query by an eligible account (if one is available) is compared with adsb.lol for the same window.
  - [ ] **M7.5 Packaging:** PyInstaller one-folder builds for Linux x86-64 and Windows x64, built by a script in `packaging/`. Check: `dopplergeo selftest` and `dopplergeo-gui --selftest` pass on a clean Ubuntu 24.04 VM and a clean Windows 11 VM with no Python installed.

**Risks and limitations**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| No clean tone in the audio (wind, speech, music) | No fit at all | Manual track drawing; denoise; report “insufficient signal” instead of guessing |
| Video timestamp wrong or stripped (messengers re-encode) | Wrong candidate set | Δt as a free parameter with a wide bound; let the user scan the window |
| Aircraft not in ADS-B (military, some GA, coverage holes) | No match | Show residual “unexplained” families; allow manual straight-line track entry |
| adsb.lol archive size per day | Long downloads, disk use | Partial download where possible, stream extraction, warn before fetch, keep only the index |
| Mirror ambiguity on single straight passes | Two answers | Show both; demand a second aircraft for grade A |
| Wind and temperature errors | 1–3% speed bias, km-scale bias at range | Open-Meteo/ERA5 lookup; widen the covariance |
| API terms change (OpenSky auth moved to OAuth2; adsb.lol may require keys) | Provider breaks | Provider abstraction, recorded fixtures, clear errors |
| Privacy | Geolocating other people's videos | Show a notice in the app; no batch scraping features |

## Sources

**Opened and checked for this spec** (facts in the Flight data table come from these):

- [OpenSky REST API](https://openskynetwork.github.io/opensky-api/rest.html): history limits, credits, OAuth2.
- [OpenSky Trino historical data](https://openskynetwork.github.io/opensky-api/trino.html): eligibility, `state_vectors_data4` schema and retention.
- [adsb.lol historical data](https://www.adsb.lol/docs/open-data/historical/): daily releases, ODbL; [readsb trace JSON format](https://github.com/wiedehopf/readsb/blob/dev/README-json.md#trace-jsons).
- [adsb.fi open data README](https://github.com/adsbfi/opendata/blob/main/README.md): endpoints, 1 req/s, terms.
- [ADS-B Exchange sample data](https://www.adsbexchange.com/data-products/sample-data/): free 1st-of-month data, URL layout.

**Found by search, not opened; read before relying on them:**

- Doppler / acoustic flight-parameter estimation: B. G. Ferguson and K. W. Lo, IEEE Journal of Oceanic Engineering papers on turboprop blade-rate instantaneous frequency (single sensor and arrays), via the [ASA 1995 abstract](https://auditory.org/asamtgs/asa95stl/4pSP/4pSP10.html). A 2022 single-hydrophone 3-D extension in [Frontiers in Marine Science](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.996493/full). A DRDO paper on UAV harmonics with STFT and a least-squares fit for altitude and speed ([Defence Science Journal](https://publications.drdo.gov.in/ojs/index.php/dsj/article/download/2238/1198/7489)). Tong (2016), *Estimation of low-altitude moving target trajectory using single acoustic array* ([catalogue record](https://katalog.hcu-hamburg.de/vufind/Search2Record/OLC1975452658)).
- [airplanes.live API guide](https://airplanes.live/api-guide): refused automated fetch; 1 req/s and non-commercial terms come from the search listing.
- Map stack: [MapLibre Leaflet migration guide](https://maplibre.org/maplibre-gl-js/docs/guides/leaflet-migration-guide/), [Protomaps FAQ](https://protomaps.com/faq), [PMTiles JS README](https://cdn.jsdelivr.net/npm/pmtiles@4.5.0/README.md), [pyqtlet implementation notes](https://pyqtlet.readthedocs.io/en/latest/technical.html) (load-order pitfall), [QWebEngineView docs](https://doc.qt.io/qtforpython-6/PySide6/QtWebEngineWidgets/QWebEngineView.html).

The physics in this spec is the standard moving-source Doppler and retarded-time formulation. The formulas are written from first principles, not copied from a source.
