"""dopplergeo locate, end to end through audio: rendered AAC file -> tone
tracking -> Doppler fits -> cached ADS-B -> solver -> GPS coordinates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.cli import main
from dopplergeo.core.geo import haversine_m
from dopplergeo.core.media import write_audio_file
from scenario import H_OBS, LAT0, LON0, OBS_LAT, OBS_LON


def run(recording, *extra):
    media, cache_dir, tmp = recording
    out = tmp / "out"
    code = main(["locate", str(media), "--near", f"{LAT0},{LON0}", "--radius", "15", "--providers", "adsblol",
                 "--cache", str(cache_dir), "--out", str(out),
                 "--weather", "none",  # the scenario is rendered in still air at 15 deg C (and no network)
                 *extra])
    return code, out


def test_locate_recovers_gps_coordinates(recording, capsys):
    code, out = run(recording, "--height", str(H_OBS))
    text = capsys.readouterr().out
    assert code == 0, text
    doc = json.loads((out / "solution.json").read_text())
    sol = doc["solution"]
    err = float(haversine_m(sol["lat"], sol["lon"], OBS_LAT, OBS_LON))
    assert err < 50.0, text
    assert doc["start_utc_source"] == "creation_time"
    # The MP4 muxer stores the location tag to 4 decimals (~11 m), so allow for that.
    assert doc["gps_tag_error_m"] == pytest.approx(err, abs=15.0)
    assert set(sol["assignment"].values()) == {"3c6444", "4840d6"}  # an aircraft may give >1 family
    assert sol["grade"] in ("A", "B")
    assert "POSITION" in text and "openstreetmap.org" in text
    gj = json.loads((out / "solution.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"solution", "ellipse_95", "candidate_track", "gps_tag"} <= kinds
    assert (out / "candidates.csv").read_text().count("\n") >= 3


def test_locate_reports_missing_inputs(recording, tmp_path, capsys):
    media, cache_dir, _ = recording
    bare = tmp_path / "bare.wav"
    import av  # noqa: F401  (codec availability is already exercised above)
    from dopplergeo.core.media import load_audio
    clip = load_audio(media)
    write_audio_file(bare, clip.samples, clip.sample_rate, codec="pcm_s16le")
    assert main(["locate", str(bare), "--near", f"{LAT0},{LON0}", "--cache", str(cache_dir),
                 "--providers", "adsblol", "--out", str(tmp_path / "o")]) == 1
    assert "creation time" in capsys.readouterr().err
    # Wrong day: the cache cannot answer and adsb.lol has no archive for the future.
    assert main(["locate", str(media), "--start", "2030-01-01T00:00:00Z", "--near", f"{LAT0},{LON0}",
                 "--cache", str(cache_dir), "--providers", "adsblol", "--out", str(tmp_path / "o")]) == 1
    assert "no flight data" in capsys.readouterr().err


def test_locate_worldwide_without_region(recording, capsys):
    media, cache_dir, tmp = recording
    out = tmp / "out_world"
    code = main(["locate", str(media), "--providers", "adsblol", "--cache", str(cache_dir),
                 "--elevation", "adsb", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0, text
    doc = json.loads((out / "solution.json").read_text())
    assert doc["region_source"] == "worldwide" and not doc["worldwide"]["ambiguous"]
    sol = doc["solution"]
    assert haversine_m(sol["lat"], sol["lon"], OBS_LAT, OBS_LON) < 50.0, text
    assert "unique place" in text


@pytest.fixture(scope="module")
def stepped(recording):
    """The GUI's path: the same chain run one step at a time."""
    from dopplergeo.core.pipeline import analyze_file
    from dopplergeo.flights import FlightCache
    from dopplergeo.flights.adsblol import AdsbLolProvider
    from dopplergeo.locate import family_observations, fetch_flights, resolve_region, resolve_start, solve_position

    media, cache_dir, _ = recording
    res = analyze_file(media)
    epoch, src = resolve_start(res, None)
    region, rsrc = resolve_region((LAT0, LON0), 15.0)
    with FlightCache(cache_dir) as cache:
        data = fetch_flights(family_observations(res), epoch, region, providers=[AdsbLolProvider()], cache=cache)
        r = solve_position(res, epoch, src, region, rsrc, data, cache=cache, height_m=H_OBS)
    return r, cache_dir


def test_stepped_api_matches_locate(stepped):
    r, _ = stepped
    assert r.solution is not None and r.error is None
    assert haversine_m(r.solution.lat, r.solution.lon, OBS_LAT, OBS_LON) < 50.0


def test_predicted_curve_sits_on_the_measured_tone(stepped):
    from dopplergeo.locate import predicted_curve

    r, _ = stepped
    sol = r.solution
    for fam in r.families:
        icao = sol.assignment[fam.id]
        pred = predicted_curve(sol, fam, icao, fam.t, k=1.0) * fam.k
        ok = np.isfinite(pred)
        assert ok.mean() > 0.9
        # Within a few Hz of the measured tone (rest frequency re-fitted).
        assert np.median(np.abs(pred[ok] - fam.f[ok])) < 1.0
        # The other aircraft's curve misses badly (the trust signal works both ways).
        other = next((c.icao24 for c in sol.candidates if c.icao24 != icao), None)
        if other is not None:
            wrong = predicted_curve(sol, fam, other, fam.t) * fam.k
            okw = np.isfinite(wrong)
            if okw.any():
                assert np.median(np.abs(wrong[okw] - fam.f[okw])) > 3.0


def test_excluded_aircraft_is_reported_but_not_used(stepped):
    from dataclasses import replace as dc_replace

    from dopplergeo.core.solver import SolverSettings
    from dopplergeo.flights import FlightCache
    from dopplergeo.locate import solve_position

    r, cache_dir = stepped
    with FlightCache(cache_dir) as cache:
        r2 = solve_position(r.analysis, r.epoch, r.epoch_source, r.region, r.region_source, r.flights,
                            cache=cache, height_m=H_OBS,
                            settings=SolverSettings(exclude_aircraft=frozenset({"3c6444"})))
    sol = r2.solution
    assert sol is not None
    assert "3c6444" not in sol.assignment.values()
    assert any(c.icao24 == "3c6444" and not c.included for c in sol.candidates)
