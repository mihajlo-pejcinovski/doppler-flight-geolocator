"""adsb.lol REST API provider against recorded-style responses (respx, no network)."""

import gzip
import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from dopplergeo.flights import BBox, FlightCache, FlightQuery, FlightService
from dopplergeo.flights.adsblol_api import API_URL, AdsbLolApiProvider, QuotaExceeded, load_aircraft_info

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc).timestamp()
REGION = BBox(53.25, -2.45, 53.55, -2.05)          # around Manchester
QUERY = FlightQuery(NOW - 600, NOW - 300, REGION)    # 10 to 5 minutes ago


def point_response():
    return {"now": NOW, "total": 4, "ac": [
        # over the region now, slow on approach: picked
        {"hex": "40756f", "flight": "EZY32NC ", "r": "G-EZGY", "t": "A320", "lat": 53.40, "lon": -2.20, "gs": 160},
        # 400 km away at 450 kt: cannot have been over the region 5-10 min ago
        {"hex": "4ca9d3", "flight": "RYR7857", "r": "EI-EVK", "t": "B738", "lat": 56.9, "lon": -2.2, "gs": 450},
        # 25 km away, parked: may have landed just after the window: picked
        {"hex": "4d2288", "flight": "RYR60SF", "r": "9H-QDA", "t": "B738", "lat": 53.36, "lon": -2.27, "gs": 0},
        # non-ICAO (TIS-B) address: has no trace
        {"hex": "~2a0b11", "lat": 53.4, "lon": -2.2, "gs": 100},
    ]}


def trace(icao, base, points, gz=False, desc=None):
    """readsb trace_full: rows [dt, lat, lon, alt_ft, gs_kt, track, flags, vrate, detail]."""
    body = {"icao": icao, "timestamp": base, "r": None, "t": None, "desc": desc, "trace": points}
    raw = json.dumps(body).encode()
    return httpx.Response(200, content=gzip.compress(raw) if gz else raw)


def row(dt, lat, lon, flight=None):
    return [dt, lat, lon, 3000, 160.0, 235.0, 0, -700, {"flight": flight} if flight else None]


def provider(**kw):
    return AdsbLolApiProvider(api_key="sk-lol-test", now=lambda: NOW, sleep=lambda s: None, **kw)


@respx.mock
def test_fetch_lists_reachable_aircraft_and_keeps_window_points(tmp_path):
    base = NOW - 3600
    point = respx.get(url__regex=rf"{API_URL}/v2/point/.*").mock(return_value=httpx.Response(200,
                                                                                          json=point_response()))
    ezy = respx.get(f"{API_URL}/1/trace/full/40756f").mock(return_value=trace("40756f", base, [
        row(0, 53.0, -1.0, "EZY32NC"),          # an hour earlier, outside the region
        row(3600 - 480, 53.42, -2.15),          # in the window, in the region
        row(3600 - 420, 53.40, -2.20),          # in the window, in the region
    ], gz=True, desc="AIRBUS A-320"))
    ryr = respx.get(f"{API_URL}/1/trace/full/4d2288").mock(return_value=trace("4d2288", base, [
        row(3600 - 400, 53.38, -2.24, "RYR60SF")]))
    far = respx.get(f"{API_URL}/1/trace/full/4ca9d3").mock(return_value=httpx.Response(200, json={}))

    p = provider()
    with FlightCache(tmp_path / "cache") as cache:
        cov = p.fetch(QUERY, cache)
        assert cov[0].provider == "adsblol-api"
        table = cache.query(QUERY, "adsblol-api")
    assert point.called and ezy.called and ryr.called and not far.called
    assert point.calls[0].request.headers["x-api-key"] == "sk-lol-test"
    assert sorted(set(table.column("icao24").to_pylist())) == ["40756f", "4d2288"]
    assert table.num_rows == 3                    # the point an hour earlier and outside is dropped
    assert p.requests_made == 3                   # 1 listing + 2 traces
    info = load_aircraft_info(tmp_path / "cache")
    assert info["40756f"] == {"registration": "G-EZGY", "type": "A320", "model": "AIRBUS A-320"}
    assert info["4d2288"]["registration"] == "9H-QDA"


@respx.mock
def test_second_query_is_served_from_cache_without_requests(tmp_path):
    respx.get(url__regex=rf"{API_URL}/v2/point/.*").mock(return_value=httpx.Response(200, json={"ac": [
        {"hex": "40756f", "lat": 53.40, "lon": -2.20, "gs": 160}]}))
    tr = respx.get(f"{API_URL}/1/trace/full/40756f").mock(
        return_value=trace("40756f", NOW - 3600, [row(3600 - 450, 53.40, -2.20, "EZY32NC")]))
    with FlightCache(tmp_path / "cache") as cache:
        svc = FlightService(cache, [provider()], now=lambda: NOW)
        first = svc.get(QUERY)
        second = svc.get(QUERY)
    assert not first.from_cache and second.from_cache
    assert tr.call_count == 1


@respx.mock
def test_quota_exhausted_is_reported():
    respx.get(url__regex=rf"{API_URL}/v2/point/.*").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "40000"}))
    with pytest.raises(QuotaExceeded, match="quota used up"):
        provider().fetch(QUERY, cache=None)


def test_only_recent_windows_are_served():
    p = provider()
    assert p.can_serve(QUERY, NOW) is None
    old = FlightQuery(NOW - 5 * 3600, NOW - 5 * 3600 + 300, REGION)
    assert "last 2 h" in p.can_serve(old, NOW)


def test_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ADSBLOL_API_KEY", "sk-lol-env")
    assert AdsbLolApiProvider()._headers() == {"x-api-key": "sk-lol-env"}
    monkeypatch.delenv("ADSBLOL_API_KEY")
    assert AdsbLolApiProvider()._headers() == {}


@respx.mock
def test_revoked_key_falls_back_to_keyless_access(tmp_path):
    """Seen live on 2026-11-14: the key started failing with 401 "api key revoked" on
    every endpoint while keyless requests still worked."""
    def by_key(request):
        if "x-api-key" in request.headers:
            return httpx.Response(401, text="api key revoked")
        return httpx.Response(200, json={"ac": [{"hex": "40756f", "lat": 53.40, "lon": -2.20, "gs": 160}]})
    respx.get(url__regex=rf"{API_URL}/v2/point/.*").mock(side_effect=by_key)
    tr = respx.get(f"{API_URL}/1/trace/full/40756f").mock(
        return_value=trace("40756f", NOW - 3600, [row(3600 - 450, 53.40, -2.20, "EZY32NC")]))
    logs = []
    p = provider(log=logs.append)
    with FlightCache(tmp_path / "cache") as cache:
        p.fetch(QUERY, cache)
        assert cache.query(QUERY, "adsblol-api").num_rows == 1
    assert p.key_rejected == "api key revoked" and p.api_key is None
    assert "x-api-key" not in tr.calls[0].request.headers
    assert any("rejected the API key" in m for m in logs)


@respx.mock
def test_short_rate_limit_is_waited_out_long_one_is_the_quota(tmp_path):
    """Keyless access was throttled live with 429 + Retry-After: 28 after ~10 requests."""
    calls = {"n": 0}

    def throttled(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "28"}, text="rate limit exceeded")
        return httpx.Response(200, json={"ac": []})
    respx.get(url__regex=rf"{API_URL}/v2/point/.*").mock(side_effect=throttled)
    slept, stages = [], []
    p = AdsbLolApiProvider(api_key=None, now=lambda: NOW, sleep=slept.append,
                           progress=lambda st, d, t: stages.append(st))
    with FlightCache(tmp_path / "cache") as cache:
        p.fetch(QUERY, cache)
    assert slept == [28.0] and p.rate_waited_s == 28.0
    assert any("waiting for adsb.lol rate limit" in st for st in stages)
