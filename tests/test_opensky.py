from datetime import datetime, timezone

import httpx
import pytest
import respx

from dopplergeo.flights import BBox, FlightCache, FlightQuery
from dopplergeo.flights.http import RateLimited
from dopplergeo.flights.opensky_rest import (
    API_URL,
    TOKEN_URL,
    CreditBudgetExceeded,
    OpenSkyCredentials,
    OpenSkyRestProvider,
    states_credit_cost,
)

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc).timestamp()
REGION = BBox(52.0, 4.0, 53.0, 5.0)  # 1 sq deg -> 1 credit
CREDS = OpenSkyCredentials("id", "secret")


def state(icao, t, lat=52.5, lon=4.5, ground=False):
    return [icao, "KLM12  ", "Netherlands", t, None if t is None else t + 1, lon, lat, 3000.0, ground, 200.0, 90.0, -5.0,
            None, 3050.0, "1234", False, 0, 4]


class Clock:
    def __init__(self, t):
        self.t = t
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def provider(clock, **kw):
    return OpenSkyRestProvider(credentials=CREDS, client=httpx.Client(), now=clock, sleep=clock.sleep, **kw)


def test_credit_cost_by_area():
    assert states_credit_cost(REGION) == 1
    assert states_credit_cost(BBox(0, 0, 8, 8)) == 2
    assert states_credit_cost(BBox(0, 0, 15, 20)) == 3
    assert states_credit_cost(BBox(0, 0, 30, 30)) == 4
    assert states_credit_cost(None) == 4


def test_states_mode_steps_dedupes_and_caches(tmp_path):
    clock = Clock(NOW)
    calls = []

    def states(req):
        t = int(req.url.params["time"])
        calls.append(t)
        assert req.headers["Authorization"].startswith("Bearer tok")
        assert float(req.url.params["lamin"]) == 52.0
        # Each snapshot reports the latest position time, which only changes every 10 s.
        tp = t - t % 10
        return httpx.Response(200, json={"time": t, "states": [state("4840D6", tp), state("aaaaaa", tp, ground=True),
                                                                 state("bbbbbb", None)]},
                              headers={"X-Rate-Limit-Remaining": str(4000 - len(calls))})

    with respx.mock as router:
        tok = router.post(TOKEN_URL).respond(200, json={"access_token": "tok1", "expires_in": 1800})
        router.get(f"{API_URL}/states/all").mock(side_effect=states)
        p = provider(clock)
        q = FlightQuery(NOW - 600, NOW - 570, REGION)
        assert p.mode_for(q) == "states"
        with FlightCache(tmp_path) as cache:
            (cov,) = p.fetch(q, cache)
            tbl = cache.query(q, "opensky")
    assert calls == list(range(int(NOW - 600), int(NOW - 569), 5))
    assert tok.call_count == 1
    assert p.credits_used == 7 and p.credits_remaining == 3993
    rows = tbl.to_pylist()
    assert {r["icao24"] for r in rows} == {"4840d6", "aaaaaa"}
    assert len([r for r in rows if r["icao24"] == "4840d6"]) == 4  # deduped per position time
    r = rows[0]
    assert r["callsign"] == "KLM12" and r["alt_geo_m"] == 3050.0 and r["alt_baro_m"] == 3000.0


def test_token_refresh_on_expiry_and_401():
    clock = Clock(NOW)
    issued = []

    def token(req):
        issued.append(clock.t)
        return httpx.Response(200, json={"access_token": f"tok{len(issued)}", "expires_in": 1800})

    seen = []

    def states(req):
        seen.append(req.headers["Authorization"])
        if req.headers["Authorization"] == "Bearer tok2" and seen.count("Bearer tok2") == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"time": 0, "states": []})

    with respx.mock as router:
        router.post(TOKEN_URL).mock(side_effect=token)
        router.get(f"{API_URL}/states/all").mock(side_effect=states)
        p = provider(clock)
        p.states(NOW - 100, REGION)
        clock.t += 1800 - 30  # within 60 s of expiry -> proactive refresh (tok2) -> 401 -> tok3
        p.states(NOW - 100, REGION)
    assert len(issued) == 3
    assert seen == ["Bearer tok1", "Bearer tok2", "Bearer tok3"]


def test_429_short_wait_retries_long_wait_raises():
    clock = Clock(NOW)
    replies = iter([httpx.Response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "7"}),
                    httpx.Response(200, json={"states": None}),
                    httpx.Response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "86400"})])
    with respx.mock as router:
        router.post(TOKEN_URL).respond(200, json={"access_token": "t", "expires_in": 1800})
        router.get(f"{API_URL}/states/all").mock(side_effect=lambda req: next(replies))
        p = provider(clock)
        assert p.states(NOW, REGION) == []
        assert clock.slept == [7.0]
        with pytest.raises(RateLimited, match="86400"):
            p.states(NOW, REGION)


def test_credit_budget_stops_before_overspending():
    clock = Clock(NOW)
    with respx.mock as router:
        router.post(TOKEN_URL).respond(200, json={"access_token": "t", "expires_in": 1800})
        route = router.get(f"{API_URL}/states/all").respond(200, json={"states": []})
        p = provider(clock, daily_credits=3)
        with pytest.raises(CreditBudgetExceeded):
            p.fetch_states(FlightQuery(NOW - 600, NOW - 580, REGION))
    assert route.call_count == 3


def test_tracks_mode_filters_window_and_area(tmp_path):
    clock = Clock(NOW)
    t0 = NOW - 5 * 86400
    flights = [{"icao24": "4840d6", "firstSeen": t0 - 3000, "lastSeen": t0 + 3000, "callsign": "KLM1"},
               {"icao24": "3c6444", "firstSeen": t0 - 9000, "lastSeen": t0 - 8000, "callsign": "DLH2"}]
    path = [[t0 - 100 + 20 * i, 52.5 + 0.01 * i, 4.5, 3000.0, 0.0, False] for i in range(10)]
    path.append([t0, 60.0, 4.5, 3000.0, 0.0, False])  # outside the bbox
    with respx.mock as router:
        router.post(TOKEN_URL).respond(200, json={"access_token": "t", "expires_in": 1800})
        fl = router.get(f"{API_URL}/flights/all").respond(200, json=flights)
        tr = router.get(f"{API_URL}/tracks/all").respond(
            200, json={"icao24": "4840d6", "callsign": "KLM1 ", "path": path})
        p = provider(clock)
        q = FlightQuery(t0 - 60, t0 + 60, REGION)
        assert p.mode_for(q) == "tracks"
        with FlightCache(tmp_path) as cache:
            p.fetch(q, cache)
            rows = cache.query(q, "opensky").to_pylist()
    assert fl.call_count == 1 and tr.call_count == 1
    assert tr.calls[0].request.url.params["icao24"] == "4840d6"
    assert len(rows) == 7 and all(  # t0-60 .. t0+60 inclusive, in the bbox
        r["gs_mps"] is None for r in rows)
    assert p.credits_used == 0


def test_tracks_mode_refuses_unfilterable_volume():
    clock = Clock(NOW)
    t0 = NOW - 86400
    many = [{"icao24": f"{i:06x}", "firstSeen": t0 - 10, "lastSeen": t0 + 10} for i in range(50)]
    with respx.mock as router:
        router.post(TOKEN_URL).respond(200, json={"access_token": "t", "expires_in": 1800})
        router.get(f"{API_URL}/flights/all").respond(200, json=many)
        with pytest.raises(Exception, match="adsb.lol"):
            provider(clock, max_tracks=20).fetch_tracks(FlightQuery(t0 - 60, t0 + 60, REGION))


def test_availability_rules():
    clock = Clock(NOW)
    q_old = FlightQuery(NOW - 40 * 86400, NOW - 40 * 86400 + 60, REGION)
    assert provider(clock).can_serve(q_old, NOW) is not None
    anon = OpenSkyRestProvider(client=httpx.Client(), now=clock)
    assert "credentials" in anon.can_serve(FlightQuery(NOW - 60, NOW, REGION), NOW)
