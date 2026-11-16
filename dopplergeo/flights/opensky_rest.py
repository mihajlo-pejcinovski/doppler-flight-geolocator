"""OpenSky Network REST API (OAuth2 client credentials).

Two modes, picked from the age of the window:

* ``states``: ``/states/all`` with a bbox every ``step_s`` seconds. Registered
  users can look back 1 hour. Costs 1-4 credits per call by area.
* ``tracks``: ``/flights/all`` (<= 2 h intervals, no area filter) then
  ``/tracks/all`` per aircraft, up to 30 days back. Marked experimental by
  OpenSky; tracks carry no speed, so the spline uses position differences.

Tokens last 30 minutes and are refreshed early or on a 401. 429 responses
honour ``X-Rate-Limit-Retry-After-Seconds``; the local credit counter stops
before the daily budget is spent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .base import (
    BBox,
    Coverage,
    FlightProvider,
    FlightQuery,
    ProviderError,
    ProviderUnavailable,
    TrackPoint,
    points_to_table,
)
from .cache import FlightCache
from .http import make_client, request

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL = "https://opensky-network.org/api"
ATTRIBUTION = ("Flight data: The OpenSky Network, https://opensky-network.org "
               "(Schäfer et al., IPSN 2014)")

STATES_LOOKBACK_S = 3600.0
TRACKS_LOOKBACK_S = 30 * 86400.0
FLIGHTS_MAX_INTERVAL_S = 2 * 3600.0


def states_credit_cost(bbox: BBox | None) -> int:
    """Credits for one /states/all call (OpenSky REST docs, 'API credit usage')."""
    if bbox is None:
        return 4
    a = bbox.area_sq_deg
    return 1 if a <= 25 else 2 if a <= 100 else 3 if a <= 400 else 4


@dataclass(frozen=True)
class OpenSkyCredentials:
    client_id: str
    client_secret: str

    @classmethod
    def from_env(cls) -> OpenSkyCredentials | None:
        cid, sec = os.environ.get("OPENSKY_CLIENT_ID"), os.environ.get("OPENSKY_CLIENT_SECRET")
        return cls(cid, sec) if cid and sec else None


class CreditBudgetExceeded(ProviderError):
    pass


@dataclass
class OpenSkyRestProvider(FlightProvider):
    credentials: OpenSkyCredentials | None = None
    client: httpx.Client = field(default_factory=make_client)
    step_s: float = 5.0
    daily_credits: int = 4000
    max_tracks: int = 300
    max_wait_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time

    credits_used: int = 0
    credits_remaining: int | None = None  # last X-Rate-Limit-Remaining seen
    _token: str | None = field(default=None, repr=False)
    _token_expiry: float = 0.0

    name = "opensky"
    attribution = ATTRIBUTION

    # -- auth & transport ---------------------------------------------------

    def _bearer(self, force: bool = False) -> str:
        if self.credentials is None:
            raise ProviderUnavailable("OpenSky history needs OAuth2 client credentials "
                                      "(OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET)")
        if force or self._token is None or self.now() > self._token_expiry - 60:
            r = request(self.client, "POST", TOKEN_URL, sleep=self.sleep, max_wait_s=self.max_wait_s, data={
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
            })
            js = r.json()
            self._token = js["access_token"]
            self._token_expiry = self.now() + float(js.get("expires_in", 1800))
        return self._token

    def _get(self, path: str, params: dict, credits: int = 0, ok_statuses: tuple[int, ...] = ()) -> httpx.Response:
        if credits and self.credits_used + credits > self.daily_credits:
            raise CreditBudgetExceeded(
                f"OpenSky credit budget reached ({self.credits_used}/{self.daily_credits} used)")
        url = f"{API_URL}{path}"
        for attempt in range(2):
            headers = {"Authorization": f"Bearer {self._bearer(force=attempt > 0)}"}
            r = request(self.client, "GET", url, params=params, headers=headers, sleep=self.sleep,
                        max_wait_s=self.max_wait_s, ok_statuses=(401, *ok_statuses))
            if r.status_code != 401:
                break
        else:
            raise ProviderError("OpenSky rejected the access token twice (check credentials)")
        self.credits_used += credits
        rem = r.headers.get("X-Rate-Limit-Remaining")
        if rem is not None:
            try:
                self.credits_remaining = int(rem)
            except ValueError:
                pass
        return r

    # -- mode selection -----------------------------------------------------

    def mode_for(self, query: FlightQuery) -> str | None:
        age = self.now() - query.t0
        if age <= STATES_LOOKBACK_S:
            return "states"
        if age <= TRACKS_LOOKBACK_S:
            return "tracks"
        return None

    def can_serve(self, query: FlightQuery, now: float) -> str | None:
        if self.credentials is None:
            return "no OpenSky credentials configured"
        if self.mode_for(query) is None:
            return "OpenSky REST keeps at most 30 days of tracks"
        return None

    def fetch(self, query: FlightQuery, cache: FlightCache) -> list[Coverage]:
        mode = self.mode_for(query)
        if mode is None:
            raise ProviderUnavailable("window is older than OpenSky REST's 30-day track history")
        points = self.fetch_states(query) if mode == "states" else self.fetch_tracks(query)
        cov = [Coverage(self.name, query.t0, query.t1, query.bbox)]
        cache.write_part(points_to_table(points), cov, meta={"mode": mode, "credits": self.credits_used})
        return cov

    # -- /states/all --------------------------------------------------------

    def states(self, t: float, bbox: BBox) -> list[TrackPoint]:
        b = bbox
        r = self._get("/states/all", {"time": int(t), "lamin": b.lat_min, "lomin": b.lon_min,
                                      "lamax": b.lat_max, "lomax": b.lon_max, "extended": 1},
                      credits=states_credit_cost(bbox))
        return parse_states(r.json())

    def fetch_states(self, query: FlightQuery) -> list[TrackPoint]:
        seen: dict[tuple[str, float], TrackPoint] = {}
        t = query.t0
        while t <= query.t1 + 1e-9:
            for p in self.states(t, query.bbox):
                if query.t0 - self.step_s <= p.t <= query.t1 + self.step_s:
                    seen[(p.icao24, p.t)] = p
            t += self.step_s
        return sorted(seen.values(), key=lambda p: (p.icao24, p.t))

    # -- /flights/all + /tracks/all ----------------------------------------

    def flights(self, t0: float, t1: float) -> list[dict]:
        out: list[dict] = []
        a = t0
        while a < t1:
            b = min(t1, a + FLIGHTS_MAX_INTERVAL_S)
            r = self._get("/flights/all", {"begin": int(a), "end": int(b) + 1}, ok_statuses=(404,))
            if r.status_code == 200:
                out += r.json() or []
            a = b
        return out

    def track(self, icao24: str, t: float) -> dict | None:
        r = self._get("/tracks/all", {"icao24": icao24, "time": int(t)}, ok_statuses=(404,))
        return r.json() if r.status_code == 200 else None

    def fetch_tracks(self, query: FlightQuery) -> list[TrackPoint]:
        flights = [f for f in self.flights(query.t0, query.t1)
                   if f.get("firstSeen", 0) <= query.t1 and f.get("lastSeen", 0) >= query.t0]
        uniq: dict[str, dict] = {}
        for f in flights:
            uniq.setdefault(f["icao24"], f)
        if len(uniq) > self.max_tracks:
            raise ProviderError(
                f"OpenSky lists {len(uniq)} flights in the window and cannot filter them by area; "
                f"fetching each track exceeds max_tracks={self.max_tracks}. Use adsb.lol for this date.")
        mid = (query.t0 + query.t1) / 2
        points: list[TrackPoint] = []
        for icao, f in uniq.items():
            when = min(max(mid, f.get("firstSeen", mid)), f.get("lastSeen", mid))
            tr = self.track(icao, when)
            if tr:
                points += [p for p in parse_track(tr)
                           if query.t0 <= p.t <= query.t1 and query.bbox.contains(p.lat, p.lon)]
        return points


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _f(x: object) -> float | None:
    return None if x is None else float(x)  # type: ignore[arg-type]


def parse_states(js: dict) -> list[TrackPoint]:
    """``/states/all`` -> TrackPoints, timed by each vector's last position update."""
    out = []
    for s in js.get("states") or []:
        t_pos, lon, lat = s[3], s[5], s[6]
        if t_pos is None or lat is None or lon is None:
            continue
        cs = (s[1] or "").strip() or None
        out.append(TrackPoint(
            t=float(t_pos), icao24=s[0].lower(), callsign=cs, lat=float(lat), lon=float(lon),
            alt_geo_m=_f(s[13]), alt_baro_m=_f(s[7]), gs_mps=_f(s[9]), track_deg=_f(s[10]),
            vrate_mps=_f(s[11]), on_ground=bool(s[8])))
    return out


def parse_track(js: dict) -> list[TrackPoint]:
    """``/tracks/all`` waypoints: [time, lat, lon, baro_altitude, true_track, on_ground]."""
    icao = js.get("icao24", "").lower()
    cs = (js.get("callsign") or "").strip() or None
    out = []
    for w in js.get("path") or []:
        if w[1] is None or w[2] is None:
            continue
        out.append(TrackPoint(t=float(w[0]), icao24=icao, callsign=cs, lat=float(w[1]), lon=float(w[2]),
                              alt_geo_m=None, alt_baro_m=_f(w[3]), gs_mps=None, track_deg=_f(w[4]),
                              vrate_mps=None, on_ground=bool(w[5])))
    return out
