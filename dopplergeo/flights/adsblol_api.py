"""adsb.lol REST API (https://api.adsb.lol/docs) for recent recordings.

The API has no historical area query: ``/1/trace/full/{hex}`` returns one
aircraft's points for the last ~24 h only, and ``/v2/point/{lat}/{lon}/{nm}``
lists the aircraft around a point *now* (with registration, type and model).
So a window is served by

1. listing the aircraft within reach of the region now (radius grown by how far
   an aircraft can have flown since the window started, at most 250 nm);
2. dropping those that cannot have been over the region during the window, from
   their current position and speed;
3. fetching each remaining aircraft's trace and keeping the points in the region
   and window (``adsblol.parse_trace``, the archive's readsb trace format).

Aircraft that have since landed and switched off are no longer listed, so only
windows up to ``max_age_s`` old (default 2 h) are served. Each trace is one
request against a daily quota (500 with an anonymous key; HTTP 429 when used
up), so at most ``max_traces`` traces are fetched per window, nearest first.
The key comes from ``ADSBLOL_API_KEY``; without one adsb.lol allows 500
requests per day per IP.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .adsblol import ATTRIBUTION, parse_trace
from .base import BBox, Coverage, FlightProvider, FlightQuery, ProviderError, ProviderUnavailable
from .base import COLUMNS
from .cache import FlightCache
from .http import RateLimited, make_client, request

API_URL = "https://api.adsb.lol"
KEY_ENV = "ADSBLOL_API_KEY"
NM = 1852.0
MAX_RADIUS_NM = 250.0
INFO_FILE = "aircraft_info.json"


class QuotaExceeded(ProviderError):
    """adsb.lol's daily request quota is used up (HTTP 429)."""


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(a)))


def _region_center_radius(b: BBox) -> tuple[float, float, float]:
    """(lat, lon, radius m) of a circle around the bbox."""
    lat0, lon0 = (b.lat_min + b.lat_max) / 2, (b.lon_min + b.lon_max) / 2
    r = max(_haversine_m(lat0, lon0, la, lo) for la in (b.lat_min, b.lat_max) for lo in (b.lon_min, b.lon_max))
    return lat0, lon0, r


def load_aircraft_info(cache_root: str | Path) -> dict[str, dict]:
    """icao24 -> {"registration", "type", "model"} gathered from API responses."""
    p = Path(cache_root) / INFO_FILE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def save_aircraft_info(cache_root: str | Path, info: dict[str, dict]) -> None:
    if not info:
        return
    p = Path(cache_root) / INFO_FILE
    merged = load_aircraft_info(cache_root)
    for icao, d in info.items():
        merged.setdefault(icao, {}).update({k: v for k, v in d.items() if v})
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=0, sort_keys=True))
    tmp.replace(p)


@dataclass
class AdsbLolApiProvider(FlightProvider):
    api_key: str | None = field(default_factory=lambda: os.environ.get(KEY_ENV) or None, repr=False)
    client: httpx.Client = field(default_factory=make_client)
    max_age_s: float = 2 * 3600.0
    max_traces: int = 150
    reach_speed_mps: float = 320.0      # fastest an aircraft is assumed to cover ground
    # adsb.lol throttles bursts with 429 + a short Retry-After (28 s seen keyless): wait it
    # out; a wait longer than this means the daily quota is used up.
    max_rate_wait_s: float = 300.0
    max_rate_waits: int = 40
    progress: Callable[[str, int, int], None] | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time
    requests_made: int = 0
    log: Callable[[str], None] = lambda m: None
    key_rejected: str | None = None     # adsb.lol's reason when it refused the key (then no key is sent)
    rate_waited_s: float = 0.0          # time spent waiting out adsb.lol's throttling

    name = "adsblol-api"
    attribution = ATTRIBUTION

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _get_json(self, path: str, ok_statuses: tuple[int, ...] = ()) -> tuple[int, dict | None]:
        for _ in range(2):
            r = self._request(path, (*ok_statuses, 401, 403))
            self.requests_made += 1
            if r.status_code in (401, 403) and self.api_key:
                # A revoked or expired key fails every request, while keyless access still
                # works (500/day per IP): carry on without it and say so.
                self.key_rejected = r.text.strip()[:80] or f"HTTP {r.status_code}"
                self.api_key = None
                self.log(f"adsb.lol rejected the API key ({self.key_rejected}); continuing without a key "
                         f"(500 requests/day per IP). Get a new key and set {KEY_ENV}.")
                continue
            break
        if r.status_code in (401, 403):
            raise ProviderError(f"adsb.lol API refused the request: HTTP {r.status_code} {r.text.strip()[:80]}")
        if r.status_code != 200:
            return r.status_code, None
        body = r.content
        if body[:2] == b"\x1f\x8b":  # traces may come gzip-encoded without a Content-Encoding header
            body = gzip.decompress(body)
        return 200, json.loads(body)

    def _request(self, path: str, ok_statuses: tuple[int, ...]) -> httpx.Response:
        """GET with adsb.lol's throttling waited out (progress says so); a long
        Retry-After, or too many waits, is the daily quota: QuotaExceeded."""
        waits = 0
        while True:
            try:
                return request(self.client, "GET", f"{API_URL}{path}", headers=self._headers(), sleep=self.sleep,
                               retries=2, max_wait_s=0.0, ok_statuses=ok_statuses)  # 429s handled here
            except RateLimited as e:
                wait = e.retry_after if e.retry_after is not None else 30.0
                if wait > self.max_rate_wait_s or waits >= self.max_rate_waits:
                    raise QuotaExceeded(
                        f"adsb.lol API daily quota used up after {self.requests_made} requests this run "
                        f"(retry after {wait / 3600:.1f} h, resets 00:00 UTC; "
                        f"{'with key' if self.api_key else 'no key: set ' + KEY_ENV})") from e
                waits += 1
                self.rate_waited_s += wait
                self._report(f"waiting for adsb.lol rate limit ({wait:.0f} s)", 0, 0)
                self.sleep(wait)

    # -- FlightProvider -------------------------------------------------------

    def can_serve(self, query: FlightQuery, now: float) -> str | None:
        age = now - query.t0
        if age > self.max_age_s:
            return (f"adsb.lol API serves only the last {self.max_age_s / 3600:.0f} h "
                    f"(window is {age / 3600:.1f} h old; older days come from the daily archive)")
        return None

    def fetch(self, query: FlightQuery, cache: FlightCache) -> list[Coverage]:
        now = self.now()
        why = self.can_serve(query, now)
        if why:
            raise ProviderUnavailable(why)
        lat0, lon0, r_region = _region_center_radius(query.bbox)
        since = max(0.0, now - query.t0)
        radius_nm = min(MAX_RADIUS_NM, (r_region + self.reach_speed_mps * since) / NM + 5.0)
        self._report("listing aircraft near the region", 0, 0)
        _, js = self._get_json(f"/v2/point/{lat0:.4f}/{lon0:.4f}/{radius_nm:.0f}")
        aircraft = (js or {}).get("ac") or []

        # Who can have been over the region during the window?
        picks: list[tuple[bool, float, str]] = []   # (on the ground, distance, icao): airborne first
        info: dict[str, dict] = {}
        for a in aircraft:
            icao = str(a.get("hex", "")).lower()
            if len(icao) != 6 or icao.startswith("~"):
                continue  # "~" marks a non-ICAO (TIS-B) address: no trace to fetch
            info[icao] = {"registration": a.get("r"), "type": a.get("t"), "model": a.get("desc")}
            lat, lon = a.get("lat"), a.get("lon")
            ground = a.get("alt_baro") == "ground"
            if lat is None or lon is None:
                picks.append((ground, 0.0, icao))  # position unknown: keep
                continue
            dist = max(0.0, _haversine_m(lat0, lon0, float(lat), float(lon)) - r_region)
            gs = (a.get("gs") or 0.0) * 0.514444
            v = max(1.3 * gs, 80.0)  # margin for speed changes; parked aircraft may have just landed
            if dist <= v * since + 2000.0:
                picks.append((ground, dist, icao))
        picks.sort()
        truncated = len(picks) > self.max_traces
        picks = picks[: self.max_traces]

        cols: dict[str, list] = {c: [] for c in COLUMNS}
        keep = query.bbox
        # Airborne aircraft first: with a cap, parked light aircraft at nearby airfields
        # otherwise crowded out the airliners (seen live around Manchester).
        for i, (_, _, icao) in enumerate(picks):
            self._report(f"fetching aircraft traces {i + 1}/{len(picks)}", i, len(picks))
            status, tr = self._get_json(f"/1/trace/full/{icao}", ok_statuses=(404,))
            if tr is None:
                continue
            for k_api, k_info in (("r", "registration"), ("t", "type"), ("desc", "model")):
                if tr.get(k_api):
                    info.setdefault(icao, {})[k_info] = tr[k_api]
            parse_trace(tr, cols, keep=keep)
        # Only the window (plus the trace sampling) is kept.
        pad = 60.0
        idx = [j for j, t in enumerate(cols["t"]) if query.t0 - pad <= t <= query.t1 + pad]
        table_cols = {c: [cols[c][j] for j in idx] for c in COLUMNS}
        cov = [Coverage(self.name, query.t0, query.t1, query.bbox)]
        cache.write_part(_table(table_cols), cov,
                         meta={"aircraft_listed": len(aircraft), "traces": len(picks), "truncated": truncated,
                               "requests": self.requests_made})
        save_aircraft_info(cache.root, info)
        self._report(f"{len(picks)} traces, {self.requests_made} API requests", len(picks), max(len(picks), 1))
        return cov

    def _report(self, stage: str, done: int, total: int) -> None:
        if self.progress:
            self.progress(stage, done, total)


def _table(cols: dict[str, list]):
    import pyarrow as pa

    from .base import SCHEMA
    return pa.table(cols, schema=SCHEMA)
