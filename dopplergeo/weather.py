"""Air temperature and wind for the recording hour from Open-Meteo (no key).

Historical reanalysis (ERA5 and friends) comes from the archive API, which lags
real time by a few days; recent hours come from the forecast API's past data.
Hourly values are interpolated linearly to the requested time. The wind used is
the 100 m wind: the sound paths run from the ground to aircraft a few hundred
metres up, where the 10 m wind is too weak (surface friction).

Requests send only the region centre rounded to 0.1 deg (~10 km), and every
answer is cached per (rounded place, UTC day). Data: Open-Meteo.com, CC BY 4.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core.atmosphere import Atmosphere, wind_from_met
from .flights.base import ProviderError
from .flights.http import make_client, request

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = ("temperature_2m", "wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_direction_10m")
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


class WeatherError(RuntimeError):
    pass


def _fetch_day(client, lat: float, lon: float, day: str) -> dict:
    params = {"latitude": f"{lat:.1f}", "longitude": f"{lon:.1f}", "start_date": day, "end_date": day,
              "hourly": ",".join(HOURLY), "wind_speed_unit": "ms", "timezone": "GMT"}
    last = None
    for url in (ARCHIVE_URL, FORECAST_URL):
        try:
            data = request(client, "GET", url, params=params).json()
        except (ProviderError, ValueError) as e:
            last = e
            continue
        h = data.get("hourly") or {}
        if h.get("time") and all(v is not None for v in h.get("temperature_2m", [None])):
            data["_source"] = "Open-Meteo archive" if url == ARCHIVE_URL else "Open-Meteo forecast (past)"
            return data
    raise WeatherError(f"no Open-Meteo data for {day} at {lat:.1f},{lon:.1f}: {last or 'empty answer'}")


def hourly_weather(lat: float, lon: float, t_utc: float, cache_dir: str | Path | None = None,
                   client=None) -> dict:
    """Open-Meteo hourly data (dict of lists) for the UTC day of ``t_utc``, cached."""
    lat, lon = round(lat, 1), round(lon, 1)
    day = datetime.fromtimestamp(t_utc, tz=timezone.utc).strftime("%Y-%m-%d")
    path = Path(cache_dir) / f"{lat:+.1f}_{lon:+.1f}_{day}.json" if cache_dir else None
    if path is not None and path.exists():
        return json.loads(path.read_text())
    own = client is None
    client = client or make_client()
    try:
        data = _fetch_day(client, lat, lon, day)
    finally:
        if own:
            client.close()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def atmosphere_at(lat: float, lon: float, t_utc: float, cache_dir: str | Path | None = None,
                  wind_level: str = "100m", client=None) -> Atmosphere:
    """Temperature (2 m) and wind (``wind_level``: "100m" or "10m") at ``t_utc``,
    linearly interpolated between the hours (fetching the next day when needed)."""
    data = hourly_weather(lat, lon, t_utc, cache_dir, client)
    h = data["hourly"]
    times = [datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() for s in h["time"]]
    if t_utc > times[-1]:  # after 23:00 UTC: append the next day's first hours
        nxt = hourly_weather(lat, lon, t_utc + 3600.0, cache_dir, client)["hourly"]
        h = {k: list(h[k]) + list(nxt[k]) for k in h}
        times = [datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() for s in h["time"]]
    t = np.asarray(times)

    def at(key: str) -> float:
        v = np.asarray([np.nan if x is None else x for x in h[key]], dtype=float)
        ok = np.isfinite(v)
        if not ok.any():
            raise WeatherError(f"Open-Meteo has no {key} for that day")
        return float(np.interp(t_utc, t[ok], v[ok]))

    # Interpolate the wind as a vector (directions wrap at 360).
    spd, frm = f"wind_speed_{wind_level}", f"wind_direction_{wind_level}"
    v = np.asarray([np.nan if x is None else x for x in h[spd]], dtype=float)
    d = np.asarray([np.nan if x is None else x for x in h[frm]], dtype=float)
    e, n = (np.array(c) for c in zip(*[wind_from_met(a, b) for a, b in zip(v, d)]))
    ok = np.isfinite(e) & np.isfinite(n)
    if not ok.any():
        raise WeatherError(f"Open-Meteo has no {wind_level} wind for that day")
    we, wn = float(np.interp(t_utc, t[ok], e[ok])), float(np.interp(t_utc, t[ok], n[ok]))
    when = datetime.fromtimestamp(t_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    return Atmosphere(temp_c=at("temperature_2m"), wind_e=we, wind_n=wn, lapse=True,
                      source=f"{data.get('_source', 'Open-Meteo')}, {wind_level} wind, {when} UTC, "
                             f"{round(lat, 1):.1f},{round(lon, 1):.1f}")
