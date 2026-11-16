"""adsb.lol daily globe_history archives (ODbL 1.0).

One GitHub release per UTC day, e.g. ``v2025.12.30-planes-readsb-prod-0`` in
``adsblol/globe_history_2025``, holding a tar split into ``.tar.aa``/``.tar.ab``
(about 3-4 GB per day). The tar starts with ~0.7 GB of ``heatmap/`` blobs, then
``traces/xx/trace_full_<icao>.json`` (readsb trace format, sometimes gzipped).

Indexing a day:

1. resolve the release (direct tag lookup, falling back to listing releases);
2. walk tar headers with small Range requests to find where ``traces/`` begins,
   so the heatmap prefix is never downloaded;
3. stream from there across the parts (resuming on dropped connections),
   parse each trace on the fly, and spill rows to a temporary Parquet file;
4. sort into one cache part per day and register its coverage.

The raw archive is not kept unless ``keep_raw_dir`` is set.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.geo import FOOT, FPM, KNOT
from .base import SCHEMA, BBox, Coverage, FlightProvider, FlightQuery, ProviderError, ProviderUnavailable
from .cache import FlightCache
from .http import make_client, request

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "adsblol/globe_history_{year}"
ATTRIBUTION = "ADS-B data: adsb.lol (https://www.adsb.lol), ODbL 1.0"

# readsb trace flags
FLAG_STALE = 1
FLAG_NEW_LEG = 2
FLAG_VRATE_GEOM = 4
FLAG_ALT_GEOM = 8

ProgressFn = Callable[[str, int, int], None]


class DownloadNeedsConfirmation(ProviderError):
    def __init__(self, plan: DownloadPlan):
        self.plan = plan
        super().__init__(f"adsb.lol download needs confirmation: {plan.describe()}")


# ---------------------------------------------------------------------------
# Byte sources (remote parts or local files), addressed as one concatenation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleasePart:
    name: str
    url: str
    size: int


@dataclass(frozen=True)
class Release:
    repo: str
    tag: str
    parts: tuple[ReleasePart, ...]

    @property
    def total_size(self) -> int:
        return sum(p.size for p in self.parts)


@dataclass(frozen=True)
class DownloadPlan:
    release: Release
    traces_offset: int

    @property
    def bytes_to_stream(self) -> int:
        return self.release.total_size - self.traces_offset

    def describe(self) -> str:
        gb = 1e9
        return (f"{self.release.tag}: stream {self.bytes_to_stream / gb:.2f} GB of "
                f"{self.release.total_size / gb:.2f} GB (skipping {self.traces_offset / gb:.2f} GB of heatmaps)")


class _Concat:
    """Several byte sources read as one: ranged reads and resumable streaming."""

    def __init__(self, sizes: Sequence[int], open_range: Callable[[int, int], Iterator[bytes]]):
        # open_range(part_index, start) yields chunks from ``start`` to the part's end.
        self.sizes = list(sizes)
        self._open = open_range
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)

    def _locate(self, offset: int) -> tuple[int, int]:
        for i, (s, n) in enumerate(zip(self.starts, self.sizes)):
            if offset < s + n:
                return i, offset - s
        raise EOFError(offset)

    def read_range(self, offset: int, n: int) -> bytes:
        out = bytearray()
        while len(out) < n and offset < self.total:
            i, local = self._locate(offset)
            gen = self._open(i, local)
            try:
                for chunk in gen:
                    take = chunk[: n - len(out)]
                    out += take
                    offset += len(take)
                    if len(out) >= n:
                        break
            finally:
                gen.close()
        return bytes(out)

    def iter_from(self, offset: int) -> Iterator[bytes]:
        while offset < self.total:
            i, local = self._locate(offset)
            for chunk in self._open(i, local):
                offset += len(chunk)
                yield chunk


def _http_concat(client: httpx.Client, parts: Sequence[ReleasePart], retries: int = 6,
                 sleep: Callable[[float], None] = time.sleep, range_hint: int | None = None) -> _Concat:
    def open_range(i: int, start: int) -> Iterator[bytes]:
        part = parts[i]
        pos, failures = start, 0
        while pos < part.size:
            end = "" if range_hint is None else str(min(part.size - 1, pos + range_hint - 1))
            try:
                with client.stream("GET", part.url, headers={"Range": f"bytes={pos}-{end}"}) as r:
                    if r.status_code not in (200, 206):
                        raise ProviderError(f"GET {part.name} -> HTTP {r.status_code}")
                    skip = pos if r.status_code == 200 else 0  # server ignored Range
                    for chunk in r.iter_raw(1 << 20):
                        if skip:
                            drop = min(skip, len(chunk))
                            chunk, skip = chunk[drop:], skip - drop
                            if not chunk:
                                continue
                        pos += len(chunk)
                        failures = 0
                        yield chunk
                    if range_hint is not None:
                        continue
                    if pos < part.size:
                        raise httpx.ReadError("connection closed early")
            except httpx.TransportError:
                failures += 1
                if failures > retries:
                    raise ProviderError(f"download of {part.name} kept failing at byte {pos}") from None
                sleep(min(60.0, 2.0 ** failures))
    return _Concat([p.size for p in parts], open_range)


def _local_concat(paths: Sequence[Path]) -> _Concat:
    def open_range(i: int, start: int) -> Iterator[bytes]:
        with open(paths[i], "rb") as fh:
            fh.seek(start)
            while chunk := fh.read(1 << 20):
                yield chunk
    return _Concat([p.stat().st_size for p in paths], open_range)


class _IterReader(io.RawIOBase):
    def __init__(self, chunks: Iterator[bytes], on_bytes: Callable[[int], None] | None = None):
        self._it = chunks
        self._buf = memoryview(b"")
        self._on = on_bytes

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        while not self._buf:
            try:
                chunk = next(self._it)
            except StopIteration:
                return 0
            if self._on:
                self._on(len(chunk))
            self._buf = memoryview(chunk)
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


# ---------------------------------------------------------------------------
# Tar walking and trace parsing
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _is_traces(name: str) -> bool:
    n = _norm(name)
    return n == "traces" or n.startswith("traces/")


def find_traces_offset(src: _Concat, max_headers: int = 100_000) -> int:
    """Byte offset of the first ``traces/`` tar header, read via ranged requests."""
    off = 0
    for _ in range(max_headers):
        hdr = src.read_range(off, 512)
        if len(hdr) < 512 or hdr == bytes(512):
            raise ProviderError("archive has no traces/ directory")
        try:
            info = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
        except tarfile.TarError as e:
            raise ProviderError(f"unreadable tar header at byte {off}: {e}") from e
        if _is_traces(info.name):
            return off
        off += 512 + (info.size + 511) // 512 * 512
    return 0


class TraceIndexer:
    """Accumulates parsed trace rows and spills them to a Parquet file in batches."""

    def __init__(self, path: Path, keep_bbox: BBox | None = None, flush_rows: int = 1_000_000):
        self.path = path
        self.keep = keep_bbox
        self.flush_rows = flush_rows
        self.n_rows = 0
        self.n_aircraft = 0
        self.n_errors = 0
        self._writer: pq.ParquetWriter | None = None
        self._reset()

    def _reset(self) -> None:
        self.cols: dict[str, list] = {n: [] for n in SCHEMA.names}

    def add_json(self, data: bytes) -> None:
        try:
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            obj = json.loads(data)
        except (OSError, ValueError):
            self.n_errors += 1
            return
        self.add(obj)

    def add(self, obj: dict) -> None:
        n = parse_trace(obj, self.cols, self.keep)
        if n:
            self.n_aircraft += 1
            self.n_rows += n
        if len(self.cols["t"]) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.cols["t"]:
            return
        tbl = pa.Table.from_pydict(self.cols, schema=SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, SCHEMA, compression="zstd")
        self._writer.write_table(tbl)
        self._reset()

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            pq.write_table(SCHEMA.empty_table(), self.path)
        else:
            self._writer.close()
            self._writer = None


def parse_trace(obj: dict, cols: dict[str, list], keep: BBox | None = None) -> int:
    """Append one readsb ``trace_full`` object to column lists; returns rows added.

    Trace rows: [dt, lat, lon, alt_ft|"ground"|null, gs_kt, track, flags,
    vrate_fpm, detail|null, source, alt_geom_ft, geom_rate_fpm, ias, roll].
    The callsign appears only in occasional detail objects; it is carried
    forward and back-filled within each leg (flag 2 starts a new leg).
    """
    icao = str(obj.get("icao", "")).lower()
    base = float(obj.get("timestamp", 0.0))
    trace = obj.get("trace") or []
    if keep is not None:
        lat_min, lon_min, lat_max, lon_max = keep.as_tuple()
    c_t, c_icao, c_cs = cols["t"], cols["icao24"], cols["callsign"]
    c_lat, c_lon, c_geo, c_baro = cols["lat"], cols["lon"], cols["alt_geo_m"], cols["alt_baro_m"]
    c_gs, c_trk, c_vr, c_gnd = cols["gs_mps"], cols["track_deg"], cols["vrate_mps"], cols["on_ground"]
    start = len(c_t)
    leg_start = start
    callsign: str | None = None
    for e in trace:
        n = len(e)
        flags = (e[6] or 0) if n > 6 else 0
        if flags & FLAG_NEW_LEG:
            callsign, leg_start = None, len(c_t)
        det = e[8] if n > 8 else None
        if det and det.get("flight"):
            cs = det["flight"].strip()
            if cs:
                if callsign is None:  # back-fill this leg's earlier rows
                    for j in range(leg_start, len(c_cs)):
                        c_cs[j] = cs
                callsign = cs
        lat, lon = e[1], e[2]
        if lat is None or lon is None:
            continue
        if keep is not None and not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        alt = e[3]
        ground = alt == "ground"
        a = None if ground or alt is None else alt * FOOT
        ag = e[10] if n > 10 else None
        if flags & FLAG_ALT_GEOM:
            geo, baro = a, None
        else:
            geo, baro = (ag * FOOT if ag is not None else None), a
        gr = e[11] if n > 11 else None
        vr = gr if gr is not None else e[7]
        gs = e[4]
        c_t.append(base + e[0])
        c_icao.append(icao)
        c_cs.append(callsign)
        c_lat.append(lat)
        c_lon.append(lon)
        c_geo.append(geo)
        c_baro.append(baro)
        c_gs.append(gs * KNOT if gs is not None else None)
        c_trk.append(e[5])
        c_vr.append(vr * FPM if vr is not None else None)
        c_gnd.append(ground)
    return len(c_t) - start


def index_tar_stream(fileobj: io.BufferedIOBase, indexer: TraceIndexer,
                     on_member: Callable[[int], None] | None = None) -> None:
    """Parse every ``traces/`` member of a (possibly mid-archive) tar stream."""
    seen = False
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not _is_traces(m.name):
                if seen:
                    break  # traces/ is contiguous; stop instead of reading the rest
                continue
            seen = True
            if not m.isfile():
                continue
            fh = tf.extractfile(m)
            if fh is not None:
                indexer.add_json(fh.read())
            if on_member:
                on_member(indexer.n_aircraft)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"^v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-(\w+)-(\d+)$")


@dataclass
class AdsbLolProvider(FlightProvider):
    """Whole-day archive provider. ``confirm`` approves each download plan;
    without it a download raises ``DownloadNeedsConfirmation``."""

    confirm: Callable[[DownloadPlan], bool] | None = None
    progress: ProgressFn | None = None
    variant: str = "prod"
    repo_template: str = DEFAULT_REPO
    keep_bbox: BBox | None = None
    keep_raw_dir: Path | None = None
    client: httpx.Client = field(default_factory=make_client)
    github_token: str | None = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN"))
    sleep: Callable[[float], None] = time.sleep

    name = "adsblol"
    attribution = ATTRIBUTION

    # -- release lookup -----------------------------------------------------

    def _gh(self, path: str, **params) -> httpx.Response:
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return request(self.client, "GET", f"{GITHUB_API}{path}", params=params or None, headers=headers,
                       ok_statuses=(404,), sleep=self.sleep)

    @staticmethod
    def _release_from_json(repo: str, js: dict) -> Release:
        assets = sorted(js.get("assets", []), key=lambda a: a["name"])
        parts = tuple(ReleasePart(a["name"], a["browser_download_url"], int(a["size"]))
                      for a in assets if re.search(r"\.tar(\.[a-z]{2})?$", a["name"]))
        if not parts:
            raise ProviderError(f"release {js.get('tag_name')} has no tar assets")
        return Release(repo, js["tag_name"], parts)

    def find_release(self, day: datetime) -> Release:
        repo = self.repo_template.format(year=day.year)
        prefix = f"v{day:%Y.%m.%d}-planes-readsb-{self.variant}-"
        r = self._gh(f"/repos/{repo}/releases/tags/{prefix}0")
        if r.status_code == 200:
            return self._release_from_json(repo, r.json())
        # Fall back to listing (re-uploads get suffix -1, -2, ...). Newest first.
        best: dict | None = None
        for page in range(1, 60):
            r = self._gh(f"/repos/{repo}/releases", per_page=100, page=page)
            items = r.json() if r.status_code == 200 else []
            if not items:
                break
            older = 0
            for js in items:
                tag = js.get("tag_name", "")
                if tag.startswith(prefix):
                    if best is None or int(tag.rsplit("-", 1)[1]) > int(best["tag_name"].rsplit("-", 1)[1]):
                        best = js
                m = _TAG_RE.match(tag)
                if m and datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc) < day:
                    older += 1
            if best is not None or older == len(items):
                break
        if best is None:
            raise ProviderUnavailable(f"no adsb.lol {self.variant} release for {day:%Y-%m-%d} in {repo}")
        return self._release_from_json(repo, best)

    def plan(self, day: datetime) -> DownloadPlan:
        rel = self.find_release(day)
        src = _http_concat(self.client, rel.parts, sleep=self.sleep, range_hint=512)
        return DownloadPlan(rel, find_traces_offset(src))

    # -- provider API -------------------------------------------------------

    def can_serve(self, query: FlightQuery, now: float) -> str | None:
        today = datetime.fromtimestamp(now, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if query.utc_dates()[-1] >= today:
            return "adsb.lol publishes a day's archive only after the UTC day ends"
        if self.keep_bbox is not None and not self.keep_bbox.covers(query.bbox):
            return "query lies outside the configured keep_bbox"
        return None

    def fetch(self, query: FlightQuery, cache: FlightCache) -> list[Coverage]:
        added: list[Coverage] = []
        for day in query.utc_dates():
            t0 = day.timestamp()
            sub = FlightQuery(max(query.t0, t0), min(query.t1, t0 + 86400), query.bbox)
            if cache.covering_parts(sub, self.name) is None:
                added += self.index_day(day, cache)
        return added

    def _coverage(self, day: datetime) -> list[Coverage]:
        t0 = day.timestamp()
        return [Coverage(self.name, t0, t0 + 86400, self.keep_bbox or BBox.world())]

    def index_day(self, day: datetime, cache: FlightCache) -> list[Coverage]:
        """Download (streaming) and index one UTC day into the cache."""
        plan = self.plan(day)
        if self.confirm is None:
            raise DownloadNeedsConfirmation(plan)
        if not self.confirm(plan):
            raise ProviderError(f"download of {plan.release.tag} declined")
        if self.keep_raw_dir is not None:
            paths = self._download_raw(plan.release)
            return self.index_archive(day, cache, paths, tag=plan.release.tag)
        src = _http_concat(self.client, plan.release.parts, sleep=self.sleep)
        return self._index(day, cache, src, plan.traces_offset, plan.release.tag)

    def index_archive(self, day: datetime, cache: FlightCache, paths: Sequence[Path | str],
                      tag: str | None = None) -> list[Coverage]:
        """Index already-downloaded archive parts (``.tar`` or ``.tar.aa``, ``.tar.ab``, ...)."""
        paths = sorted(Path(p) for p in paths)
        src = _local_concat(paths)
        return self._index(day, cache, src, find_traces_offset(src), tag or paths[0].name)

    def _index(self, day: datetime, cache: FlightCache, src: _Concat, offset: int, tag: str) -> list[Coverage]:
        dest = cache.part_path(self.name, day.timestamp(), f"{day:%Y%m%d}-{self.variant}")
        tmp = dest.with_name(dest.stem + ".raw.parquet")
        indexer = TraceIndexer(tmp, self.keep_bbox)
        total = src.total - offset
        done = 0

        def on_bytes(n: int) -> None:
            nonlocal done
            done += n
            if self.progress:
                self.progress("download", done, total)

        try:
            reader = io.BufferedReader(_IterReader(src.iter_from(offset), on_bytes), 1 << 20)
            index_tar_stream(reader, indexer)
            indexer.close()
            if self.progress:
                self.progress("sort", 0, indexer.n_rows)
            n = cache.sort_into([tmp], dest)
        except BaseException:
            indexer.abort()
            raise
        finally:
            tmp.unlink(missing_ok=True)
        cov = self._coverage(day)
        cache.register(dest, cov, n, {"tag": tag, "aircraft": indexer.n_aircraft,
                                      "parse_errors": indexer.n_errors})
        return cov

    def _download_raw(self, rel: Release) -> list[Path]:
        assert self.keep_raw_dir is not None
        out_dir = Path(self.keep_raw_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, part in enumerate(rel.parts):
            path = out_dir / part.name
            have = path.stat().st_size if path.exists() else 0
            if have < part.size:
                src = _http_concat(self.client, [part], sleep=self.sleep)
                with open(path, "ab") as fh:
                    for chunk in src.iter_from(have):
                        fh.write(chunk)
                        have += len(chunk)
                        if self.progress:
                            self.progress(f"download {part.name}", have, part.size)
            paths.append(path)
        return paths
