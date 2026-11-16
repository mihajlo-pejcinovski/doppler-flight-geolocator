import gzip
import io
import json
import re
import tarfile
from datetime import datetime, timezone

import httpx
import pytest
import respx

from dopplergeo.core.geo import FOOT, FPM, KNOT
from dopplergeo.flights import BBox, FlightCache, FlightQuery, FlightService
from dopplergeo.flights.adsblol import (
    AdsbLolProvider,
    DownloadNeedsConfirmation,
    TraceIndexer,
    find_traces_offset,
    _local_concat,
    parse_trace,
)
from dopplergeo.flights.base import SCHEMA

DAY = datetime(2025, 12, 30, tzinfo=timezone.utc)
BASE = DAY.timestamp()
TAG = "v2025.12.30-planes-readsb-prod-0"
REPO = "adsblol/globe_history_2025"
DL = f"https://github.com/{REPO}/releases/download/{TAG}"


def trace(icao, lat0, lon0, n=20, t0=50_000.0, callsign_at=5, flight="KLM1234 ", leg_at=None):
    rows = []
    for i in range(n):
        det = {"type": "adsb_icao", "flight": flight} if i == callsign_at else None
        flags = 2 if i == leg_at else 0
        rows.append([t0 + 4 * i, lat0 + i * 0.002, lon0, 10_000 + 10 * i, 300.0, 0.0, flags, 640, det,
                     "adsb_icao", 10_200 + 10 * i, 600, 280, 0.1])
    return {"icao": icao, "r": "PH-BXA", "t": "B738", "timestamp": BASE, "trace": rows}


def make_archive(traces, heatmap_bytes=300_000):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        def add(name, data=None):
            ti = tarfile.TarInfo(name)
            if data is None:
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        add("./")
        add("./heatmap")
        for k in range(3):
            add(f"./heatmap/{k}.bin.ttf", bytes([k]) * heatmap_bytes)
        add("./README.txt", b"readme")
        add("./traces")
        for i, tr in enumerate(traces):
            raw = json.dumps(tr).encode()
            add(f"./traces/{tr['icao'][-2:]}/trace_full_{tr['icao']}.json", gzip.compress(raw) if i % 2 else raw)
        add("./zzz_after_traces.txt", b"x" * 1000)
    data = buf.getvalue()
    cut = len(data) // 2 + 123  # split mid-member, like split(1)
    return data, [("tar.aa", data[:cut]), ("tar.ab", data[cut:])]


TRACES = [trace("4840d6", 52.30, 4.70), trace("3c6444", 52.60, 4.90, callsign_at=-1),
          trace("a1b2c3", 40.0, -74.0), trace("~23ab01", 52.31, 4.71, leg_at=10)]


class RangeServer:
    """Serves byte ranges of the archive parts; can cut the first full-body stream short."""

    def __init__(self, parts, cut_first_stream=False):
        self.parts = dict(parts)
        self.cut = cut_first_stream
        self.served: list[tuple[str, int, int]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        part = request.url.path.rsplit(".", 1)[-1]
        data = self.parts[f"tar.{part}"]
        m = re.match(r"bytes=(\d+)-(\d*)", request.headers.get("Range", ""))
        a = int(m[1]) if m else 0
        b = int(m[2]) + 1 if m and m[2] else len(data)
        body = data[a:b]
        if self.cut and not m[2] and len(body) > 4096:
            self.cut = False
            body = body[: len(body) // 3]  # connection drops mid-download
        self.served.append((part, a, len(body)))
        return httpx.Response(206, content=body,
                              headers={"Content-Range": f"bytes {a}-{a + len(body) - 1}/{len(data)}"})


def release_json(parts):
    return {"tag_name": TAG, "assets": [
        {"name": f"{TAG}.{name}", "size": len(body), "browser_download_url": f"{DL}/{TAG}.{name}"}
        for name, body in parts]}


def mock_github(router, parts, direct=True):
    tag_url = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    if direct:
        router.get(tag_url).respond(200, json=release_json(parts))
    else:
        router.get(tag_url).respond(404, json={"message": "Not Found"})
        older = {"tag_name": "v2025.12.29-planes-readsb-prod-0", "assets": []}
        router.get(f"https://api.github.com/repos/{REPO}/releases").respond(
            200, json=[{**release_json(parts), "tag_name": TAG.replace("-0", "-1")}, older])


def cols():
    return {n: [] for n in SCHEMA.names}


def test_parse_trace_units_callsign_and_legs():
    c = cols()
    n = parse_trace(TRACES[3], c)
    assert n == 20
    assert c["t"][0] == BASE + 50_000.0
    assert c["alt_baro_m"][0] == pytest.approx(10_000 * FOOT)
    assert c["alt_geo_m"][0] == pytest.approx(10_200 * FOOT)
    assert c["gs_mps"][0] == pytest.approx(300 * KNOT)
    assert c["vrate_mps"][0] == pytest.approx(600 * FPM)  # geometric rate preferred
    assert c["callsign"][:10] == ["KLM1234"] * 10  # back-filled before first report
    assert c["callsign"][10:] == [None] * 10  # new leg resets it
    assert c["icao24"][0] == "~23ab01"


def test_parse_trace_ground_geom_flag_and_bbox():
    tr = trace("abcdef", 52.0, 4.0, n=3)
    tr["trace"][0][3] = "ground"
    tr["trace"][1][6] = 8  # altitude field is geometric
    c = cols()
    parse_trace(tr, c)
    assert c["on_ground"] == [True, False, False] and c["alt_baro_m"][0] is None
    assert c["alt_geo_m"][1] == pytest.approx(10_010 * FOOT) and c["alt_baro_m"][1] is None
    c = cols()
    assert parse_trace(tr, c, keep=BBox(52.001, 3.0, 53.0, 5.0)) == 2


def test_find_traces_offset_skips_heatmaps(tmp_path):
    data, parts = make_archive(TRACES)
    paths = []
    for name, body in parts:
        p = tmp_path / f"x.{name}"
        p.write_bytes(body)
        paths.append(p)
    off = find_traces_offset(_local_concat(paths))
    assert off > 900_000
    assert tarfile.TarInfo.frombuf(data[off:off + 512], "utf-8", "surrogateescape").name == "./traces"


def test_index_local_archive(tmp_path):
    _, parts = make_archive(TRACES)
    paths = []
    for name, body in parts:
        (tmp_path / f"day.{name}").write_bytes(body)
        paths.append(tmp_path / f"day.{name}")
    with FlightCache(tmp_path / "c") as cache:
        AdsbLolProvider(client=httpx.Client()).index_archive(DAY, cache, paths)
        q = FlightQuery(BASE + 49_000, BASE + 52_000, BBox(52.0, 4.0, 53.0, 5.0))
        tbl = cache.query(q, "adsblol")
        assert set(tbl.column("icao24").to_pylist()) == {"4840d6", "3c6444", "~23ab01"}
        assert tbl.num_rows == 60
        rec = cache.records("adsblol")[0]
        assert json.loads(rec["meta"])["aircraft"] == 4


@pytest.mark.parametrize("direct", [True, False])
def test_download_streams_only_traces_and_resumes(tmp_path, direct):
    data, parts = make_archive(TRACES)
    server = RangeServer(parts, cut_first_stream=True)
    with respx.mock(assert_all_called=False) as router:
        mock_github(router, parts, direct)
        router.get(url__startswith=DL).mock(side_effect=server)
        plans = []
        prov = AdsbLolProvider(client=httpx.Client(), confirm=lambda p: plans.append(p) or True,
                               sleep=lambda s: None, github_token=None)
        with FlightCache(tmp_path) as cache:
            svc = FlightService(cache, [prov], now=lambda: BASE + 3 * 86400)
            q = FlightQuery(BASE + 49_000, BASE + 52_000, BBox(52.0, 4.0, 53.0, 5.0))
            res = svc.get(q)
            assert res.provider == "adsblol" and not res.from_cache
            assert res.table.num_rows == 60
            assert "ODbL" in res.attribution
            # The second query on the same day never touches the network.
            n_requests = len(server.served)
            assert svc.get(FlightQuery(BASE + 100, BASE + 200, BBox.world())).from_cache
            assert len(server.served) == n_requests

    (plan,) = plans
    assert plan.release.tag.startswith("v2025.12.30-planes-readsb-prod-")
    size_aa = len(parts[0][1])
    streams = [(a + (size_aa if part == "ab" else 0), n) for part, a, n in server.served if n > 512]
    # Streaming started at the traces/ header, never at byte 0 of the heatmaps ...
    assert min(a for a, _ in streams) == plan.traces_offset > 900_000
    # ... and covered the rest of the archive exactly once, despite the dropped connection.
    assert sum(n for _, n in streams) == len(data) - plan.traces_offset
    assert len(streams) == 2  # first stream cut short, then resumed with a Range request


def test_download_requires_confirmation(tmp_path):
    _, parts = make_archive(TRACES)
    with respx.mock(assert_all_called=False) as router:
        mock_github(router, parts)
        router.get(url__startswith=DL).mock(side_effect=RangeServer(parts))
        prov = AdsbLolProvider(client=httpx.Client(), github_token=None)
        with FlightCache(tmp_path) as cache:
            svc = FlightService(cache, [prov], now=lambda: BASE + 3 * 86400)
            with pytest.raises(DownloadNeedsConfirmation) as e:
                svc.get(FlightQuery(BASE + 10, BASE + 20, BBox.world()))
    assert "skipping" in e.value.plan.describe()


def test_can_serve_only_finished_days():
    prov = AdsbLolProvider(client=httpx.Client())
    q = FlightQuery(BASE + 10, BASE + 20, BBox.world())
    assert prov.can_serve(q, BASE + 3600) is not None
    assert prov.can_serve(q, BASE + 86400 + 1) is None
    keep = AdsbLolProvider(client=httpx.Client(), keep_bbox=BBox(50, 0, 55, 10))
    assert keep.can_serve(q, BASE + 2 * 86400) is not None


def test_indexer_skips_corrupt_members(tmp_path):
    ix = TraceIndexer(tmp_path / "x.parquet", flush_rows=5)
    ix.add_json(b"not json")
    ix.add_json(gzip.compress(json.dumps(TRACES[0]).encode()))
    ix.close()
    assert (ix.n_errors, ix.n_aircraft, ix.n_rows) == (1, 1, 20)
