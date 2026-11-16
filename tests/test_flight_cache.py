from datetime import datetime, timezone

import pytest

from dopplergeo.flights import (
    BBox,
    Coverage,
    FlightCache,
    FlightProvider,
    FlightQuery,
    FlightService,
    NoFlightData,
    ProviderError,
    TrackPoint,
    points_to_table,
)

T0 = datetime(2025, 12, 30, 14, 0, tzinfo=timezone.utc).timestamp()
REGION = BBox(52.0, 4.0, 53.0, 5.0)


def pts(icao="abc123", lat=52.5, lon=4.5, n=10, t0=T0, dt=5.0):
    return [TrackPoint(t0 + i * dt, icao, "CS1", lat + i * 1e-3, lon, 3000.0, 2950.0, 200.0, 0.0, 0.0)
            for i in range(n)]


@pytest.fixture
def cache(tmp_path):
    with FlightCache(tmp_path / "cache") as c:
        yield c


def test_write_and_query_filters_time_and_bbox(cache):
    rows = pts() + pts("out001", lat=40.0) + pts("late01", t0=T0 + 10_000)
    cache.write_part(points_to_table(rows), [Coverage("p", T0 - 100, T0 + 20_000, BBox.world())])
    q = FlightQuery(T0 - 10, T0 + 1000, REGION)
    tbl = cache.query(q, "p")
    assert set(tbl.column("icao24").to_pylist()) == {"abc123"}
    assert tbl.num_rows == 10
    assert tbl.column("t").to_pylist() == sorted(tbl.column("t").to_pylist())


def test_coverage_union_and_misses(cache):
    a = Coverage("p", T0, T0 + 100, REGION)
    b = Coverage("p", T0 + 100, T0 + 200, REGION)
    cache.write_part(points_to_table(pts(n=3)), [a])
    path_b = cache.write_part(points_to_table(pts(n=3, t0=T0 + 120)), [b])
    assert cache.covering_parts(FlightQuery(T0 + 10, T0 + 190, REGION), "p") is not None
    assert cache.covering_parts(FlightQuery(T0 + 10, T0 + 250, REGION), "p") is None  # time gap
    assert cache.covering_parts(FlightQuery(T0 + 10, T0 + 90, REGION.expanded(5)), "p") is None  # bbox
    assert cache.covering_parts(FlightQuery(T0 + 10, T0 + 90, REGION), "other") is None
    assert cache.covering_provider(FlightQuery(T0 + 10, T0 + 190, REGION)) == "p"
    path_b.unlink()
    assert cache.covering_parts(FlightQuery(T0 + 10, T0 + 190, REGION), "p") is None


def test_overlapping_parts_are_deduplicated(cache):
    cov = [Coverage("p", T0 - 10, T0 + 100, REGION)]
    cache.write_part(points_to_table(pts()), cov)
    cache.write_part(points_to_table(pts()), cov)
    assert cache.query(FlightQuery(T0 - 5, T0 + 90, REGION), "p").num_rows == 10


def test_cache_persists_across_instances(tmp_path):
    with FlightCache(tmp_path) as c:
        c.write_part(points_to_table(pts()), [Coverage("p", T0 - 10, T0 + 100, REGION)])
    with FlightCache(tmp_path) as c:
        assert c.query(FlightQuery(T0, T0 + 50, REGION), "p").num_rows == 10


class FakeProvider(FlightProvider):
    def __init__(self, name, fail=None, refuse=None):
        self.name = name
        self.attribution = f"data by {name}"
        self.fail, self.refuse = fail, refuse
        self.calls = 0

    def can_serve(self, query, now):
        return self.refuse

    def fetch(self, query, cache):
        self.calls += 1
        if self.fail:
            raise ProviderError(self.fail)
        cov = [Coverage(self.name, query.t0, query.t1, query.bbox)]
        cache.write_part(points_to_table(pts()), cov)
        return cov


def test_service_falls_back_then_serves_from_cache(cache):
    broken, refusing, good = FakeProvider("a", fail="HTTP 503"), FakeProvider("b", refuse="too old"), \
        FakeProvider("c")
    svc = FlightService(cache, [broken, refusing, good])
    q = FlightQuery(T0 - 10, T0 + 100, REGION)
    first = svc.get(q)
    assert (first.provider, first.from_cache, first.table.num_rows) == ("c", False, 10)
    assert first.attribution == "data by c"
    assert refusing.calls == 0
    again = svc.get(FlightQuery(T0, T0 + 50, REGION))
    assert again.from_cache and good.calls == 1 and broken.calls == 1
    (track,) = again.tracks()
    assert track.icao24 == "abc123"


def test_service_reports_every_reason(cache):
    svc = FlightService(cache, [FakeProvider("a", fail="boom"), FakeProvider("b", refuse="no creds")])
    with pytest.raises(NoFlightData) as e:
        svc.get(FlightQuery(T0, T0 + 10, REGION))
    assert e.value.reasons == {"a": "boom", "b": "no creds"}
