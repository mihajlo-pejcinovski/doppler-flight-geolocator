"""Local flight cache: Parquet parts per provider/date plus a DuckDB coverage index.

Every provider writes what it fetched as a Parquet part and registers the
(time, bbox) region that part answers completely. A query is served locally
when the union of one provider's coverage contains it. Parts are sorted by
1-degree cell then time, so DuckDB's row-group statistics prune a bbox query
down to a few row groups even for a whole-world day.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pyarrow as pa

from .base import COLUMNS, SCHEMA, BBox, Coverage, FlightQuery

ROW_GROUP_SIZE = 65_536
SORT_KEY = "floor(lat), floor(lon), icao24, t"


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "dopplergeo" / "flights"


def _arrow(rel: duckdb.DuckDBPyRelation) -> pa.Table:
    """Relation as an Arrow table on DuckDB 1.x (``arrow``) and 2.x (``to_arrow_table``)."""
    fn = getattr(rel, "to_arrow_table", None) or rel.arrow
    return fn()


def _sql_list(paths: Sequence[Path | str]) -> str:
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


class FlightCache:
    def __init__(self, root: Path | str | None = None, memory_limit: str = "2GB"):
        self.root = Path(root) if root is not None else default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.root / "index.duckdb"))
        # A whole adsb.lol day is ~100M rows; without a cap DuckDB's ORDER BY grew past 11 GB
        # and was OOM-killed. With a limit and a spill folder it sorts out of core.
        spill = self.root / ".duckdb_tmp"
        self.db.execute(f"SET memory_limit = '{memory_limit}'")
        self.db.execute(f"SET temp_directory = '{spill}'")
        self.db.execute("SET preserve_insertion_order = false")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS coverage (
                provider VARCHAR, path VARCHAR, t0 DOUBLE, t1 DOUBLE,
                lat_min DOUBLE, lon_min DOUBLE, lat_max DOUBLE, lon_max DOUBLE,
                n_points BIGINT, fetched_at DOUBLE, meta VARCHAR)""")

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> FlightCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing ------------------------------------------------------------

    def part_path(self, provider: str, t: float, name: str | None = None) -> Path:
        day = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        d = self.root / provider / f"date={day}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name or uuid.uuid4().hex[:12]}.parquet"

    def sort_into(self, sources: Sequence[Path | str] | pa.Table, dest: Path) -> int:
        """Write rows (a table or Parquet files) to ``dest`` sorted for bbox pruning. Returns row count."""
        tmp = dest.with_suffix(".parquet.tmp")
        if isinstance(sources, pa.Table):
            self.db.register("_incoming", sources.select(COLUMNS).cast(SCHEMA))
            src = "_incoming"
        else:
            src = f"read_parquet({_sql_list(sources)})"
        try:
            self.db.execute(
                f"COPY (SELECT {', '.join(COLUMNS)} FROM {src} ORDER BY {SORT_KEY}) TO '{tmp}' "
                f"(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP_SIZE})")
        finally:
            if src == "_incoming":
                self.db.unregister("_incoming")
        os.replace(tmp, dest)
        return self.db.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]

    def register(self, path: Path, coverage: Iterable[Coverage], n_points: int,
                 meta: dict | None = None) -> None:
        now = time.time()
        for c in coverage:
            self.db.execute(
                "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [c.provider, str(path), c.t0, c.t1, *c.bbox.as_tuple(), n_points, now,
                 json.dumps(meta or {})])

    def write_part(self, table: pa.Table, coverage: Sequence[Coverage], name: str | None = None,
                   meta: dict | None = None) -> Path:
        """Store a fetched table and register the regions it answers completely."""
        if not coverage:
            raise ValueError("a part must cover something")
        path = self.part_path(coverage[0].provider, coverage[0].t0, name)
        n = self.sort_into(table, path)
        self.register(path, coverage, n, meta)
        return path

    # -- reading ------------------------------------------------------------

    def records(self, provider: str | None = None) -> list[dict]:
        sql = "SELECT * FROM coverage" + (" WHERE provider = ?" if provider else "") + " ORDER BY t0"
        cur = self.db.execute(sql, [provider] if provider else [])
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def covering_parts(self, query: FlightQuery, provider: str) -> list[str] | None:
        """Parts of ``provider`` that together answer ``query``, or None if not covered."""
        rows = self.db.execute(
            """SELECT path, t0, t1 FROM coverage
               WHERE provider = ? AND lat_min <= ? AND lon_min <= ? AND lat_max >= ? AND lon_max >= ?
                 AND t1 > ? AND t0 < ? ORDER BY t0""",
            [provider, *query.bbox.as_tuple(), query.t0, query.t1]).fetchall()
        reach = query.t0
        paths: list[str] = []
        for path, t0, t1 in rows:
            if t0 > reach:
                break
            if t1 > reach:
                reach = t1
            if Path(path).exists() and path not in paths:
                paths.append(path)
            elif not Path(path).exists():
                return None  # index points at a deleted file; treat as a miss
        return paths if reach >= query.t1 and paths else None

    def covering_provider(self, query: FlightQuery, providers: Sequence[str] | None = None) -> str | None:
        names = providers or [r[0] for r in self.db.execute("SELECT DISTINCT provider FROM coverage").fetchall()]
        for p in names:
            if self.covering_parts(query, p) is not None:
                return p
        return None

    def query(self, query: FlightQuery, provider: str) -> pa.Table:
        """Rows inside the query window and bbox from one provider's covering parts."""
        paths = self.covering_parts(query, provider)
        if paths is None:
            raise KeyError(f"cache does not cover {query} for {provider}")
        b = query.bbox
        rel = self.db.sql(
            f"""SELECT DISTINCT ON (icao24, t) {', '.join(COLUMNS)} FROM read_parquet({_sql_list(paths)})
                WHERE t >= ? AND t <= ? AND lat >= ? AND lat <= ? AND lon >= ? AND lon <= ?
                ORDER BY icao24, t""",
            params=[query.t0, query.t1, b.lat_min, b.lat_max, b.lon_min, b.lon_max])
        return _arrow(rel).cast(SCHEMA)

    def remove(self, provider: str, path: str | None = None) -> None:
        if path is None:
            rows = self.db.execute("SELECT DISTINCT path FROM coverage WHERE provider = ?", [provider]).fetchall()
            self.db.execute("DELETE FROM coverage WHERE provider = ?", [provider])
            for (p,) in rows:
                Path(p).unlink(missing_ok=True)
        else:
            self.db.execute("DELETE FROM coverage WHERE provider = ? AND path = ?", [provider, path])
            Path(path).unlink(missing_ok=True)


__all__ = ["BBox", "FlightCache", "default_cache_dir"]
