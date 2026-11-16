"""Ordered provider fallback over the local cache."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pyarrow as pa

from .base import FlightProvider, FlightQuery, FlightTrack, ProviderError, tracks_from_table
from .cache import FlightCache


class NoFlightData(ProviderError):
    def __init__(self, query: FlightQuery, reasons: dict[str, str]):
        self.reasons = reasons
        detail = "; ".join(f"{k}: {v}" for k, v in reasons.items()) or "no providers configured"
        super().__init__(f"no flight data for the window ({detail})")


@dataclass
class FlightData:
    query: FlightQuery
    provider: str
    attribution: str
    table: pa.Table
    from_cache: bool

    def tracks(self, max_gap_s: float = 30.0) -> list[FlightTrack]:
        return tracks_from_table(self.table, max_gap_s=max_gap_s)


@dataclass
class FlightService:
    cache: FlightCache
    providers: Sequence[FlightProvider]
    now: Callable[[], float] = field(default=time.time)

    def get(self, query: FlightQuery) -> FlightData:
        """Serve from the cache when any provider's coverage already answers the
        query; otherwise try providers in order and serve from what they cached."""
        names = [p.name for p in self.providers]
        by_name = {p.name: p for p in self.providers}
        hit = self.cache.covering_provider(query, names)
        if hit is not None:
            return FlightData(query, hit, by_name[hit].attribution, self.cache.query(query, hit), True)

        reasons: dict[str, str] = {}
        for p in self.providers:
            why = p.can_serve(query, self.now())
            if why:
                reasons[p.name] = why
                continue
            try:
                p.fetch(query, self.cache)
            except ProviderError as e:
                reasons[p.name] = str(e)
                if getattr(e, "plan", None) is not None:  # needs user confirmation: surface it
                    raise
                continue
            if self.cache.covering_parts(query, p.name) is not None:
                return FlightData(query, p.name, p.attribution, self.cache.query(query, p.name), False)
            reasons[p.name] = "fetch finished but did not cover the query"
        raise NoFlightData(query, reasons)
