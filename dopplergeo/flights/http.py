"""Shared HTTP plumbing: one client factory and a retrying request helper."""

from __future__ import annotations

import time
from typing import Callable

import httpx

from .. import __version__
from .base import ProviderError

USER_AGENT = f"dopplergeo/{__version__}"


class RateLimited(ProviderError):
    def __init__(self, url: str, retry_after: float | None):
        self.retry_after = retry_after
        wait = f"; retry after {retry_after:.0f} s" if retry_after is not None else ""
        super().__init__(f"rate limited by {httpx.URL(url).host}{wait}")


def make_client(timeout: float = 30.0, headers: dict[str, str] | None = None) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0), follow_redirects=True,
                        headers={"User-Agent": USER_AGENT, **(headers or {})})


def retry_after(resp: httpx.Response) -> float | None:
    for h in ("X-Rate-Limit-Retry-After-Seconds", "Retry-After"):
        v = resp.headers.get(h)
        if v is not None:
            try:
                return max(0.0, float(v))
            except ValueError:
                return None
    return None


def request(client: httpx.Client, method: str, url: str, *, retries: int = 3, backoff_s: float = 1.0,
            max_wait_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
            ok_statuses: tuple[int, ...] = (), **kw) -> httpx.Response:
    """Send with retries on transport errors, 5xx and short 429s.

    Returns the response for any 2xx status or one listed in ``ok_statuses``;
    raises ``RateLimited`` when told to wait longer than ``max_wait_s`` and
    ``ProviderError`` for other failures.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.request(method, url, **kw)
        except httpx.TransportError as e:
            last = e
            if attempt < retries:
                sleep(backoff_s * 2 ** attempt)
            continue
        if resp.is_success or resp.status_code in ok_statuses:
            return resp
        if resp.status_code == 429:
            wait = retry_after(resp)
            if attempt >= retries or (wait or 0.0) > max_wait_s:
                raise RateLimited(url, wait)
            sleep(wait if wait is not None else backoff_s * 2 ** attempt)
            continue
        if resp.status_code >= 500 and attempt < retries:
            sleep(backoff_s * 2 ** attempt)
            continue
        raise ProviderError(f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
    raise ProviderError(f"{method} {url} failed after {retries + 1} attempts: {last}")
