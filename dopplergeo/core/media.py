"""Audio ingest: demux any FFmpeg-readable file, read time/location metadata,
downmix to mono float32 and resample.

Timeline convention: ``AudioClip.samples[i]`` is at media time
``pts_offset + i / sample_rate`` seconds, and at UTC
``start_utc + that`` when ``start_utc`` is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

try:  # soxr is the preferred resampler; fall back to polyphase if missing.
    import soxr
except ImportError:  # pragma: no cover
    soxr = None

DEFAULT_SAMPLE_RATE = 16_000

# Keys seen in phone videos, in priority order.
_TIME_KEYS = (
    "com.apple.quicktime.creationdate",  # iPhone: local time with offset
    "creation_time",  # MP4 mvhd, UTC
    "date",
)
_LOCATION_KEYS = (
    "com.apple.quicktime.location.ISO6709",
    "location",  # Android / ffmpeg ©xyz
    "location-eng",
)
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")


@dataclass
class AudioClip:
    samples: NDArray[np.float32]
    sample_rate: int
    native_sample_rate: int
    pts_offset: float
    start_utc: datetime | None
    start_utc_source: str  # metadata key it came from, or "none"
    gps: tuple[float, float, float | None] | None  # lat, lon, alt_m
    path: str
    metadata: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate

    def times(self) -> NDArray[np.float64]:
        """Media time (s) of every sample."""
        return self.pts_offset + np.arange(self.samples.size) / self.sample_rate

    def utc_epoch(self) -> float | None:
        """Unix time of media time 0, if known."""
        return None if self.start_utc is None else self.start_utc.timestamp()


def parse_iso6709(text: str) -> tuple[float, float, float | None] | None:
    """Parse '+37.7749-122.4194+012.300/' style strings (degrees, metres)."""
    m = _ISO6709.search(text.strip())
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    alt = float(m.group(3)) if m.group(3) else None
    return lat, lon, alt


def parse_timestamp(text: str) -> datetime | None:
    """Parse ISO-8601 timestamps as written by cameras and ffmpeg. Naive times are
    taken as UTC (the MP4 ``creation_time`` convention)."""
    s = text.split(";")[0].strip().replace("Z", "+00:00")  # muxers may repeat a tag: "a;a"
    # '2026-11-13T10:22:31+0100' -> '+01:00' for fromisoformat on 3.11
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if dt.year < 1990:  # unset QuickTime clocks show 1904/1970
        return None
    return dt.astimezone(timezone.utc)


def _collect_metadata(container: av.container.InputContainer) -> dict[str, str]:
    meta = {str(k): str(v) for k, v in container.metadata.items()}
    for stream in container.streams:
        for k, v in stream.metadata.items():
            meta.setdefault(str(k), str(v))
    return meta


def _resample(x: NDArray[np.float32], sr_in: int, sr_out: int) -> NDArray[np.float32]:
    if sr_in == sr_out:
        return x
    if soxr is not None:
        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_audio(
    path: str | Path,
    sample_rate: int | None = DEFAULT_SAMPLE_RATE,
    stream_index: int = 0,
) -> AudioClip:
    """Decode the ``stream_index``-th audio stream of ``path``.

    ``sample_rate=None`` keeps the native rate. Channels are averaged to mono.
    """
    path = str(path)
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"{path}: no audio stream")
        stream = container.streams.audio[stream_index]
        native_sr = int(stream.rate or stream.codec_context.sample_rate)
        pts_offset = (
            float(stream.start_time * stream.time_base)
            if stream.start_time is not None and stream.time_base is not None
            else 0.0
        )
        chunks = []
        for frame in container.decode(stream):
            a = frame.to_ndarray().astype(np.float32)
            if frame.format.is_planar:  # (channels, n)
                a = a.mean(axis=0)
            else:  # packed: (1, n * channels)
                a = a.reshape(-1, frame.layout.nb_channels).mean(axis=1)
            if np.issubdtype(frame.to_ndarray().dtype, np.integer):
                a /= float(np.iinfo(frame.to_ndarray().dtype).max)
            chunks.append(a)
        meta = _collect_metadata(container)
        container_s = float(container.duration) / 1e6 if container.duration else None

    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    out_sr = native_sr if sample_rate is None else int(sample_rate)
    x = _resample(x, native_sr, out_sr)

    start, source = None, "none"
    for key in _TIME_KEYS:
        if key in meta and (dt := parse_timestamp(meta[key])) is not None:
            start, source = dt, key
            break
    if source == "creation_time" and "com.android.version" in meta:
        # Android's MPEG4Writer stamps creation_time when recording *stops*
        # (seen on a real phone video: tag 12:12:07Z, 60.8 s long, file name
        # 13:11:05 local = 12:11:05Z). Shift to the start.
        length = container_s if container_s else x.size / out_sr
        start = start - timedelta(seconds=length)
        source = "creation_time (android stop time - duration)"
    gps = None
    for key in _LOCATION_KEYS:
        if key in meta and (g := parse_iso6709(meta[key])) is not None:
            gps = g
            break

    return AudioClip(
        samples=x,
        sample_rate=out_sr,
        native_sample_rate=native_sr,
        pts_offset=pts_offset,
        start_utc=start,
        start_utc_source=source,
        gps=gps,
        path=path,
        metadata=meta,
    )


def write_audio_file(
    path: str | Path,
    samples: NDArray,
    sample_rate: int,
    codec: str = "aac",
    bit_rate: int = 128_000,
    metadata: dict[str, str] | None = None,
) -> None:
    """Encode mono float samples to a file (``.m4a``/``.mp4`` with AAC, or
    ``.wav`` with ``codec='pcm_s16le'``). Used for tests and fixtures."""
    x = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    # Keep custom keys (e.g. com.android.version) in MP4 files.
    opts = {"movflags": "use_metadata_tags"} if str(path).endswith((".mp4", ".m4a", ".mov")) else {}
    with av.open(str(path), mode="w", options=opts) as container:
        for k, v in (metadata or {}).items():
            container.metadata[k] = v
        stream = container.add_stream(codec, rate=sample_rate, layout="mono")
        if codec != "pcm_s16le":
            stream.bit_rate = bit_rate
        fmt = "s16" if codec == "pcm_s16le" else "fltp"
        frame_len = 1024
        for i in range(0, x.size, frame_len):
            chunk = x[i : i + frame_len]
            if fmt == "s16":
                data = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)[None, :]
            else:
                data = chunk[None, :]
            frame = av.AudioFrame.from_ndarray(data, format=fmt, layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
