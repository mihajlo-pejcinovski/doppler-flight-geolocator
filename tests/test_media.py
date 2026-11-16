"""Ingest: decode, resample, metadata."""

from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core.media import load_audio, parse_iso6709, parse_timestamp, write_audio_file


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+48.8566+002.3522+035.000/", (48.8566, 2.3522, 35.0)),
        ("+37.7749-122.4194/", (37.7749, -122.4194, None)),
        ("-33.8688+151.2093+012.3/", (-33.8688, 151.2093, 12.3)),
        ("garbage", None),
        ("+95.0+010.0/", None),  # latitude out of range
    ],
)
def test_iso6709(text, expected):
    assert parse_iso6709(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-11-13T10:22:31.000000Z", datetime(2026, 11, 13, 10, 22, 31, tzinfo=timezone.utc)),
        ("2026-11-13T11:22:31+0100", datetime(2026, 11, 13, 10, 22, 31, tzinfo=timezone.utc)),
        ("2026-11-13 10:22:31", datetime(2026, 11, 13, 10, 22, 31, tzinfo=timezone.utc)),
        ("1904-01-01T00:00:00Z", None),  # unset QuickTime clock
        ("not a date", None),
    ],
)
def test_timestamps(text, expected):
    assert parse_timestamp(text) == expected


@pytest.fixture
def tone():
    sr = 44_100
    t = np.arange(int(sr * 3.0)) / sr
    return sr, 0.5 * np.sin(2 * np.pi * 440.0 * t)


def test_aac_roundtrip_metadata_and_resample(tmp_path, tone):
    sr, x = tone
    path = tmp_path / "clip.m4a"
    write_audio_file(path, x, sr, metadata={
        "creation_time": "2026-11-13T10:22:31.000000Z",
        "location": "+48.8566+002.3522+035.000/",
    })
    clip = load_audio(path, sample_rate=16_000)
    assert clip.sample_rate == 16_000
    assert clip.native_sample_rate == sr
    assert clip.duration == pytest.approx(3.0, abs=0.05)
    assert clip.start_utc == datetime(2026, 11, 13, 10, 22, 31, tzinfo=timezone.utc)
    assert clip.start_utc_source == "creation_time"
    assert clip.gps == pytest.approx((48.8566, 2.3522, 35.0))
    # The 440 Hz tone survives encode + resample.
    spec = np.abs(np.fft.rfft(clip.samples[4000:20000] * np.hanning(16000)))
    f_peak = np.argmax(spec) * 16_000 / 16000
    assert f_peak == pytest.approx(440.0, abs=1.0)


def test_wav_without_metadata_and_native_rate(tmp_path, tone):
    sr, x = tone
    path = tmp_path / "clip.wav"
    write_audio_file(path, x, sr, codec="pcm_s16le")
    clip = load_audio(path, sample_rate=None)
    assert clip.sample_rate == sr
    assert clip.start_utc is None and clip.start_utc_source == "none"
    assert clip.gps is None
    np.testing.assert_allclose(clip.samples, x, atol=1e-4)


def test_no_audio_stream_is_an_error(tmp_path):
    import av

    path = tmp_path / "video_only.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height = 32, 32
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
        for p in s.encode(frame):
            c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    with pytest.raises(ValueError, match="no audio"):
        load_audio(path)


def test_android_creation_time_is_the_stop_time(tmp_path, tone):
    # Android stamps creation_time when recording stops; the start is earlier
    # by the file's duration (seen on a real Samsung video).
    sr, x = tone
    path = tmp_path / "VID.mp4"
    write_audio_file(path, x, sr, metadata={"creation_time": "2026-11-13T12:12:07.000000Z",
                                            "com.android.version": "13"})
    clip = load_audio(path)
    assert clip.metadata.get("com.android.version") == "13"
    assert clip.start_utc_source.startswith("creation_time (android")
    expected = datetime(2026, 11, 13, 12, 12, 7, tzinfo=timezone.utc).timestamp() - 3.0
    assert clip.start_utc.timestamp() == pytest.approx(expected, abs=0.1)
