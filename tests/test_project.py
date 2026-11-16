"""Project files: save/load round trip, edited tracks re-fit as labelled, media check."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core.pipeline import analyze_file, refit_families
from dopplergeo.project import (
    MediaSource,
    Project,
    ProjectError,
    RegionModel,
    check_media,
    fit_models,
    load_project,
    save_project,
    sha256_file,
)


@pytest.fixture(scope="module")
def analysis(recording):
    media, _, _ = recording
    return media, analyze_file(media)


def test_roundtrip_keeps_inputs_tracks_and_fits(analysis, tmp_path):
    media, res = analysis
    p = Project(name="schiphol", media=MediaSource(path=str(media), sha256=sha256_file(media),
                                                   duration_s=150.0, sample_rate=16000),
                start_utc_override=datetime(2026, 11, 12, 14, 0, 1, tzinfo=timezone.utc), clock_bound_s=5,
                region=RegionModel(kind="near", lat=52.3, lon=4.76, radius_km=15), mutes=[(10.0, 12.5)],
                excluded_aircraft=["abc123"], fits=fit_models(res.families))
    save_project(tmp_path / "proj", p, res.tracks)
    q, tracks = load_project(tmp_path / "proj")
    assert q == p
    assert [t.id for t in tracks] == [t.id for t in res.tracks]
    for a, b in zip(res.tracks, tracks):
        np.testing.assert_array_equal(a.t, b.t)
        np.testing.assert_array_equal(a.f, b.f)
        np.testing.assert_array_equal(a.frame, b.frame)
        assert (a.family_id, a.k, a.manual) == (b.family_id, b.k, b.manual)
    assert check_media(q) == media


def test_restored_tracks_refit_to_the_same_parameters(analysis, tmp_path):
    _, res = analysis
    save_project(tmp_path, Project(), res.tracks)
    _, tracks = load_project(tmp_path)
    again = refit_families(tracks)
    before = {fr.family.id: fr.fit for fr in res.families if fr.fit}
    assert len(again) == len(before)
    for fr in again:
        a, b = before[fr.family.id], fr.fit
        assert b.v == pytest.approx(a.v, rel=1e-6) and b.d == pytest.approx(a.d, rel=1e-6)


def test_deleting_a_track_changes_only_its_family(analysis):
    _, res = analysis
    fam = next(fr.family for fr in res.families if len(fr.family.tracks) >= 2)
    drop = fam.tracks[-1].id
    kept = [t for t in res.tracks if t.id != drop]
    again = {fr.family.id: fr for fr in refit_families(kept)}
    assert drop not in [t.id for t in again[fam.id].family.tracks]
    assert again[fam.id].fit is not None


def test_changed_media_is_refused(analysis, tmp_path):
    media, _ = analysis
    copy = tmp_path / "m.m4a"
    copy.write_bytes(media.read_bytes())
    p = Project(media=MediaSource(path=str(copy), sha256=sha256_file(copy)))
    copy.write_bytes(media.read_bytes() + b"x")
    with pytest.raises(ProjectError, match="changed"):
        check_media(p)


def test_newer_schema_is_refused(tmp_path):
    (tmp_path / "project.json").write_text('{"version": 99}')
    with pytest.raises(ProjectError, match="newer"):
        load_project(tmp_path)
