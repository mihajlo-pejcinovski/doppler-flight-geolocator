"""Project files (spec "Data model and files"): a folder with ``project.json``
(versioned pydantic schema: inputs, user edits, result summaries) and
``tracks.parquet`` (tone tracks as edited by the user). The media file is
referenced by path and SHA-256, never copied.

Qt-free: the GUI saves and restores through this module, and so can scripts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from .core.spectral import SpectroSettings
from .core.tracking import ToneTrack

SCHEMA_VERSION = 1
PROJECT_JSON = "project.json"
TRACKS_PARQUET = "tracks.parquet"


class ProjectError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


class MediaSource(BaseModel):
    path: str
    sha256: str
    duration_s: float = 0.0
    sample_rate: int = 0
    start_utc: datetime | None = None
    start_utc_source: str = "none"
    gps_tag: tuple[float, float, float | None] | None = None


class SpectroModel(BaseModel):
    window_s: float = 1.0
    hop_s: float = 0.1
    zero_pad: int = 2
    fmin: float = 20.0
    fmax: float = 5000.0

    @classmethod
    def from_settings(cls, s: SpectroSettings) -> SpectroModel:
        return cls(window_s=s.window_s, hop_s=s.hop_s, zero_pad=s.zero_pad, fmin=s.fmin, fmax=s.fmax)

    def to_settings(self) -> SpectroSettings:
        return SpectroSettings(window_s=self.window_s, hop_s=self.hop_s, zero_pad=self.zero_pad,
                               fmin=self.fmin, fmax=self.fmax)


class RegionModel(BaseModel):
    kind: Literal["near", "bbox", "worldwide"] = "worldwide"
    lat: float | None = None
    lon: float | None = None
    radius_km: float = 50.0
    bbox: tuple[float, float, float, float] | None = None  # lat_min, lon_min, lat_max, lon_max


class FitModel(BaseModel):
    family_id: int
    ks: list[float]
    f0: float
    v: float
    d: float
    tau0: float
    t_cpa: float
    cov: list[list[float]]
    param_names: list[str]
    chi2_red: float
    grade: str
    sigma_v: float
    sigma_d: float
    sigma_t_cpa: float


class FlightQueryModel(BaseModel):
    provider: str
    t0: float
    t1: float
    bbox: tuple[float, float, float, float]
    fetched_at: datetime
    n_aircraft: int
    from_cache: bool = False


class CandidateModel(BaseModel):
    icao24: str
    callsign: str | None = None
    family_id: int
    cpa_time_utc: float
    cpa_range_m: float
    speed_match: float
    chi2: float | None = None
    included: bool = True
    used: bool = False


class Project(BaseModel):
    version: int = SCHEMA_VERSION
    name: str = "untitled"
    media: MediaSource | None = None
    # inputs and user edits
    start_utc_override: datetime | None = None
    clock_bound_s: float = 0.0
    region: RegionModel = Field(default_factory=RegionModel)
    spectro: SpectroModel = Field(default_factory=SpectroModel)
    mutes: list[tuple[float, float]] = Field(default_factory=list)
    noise_range: tuple[float, float] | None = None
    providers: list[str] = Field(default_factory=lambda: ["adsblol-api", "opensky", "adsblol"])
    elevation: str = "auto"
    height_m: float | None = None
    excluded_aircraft: list[str] = Field(default_factory=list)
    # A known true recording position (lat, lon) for files without a GPS tag, e.g. a
    # recordist's published geotag: shown on the map and compared with the solution.
    # Fit parameters the user locked per family: {family_id: {"v": 82.0, "d": 750.0}}
    fit_locks: dict[int, dict[str, float]] = Field(default_factory=dict)
    reference_position: tuple[float, float] | None = None
    reference_note: str = ""
    # result summaries (tracks themselves live in tracks.parquet)
    fits: list[FitModel] = Field(default_factory=list)
    flight_query: FlightQueryModel | None = None
    candidates: list[CandidateModel] = Field(default_factory=list)
    solution: dict | None = None


# ---------------------------------------------------------------------------
# Tracks <-> Parquet
# ---------------------------------------------------------------------------


def tracks_to_table(tracks: list[ToneTrack]) -> pa.Table:
    """One row per track point (long format), like ``ToneTrack`` in the spec."""
    cols: dict[str, list] = {k: [] for k in ("track_id", "family_id", "k", "manual", "frame", "t", "f",
                                             "snr_db", "sigma_f")}
    for tr in tracks:
        n = int(tr.t.size)
        cols["track_id"] += [int(tr.id)] * n
        cols["family_id"] += [None if tr.family_id is None else int(tr.family_id)] * n
        cols["k"] += [float(tr.k)] * n
        cols["manual"] += [bool(tr.manual)] * n
        cols["frame"] += np.asarray(tr.frame, dtype=np.int64).tolist()
        cols["t"] += np.asarray(tr.t, dtype=float).tolist()
        cols["f"] += np.asarray(tr.f, dtype=float).tolist()
        cols["snr_db"] += np.asarray(tr.snr_db, dtype=float).tolist()
        cols["sigma_f"] += np.asarray(tr.sigma_f, dtype=float).tolist()
    schema = pa.schema([("track_id", pa.int32()), ("family_id", pa.int32()), ("k", pa.float64()),
                        ("manual", pa.bool_()), ("frame", pa.int64()), ("t", pa.float64()), ("f", pa.float64()),
                        ("snr_db", pa.float64()), ("sigma_f", pa.float64())])
    return pa.table(cols, schema=schema)


def tracks_from_table(table: pa.Table) -> list[ToneTrack]:
    if table.num_rows == 0:
        return []
    d = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    out: list[ToneTrack] = []
    ids = d["track_id"].astype(int)
    for tid in np.unique(ids):
        m = ids == tid
        order = np.argsort(d["t"][m], kind="stable")
        fam = d["family_id"][m][0]
        out.append(ToneTrack(
            id=int(tid), frame=d["frame"][m][order].astype(np.intp), t=d["t"][m][order].astype(float),
            f=d["f"][m][order].astype(float), snr_db=d["snr_db"][m][order].astype(float),
            sigma_f=d["sigma_f"][m][order].astype(float), manual=bool(d["manual"][m][0]),
            family_id=None if fam is None or (isinstance(fam, float) and np.isnan(fam)) else int(fam),
            k=float(d["k"][m][0])))
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_project(folder: str | Path, project: Project, tracks: list[ToneTrack] | None = None) -> Path:
    """Write ``project.json`` (and ``tracks.parquet`` when tracks are given) into
    ``folder``. Writes go to a temporary file first, so a crash never leaves a
    half-written project."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PROJECT_JSON
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(project.model_dump_json(indent=2))
    tmp.replace(path)
    tp = folder / TRACKS_PARQUET
    if tracks is not None:
        ttmp = tp.with_suffix(".parquet.tmp")
        pq.write_table(tracks_to_table(tracks), ttmp)
        ttmp.replace(tp)
    return path


def load_project(folder: str | Path) -> tuple[Project, list[ToneTrack] | None]:
    folder = Path(folder)
    path = folder / PROJECT_JSON if folder.is_dir() else folder
    if not path.exists():
        raise ProjectError(f"no {PROJECT_JSON} in {folder}")
    raw = json.loads(path.read_text())
    v = raw.get("version", 0)
    if v > SCHEMA_VERSION:
        raise ProjectError(f"project version {v} is newer than this program ({SCHEMA_VERSION})")
    project = Project.model_validate(raw)
    tp = path.parent / TRACKS_PARQUET
    tracks = tracks_from_table(pq.read_table(tp)) if tp.exists() else None
    return project, tracks


def check_media(project: Project, base: Path | None = None) -> Path:
    """The project's media file, verified by SHA-256. A moved project may hold a
    path relative to its folder."""
    if project.media is None:
        raise ProjectError("project has no media file")
    p = Path(project.media.path)
    if not p.is_absolute() and base is not None:
        p = base / p
    if not p.exists():
        raise ProjectError(f"media file not found: {p}")
    if sha256_file(p) != project.media.sha256:
        raise ProjectError(f"media file changed since the project was saved: {p}")
    return p


# ---------------------------------------------------------------------------
# Result summaries
# ---------------------------------------------------------------------------


def fit_models(results) -> list[FitModel]:
    """From ``pipeline.FamilyResult`` objects (fitted ones only)."""
    out = []
    for fr in results:
        f = fr.fit
        if f is None:
            continue
        out.append(FitModel(
            family_id=fr.family.id, ks=[float(k) for k in fr.family.ks], f0=f.f0, v=f.v, d=f.d, tau0=f.tau0,
            t_cpa=f.t_cpa, cov=np.asarray(f.cov, float).tolist(), param_names=list(f.param_names),
            chi2_red=f.chi2_red, grade=f.grade, sigma_v=f.stderr("v"), sigma_d=f.stderr("d"),
            sigma_t_cpa=f.stderr("t_cpa")))
    return out


def candidate_models(solution) -> list[CandidateModel]:
    if solution is None:
        return []
    out = []
    for c in solution.candidates:
        out.append(CandidateModel(
            icao24=c.icao24, callsign=c.callsign, family_id=c.family_id, cpa_time_utc=c.t_cpa_utc,
            cpa_range_m=c.cpa_range_m, speed_match=c.speed_match,
            chi2=None if not np.isfinite(c.chi2) else float(c.chi2), included=c.included,
            used=solution.assignment.get(c.family_id) == c.icao24))
    return out
