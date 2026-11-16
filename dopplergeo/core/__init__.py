"""Qt-free core: physics, signal processing, fitting, solving."""

from .atmosphere import SPEED_OF_SOUND_15C, path_mean_speed_of_sound, speed_of_sound
from .doppler import (
    DopplerFit,
    bootstrap_fit,
    cpa_reception_time,
    doppler_frequency,
    fit_doppler,
    frequency_from_trajectory,
    initial_guess,
    radial_speed,
    retarded_offset,
    retarded_time,
)
from .synthetic import StraightLineFlight, render_audio, synthesize_track

__all__ = [
    "SPEED_OF_SOUND_15C",
    "DopplerFit",
    "StraightLineFlight",
    "bootstrap_fit",
    "cpa_reception_time",
    "doppler_frequency",
    "fit_doppler",
    "frequency_from_trajectory",
    "initial_guess",
    "path_mean_speed_of_sound",
    "radial_speed",
    "render_audio",
    "retarded_offset",
    "retarded_time",
    "speed_of_sound",
    "synthesize_track",
]
