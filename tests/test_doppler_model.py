"""Physics checks on the closed-form model against the general trajectory solver."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dopplergeo.core import (
    SPEED_OF_SOUND_15C,
    StraightLineFlight,
    cpa_reception_time,
    doppler_frequency,
    frequency_from_trajectory,
    path_mean_speed_of_sound,
    radial_speed,
    retarded_offset,
    retarded_time,
    speed_of_sound,
)

C = SPEED_OF_SOUND_15C


def test_speed_of_sound_reference_values():
    assert speed_of_sound(15.0) == pytest.approx(340.29, abs=0.05)
    assert speed_of_sound(0.0) == pytest.approx(331.3, abs=0.1)


def test_path_mean_speed_of_sound_is_between_end_values():
    c_mean = path_mean_speed_of_sound(0.0, 10_000.0, surface_temp_c=15.0)
    assert speed_of_sound(15.0 - 65.0) < c_mean < speed_of_sound(15.0)


@settings(max_examples=300, deadline=None)
@given(
    v=st.floats(30, 320),
    d=st.floats(100, 20_000),
    alt_frac=st.floats(0.05, 1.0),
    heading=st.floats(0, 360),
    tau0=st.floats(-1e3, 1e3),
    T=st.floats(-300, 300),
)
def test_closed_form_retarded_time_matches_numeric(v, d, alt_frac, heading, tau0, T):
    flight = StraightLineFlight.from_cpa(v, d, tau0, heading, altitude=alt_frac * d)
    t = np.array([tau0 + d / C + T])
    s_closed = retarded_offset(t, tau0, v, d, C)
    tau_num = retarded_time(t, flight.position, (0, 0, 0), C, flight.velocity)
    assert abs((tau0 + s_closed[0]) - tau_num[0]) < 1e-6  # < 1 us


@settings(max_examples=200, deadline=None)
@given(
    v=st.floats(30, 320),
    d=st.floats(100, 20_000),
    heading=st.floats(0, 360),
    f0=st.floats(20, 3000),
    T=st.floats(-300, 300),
)
def test_closed_form_frequency_matches_trajectory_model(v, d, heading, f0, T):
    tau0 = 50.0
    flight = StraightLineFlight.from_cpa(v, d, tau0, heading)
    t = np.array([tau0 + d / C + T])
    f_closed = doppler_frequency(t, f0, v, d, tau0, C)
    f_traj, _ = frequency_from_trajectory(t, f0, flight.position, flight.velocity, (0, 0, 0), C)
    assert f_closed[0] == pytest.approx(f_traj[0], rel=1e-9)


def test_pitch_is_high_on_approach_low_on_departure_and_f0_at_cpa():
    f0, v, d, tau0 = 200.0, 120.0, 1500.0, 0.0
    m = v / C
    t_far = np.array([-1e6, 1e6])
    f_far = doppler_frequency(t_far, f0, v, d, tau0)
    assert f_far[0] == pytest.approx(f0 / (1 - m), rel=1e-6)
    assert f_far[1] == pytest.approx(f0 / (1 + m), rel=1e-6)
    assert doppler_frequency([cpa_reception_time(tau0, d)], f0, v, d, tau0)[0] == pytest.approx(
        f0, rel=1e-12
    )


@given(s=st.floats(0, 600), v=st.floats(30, 320), d=st.floats(100, 20_000))
def test_symmetry_in_emission_time(s, v, d):
    # 1/f(+s) + 1/f(-s) = 2/f0 for emission offsets +-s.
    f0 = 100.0
    inv = lambda x: (1 + radial_speed(x, v, d) / C) / f0  # noqa: E731
    assert inv(s) + inv(-s) == pytest.approx(2 / f0, rel=1e-12)


def test_slope_at_cpa():
    f0, v, d, tau0 = 150.0, 200.0, 2500.0, 10.0
    tc = cpa_reception_time(tau0, d)
    h = 1e-4
    num = (doppler_frequency([tc + h], f0, v, d, tau0)[0] - doppler_frequency([tc - h], f0, v, d, tau0)[0]) / (2 * h)
    assert num == pytest.approx(-f0 * v * v / (C * d), rel=1e-6)


def test_retarded_time_handles_utc_epochs():
    tau0 = 1.79e9  # 2026 in Unix seconds
    flight = StraightLineFlight.from_cpa(220.0, 3000.0, tau0, 45.0)
    t = tau0 + np.linspace(-120, 120, 50)
    tau = retarded_time(t, flight.position, (0, 0, 0), C, flight.velocity)
    s = retarded_offset(t, tau0, 220.0, 3000.0, C)
    assert np.max(np.abs((tau - tau0) - s)) < 1e-5


def test_mirror_geometry_gives_identical_curves():
    # One microphone cannot tell which side of the track it is on.
    t = np.linspace(-60, 60, 200)
    a = StraightLineFlight.from_cpa(180.0, 2000.0, 0.0, 30.0, right_side=True)
    b = StraightLineFlight.from_cpa(180.0, 2000.0, 0.0, 30.0, right_side=False)
    fa, _ = frequency_from_trajectory(t, 300.0, a.position, a.velocity, (0, 0, 0))
    fb, _ = frequency_from_trajectory(t, 300.0, b.position, b.velocity, (0, 0, 0))
    np.testing.assert_allclose(fa, fb, rtol=1e-12)
