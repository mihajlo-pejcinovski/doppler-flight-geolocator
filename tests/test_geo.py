import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dopplergeo.core.geo import (
    WGS84_A,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_rotation,
    enu_to_ecef,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    haversine_m,
    velocity_enu,
    velocity_enu_to_ecef,
)

lat_s = st.floats(-89.9, 89.9)
lon_s = st.floats(-180.0, 180.0)
h_s = st.floats(-500.0, 15_000.0)


def test_known_points():
    np.testing.assert_allclose(geodetic_to_ecef(0, 0, 0), [WGS84_A, 0, 0], atol=1e-9)
    np.testing.assert_allclose(geodetic_to_ecef(0, 90, 100), [0, WGS84_A + 100, 0], atol=1e-6)
    lat, lon, h = ecef_to_geodetic([0.0, 0.0, 6_356_752.314245 + 10])
    assert abs(lat - 90) < 1e-9 and abs(h - 10) < 1e-6


@settings(max_examples=300, deadline=None)
@given(lat_s, lon_s, h_s)
def test_geodetic_round_trip(lat, lon, h):
    la, lo, hh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert abs(hh - h) < 1e-3
    assert abs(la - lat) < 1e-9
    assert abs((lo - lon + 180) % 360 - 180) < 1e-9


@settings(max_examples=300, deadline=None)
@given(lat_s, lon_s, h_s, st.floats(-1e5, 1e5), st.floats(-1e5, 1e5), st.floats(-500, 15_000))
def test_enu_round_trip_sub_mm(lat0, lon0, h0, e, n, u):
    """Spec: geodesy ENU round-trip error < 1 mm."""
    enu = np.array([e, n, u])
    back = geodetic_to_enu(*enu_to_geodetic(enu, lat0, lon0, h0), lat0, lon0, h0)
    assert np.linalg.norm(back - enu) < 1e-3
    np.testing.assert_allclose(ecef_to_enu(enu_to_ecef(enu, lat0, lon0, h0), lat0, lon0, h0), enu, atol=1e-6)


def test_enu_axes_point_the_right_way():
    up = geodetic_to_enu(52.0, 4.0, 1000.0, 52.0, 4.0, 0.0)
    np.testing.assert_allclose(up, [0, 0, 1000], atol=1e-6)
    north = geodetic_to_enu(52.01, 4.0, 0.0, 52.0, 4.0)
    assert north[1] > 1100 and abs(north[0]) < 1e-6
    east = geodetic_to_enu(52.0, 4.01, 0.0, 52.0, 4.0)
    assert east[0] > 600 and abs(east[1]) < 1.0


def test_velocity_rotation_matches_enu_matrix():
    v = velocity_enu(100.0, 90.0, 5.0)  # due east, climbing
    np.testing.assert_allclose(v, [100, 0, 5], atol=1e-9)
    ecef = velocity_enu_to_ecef(v, 40.0, -3.0)
    np.testing.assert_allclose(ecef, v @ enu_rotation(40.0, -3.0), atol=1e-9)


def test_haversine_scale():
    assert abs(haversine_m(0, 0, 1, 0) - 111_195) < 10
