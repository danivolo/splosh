import numpy as np
import pytest

from splosh import load_snapshot, make_frame
from splosh.frames import Frame, resolve_center, resolve_angle


def test_bh_centre_is_dynamical_mass_weighted_com(snapfile):
    s = load_snapshot(snapfile)
    f = make_frame(s, "bh")
    expect = (3.0 * np.array([1.0, 0.5, 0.1]) + 1.0 * np.array([-1.0, -0.5, 0.0])) / 4.0      # Masses, not BH_Mass (they are 0.9x: same ratio, so also test velocity)
    assert np.allclose(f.center, expect)
    assert np.allclose(f.velocity, (3.0 * np.array([0, 1.0, 0]) + np.array([0, -3.0, 0])) / 4.0)


def test_bh_mass_field_is_the_dynamical_one(snapfile):
    s = load_snapshot(snapfile)
    s["bh_mdyn"] = np.array([1.0, 3.0])           # swap: the rule must follow bh_mdyn, not bh_mass
    c, _ = resolve_center(s, "bh")
    assert np.allclose(c, (1.0 * np.array([1.0, 0.5, 0.1]) + 3.0 * np.array([-1.0, -0.5, 0.0])) / 4.0)


def test_align_bh_puts_lowest_id_to_second_lowest_along_plus_x(snapfile):
    s = load_snapshot(snapfile)               # ids (7, 3): the lowest id 3 is the second row in the file
    f = make_frame(s, "bh", "bh")
    p = f.pos(s["bh_pos"])
    i3, i7 = 1, 0
    d = p[i7] - p[i3]
    assert d[0] > 0 and abs(d[1]) < 1e-12                                   # id 3 -> id 7 along +x
    assert np.isclose(np.linalg.norm(d), np.linalg.norm(s["bh_pos"][i7] - s["bh_pos"][i3]))
    assert abs(p[i7][2] - p[i3][2] - (0.1 - 0.0)) < 1e-12                   # z untouched
    # the centre of mass is the origin
    m = s["bh_mdyn"]
    assert np.allclose((p * m[:, None]).sum(0) / m.sum(), 0, atol=1e-12)
    # velocity: COM subtracted and rotated like the positions
    v = f.vel(s["bh_vel"])
    assert np.allclose((v * m[:, None]).sum(0) / m.sum(), 0, atol=1e-6)


def test_pair_rule_and_xyz_and_com(snapfile):
    s = load_snapshot(snapfile, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities", "ParticleIDs"))
    f = make_frame(s, "pair:7,3", "pair:7,3")                                # reversed pair: 7 -> 3 along +x
    p = f.pos(s["bh_pos"])
    assert (p[1] - p[0])[0] > 0 and abs((p[1] - p[0])[1]) < 1e-12
    # a gas particle by id
    c, _ = resolve_center(s, "ids:1005")
    assert np.allclose(c, s["Coordinates"][5])
    assert np.allclose(resolve_center(s, "xyz:1,2,3")[0], [1, 2, 3]) and np.allclose(resolve_center(s, "1,2,3")[0], [1, 2, 3])
    mw, _ = resolve_center(s, "mass-weighted")
    assert np.allclose(mw, (s["Coordinates"] * s["Masses"][:, None]).sum(0) / s["Masses"].sum())
    com, _ = resolve_center(s, "com")
    mt = np.concatenate([s["Masses"], s["bh_mdyn"]]); pt = np.vstack([s["Coordinates"], s["bh_pos"]])
    assert np.allclose(com, (pt * mt[:, None]).sum(0) / mt.sum())
    assert np.allclose(resolve_center(s, "peak")[0], s["Coordinates"][np.argmax(s["Density"])])


def test_errors(snapfile):
    s = load_snapshot(snapfile)
    with pytest.raises(ValueError):
        resolve_center(s, "nonsense")
    with pytest.raises(KeyError):
        resolve_center(s, "ids:999999")
    with pytest.raises(ValueError):
        resolve_angle(s, "vec:0,0,1")
    with pytest.raises(ValueError):
        resolve_angle(s, "bogus")


def test_rotation_matches_pynbody_rotate_z():
    pynbody = pytest.importorskip("pynbody")
    pts = np.array([[1.0, 2.0, 0.3], [0.0, 0.0, 0.0], [3.0, -1.0, 2.0]])
    vel = np.eye(3)
    th = np.arctan2(2.0, 1.0)
    f = pynbody.new(gas=3)
    f["pos"] = pts
    f["vel"] = vel
    f.rotate_z(-np.degrees(th))
    fr = Frame(angle=th)
    assert np.allclose(fr.pos(pts), np.array(f["pos"]), atol=1e-12)
    assert np.allclose(fr.vel(vel), np.array(f["vel"]), atol=1e-12)


def test_binary_mass_is_the_dynamical_mass_not_bh_mass(tmp_path):
    """Dynamical Masses 3:1 but the sub-grid BH_Mass field 1:3 (a different RATIO, so any use of BH_Mass changes the centre): every centre rule, the alignment-free
    velocity, the pair rule, the ids rule and splosh.binary_com must follow Masses."""
    import splosh
    from conftest import write_snapshot
    p = str(tmp_path / "s.hdf5")
    write_snapshot(p, bh_mass=(3.0, 1.0), bh_mass_field=(1.0, 3.0))       # ids (7, 3), positions (1,.5,.1) and (-1,-.5,0)
    s = load_snapshot(p)
    assert np.allclose(s["bh_mdyn"], [3.0, 1.0]) and np.allclose(s["bh_mass"], [1.0, 3.0])
    x7, x3 = np.array([1.0, 0.5, 0.1]), np.array([-1.0, -0.5, 0.0])
    v7, v3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, -3.0, 0.0])
    dyn, wrong = (3 * x7 + x3) / 4, (x7 + 3 * x3) / 4
    assert not np.allclose(dyn, wrong)
    for spec in ("bh", "pair:7,3", "ids:7,3"):
        c, v = resolve_center(s, spec)
        assert np.allclose(c, dyn), spec
        assert np.allclose(v, (3 * v7 + v3) / 4), spec
    assert np.allclose(splosh.binary_com(s), dyn)
    # com: gas + both BHs, with the dynamical masses
    c, _ = resolve_center(s, "com")
    pos = np.concatenate([s["Coordinates"].astype(np.float64), s["bh_pos"]]); m = np.concatenate([s["Masses"].astype(np.float64), [3.0, 1.0]])
    assert np.allclose(c, (pos * m[:, None]).sum(0) / m.sum())
    from splosh.maps import bh_markers, View
    assert [round(d["mass"], 6) for d in bh_markers(s, View())] == [3.0, 1.0]


# ---- tilt: arbitrary 3D rotations after centring and alignment
from splosh.frames import resolve_tilt, to_z  # noqa: E402
from splosh import View, render_map  # noqa: E402


def _is_rotation(R):
    return np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("n", [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0.3, -0.4, 0.86), (1e-9, 0, -1)])
def test_to_z_maps_the_direction_onto_z(n):
    R = to_z(n)
    assert _is_rotation(R)
    assert np.allclose(R @ (np.array(n) / np.linalg.norm(n)), [0, 0, 1], atol=1e-9)


def test_tilt_rules(snapfile):
    s = load_snapshot(snapfile, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Velocities"))
    f0 = make_frame(s, "bh")
    assert np.allclose(make_frame(s, "bh", "none", "none").rot, f0.rot)
    T = resolve_tilt(s, "incl:90", f0)
    assert _is_rotation(T) and np.allclose(T @ [0, 0, 1], [0, -1, 0]) and np.allclose(T @ [1, 0, 0], [1, 0, 0])
    assert np.allclose(resolve_tilt(s, "incl:30,40", f0), resolve_tilt(s, "euler:40,30,0", f0))
    assert np.allclose(resolve_tilt(s, "incl:30+incl:60", f0), resolve_tilt(s, "incl:90", f0))          # terms compose left to right
    assert np.allclose(resolve_tilt(s, "los:0,1,0", f0) @ [0, 1, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        resolve_tilt(s, "spin:3", f0)
    with pytest.raises(ValueError):
        resolve_tilt(s, "incl:", f0)


def test_tilt_L_makes_the_angular_momentum_the_line_of_sight(snapfile):
    s = load_snapshot(snapfile, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Velocities"))
    # give the gas a rigid rotation about a tilted axis a: v = a x r
    a = np.array([0.5, -0.3, 0.8]); a /= np.linalg.norm(a)
    s["Velocities"] = np.cross(a, s["Coordinates"] - make_frame(s, "bh").center).astype(np.float32) + make_frame(s, "bh").velocity.astype(np.float32)
    f = make_frame(s, "bh", "none", "L")
    f0 = make_frame(s, "bh")
    L0 = (s["Masses"][:, None] * np.cross(f0.pos(s["Coordinates"]), f0.vel(s["Velocities"]))).sum(0)     # = I a, not parallel to a for a non-spherical cloud
    assert np.allclose(f.tilt @ (L0 / np.linalg.norm(L0)), [0, 0, 1], atol=1e-6)
    p, v = f.pos(s["Coordinates"]), f.vel(s["Velocities"])
    L = (s["Masses"][:, None] * np.cross(p, v)).sum(0)
    assert abs(L[0]) < 1e-4 * abs(L[2]) and abs(L[1]) < 1e-4 * abs(L[2]) and L[2] > 0
    assert f.describe()["tilt_spec"] == "L"
    fb = make_frame(s, "bh", "none", "Lbh")
    pb, vb = fb.pos(s["bh_pos"]), fb.vel(s["bh_vel"])
    Lb = (s["bh_mdyn"][:, None] * np.cross(pb, vb)).sum(0)
    assert np.allclose(Lb[:2], 0, atol=1e-9 * np.abs(Lb).max()) and Lb[2] > 0
    with pytest.raises(ValueError):
        make_frame(s, "bh", "none", "L:1e-9")                  # no gas inside


def test_tilt_in_maps_conserves_mass_and_matches_the_axes_view(snapfile):
    s = load_snapshot(snapfile, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Velocities"))
    v = View(center="bh", width=40, npix=128, exact=True)
    a, _ = render_map(s, v)
    b, mb = render_map(s, v.with_(tilt="euler:17,63,-41"))
    pix = (40 / 128) ** 2
    assert np.isclose(a.sum() * pix, b.sum() * pix, rtol=1e-3)                 # a rotation moves mass, it does not create it
    assert mb["frame"]["tilt_spec"] == "euler:17,63,-41"
    # incl:90 seen along z equals the xz view seen from -y: identical up to the vertical flip (the old +z points to -y)
    c, _ = render_map(s, v.with_(tilt="incl:90"))
    d, _ = render_map(s, v.with_(axes="xz"))
    assert np.allclose(c, d[::-1, :], rtol=1e-4, atol=1e-6 * d.max())
