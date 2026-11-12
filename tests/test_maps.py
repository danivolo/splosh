"""Slices, weighted maps and frames, validated against the brute-force reference (refmaps.py)."""
import numpy as np
import pytest

from splosh import load_snapshot, splat, render_map, View
from splosh.frames import make_frame
from splosh.maps import AXES, bh_markers
import refmaps


def cloud(n=300, seed=5, hmin=0.5, hmax=1.2):
    rng = np.random.default_rng(seed)
    x, y, z = (rng.normal(size=n) * 0.8 for _ in range(3))
    H = hmin + (hmax - hmin) * rng.random(n)
    w = rng.random(n) + 0.2
    q = rng.random((2, n)).astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), H.astype(np.float32), w, q


@pytest.mark.parametrize("pyramid", [False, True])
def test_slice_matches_brute_force(pyramid):
    x, y, z, H, w, q = cloud(hmin=0.15, hmax=1.2)
    center, width, n = (0.1, -0.2, 0.15), 4.0, 64
    ref = refmaps.brute(x, y, z, H, w, q, center, width, n, "slice")
    img = splat.splat_slice(x, y, z, H, w, q, center, width, n, pyramid=pyramid)
    tol = 2e-2 if pyramid else 5e-3
    scale = ref[..., 0].max()
    for c in range(3):
        assert np.abs(img[..., c] - ref[..., c]).max() < tol * ref[..., c].max()
    assert scale > 0
    if not pyramid:                                                       # the exact engine differs from the reference only by the 4096-bin table
        m = ref[..., 0] > 0.05 * scale
        assert np.median(np.abs(img[..., 0][m] - ref[..., 0][m]) / ref[..., 0][m]) < 1e-3


def test_slice_single_particle_profile_and_empty():
    x = np.zeros(1, np.float32)
    H = np.full(1, 2.0, np.float32)
    img = splat.splat_slice(x, x.copy(), np.full(1, 0.8, np.float32), H, np.ones(1), np.ones((1, 1), np.float32), (0, 0, 0.0), 6.0, 120)
    dx = 6.0 / 120
    for r in (0.25, 0.75, 1.25, 1.55):
        i = int(60 + r / dx)
        rr = (i + 0.5 - 60) * dx
        assert img[60, i, 0] == pytest.approx(float(refmaps.w3(np.sqrt(rr**2 + 0.8**2) / 2.0)) / 8.0, rel=2e-2, abs=1e-4)
    far = splat.splat_slice(x, x.copy(), np.full(1, 2.5, np.float32), H, np.ones(1), np.ones((1, 1), np.float32), (0, 0, 0.0), 6.0, 64)
    assert far.max() == 0.0                                               # |dz| > H: not in the slice


@pytest.mark.parametrize("exact", [True, False])
def test_projection_matches_brute_force(exact):
    x, y, z, H, w, q = cloud(hmin=0.55, hmax=1.3)                         # H >= 8.8 pixels at width 4 / 64 px: above the small-particle renormalisation
    center, width, n = (0.0, 0.0), 4.0, 64
    ref = refmaps.brute(x, y, z, H, w, q, center, width, n, "proj")
    fn = splat.splat_bands if exact else splat.splat_bands_pyr
    img = fn(x, y, H, w, q, center, width, n)
    for c in range(3):
        assert np.abs(img[..., c] - ref[..., c]).max() < (4e-3 if exact else 3e-2) * ref[..., c].max()


def synthetic_snap(n=400, seed=2, hmin=0.15):
    x, y, z, H, w, q = cloud(n, seed, hmin, 1.0)
    rng = np.random.default_rng(seed + 1)
    pos = np.column_stack([x, y, z]).astype(np.float64)
    vel = np.column_stack([np.ones(n), 0.5 * x, np.zeros(n)]).astype(np.float32)
    m = w.astype(np.float32)
    rho = (m / (H.astype(np.float64) ** 3)).astype(np.float32)
    return {"time": 1.0, "Coordinates": pos, "SmoothingLength": H, "Masses": m, "Density": rho,
            "InternalEnergy": (1.0 + q[0]).astype(np.float32), "Velocities": vel,
            "bh_pos": np.array([[0.4, 0.1, 0.0], [-0.4, -0.1, 0.0]]), "bh_mass": np.array([1.0, 1.0]), "bh_mdyn": np.array([2.0, 1.0]),
            "bh_vel": np.zeros((2, 3)), "bh_id": np.array([11, 10])}


@pytest.mark.parametrize("mode,weight", [("proj", "auto"), ("slice", "auto"), ("slice", "mass"), ("proj", "volume")])
def test_weighted_quantity_map_matches_brute_force(mode, weight):
    s = synthetic_snap(hmin=0.15 if mode == "slice" else 0.55)          # projections: H above 6 pixels (no small-particle renormalisation)
    view = View(center="bh", align="bh", mode=mode, field="u", width=4.0, npix=48, weight=weight, z0=0.05, exact=True)
    img, meta = render_map(s, view)
    fr = make_frame(s, "bh", "bh")
    p = fr.pos(s["Coordinates"])
    wts = s["Masses"].astype(np.float64)
    wmode = meta["weight"]
    if wmode == "volume":
        wts = wts / s["Density"]
    ref = refmaps.brute(p[:, 0], p[:, 1], p[:, 2], s["SmoothingLength"].astype(np.float64), wts, s["InternalEnergy"][None, :].astype(np.float64), (0, 0, view.z0), 4.0, 48, mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = ref[..., 1] / ref[..., 0]
    ok = np.isfinite(img) & (ref[..., 0] > 0.02 * ref[..., 0].max())
    assert ok.sum() > 300
    assert np.abs(img[ok] - ratio[ok]).max() < 6e-3 * ratio[ok].max()
    assert meta["weight"] == {"auto": "mass" if mode == "proj" else "volume"}.get(weight, weight)


def test_rho_maps_and_constant_field():
    s = synthetic_snap()
    s["InternalEnergy"][:] = 3.5
    for mode in ("proj", "slice"):
        img, meta = render_map(s, View(center="com", mode=mode, field="u", width=6.0, npix=64, z0=0.0))
        assert np.nanmax(np.abs(img - 3.5)) < 3.5 * 2e-5 and np.isfinite(img).sum() > 100       # a constant field is reproduced where defined
    img, meta = render_map(s, View(center="com", mode="proj", field="rho", width=40.0, npix=64))
    dx = 40.0 / 64
    assert img.sum() * dx * dx == pytest.approx(s["Masses"].astype(np.float64).sum(), rel=2e-3)        # the column density conserves the mass
    assert meta["label"] == "column density" and meta["origin"] == "lower"


def test_velocity_fields_follow_the_frame():
    s = synthetic_snap()
    s["Velocities"][:] = [1.0, 0.0, 0.0]
    s["bh_vel"] = np.zeros((2, 3))
    view = View(center="bh", align="bh", field="vx", width=8.0, npix=32)
    th = make_frame(s, "bh", "bh").angle
    vx, _ = render_map(s, view)
    vy, _ = render_map(s, view.with_(field="vy"))
    ok = np.isfinite(vx)
    assert np.allclose(vx[ok], np.cos(th), atol=1e-5) and np.allclose(vy[ok], -np.sin(th), atol=1e-5)          # (1,0,0) rotated by -theta
    vl, _ = render_map(s, view.with_(field="vlos", axes="xz"))                                               # line of sight = y
    assert np.allclose(vl[np.isfinite(vl)], -np.sin(th), atol=1e-5)


def test_axes_permutations_and_bh_markers():
    s = synthetic_snap()
    v = View(center="bh", align="bh", width=6.0, npix=32)
    xy, _ = render_map(s, v)
    xz, _ = render_map(s, v.with_(axes="xz"))
    assert xy.shape == xz.shape and not np.allclose(xy, xz)
    mk = bh_markers(s, v)
    assert [m["id"] for m in mk] == [11, 10] and abs(mk[0]["y"]) < 1e-12 and mk[0]["x"] != mk[1]["x"]
    mk2 = bh_markers(s, v.with_(axes="xz"))
    assert np.isclose(mk2[0]["y"], mk[0]["depth"]) and np.isclose(mk2[0]["depth"], mk[0]["y"])


def test_view_validation():
    with pytest.raises(ValueError):
        View(axes="zz").validate()
    with pytest.raises(ValueError):
        View(npix=2).validate()
    with pytest.raises(KeyError):
        render_map(synthetic_snap(), View(field="not_a_field", center="com"))


def test_density_map_needs_no_density_dataset():
    s = synthetic_snap()
    del s["Density"]
    img, _ = render_map(s, View(center="com", mode="slice", field="rho", width=6.0, npix=32))
    assert img.shape == (32, 32) and img.max() > 0
