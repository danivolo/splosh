"""Volume rendering (splat.splat_volume, View(mode="volume")): the compositing law, the limits it must reach, and its behaviour in maps."""
import numpy as np
import pytest

from splosh import View, load_snapshot, render_map
from splosh.splat import splat_volume, splat_bands

FIELDS = ("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities")


def test_two_screens_front_hides_back():
    """two identical big particles on the line of sight: the opacity is 1 - e^(-2 tau); the front source dominates; swapping depths swaps the result."""
    x = np.zeros(2); y = np.zeros(2); h = np.full(2, 10.0); w = np.ones(2)
    k = 50.0
    a = splat_volume(x, y, np.array([1.0, -1.0]), h, w, np.array([2.0, 7.0]), k, (0, 0), 0.5, 8)       # source 2 in front
    b = splat_volume(x, y, np.array([-1.0, 1.0]), h, w, np.array([2.0, 7.0]), k, (0, 0), 0.5, 8)       # source 7 in front
    col = splat_bands(x[:1], y[:1], h[:1], w[:1], np.zeros((0, 1), np.float32), (0, 0), 0.5, 8)[..., 0]    # column density of one particle
    tau = k * col
    assert np.allclose(a[..., 1], 1 - np.exp(-2 * tau), rtol=1e-4)
    E_front2 = (1 - np.exp(-tau)) * 2 + np.exp(-tau) * (1 - np.exp(-tau)) * 7
    assert np.allclose(a[..., 0], E_front2, rtol=1e-4)
    assert (a[..., 0] / a[..., 1] < 4.5).all() and (b[..., 0] / b[..., 1] > 4.5).all()            # what you see is mostly the front particle


def test_thin_limit_is_the_projection():
    """optically thin (small kappa): the opacity is kappa times the column density, the visible field the mass-weighted mean."""
    rng = np.random.default_rng(1)
    n = 3000
    x, y, z = rng.normal(size=(3, n)); h = rng.uniform(0.3, 0.6, n); w = rng.uniform(0.5, 1.5, n); q = rng.uniform(1, 3, n)
    k = 1e-6                                                              # tau <= 5.5e-4 at the centre: 1 - e^-tau = tau to 3e-4
    v = splat_volume(x, y, z, h, w, q, k, (0, 0), 4.0, 64)
    p = splat_bands(x, y, h, w, np.ascontiguousarray(q[None, :], np.float32), (0, 0), 4.0, 64)
    assert np.allclose(v[..., 1], k * p[..., 0], rtol=2e-3, atol=1e-9)
    m = p[..., 0] > 1e-3 * p[..., 0].max()
    assert np.allclose((v[..., 0] / v[..., 1])[m], (p[..., 1] / p[..., 0])[m], rtol=2e-3)


def test_volume_maps(snapfile):
    s = load_snapshot(snapfile, gas_fields=FIELDS)
    base = View(center="bh", width=6.0, npix=64, mode="volume")
    op, meta = render_map(s, base)
    assert meta["label"] == "opacity (volume)" and meta["kappa"] > 0 and np.nanmin(op) >= 0 and np.nanmax(op) <= 1
    thick, _ = render_map(s, base.with_(kappa=100 * meta["kappa"]))
    assert thick.mean() > op.mean()                                       # more opaque
    u, mu = render_map(s, base.with_(field="u"))
    assert mu["label"] == "visible u (volume)" and np.nanmin(u) >= s["InternalEnergy"].min() * (1 - 1e-5) and np.nanmax(u) <= s["InternalEnergy"].max() * (1 + 1e-5)
    # seen from the other side (axes xy viewed from below = tilt incl:180) a thick volume shows other gas
    back, _ = render_map(s, base.with_(field="u", kappa=100 * meta["kappa"], tilt="incl:180"))
    front, _ = render_map(s, base.with_(field="u", kappa=100 * meta["kappa"]))
    assert not np.allclose(np.nan_to_num(front), np.nan_to_num(back[::-1, :]))
    with pytest.raises(ValueError):
        base.with_(kappa=-1).validate()
