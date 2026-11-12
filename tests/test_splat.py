import numpy as np
import pytest
from scipy.integrate import quad

from splosh import splat, kernel

VARIANTS = [splat.splat_serial, splat.splat_tl, splat.splat_bands, splat.splat_bands_pyr]


def w3(q):
    return (8 / np.pi) * (1 - 6 * q**2 + 6 * q**3) if q < 0.5 else ((8 / np.pi) * 2 * (1 - q) ** 3 if q < 1 else 0.0)


def f_exact(s):
    tm = np.sqrt(max(1 - s * s, 0.0))
    pts = [np.sqrt(0.25 - s * s)] if s < 0.5 else None
    return 2 * quad(lambda t: w3(np.sqrt(s * s + t * t)), 0, tm, points=pts, limit=200)[0]


def one(fn, H, W=8.0, N=256, m=1.0):
    x = np.zeros(1, np.float32)
    return fn(x, x.copy(), np.full(1, H, np.float32), np.full(1, m), np.ones((1, 1), np.float32), (0.0, 0.0), W, N)


def test_kernel_tables_normalised():
    t2, t3 = kernel.tables()
    s = np.sqrt(np.linspace(0, 1, len(t2)))
    assert abs(np.trapezoid(2 * np.pi * s * t2, s) - 1) < 1e-4
    assert abs(np.trapezoid(4 * np.pi * s**2 * t3, s) - 1) < 1e-3


@pytest.mark.parametrize("fn", VARIANTS)
def test_single_particle_matches_exact_projected_kernel(fn):
    W, N, H = 8.0, 256, 3.0
    img = one(fn, H, W, N)[..., 0]
    dx = W / N
    c = N // 2
    for r in (0.3, 0.9, 1.5, 2.1, 2.7):
        j = int(c + r / dx)
        s = (j + 0.5 - c) * dx / H
        assert img[c, j] == pytest.approx(f_exact(s) / H**2, rel=5e-3, abs=2e-3 * f_exact(0) / H**2 * 0.01)   # nearest-bin table: rim pixels carry <0.1% of the weight
    assert img.sum() * dx * dx == pytest.approx(1.0, rel=2e-3)


@pytest.mark.parametrize("fn", VARIANTS)
def test_weight_and_quantity_columns(fn):
    img = one(fn, 2.0, m=3.0)
    assert img.shape == (256, 256, 2)
    # the quantity column of a particle with q = 1 equals the weight column
    assert np.allclose(img[..., 1], img[..., 0], rtol=1e-6, atol=0)


@pytest.mark.parametrize("H_pix", [0.2, 0.6, 1.0, 1.7, 3.0])
def test_small_particles_conserve_weight(H_pix):
    W, N = 8.0, 256
    dx = W / N
    for fn in VARIANTS:
        img = one(fn, H_pix * dx, W, N, m=2.5)[..., 0]
        assert img.sum() * dx * dx == pytest.approx(2.5, rel=1e-3)


def test_variants_agree_on_a_random_cloud():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(size=n).astype(np.float32); y = rng.normal(size=n).astype(np.float32)
    H = (0.05 + 0.6 * rng.random(n)).astype(np.float32); w = rng.random(n) + 0.1
    vals = rng.random((2, n)).astype(np.float32)
    ref = splat.splat_serial(x, y, H, w, vals, (0.0, 0.0), 4.0, 256)
    for fn in (splat.splat_tl, splat.splat_bands):
        np.testing.assert_allclose(fn(x, y, H, w, vals, (0.0, 0.0), 4.0, 256), ref, rtol=2e-4, atol=1e-4 * ref.max())
    pyr = splat.splat_bands_pyr(x, y, H, w, vals, (0.0, 0.0), 4.0, 256)          # default big = 12
    mk = ref[..., 0] > 1e-2 * ref[..., 0].max()
    assert np.median(np.abs(pyr[..., 0][mk] - ref[..., 0][mk]) / ref[..., 0][mk]) < 2e-3
    assert pyr[..., 0].sum() == pytest.approx(ref[..., 0].sum(), rel=2e-3)


def test_clipping_and_offcentre_view():
    # a particle half outside the image still deposits the part inside; a particle far outside deposits nothing
    x = np.array([3.9, 100.0], np.float32); y = np.zeros(2, np.float32); H = np.array([1.0, 1.0], np.float32)
    for fn in VARIANTS:
        img = fn(x, y, H, np.ones(2), np.ones((1, 2), np.float32), (0.0, 0.0), 8.0, 128)
        m_in = img[..., 0].sum() * (8.0 / 128) ** 2
        assert 0.2 < m_in < 0.8
    img = splat.splat_bands(np.array([5.0], np.float32), np.zeros(1, np.float32), np.ones(1, np.float32), np.ones(1), np.ones((1, 1), np.float32), (5.0, 0.0), 4.0, 64)
    assert img[..., 0].sum() * (4.0 / 64) ** 2 == pytest.approx(1.0, rel=2e-3)
