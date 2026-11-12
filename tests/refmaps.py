"""Brute-force reference maps for the tests: every pixel against every particle with the exact kernel (closed form for the slice, quadrature for the column integral)."""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


def w3(q):
    q = np.asarray(q, dtype=np.float64)
    return np.where(q < 0.5, 1 - 6 * q**2 + 6 * q**3, np.where(q < 1, 2 * (1 - q) ** 3, 0.0)) * (8.0 / np.pi)


def _f_exact(s):
    tm = np.sqrt(max(1 - s * s, 0.0))
    pts = [np.sqrt(0.25 - s * s)] if s < 0.5 else None
    return 2 * quad(lambda t: float(w3(np.sqrt(s * s + t * t))), 0, tm, points=pts, limit=200)[0]


_SPL = None


def f_proj(s):
    """column-integrated kernel f(s) (s = projected distance / H, integral of W dz times H^2), cubic spline through exact quadrature values."""
    global _SPL
    if _SPL is None:
        g = np.linspace(0, 1, 801)
        _SPL = CubicSpline(g, [_f_exact(x) for x in g])
    s = np.asarray(s, dtype=np.float64)
    return np.where(s < 1, _SPL(np.minimum(s, 1.0)), 0.0)


def pixel_centres(center, width, npix):
    dx = width / npix
    c = (np.arange(npix) + 0.5) * dx - 0.5 * width
    return c + center[0], c + center[1]


def brute(x, y, z, h, w, vals, center, width, npix, mode):
    """(npix, npix, nq+1) like the splat functions: out0 = sum w K, out_k = sum w q_k K, K = f(s)/H^2 (proj) or w3(q)/H^3 (slice at z = center[2])."""
    xs, ys = pixel_centres(center, width, npix)
    X, Y = np.meshgrid(xs, ys)                       # [j, i]
    out = np.zeros((npix, npix, vals.shape[0] + 1))
    for p in range(len(x)):
        r2 = (X - x[p]) ** 2 + (Y - y[p]) ** 2
        if mode == "proj":
            k = f_proj(np.sqrt(r2) / h[p]) / h[p] ** 2
        else:
            k = w3(np.sqrt(r2 + (z[p] - center[2]) ** 2) / h[p]) / h[p] ** 3
        out[..., 0] += w[p] * k
        for c in range(vals.shape[0]):
            out[..., c + 1] += w[p] * vals[c, p] * k
    return out
