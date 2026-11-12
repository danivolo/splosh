"""SPH kernel tables for the GIZMO cubic spline (KERNEL_FUNCTION 3).

GIZMO writes SmoothingLength = H, the kernel SUPPORT radius (W(r>=H) = 0), and uses
    W(q) = 8/(pi H^3) * (1 - 6 q^2 + 6 q^3)   for 0 <= q < 1/2
         = 8/(pi H^3) * 2 (1 - q)^3           for 1/2 <= q < 1,         q = r/H     (kernel.h: KERNEL_NORM 8/pi, NUMDIMS 3)
which is Monaghan's cubic spline with h_sph = H/2.

Tables are indexed by q^2 in [0,1] (no square root in the inner loop):
    table3d[k]  = w(q)               with w = W * H^3                 (cross-section / slice)
    table2d[k]  = f(s)               with f = (integral of W dz) * H^2 (column integral, s = projected distance / H)
"""
import numpy as np

NTAB = 4096


def w3(q):
    q = np.asarray(q, dtype=np.float64)
    return np.where(q < 0.5, 1 - 6 * q**2 + 6 * q**3, np.where(q < 1, 2 * (1 - q) ** 3, 0.0)) * (8.0 / np.pi)


def tables(n=NTAB):
    """(table2d, table3d), each n+1 float32 entries at q^2 = k/n (computed once, cached next to this file)."""
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_tables_{n}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["t2"], d["t3"]
    t2, t3 = _compute_tables(n)
    try:
        np.savez(cache, t2=t2, t3=t3)
    except OSError:
        pass
    return t2, t3


def _compute_tables(n):
    q2 = np.linspace(0.0, 1.0, n + 1)
    t3 = w3(np.sqrt(q2))
    # projected: f(s) = 2 * int_0^{sqrt(1-s^2)} w(sqrt(s^2+t^2)) dt, by Gauss-Legendre with the integrand's kinks at q = 1/2 split off
    xg, wg = np.polynomial.legendre.leggauss(64)
    t2 = np.zeros_like(q2)
    for i, s2 in enumerate(q2):
        tmax = np.sqrt(max(1.0 - s2, 0.0))
        edges = [0.0, tmax]
        if s2 < 0.25:
            edges = [0.0, np.sqrt(0.25 - s2), tmax]       # q = 1/2 inside the line of sight
        acc = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            t = 0.5 * (b - a) * xg + 0.5 * (b + a)
            acc += 0.5 * (b - a) * np.sum(wg * w3(np.sqrt(s2 + t**2)))
        t2[i] = 2.0 * acc
    return t2.astype(np.float32), t3.astype(np.float32)


if __name__ == "__main__":
    t2, t3 = tables()
    s = np.sqrt(np.linspace(0, 1, len(t2)))
    print("int w 4 pi q^2 dq =", np.trapezoid(4 * np.pi * s**2 * t3, s), "  int f 2 pi s ds =", np.trapezoid(2 * np.pi * s * t2, s))
