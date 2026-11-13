"""Spatial power spectra of maps: the layout panel type `pk` (the isotropic 2D power spectrum of any map a map panel draws) and the snapshot product `pk`
(P(k) families per snapshot, for heatmaps against time).

    power_spectrum(img, width, window="hann", subtract="delta", nk=None) -> {"k", "P", "n", "kf", "var", "kn"}

Conventions: the map q on N x N pixels of size d = width / N (L = width); the field f is prepared by `subtract` (delta: q / <q> - 1; mean: q - <q>;
log: log10 q - <log10 q>; none: q; pixels without gas (NaN, or q <= 0 for log) are set to the mean of the others); multiplied by the window w (hann: a
2D Hann taper, tukey: a Tukey taper with 25% cosine edges, none); F(k) = sum f w exp(-i k.x) d^2 (an FFT), P_2D(k) = |F|^2 / (L^2 <w^2>) (the window's
loss of power restored), k = 2 pi / wavelength in 1/length. With these, the variance of f is sum P_2D / L^2 = integral P_2D d^2k / (2 pi)^2 (Parseval).
P(k) is the mean of P_2D over the annulus |k| in a bin (log-spaced bins between the fundamental kf = 2 pi / L and the Nyquist pi / d), drawn at the bin
centre (bins without a Fourier mode, at the lowest k, are left out); `n` counts the Fourier modes of a bin. kn = k / kf: the number of wavelengths across the window (the product's family index).

Product `pk` (series source prod_pk): columns P_<kn> (kn = k L / 2 pi, fixed by npix and nk: the same rows in every snapshot), kf (2 pi / L in 1/code
length), var (the variance of the prepared field), slope (the least-squares log-slope of P over kn_lo < kn < kn_hi). Parameters: field (rho), mode (proj),
center (auto: the black holes' centre of mass, else the gas centre of mass), align, tilt (auto: Lbh, face-on onto the binary's orbit, with two black holes,
else none), axes, width (code length; None: width_a x a with a binary, else 2.2 x the radius holding 95% of the gas), width_a (8), npix (256), window,
subtract, nk (24), kn_lo / kn_hi (4, 32). Opt-in (it renders a map per snapshot): computed when named (`splosh products RUN --names pk`) or when the run's
analysis_viewer/products/params.toml has a [pk] table."""
from fractions import Fraction

import numpy as np

from . import plugins

WINDOWS = ("hann", "tukey", "none")
SUBTRACT = ("delta", "mean", "log", "none")


def window1d(n, kind):
    if kind == "hann":
        return np.hanning(n + 2)[1:-1]                    # no zero at the edges' pixel centres
    if kind == "tukey":
        from scipy.signal.windows import tukey
        return tukey(n, alpha=0.25)
    if kind == "none":
        return np.ones(n)
    raise ValueError(f"window must be one of {WINDOWS}")


def prepare(img, subtract="delta"):
    """the field whose spectrum is taken (see the module docstring), float64."""
    q = np.asarray(img, np.float64)
    if subtract == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(q > 0, np.log10(np.where(q > 0, q, 1.0)), np.nan)
    ok = np.isfinite(q)
    if not ok.any():
        return np.zeros_like(q)
    mu = float(q[ok].mean())
    q = np.where(ok, q, mu)
    if subtract in ("mean", "log"):
        return q - mu
    if subtract == "delta":
        return q / mu - 1.0 if mu != 0 else q - mu
    if subtract == "none":
        return q
    raise ValueError(f"subtract must be one of {SUBTRACT}")


def kbins(npix, nk=None):
    """log-spaced bin edges in kn = k / kf from 1 to the Nyquist npix / 2 (nk bins; default ~ 6 per octave, at most 48)."""
    hi = npix / 2.0
    nk = int(nk or min(48, max(8, round(6 * np.log2(hi)))))
    return np.geomspace(1.0, hi, nk + 1)


def power_spectrum(img, width, window="hann", subtract="delta", nk=None):
    img = np.asarray(img)
    n = img.shape[0]
    if img.ndim != 2 or img.shape[1] != n:
        raise ValueError("power_spectrum: a square map")
    L = float(width)
    d = L / n
    f = prepare(img, subtract)
    w1 = window1d(n, window)
    w = np.outer(w1, w1)
    F = np.fft.fft2(f * w) * d * d
    P2 = np.abs(F) ** 2 / (L * L * float(np.mean(w * w)))
    kf = 2 * np.pi / L
    fr = np.fft.fftfreq(n) * n                           # integer mode numbers
    kn = np.hypot(*np.meshgrid(fr, fr, indexing="ij"))
    edges = kbins(n, nk)
    idx = np.digitize(kn.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & (kn.ravel() > 0)
    cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
    tot = np.bincount(idx[ok], weights=P2.ravel()[ok], minlength=len(edges) - 1)
    has = cnt > 0                                        # the lowest bins can hold no integer mode number: left out (the same bins for every map of this npix)
    P = tot[has] / cnt[has]
    knc = np.sqrt(edges[1:] * edges[:-1])[has]
    return {"k": knc * kf, "kn": knc, "P": P, "n": cnt[has], "kf": kf, "var": float(np.mean((f * w) ** 2) / np.mean(w * w)), "edges_kn": edges}


def slope(k, P, lo=None, hi=None):
    """least-squares d ln P / d ln k over lo < k < hi (finite, positive P)."""
    k, P = np.asarray(k, float), np.asarray(P, float)
    ok = np.isfinite(P) & (P > 0) & (k > (lo if lo is not None else 0)) & (k < (hi if hi is not None else np.inf))
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(k[ok]), np.log(P[ok]), 1)[0])


def power_text(s):
    fr = Fraction(float(s)).limit_denominator(6)
    return f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else f"{fr.numerator}"


def slope_label(s):
    return "k^" + power_text(s)


def reference_lines(k, P, slopes, compensate=0.0):
    """[(label, k, P_ref)]: lines of the given log-slopes (of P; the curve may be multiplied by k^compensate) through the curve at the geometric middle
    of its finite range, a factor 2 above it."""
    k, P = np.asarray(k, float), np.asarray(P, float)
    ok = np.isfinite(P) & (P > 0)
    if ok.sum() < 2:
        return []
    kk, pp = k[ok], P[ok]
    kmid = float(np.sqrt(kk[0] * kk[-1]))
    pmid = float(np.exp(np.interp(np.log(kmid), np.log(kk), np.log(pp))))
    return [(slope_label(s), kk, 2.0 * pmid * (kk / kmid) ** (float(s) + float(compensate))) for s in slopes]


# ----------------------------------------------------------------------------- the snapshot product `pk`

def _view(s, params):
    from .maps import View
    nbh = len(s["bh_pos"]) if "bh_pos" in s else 0
    center = params.get("center") or "auto"
    if center == "auto":
        center = "bh" if nbh else "mass-weighted"
    tilt = params.get("tilt") or "auto"
    if tilt == "auto":
        tilt = "Lbh" if nbh >= 2 else "none"
    width = params.get("width")
    if width is None:
        if nbh >= 2:
            from .modes import disc_frame
            fr = disc_frame(s, np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1))
            width = float(params.get("width_a") or 8.0) * fr["a"]
        else:
            from .analyze import _auto_radius
            width = 2.2 * _auto_radius(s)
    return View(field=params.get("field") or "rho", mode=params.get("mode") or "proj", center=center, align=params.get("align") or "none", tilt=tilt,
                axes=params.get("axes") or "xy", width=float(width), npix=int(params.get("npix") or 256)).validate()


def pk(path, params):
    from .io import load_snapshot
    from .maps import render_map, required_gas_fields
    from .fields import gravitational_constant  # noqa: F401  (disc_frame reads G)
    probe = load_snapshot(path, gas_fields=("Coordinates", "Masses"))
    view = _view(probe, params)
    s = load_snapshot(path, gas_fields=tuple(sorted(required_gas_fields(view) | {"Coordinates", "Masses", "Velocities"})))
    img, meta = render_map(s, view)
    r = power_spectrum(img, view.width, params.get("window") or "hann", params.get("subtract") or "delta", params.get("nk") or 24)
    out = {f"P_{kn:.4g}": p for kn, p in zip(r["kn"], r["P"])}
    out.update(kf=r["kf"], var=r["var"], width=view.width,
               slope=slope(r["kn"], r["P"], float(params.get("kn_lo") or 4), float(params.get("kn_hi") or 32)))
    return {"scalars": out}


plugins.register_product("pk", pk, needs=("Coordinates", "SmoothingLength", "Masses", "(the field's datasets)"), version=1,
                         label="pk: spatial power spectrum P(k) of a map (opt-in)", applies=None,
                         params={"field": "rho", "mode": "proj", "center": "auto", "align": "none", "tilt": "auto", "axes": "xy", "width": None, "width_a": 8.0,
                                 "npix": 256, "window": "hann", "subtract": "delta", "nk": 24, "kn_lo": 4, "kn_hi": 32},
                         defaults=["slope"],
                         meta={"P_": {"label": "P(k), k in units of 2 pi / L", "group": "spectrum"}, "kf": {"label": "2 pi / L [1/code length]", "group": "pk"},
                               "var": {"label": "variance of the prepared field", "group": "pk"}, "slope": {"label": "log-slope of P(k)", "group": "pk"},
                               "width": {"label": "window width [code length]", "group": "pk"}})
plugins.PRODUCTS["pk"]["auto"] = False
