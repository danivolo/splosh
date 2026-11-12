"""Turning a float map into a picture: stretch, colour map, PNG writer (numpy + zlib only)."""
import struct
import zlib

import numpy as np

from . import colormaps


SCALES = ("log", "linear", "asinh", "histeq")


class Stretch:
    """A stretch that carries what it needs: 'histeq' needs the distribution of the map it will colour (a table of quantiles of its non-zero pixels), 'asinh:F' a softening fraction F.
    `==` compares with the bare name ('asinh', 'histeq'), so code written for string scales keeps working; str() is the full spec."""
    def __init__(self, name, soft=0.1, table=None):
        self.name, self.soft, self.table = name, soft, table

    def __eq__(self, other):
        return self.name == (other.name if isinstance(other, Stretch) else other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return {"asinh": f"asinh:{self.soft:g}"}.get(self.name, self.name)

    __repr__ = __str__


def parse_scale(scale):
    """'log' | 'linear' | 'asinh[:F]' | 'histeq' (or a Stretch) -> str for the plain ones, a Stretch otherwise. F in (0, 100] is the linear-to-logarithmic transition as a fraction of the
    mean of |vmin| and |vmax| (default 0.1: the long-standing behaviour; smaller = more logarithmic, larger = more linear)."""
    if isinstance(scale, Stretch):
        return scale
    name, _, arg = str(scale).partition(":")
    if name not in SCALES:
        raise ValueError(f"scale must be one of {SCALES} (asinh takes :F), not {scale!r}")
    if name == "asinh":
        try:
            f = float(arg) if arg else 0.1
        except ValueError:
            raise ValueError(f"scale {scale!r}: F must be a number") from None
        if not 0 < f <= 100:
            raise ValueError("asinh softening F must be in (0, 100]")
        return scale if (name, f) == ("asinh", 0.1) and not arg else Stretch("asinh", f)
    if arg:
        raise ValueError(f"scale {name} takes no parameter")
    return name


def fit_scale(img, vmin, vmax, scale, n=257, cap=1_000_000):
    """The scale ready for use with `img`: str scales unchanged; histeq gets its quantile table (n values: the pixel values at the ranks 0..1, clipped to [vmin, vmax]; a
    regular subsample of at most `cap` pixels)."""
    sc = parse_scale(scale)
    if isinstance(sc, Stretch) or sc != "histeq":
        return sc
    v = np.asarray(img, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    v = v[v != 0] if (v != 0).any() else v                    # exact zeros (the empty background of a projection) would take their share of the colours: they are left out of the ranking
    v = np.clip(v, vmin, vmax)
    if v.size > cap:
        v = v[::-(-v.size // cap)]
    table = np.quantile(v, np.linspace(0, 1, n)) if v.size else np.array([vmin, vmax])
    return Stretch("histeq", table=table)


def limits(img, pclip=(1.0, 99.9), scale="log", symmetric=False):
    """(vmin, vmax) from percentiles of the finite (and, for log, positive) pixels. symmetric: (-v, v) with v the pclip[1] percentile of |pixel| (for contrast maps: zero in the middle)."""
    v = img[np.isfinite(img)]
    if scale == "log":
        v = v[v > 0]
    if v.size == 0:
        return 0.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(v), pclip[1]))
        m = m if m > 0 else 1e-30
        return -m, m
    lo, hi = np.percentile(v, pclip)
    return float(lo), float(hi if hi > lo else lo + 1e-30)


def stretch(img, vmin, vmax, scale="log"):
    """map to [0,1]; NaN stays NaN."""
    a = np.asarray(img, dtype=np.float64)
    sc = parse_scale(scale)
    name = sc.name if isinstance(sc, Stretch) else sc
    with np.errstate(invalid="ignore", divide="ignore"):
        if name == "log":
            t = (np.log10(np.where(a > 0, a, np.nan)) - np.log10(max(vmin, 1e-300))) / (np.log10(vmax) - np.log10(max(vmin, 1e-300)))
        elif name == "asinh":
            soft = sc.soft if isinstance(sc, Stretch) else 0.1
            s = soft * 0.5 * (abs(vmax) + abs(vmin)) or 1.0
            t = (np.arcsinh(a / s) - np.arcsinh(vmin / s)) / (np.arcsinh(vmax / s) - np.arcsinh(vmin / s))
        elif name == "histeq":
            tab = sc.table if isinstance(sc, Stretch) and sc.table is not None else fit_scale(a, vmin, vmax, "histeq").table
            ranks = np.linspace(0, 1, len(tab))
            u, first = np.unique(tab, return_index=True)           # a flat part of the distribution (a plateau): its rank is the middle of the plateau's ranks
            hi_idx = len(tab) - 1 - np.unique(tab[::-1], return_index=True)[1]
            t = np.where(np.isfinite(a), np.interp(np.clip(a, vmin, vmax), u, 0.5 * (ranks[first] + ranks[hi_idx])), np.nan)
        else:
            t = (a - vmin) / (vmax - vmin)
    return np.clip(t, 0, 1)


def colorize(img, cmap="viridis", scale="log", vmin=None, vmax=None, nan_color=(40, 40, 40), symmetric=False):
    """(ny, nx, 3) uint8 with row 0 at the TOP (the map's rows are flipped: +y up)."""
    if vmin is None or vmax is None:
        lo, hi = limits(img, scale=scale, symmetric=symmetric)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    t = stretch(img, vmin, vmax, fit_scale(img, vmin, vmax, scale))
    lut = colormaps.lut(cmap)
    idx = np.nan_to_num(t, nan=0.0) * 255
    rgb = lut[idx.round().astype(np.uint8)]
    rgb[~np.isfinite(t)] = nan_color
    return np.ascontiguousarray(rgb[::-1]), (vmin, vmax)


def png_bytes(rgb):
    ny, nx, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[j].tobytes() for j in range(ny))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def write_png(path, img, **kw):
    rgb, lim = colorize(img, **kw)
    with open(path, "wb") as f:
        f.write(png_bytes(rgb))
    return lim
