"""Maps of a snapshot: a View (centre rule, alignment, projection axes, projection or slice, field, window, resolution) -> a 2D float32 array.

Image convention: array[j, i] with i along the horizontal view axis and j along the vertical one, both increasing (origin lower-left, like the splat engine);
flip rows for display. The window is [cx - width/2, cx + width/2] x [cy - width/2, cy + width/2] in the frame coordinates (after centring and alignment).

Fields (name -> meaning):
    rho      projection: column density (mass / length^2);  slice: SPH density sum_j m_j W (mass / length^3)
    u, vx, vy, vz, speed, vlos, H     per-particle quantity q. Weighted maps are  sum w q K / sum w K  with w = mass (density weighted: the ordinary mean along the line of
             sight, or at the plane) or w = mass / density ('volume': the normalised SPH interpolation, the usual choice for a slice).  vx, vy, vz are in the aligned frame
             (centre velocity subtracted); vlos is the component along the line of sight.  H is the kernel support radius.
`weight='auto'` is 'mass' for projections and 'volume' for slices.  Derived physics fields (T, cs, pressure, entropy, mach, jz, j, vR, vphi, beta) are in fields.py; plugins add
fields through splosh.plugins.register_field.  View.units ('code' | 'cgs' | 'astro') converts the map values; meta['quantity_unit'] says what the values are in."""
from dataclasses import dataclass, asdict, replace

import numpy as np

from . import plugins, transforms
from .frames import make_frame, tilt_needs
from .units import Units, SYSTEMS, quantity_meta, COLUMN, DENSITY, ENERGY_PER_MASS, LENGTH, VELOCITY
from .splat import splat_bands_pyr, splat_bands, splat_slice, splat_volume

AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}      # (horizontal, vertical, line of sight) as indices of the frame axes
KERNEL_NOTE = "GIZMO cubic spline, SmoothingLength = H = kernel support radius"


@dataclass(frozen=True)
class View:
    center: str = "bh"          # frame.py centre rule
    align: str = "none"         # frame.py alignment rule
    tilt: str = "none"          # frame.py tilt rule: any 3D rotation after the alignment (incl:I[,PA] | euler:A,B,C | los:x,y,z | L[:R] | Lbh, joined by '+')
    axes: str = "xy"            # xy | xz | yz  (third letter = line of sight)
    mode: str = "proj"          # proj | slice | volume (front-to-back emission + absorption, seen from +line of sight: splat.splat_volume)
    field: str = "rho"
    width: float = 10.0         # window width = height in code length units
    npix: int = 512
    cx: float = 0.0             # window centre in frame coordinates (pan)
    cy: float = 0.0
    z0: float = 0.0             # slice plane position along the line of sight (frame coordinates)
    weight: str = "auto"        # auto | mass | volume | column (sum of m q K, not normalised: the column density of q in a projection, rho q in a slice)
    hscale: float = 1.0         # multiplies the stored smoothing length before splatting: 1 = the code's own kernel (H = support radius); 2 = what a reader that treats H as
                                # the Monaghan h (support 2h), e.g. pynbody fed `smooth = H`, draws
    units: str = "code"         # code | cgs | astro: the unit system of the map values (the window, width, cx, cy, z0 stay in code length units)
    gamma: float = 5.0 / 3.0    # adiabatic index used by the derived fields (T, cs, pressure, entropy, mach, beta)
    xh: float = 0.0             # hydrogen mass fraction for mu of the temperature: 0 = from the snapshot's Metallicity (X = 1 - Z - Y, columns 0 and 1) when present, else 0.76; > 0 fixes it (metals neglected)
    mu: float = 0.0             # mean molecular weight override for T (0 = from ElectronAbundance and xh)
    bunit: float = 0.0          # UnitMagneticField_in_gauss of the run, for beta: 0 = read it from the run's parameter file next to the snapshot, else GIZMO's default sqrt(4 pi)
    post: str = "none"          # transforms.py: none | resid[:azim|mean|smooth:S] | grad[:S]  (contrast against a reference, or gradient magnitude, of the map)
    exact: bool = False         # projection: True = exact row-band engine instead of the multi-resolution default (slices are always exact)
    kappa: float = 0.0          # volume: opacity per unit mass (code length^2 / mass), tau = kappa * column density; 0 = automatic (tau = 1 at the window's mean column density)

    def as_dict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)

    def validate(self):
        if self.axes not in AXES:
            raise ValueError(f"axes must be one of {list(AXES)}")
        if self.mode not in ("proj", "slice", "volume"):
            raise ValueError("mode must be proj, slice or volume")
        if not self.kappa >= 0:
            raise ValueError("kappa must be >= 0 (0 = automatic)")
        if self.weight not in ("auto", "mass", "volume", "column"):
            raise ValueError("weight must be auto, mass, volume or column")
        if self.weight == "column" and self.mode == "volume":
            raise ValueError("weight 'column' is for projections and slices")
        if self.units not in SYSTEMS:
            raise ValueError(f"units must be one of {list(SYSTEMS)}")
        if not self.gamma > 1.0:
            raise ValueError("gamma must be > 1")
        if not (0.0 <= self.xh <= 1.0 and self.mu >= 0.0 and self.bunit >= 0.0):
            raise ValueError("xh must be in [0, 1] (0 = automatic), mu >= 0 and bunit >= 0")
        if not (self.hscale > 0):
            raise ValueError("hscale must be > 0")
        transforms.parse(self.post)
        if not (self.width > 0 and 8 <= self.npix <= 8192):
            raise ValueError("width must be > 0 and 8 <= npix <= 8192")
        return self


class FieldContext:
    """What a field function sees: the snapshot dict, the frame, the velocities in the frame, the view's axis indices."""
    def __init__(self, snap, frame, view):
        self.snap, self.frame, self.view = snap, frame, view
        self._vel = None

    @property
    def vel(self):
        if self._vel is None:
            self._vel = self.frame.vel(self.snap["Velocities"])
        return self._vel

    @property
    def los(self):
        return AXES[self.view.axes][2]


def _speed(c):
    return np.sqrt((c.vel ** 2).sum(1))


BUILTIN = {
    "u": (lambda c: c.snap["InternalEnergy"], "specific internal energy"),
    "vx": (lambda c: c.vel[:, 0], "velocity x (frame)"),
    "vy": (lambda c: c.vel[:, 1], "velocity y (frame)"),
    "vz": (lambda c: c.vel[:, 2], "velocity z (frame)"),
    "vlos": (lambda c: c.vel[:, c.los], "velocity along the line of sight"),
    "speed": (_speed, "speed in the frame"),
    "H": (lambda c: c.snap["SmoothingLength"], "kernel support radius"),
}


# dimension of each built-in field in code units (units.py) and the PartType0 datasets it reads besides Coordinates, SmoothingLength, Masses
BUILTIN_UNITS = {"u": ENERGY_PER_MASS, "vx": VELOCITY, "vy": VELOCITY, "vz": VELOCITY, "vlos": VELOCITY, "speed": VELOCITY, "H": LENGTH}
BUILTIN_NEEDS = {"u": ("InternalEnergy",), "vx": ("Velocities",), "vy": ("Velocities",), "vz": ("Velocities",), "vlos": ("Velocities",), "speed": ("Velocities",), "H": ()}


def field_names():
    plugins.load_plugins()
    return ["rho"] + sorted(set(BUILTIN) | set(plugins.FIELDS))


def _field_values(name, ctx):
    if name not in plugins.FIELDS and name not in BUILTIN:
        from . import datasets
        datasets.ensure(name)                           # a snapshot dataset by its name (datasets.py)
    if name in plugins.FIELDS:
        return np.asarray(plugins.FIELDS[name][0](ctx), dtype=np.float32)
    if name in BUILTIN:
        return np.asarray(BUILTIN[name][0](ctx), dtype=np.float32)
    raise KeyError(f"unknown field {name!r}; available: {field_names()}")


def resolve_weight(view):
    return view.weight if view.weight != "auto" else ("mass" if view.mode == "proj" else "volume")


def required_gas_fields(view):
    """PartType0 datasets a render of `view` reads (the loader takes this as gas_fields)."""
    f = {"Coordinates", "SmoothingLength", "Masses"}
    if view.field not in plugins.FIELDS and view.field not in BUILTIN:
        from . import datasets
        datasets.ensure(view.field)                     # a snapshot dataset by its name (datasets.py)
    ckey = view.center.split(":")[0]
    if ckey in ("com", "mass-weighted", "peak"):
        f.add("Velocities")
    f.update(BUILTIN_NEEDS.get(view.field, ()))
    if (view.field != "rho" and resolve_weight(view) == "volume") or ckey == "peak":
        f.add("Density")
    if view.field in plugins.FIELDS:
        needs = plugins.FIELDS[view.field][3]
        f.update(needs(view) if callable(needs) else needs)
    if ckey in ("ids", "pair") or "pair" in view.align:
        f.add("ParticleIDs")
    f |= tilt_needs(view.tilt)
    return f


def render_fields(snap, view, fields, frame=None):
    """Several fields of one view (projection or slice; not rho, not volume) in ONE splat pass: [(npix, npix) float64, ...] in code units, NaN where
    undefined, weighted as render_map weights them (resolve_weight). What vector overlays use for their two components."""
    view.validate()
    if view.mode not in ("proj", "slice") or "rho" in fields:
        raise ValueError("render_fields: projection or slice, fields other than rho")
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    ia, ib, il = AXES[view.axes]
    pos = frame.pos(snap["Coordinates"])
    a, b, l = pos[:, ia], pos[:, ib], pos[:, il]
    h = np.asarray(snap["SmoothingLength"]) * np.float32(view.hscale) if view.hscale != 1.0 else np.asarray(snap["SmoothingLength"])
    half = 0.5 * view.width
    keep = (np.abs(a - view.cx) < half + h) & (np.abs(b - view.cy) < half + h)
    if view.mode == "slice":
        keep &= np.abs(l - view.z0) < h
    idx = np.nonzero(keep)[0]
    weight = resolve_weight(view.with_(field=fields[0]))
    mass = np.asarray(snap["Masses"], dtype=np.float64)
    w = mass[idx] if weight in ("mass", "column") else mass[idx] / np.asarray(snap["Density"], dtype=np.float64)[idx]
    if len(idx) == 0:
        return [np.full((view.npix, view.npix), np.nan) for _ in fields]
    ctx = FieldContext(snap, frame, view)
    vals = np.ascontiguousarray(np.stack([np.asarray(_field_values(f, ctx))[idx] for f in fields]), dtype=np.float32)
    x, y, z, hh = (np.ascontiguousarray(v[idx]) for v in (a, b, l, h))
    if view.mode == "proj":
        img = (splat_bands if view.exact else splat_bands_pyr)(x, y, hh, w, vals, (view.cx, view.cy), view.width, view.npix)
        thr = 0.0
    else:
        img = splat_slice(x, y, z, hh, w, vals, (view.cx, view.cy, view.z0), view.width, view.npix, pyramid=False)
        w0 = img[..., 0]
        thr = (1e-6 * float(w0.max()) if w0.size and w0.max() > 0 else 0.0) if weight == "mass" else 0.05
    w0 = img[..., 0]
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(len(fields)):                        # the same float32 arithmetic as render_map (bitwise the same maps)
            if weight == "column":
                out.append(img[..., k + 1].astype(np.float64))
            else:
                out.append(np.where(w0 > thr, img[..., k + 1] / w0, np.nan).astype(np.float32).astype(np.float64))
    return out


def render_map(snap, view, frame=None, engine_kw=None):
    """Render `view` of the loaded snapshot dict. Returns (image (npix, npix) float32 with NaN where undefined, meta dict).
    meta['weight_image'] is the second splat output normalisation (out0) as float32 for diagnostics.
    engine_kw: extra keyword arguments for the projection engine (e.g. min_npix of the multi-resolution levels; used by tiles.py)."""
    view.validate()
    plugins.load_plugins()
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    from .frames import select_in_frame, gather
    ia, ib, il = AXES[view.axes]
    half = 0.5 * view.width
    # the frame transform, the window cut and the gathers in one parallel pass (they were ~0.5 s of serial numpy per render at 9M particles)
    x, y, z, hh, idx = select_in_frame(snap, frame, AXES[view.axes], view.width, view.cx, view.cy, view.hscale, view.z0, view.mode == "slice")
    weight = resolve_weight(view)
    mass = np.asarray(snap["Masses"])
    if view.field == "rho" or weight in ("mass", "column"):
        w = gather(mass, idx).astype(np.float64)
    else:
        w = gather(mass, idx).astype(np.float64) / gather(np.asarray(snap["Density"]), idx).astype(np.float64)
    if view.field == "rho":
        vals = np.zeros((0, len(idx)), np.float32)
    else:
        ctx = FieldContext(snap, frame, view)
        q = np.asarray(_field_values(view.field, ctx))
        vals = gather(np.ascontiguousarray(q, dtype=np.float32), idx)[None, :] if q.ndim == 1 else np.ascontiguousarray(q[idx][None, :], dtype=np.float32)
    kappa = None
    if view.mode == "volume":
        mass_i = gather(mass, idx).astype(np.float64)
        src = np.ones(len(idx)) if view.field == "rho" else vals[0].astype(np.float64)
        kappa = view.kappa
        if not kappa > 0:                                   # automatic: tau = 1 at the window's mean column density (dense structures opaque, diffuse gas transparent)
            mwin = float(mass_i[(np.abs(x - view.cx) < 0.5 * view.width) & (np.abs(y - view.cy) < 0.5 * view.width)].sum())
            kappa = view.width ** 2 / mwin if mwin > 0 else 1.0
        vol = splat_volume(x, y, z, hh, mass_i, src, kappa, (view.cx, view.cy), view.width, view.npix) if len(idx) else np.zeros((view.npix, view.npix, 2), np.float32)
        E, A = vol[..., 0], vol[..., 1]
        if view.field == "rho":
            out, label = A.astype(np.float32), "opacity (volume)"
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(A > 1e-3, E / A, np.nan).astype(np.float32)
            label = f"visible {view.field} (volume)"
        img = None
    elif len(idx) == 0:
        img = np.zeros((view.npix, view.npix, vals.shape[0] + 1), np.float32)
    elif view.mode == "proj":
        fn = splat_bands if view.exact else splat_bands_pyr
        img = fn(x, y, hh, w, vals, (view.cx, view.cy), view.width, view.npix, **(engine_kw or {}))
    else:
        img = splat_slice(x, y, z, hh, w, vals, (view.cx, view.cy, view.z0), view.width, view.npix, pyramid=False)   # exact: only particles within H of the plane count, and the pyramid was not faster (bench/compare_slice.py)
    if img is None:
        pass                                                # volume: out and label are set above
    elif view.field == "rho":
        w0 = img[..., 0]
        out = w0.copy()
        label = "column density" if view.mode == "proj" else "density"
    elif weight == "column":                               # not normalised: sum m q K = the column density of q (projection) or rho q (slice)
        w0 = img[..., 0]
        out = img[..., 1].astype(np.float32)
        label = f"column of {view.field}" if view.mode == "proj" else f"rho {view.field}"
    else:
        w0 = img[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = img[..., 1] / w0
        thr = 0.0 if view.mode == "proj" else 0.05          # slice with w = m/rho: out0 ~ 1 inside the gas; under 5% the interpolation is not defined
        if weight == "mass" and view.mode == "slice":
            thr = 1e-6 * float(w0.max()) if w0.size and w0.max() > 0 else 0.0
        out = np.where(w0 > thr, out, np.nan).astype(np.float32)
        label = f"{weight}-weighted {view.field}"
    units = Units.from_header(snap.get("header", {}))
    from .fields import resolve_unit
    dim = resolve_unit(view, view.field, view.mode)
    if weight == "column" and view.field != "rho" and isinstance(dim, tuple):
        dim = tuple(a + b for a, b in zip(dim, COLUMN if view.mode == "proj" else DENSITY))
    qmeta = quantity_meta(units, dim, view.field, view.units)
    out = out * np.float32(qmeta["factor"]) if qmeta["factor"] != 1.0 else out
    lu, lf = units.length_unit(view.units)
    tu, tf = units.time_unit(view.units)
    meta = {
        "quantity_unit": qmeta, "unit_system": view.units, "unit_scales": units.describe(),
        "length_unit": {"unit": lu, "factor": lf}, "time_unit": {"unit": tu, "factor": tf}, "time_physical": float(snap["time"]) * tf,
        "extent_physical": [e * lf for e in (view.cx - half, view.cx + half, view.cy - half, view.cy + half)],
        "view": view.as_dict(), "time": float(snap["time"]), "frame": frame.describe(), "weight": weight, "label": label,
        "extent": [view.cx - half, view.cx + half, view.cy - half, view.cy + half], "n_particles": int(len(idx)),
        "kernel": KERNEL_NOTE + ("" if view.hscale == 1.0 else f"; H multiplied by hscale = {view.hscale:g} (legacy convention, NOT the code's kernel)"), "origin": "lower", "units": snap.get("header", {}),
        "xlabel": "xyz"[ia], "ylabel": "xyz"[ib],
    }
    if kappa is not None:
        meta["kappa"] = float(kappa)
    out = out.astype(np.float32)
    out, meta = transforms.finish(out, meta, view.post)
    return out, meta


def bh_markers(snap, view, frame=None):
    """Black-hole particles in view coordinates: list of dict(id, mass, x, y, depth) (x, y: horizontal / vertical window coordinates, depth along the line of sight)."""
    if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
        return []
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    ia, ib, il = AXES[view.axes]
    p = frame.pos(snap["bh_pos"])
    m = snap["bh_mdyn"] if "bh_mdyn" in snap else snap["bh_mass"]
    return [{"id": int(snap["bh_id"][k]), "mass": float(m[k]), "x": float(p[k, ia]), "y": float(p[k, ib]), "depth": float(p[k, il])} for k in range(len(p))]


from . import fields as _fields          # noqa: E402  (registers the built-in derived fields in the plugin registry)
_fields.register_builtin_fields()
