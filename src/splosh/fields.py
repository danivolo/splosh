"""Built-in derived gas fields, registered through the plugin registry (plugins.register_field) when splosh.maps is imported.

All fields are per-particle values in code units (velocities, positions and the centre are those of the view's frame: centred, velocity subtracted, rotated about z),
then splatted like any field. Options of the View: gamma (adiabatic index, default 5/3), xh (hydrogen mass fraction; 0 = automatic: X = 1 - Z - Y from the snapshot's Metallicity columns 0 and 1 when present, else 0.76), mu (mean molecular weight; 0 = derive it),
bunit (UnitMagneticField_in_gauss of the run, for beta).

  T         temperature [K] = (gamma - 1) * mu * m_p / k_B * u * UnitVelocity^2, with u = InternalEnergy.  mu as GIZMO/GADGET:
            mu = 4 / (1 + 3 X + 4 X x_e) = (1 + 4 y) / (1 + y + x_e),  y = (1 - X) / (4 X) the helium number per hydrogen, X = View.xh or, in automatic mode, 1 - Z - Y per particle from Metallicity[:, 0] (total metals) and Metallicity[:, 1] (helium mass fraction, the FIRE-type layout) when the snapshot has it,
            else 0.76 (other metals' effect on mu is neglected),
            x_e = ElectronAbundance (electrons per hydrogen nucleus).  View.mu > 0 replaces it (then ElectronAbundance is not needed).  m_p = 1.67262178e-24 g, k_B = 1.38065e-16 erg/K.
  cs        sound speed = sqrt(gamma (gamma - 1) u)                                   [code velocity]
  pressure  P = (gamma - 1) rho u                                                      [code mass/length/time^2]
  entropy   P / rho^gamma = (gamma - 1) u rho^(1 - gamma)  (the adiabatic constant; not k_B-normalised)
  mach      |v - v_frame| / cs                                                         [dimensionless]
  jz        specific angular momentum about the z axis through the frame centre: x vy - y vx   [length velocity]
  j         |r x v| about the frame centre (3D)                                         [length velocity]
  vR, vphi  cylindrical radial and azimuthal velocity about the z axis through the centre: (x vx + y vy)/R, (x vy - y vx)/R  (NaN at R = 0)
  beta      plasma beta = P_thermal / P_magnetic, needs MagneticField.  GIZMO works internally in code units with mu_0 = 1, so P_magnetic = B_int^2 / 2 in code pressure units, and writes
            the field as B_out = B_int * UNIT_B_IN_GAUSS / UnitMagneticField_in_gauss with UNIT_B_IN_GAUSS = sqrt(4 pi P_unit) (gizmo/io.c gizmo2gauss, allvars.h), P_unit the code pressure
            unit in cgs.  Hence B_int = B_out * bunit / sqrt(4 pi P_unit) and beta = (gamma - 1) rho u / (B_int^2 / 2); equivalently B_gauss = B_out * bunit and
            beta = P_cgs / (B_gauss^2 / 8 pi).  bunit = View.bunit, else UnitMagneticField_in_gauss read from the parameter file next to the snapshot ('*usedvalues*' or
            'gizmo.params' in its directory or its parent), else GIZMO's default sqrt(4 pi) (begrun.c).  The snapshot header does not carry it.  Cosmological factors are not applied.
  B         |B| in gauss = |MagneticField| * bunit (bunit as for beta).
  Pmag      magnetic pressure B_int^2 / 2 [code pressure] = B_gauss^2 / 8 pi in cgs; beta = pressure / Pmag.
  divB      H |div B| / |B|, the relative divergence error (H = SmoothingLength); needs DivergenceOfMagneticField, written by GIZMO only with
            OUTPUT_BFIELD_DIVCLEAN_INFO (splosh does no neighbour search, so it cannot compute div B itself).
No neighbour searches are used (no vorticity, divergence, gradients). A dataset missing from the snapshot raises a KeyError naming it."""
import numpy as np

from . import plugins
from .units import BOLTZMANN, PROTONMASS, COLUMN, DENSITY, GAUSS, PRESSURE, DIMLESS, SPECIFIC_J, TEMPERATURE, VELOCITY, Units

DEFAULT_GAMMA = 5.0 / 3.0
DEFAULT_XH = 0.76     # used when neither View.xh nor a Metallicity dataset gives X


def mean_molecular_weight(electron_abundance, xh=DEFAULT_XH):
    """mu = 4 / (1 + 3 xh + 4 xh x_e), the GADGET/GIZMO expression (x_e electrons per hydrogen nucleus, xh hydrogen mass fraction)."""
    return 4.0 / (1.0 + 3.0 * xh + 4.0 * xh * np.asarray(electron_abundance, dtype=np.float64))


def _need(ctx, name):
    if name not in ctx.snap:
        raise KeyError(f"PartType0 dataset {name!r} is not in the snapshot (or was not loaded) but this field needs it")
    return np.asarray(ctx.snap[name], dtype=np.float64)


def _gamma(ctx):
    g = float(getattr(ctx.view, "gamma", DEFAULT_GAMMA))
    if not g > 1.0:
        raise ValueError("gamma must be > 1")
    return g


def hydrogen_fraction(ctx):
    """X: View.xh if > 0, else 1 - Z - Y per particle from Metallicity columns 0 (total metals) and 1 (helium) when the snapshot has them, else DEFAULT_XH."""
    xh = float(getattr(ctx.view, "xh", 0.0) or 0.0)
    if xh > 0:
        return xh
    z = ctx.snap.get("Metallicity")
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2 and z.shape[1] >= 2:
            return np.clip(1.0 - z[:, 0] - z[:, 1], 0.05, 1.0)
    return DEFAULT_XH


def _mu(ctx):
    mu = float(getattr(ctx.view, "mu", 0.0) or 0.0)
    if mu > 0:
        return mu
    return mean_molecular_weight(_need(ctx, "ElectronAbundance"), hydrogen_fraction(ctx))


def _units(ctx):
    return Units.from_header(ctx.snap.get("header", {}))


def f_temperature(ctx):
    u = _need(ctx, "InternalEnergy")
    return (_gamma(ctx) - 1.0) * _mu(ctx) * PROTONMASS / BOLTZMANN * u * _units(ctx).velocity_cm_s ** 2


def f_cs(ctx):
    g = _gamma(ctx)
    return np.sqrt(g * (g - 1.0) * _need(ctx, "InternalEnergy"))


def _pressure(ctx):
    return (_gamma(ctx) - 1.0) * _need(ctx, "Density") * _need(ctx, "InternalEnergy")


def f_entropy(ctx):
    g = _gamma(ctx)
    return (g - 1.0) * _need(ctx, "InternalEnergy") * _need(ctx, "Density") ** (1.0 - g)


def f_mach(ctx):
    return np.sqrt((ctx.vel ** 2).sum(1)) / f_cs(ctx)


def _rv(ctx):
    _need(ctx, "Coordinates"); _need(ctx, "Velocities")
    return ctx.frame.pos(ctx.snap["Coordinates"]), ctx.vel


def f_jz(ctx):
    r, v = _rv(ctx)
    return r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]


def f_j(ctx):
    r, v = _rv(ctx)
    return np.sqrt((np.cross(r, v) ** 2).sum(1))


def _cyl(ctx):
    r, v = _rv(ctx)
    R = np.hypot(r[:, 0], r[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return r, v, np.where(R > 0, R, np.nan)


def f_vR(ctx):
    r, v, R = _cyl(ctx)
    return (r[:, 0] * v[:, 0] + r[:, 1] * v[:, 1]) / R


def f_vphi(ctx):
    r, v, R = _cyl(ctx)
    return (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) / R


GIZMO_DEFAULT_BUNIT = 3.5449077018110318      # sqrt(4 pi): UnitMagneticField_in_gauss when the parameter is absent (gizmo begrun.c)


def magnetic_unit_gauss(snap, view):
    """(UnitMagneticField_in_gauss, source): View.bunit, else the run's parameter file near the snapshot, else GIZMO's default."""
    import glob
    import os
    b = float(getattr(view, "bunit", 0.0) or 0.0)
    if b > 0:
        return b, "option"
    path = snap.get("path")
    if path:
        d = os.path.dirname(os.path.abspath(path))
        for folder in (d, os.path.dirname(d)):
            for cand in sorted(glob.glob(os.path.join(folder, "*usedvalues*"))) + [os.path.join(folder, "gizmo.params")]:
                try:
                    for line in open(cand):
                        w = line.split("%")[0].split()
                        if len(w) >= 2 and w[0] == "UnitMagneticField_in_gauss":
                            return float(w[1]), cand
                except (OSError, ValueError):
                    pass
    return GIZMO_DEFAULT_BUNIT, "GIZMO default sqrt(4 pi)"


def f_beta(ctx):
    B = _need(ctx, "MagneticField")
    u = _units(ctx)
    p_unit = u.density_cgs * u.velocity_cm_s ** 2                       # code pressure unit in cgs
    bunit, _ = magnetic_unit_gauss(ctx.snap, ctx.view)
    b_int2 = (B ** 2).sum(1) * (bunit / np.sqrt(4.0 * np.pi * p_unit)) ** 2
    with np.errstate(divide="ignore"):
        return _pressure(ctx) / (0.5 * b_int2)


def f_bmag(ctx):
    """|B| in gauss: |MagneticField| * UnitMagneticField_in_gauss (the same conversion as beta)."""
    B = _need(ctx, "MagneticField")
    bunit, _ = magnetic_unit_gauss(ctx.snap, ctx.view)
    return np.sqrt((B ** 2).sum(1)) * bunit


def f_pmag(ctx):
    """magnetic pressure |B_int|^2 / 2 in code pressure units (GIZMO's mu_0 = 1 convention, as beta: beta = pressure / Pmag)."""
    B = _need(ctx, "MagneticField")
    u = _units(ctx)
    bunit, _ = magnetic_unit_gauss(ctx.snap, ctx.view)
    return 0.5 * (B ** 2).sum(1) * bunit ** 2 / (4.0 * np.pi * u.density_cgs * u.velocity_cm_s ** 2)


def f_divb(ctx):
    """Relative divergence error H |div B| / |B| (H = SmoothingLength, the kernel support): the usual MHD accuracy diagnostic, independent of the B unit.
    Needs DivergenceOfMagneticField, which GIZMO writes only when built with OUTPUT_BFIELD_DIVCLEAN_INFO (io.c, IO_DIVB)."""
    if "DivergenceOfMagneticField" not in ctx.snap:
        raise KeyError("PartType0 dataset 'DivergenceOfMagneticField' is not in the snapshot: GIZMO writes it only when built with OUTPUT_BFIELD_DIVCLEAN_INFO")
    d = np.abs(_need(ctx, "DivergenceOfMagneticField"))
    b = np.sqrt((_need(ctx, "MagneticField") ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _need(ctx, "SmoothingLength") * d / b


def _entropy_dim(view):
    g = float(getattr(view, "gamma", DEFAULT_GAMMA))
    return (-3.0 * (1.0 - g), 1.0 - g, 2.0)


def _T_needs(view):
    if float(getattr(view, "mu", 0.0) or 0.0) > 0:
        return ("InternalEnergy",)
    auto = not float(getattr(view, "xh", 0.0) or 0.0) > 0
    return ("InternalEnergy", "ElectronAbundance") + (("?Metallicity",) if auto else ())


_ALL = [
    # name, function, label, needs, unit
    ("T", f_temperature, "temperature (ideal gas, mu from ElectronAbundance)", _T_needs, TEMPERATURE),
    ("cs", f_cs, "sound speed sqrt(gamma (gamma-1) u)", ("InternalEnergy",), VELOCITY),
    ("pressure", _pressure, "pressure (gamma-1) rho u", ("Density", "InternalEnergy"), PRESSURE),
    ("entropy", f_entropy, "entropy P / rho^gamma", ("Density", "InternalEnergy"), _entropy_dim),
    ("mach", f_mach, "Mach number |v - v_frame| / cs", ("Velocities", "InternalEnergy"), DIMLESS),
    ("jz", f_jz, "specific angular momentum j_z about the frame centre", ("Velocities",), SPECIFIC_J),
    ("j", f_j, "specific angular momentum |r x v| about the frame centre", ("Velocities",), SPECIFIC_J),
    ("vR", f_vR, "cylindrical radial velocity about the centre", ("Velocities",), VELOCITY),
    ("vphi", f_vphi, "azimuthal velocity about the centre", ("Velocities",), VELOCITY),
    ("beta", f_beta, "plasma beta 2 P / |B|^2", ("Density", "InternalEnergy", "MagneticField"), DIMLESS),
    ("B", f_bmag, "magnetic field strength |B| [gauss]", ("MagneticField",), GAUSS),
    ("Pmag", f_pmag, "magnetic pressure |B|^2 / 2 (code units, mu_0 = 1)", ("MagneticField",), PRESSURE),
    ("divB", f_divb, "relative divergence error H |div B| / |B| (GIZMO: build with OUTPUT_BFIELD_DIVCLEAN_INFO)", ("MagneticField", "DivergenceOfMagneticField", "SmoothingLength"), DIMLESS),
]


def register_builtin_fields():
    for name, fn, label, needs, unit in _ALL:
        if name not in plugins.FIELDS:                  # a plugin that registered the name first keeps it
            plugins.register_field(name, fn, label=label, needs=needs, unit=unit, builtin=True, doc=fn.__doc__ or "")


def resolve_unit(view, name, mode="proj"):
    """The dimension of field `name` for this view (tuple, 'K', or None when unknown).  `rho` is a column density in a projection and a density in a slice."""
    from .maps import BUILTIN_UNITS
    if name == "rho":
        return COLUMN if mode == "proj" else DENSITY if mode == "slice" else None      # volume: the opacity, dimensionless
    u = plugins.FIELD_META[name]["unit"] if name in plugins.FIELD_META else BUILTIN_UNITS.get(name)
    return u(view) if callable(u) else u


def list_fields(view=None):
    """[(name, label, unit string in cgs, needs tuple, origin)] for every field, for `splosh fields`."""
    from .maps import BUILTIN, BUILTIN_NEEDS, View
    view = view or View()
    un = Units()
    rows = [("rho", "column density (projection) / SPH density (slice)", "g cm^-2 | g cm^-3", ("Masses",), "builtin")]
    for name in sorted(set(BUILTIN) | set(plugins.FIELDS)):
        if name in plugins.FIELDS:
            _, label, _, needs = plugins.FIELDS[name]
            origin = "derived" if plugins.FIELD_META[name]["builtin"] else "plugin"
            if callable(needs):
                needs = sorted(set(needs(view)) | set(needs(view.with_(mu=1.0))))
        else:
            label, needs, origin = BUILTIN[name][1], BUILTIN_NEEDS[name], "builtin"
        d = resolve_unit(view, name)
        rows.append((name, label, un.quantity_unit(d, "cgs")[0] if d is not None else "(unknown)", tuple(needs), origin))
    return rows


def format_fields(view=None):
    rows = list_fields(view)
    w = max(len(r[0]) for r in rows)
    lines = [f"{'field':{w}s}  {'unit (cgs)':22s}  {'PartType0 datasets':44s}  meaning"]
    for name, label, unit, needs, origin in rows:
        lines.append(f"{name:{w}s}  {unit:22s}  {', '.join(needs) or '-':44s}  {label}" + ("" if origin == "builtin" else f"  [{origin}]"))
    return "\n".join(lines)
