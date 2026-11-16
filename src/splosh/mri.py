"""MRI / dynamo diagnostics per snapshot: the snapshot product `mri` (products.py), read as the series source prod_mri.

Frame: modes.disc_frame (the rules of disc.py: two black holes: their centre of mass, z along the orbit normal, radii in units of a; one: about it, z along
the gas angular momentum, code length; none: the gas centre of mass). B in code units with mu_0 = 1 (B^2 / 2 is the magnetic pressure, as the disc
product and the map field Pmag; the map fields BR / Bphi are in gauss).

Per radius R0 of `radii` (the tag <r> is R0 as printed, e.g. 4 or 0.5), the gas of the annulus R0 (1 - dr) < R < R0 (1 + dr):
    Bphi<r>z_<z>   the volume-weighted mean B_phi in bins of height (the butterfly diagram: a heatmap of the family Bphi4z_ against time)
    BR<r>z_<z>     the same for B_R
    Mx<r>z_<z>     the mean Maxwell stress -B_R B_phi
    rho<r>z_<z>    the mean density (volume-weighted: sum m / sum V)
  z in units of the annulus' scale height H (z_unit = "H": H the mass-weighted rms height about the annulus' mean height, recomputed per snapshot) or of
  R0 (z_unit = "R"), nz bins over [-z_max, z_max].
  Scalars: Bphi<r>up / Bphi<r>lo (mean B_phi over 0 < z < z_max and -z_max < z < 0: the dynamo's sign flips; a `spectrum` panel on them gives the
  dynamo period), Bphi<r> (all heights), BR<r>, Bz<r>, H<r> (H / R0), aM<r> (sum -B_R B_phi V / sum P V: the Maxwell alpha over |z| < z_max),
  beta<r> (sum P V / sum B^2 / 2 V), Brms<r>, M<r> (gas mass of the annulus), N<r> (cells).
Parameters: radii ([2, 4, 8], units of the frame), dr (0.1), z_max (3), nz (30), z_unit (H | R), gamma (5/3), exclude_ids, center (auto | com).
Annuli with fewer than 50 cells, and z bins with none, are NaN."""
import numpy as np

from . import plugins
from .modes import disc_frame

GAMMA = 5.0 / 3.0


def _y(v):
    return f"{v:.4g}"


def b_code(s, keep=None):
    """the snapshot's MagneticField in code units with mu_0 = 1 (None without one)."""
    if "MagneticField" not in s:
        return None
    from .fields import magnetic_unit_gauss
    from .maps import View
    from .units import Units
    un = Units.from_header(s.get("header", {}))
    bunit, _ = magnetic_unit_gauss(s, View())
    B = np.asarray(s["MagneticField"], np.float64)
    return (B if keep is None else B[keep]) * bunit / np.sqrt(4 * np.pi * un.density_cgs * un.velocity_cm_s ** 2)


def load_disc(path, params, extra=()):
    """the gas of a snapshot in the disc frame: dict with the frame (modes.disc_frame) and per cell R (frame units), phi, z (frame units), m, V, rho, u,
    B (code units, or None) and its cylindrical components BR / Bphi / Bz, the velocities about the centre vR / vphi / vz; None without gas."""
    from .io import load_snapshot
    s = load_snapshot(path, gas_fields=("Coordinates", "Velocities", "Masses", "Density", "?InternalEnergy", "?MagneticField", "?ParticleIDs") + tuple(extra))
    keep = np.ones(len(s["Masses"]), bool)
    ex = params.get("exclude_ids") or []
    if ex and "ParticleIDs" in s:
        keep &= ~np.isin(s["ParticleIDs"], np.asarray(ex, dtype=np.int64))
    if not keep.any():
        return None
    pos = np.asarray(s["Coordinates"], np.float64)[keep]
    vel = np.asarray(s["Velocities"], np.float64)[keep]
    m = np.asarray(s["Masses"], np.float64)[keep]
    rho = np.asarray(s["Density"], np.float64)[keep]
    u = np.asarray(s["InternalEnergy"], np.float64)[keep] if "InternalEnergy" in s else np.zeros(len(m))
    r_out = float(params.get("r_frame") or 12.0)
    fr = disc_frame(s, pos, vel, m, r_max=r_out, center=params.get("center") or "auto")
    sc = fr["scale"]
    rel, vrel = pos - fr["com"], vel - fr["vcom"]
    x1, x2, z = rel @ fr["e1"], rel @ fr["e2"], rel @ fr["zhat"]
    R = np.hypot(x1, x2)
    Rs = np.maximum(R, 1e-300)
    c, sn = x1 / Rs, x2 / Rs
    v1, v2 = vrel @ fr["e1"], vrel @ fr["e2"]
    d = {"frame": fr, "R": R / sc, "phi": np.arctan2(x2, x1), "z": z / sc, "m": m, "rho": rho, "V": m / rho, "u": u,
         "vR": v1 * c + v2 * sn, "vphi": -v1 * sn + v2 * c, "vz": vrel @ fr["zhat"], "B": None, "snap": s, "keep": keep}
    B = b_code(s, keep)
    if B is not None:
        b1, b2 = B @ fr["e1"], B @ fr["e2"]
        d.update(B=B, BR=b1 * c + b2 * sn, Bphi=-b1 * sn + b2 * c, Bz=B @ fr["zhat"])
    return d


def _bin(i, n, w):
    return np.bincount(i[i >= 0], weights=w[i >= 0], minlength=n)[:n]


def mri(path, params):
    radii = [float(r) for r in (params.get("radii") or [2.0, 4.0, 8.0])]
    dr = float(params.get("dr") or 0.1)
    d = load_disc(path, {**params, "r_frame": 1.5 * max(radii) * (1 + dr)})
    if d is None or d["B"] is None:
        return {}
    gamma = float(params.get("gamma") or GAMMA)
    zmax, nz = float(params.get("z_max") or 3.0), int(params.get("nz") or 30)
    unit_h = (params.get("z_unit") or "H").upper() == "H"
    ze = np.linspace(-zmax, zmax, nz + 1)
    zc = 0.5 * (ze[1:] + ze[:-1])
    R, z, m, V = d["R"], d["z"], d["m"], d["V"]
    P = (gamma - 1.0) * d["rho"] * np.maximum(d["u"], 0.0)
    BR, Bphi, Bz = d["BR"], d["Bphi"], d["Bz"]
    B2 = BR ** 2 + Bphi ** 2 + Bz ** 2
    out = {}
    for r0 in radii:
        t = _y(r0)
        sel = (R > r0 * (1 - dr)) & (R < r0 * (1 + dr))
        n = int(sel.sum())
        out[f"N{t}"], out[f"M{t}"] = n, float(m[sel].sum())
        nanz = {f"{k}{t}z_{_y(y)}": np.nan for k in ("Bphi", "BR", "Mx", "rho") for y in zc}
        if n < 50:
            out.update(nanz)
            out.update({f"{k}{t}": np.nan for k in ("Bphi", "Bphiup", "Bphilo", "BR", "Bz", "H", "aM", "beta", "Brms")})
            continue
        zs, ms, Vs = z[sel], m[sel], V[sel]
        z0 = np.sum(ms * zs) / ms.sum()
        H = float(np.sqrt(np.sum(ms * (zs - z0) ** 2) / ms.sum()))
        zz = (zs - z0) / (H if unit_h else r0)
        iz = np.digitize(zz, ze) - 1
        iz = np.where((iz >= 0) & (iz < nz), iz, -1)
        vol = _bin(iz, nz, Vs)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, f in (("Bphi", Bphi[sel]), ("BR", BR[sel]), ("Mx", -BR[sel] * Bphi[sel])):
                prof = _bin(iz, nz, Vs * f) / vol
                out.update({f"{k}{t}z_{_y(y)}": float(v) for y, v in zip(zc, prof)})
            out.update({f"rho{t}z_{_y(y)}": float(v) for y, v in zip(zc, _bin(iz, nz, ms) / vol)})
        inz = iz >= 0
        Vi = Vs[inz]
        up, lo = inz & (zz > 0), inz & (zz <= 0)
        mean = lambda f, w: float(np.sum(Vs[w] * f[w]) / np.sum(Vs[w])) if w.any() else np.nan
        out[f"Bphi{t}"], out[f"Bphiup{t}"], out[f"Bphilo{t}"] = mean(Bphi[sel], inz), mean(Bphi[sel], up), mean(Bphi[sel], lo)
        out[f"BR{t}"], out[f"Bz{t}"] = mean(BR[sel], inz), mean(Bz[sel], inz)
        PV, B2V = np.sum(P[sel][inz] * Vi), np.sum(B2[sel][inz] * Vi)
        out[f"H{t}"] = H / r0
        out[f"aM{t}"] = float(-np.sum((BR[sel] * Bphi[sel])[inz] * Vi) / PV) if PV > 0 else np.nan
        out[f"beta{t}"] = float(PV / (0.5 * B2V)) if B2V > 0 else np.nan
        out[f"Brms{t}"] = float(np.sqrt(B2V / Vi.sum()))
    fr = d["frame"]
    if fr["binary"]:
        out.update(a=fr["a"], e=fr["e"])
    from .fields import bunit_scalars
    out.update(bunit_scalars(d["snap"]))
    return {"scalars": out}


def _meta(radii=(2.0, 4.0, 8.0)):
    m = {}
    for r in radii:
        t = _y(r)
        m.update({f"Bphi{t}z_": {"label": f"B_phi(z) at R = {t} (butterfly)", "group": "butterfly"},
                  f"BR{t}z_": {"label": f"B_R(z) at R = {t}", "group": "butterfly"},
                  f"Mx{t}z_": {"label": f"-B_R B_phi (z) at R = {t}", "group": "butterfly"},
                  f"rho{t}z_": {"label": f"rho(z) at R = {t}", "group": "butterfly"},
                  f"Bphiup{t}": {"label": f"mean B_phi, upper half, R = {t}", "group": "dynamo"},
                  f"Bphilo{t}": {"label": f"mean B_phi, lower half, R = {t}", "group": "dynamo"},
                  f"Bphi{t}": {"label": f"mean B_phi, R = {t}", "group": "dynamo"},
                  f"aM{t}": {"label": f"Maxwell alpha, R = {t}", "group": "annulus"},
                  f"beta{t}": {"label": f"plasma beta <P> / <B^2/2>, R = {t}", "group": "annulus"},
                  f"H{t}": {"label": f"H / R, R = {t}", "group": "annulus"},
                  f"Brms{t}": {"label": f"B rms, R = {t}", "group": "annulus"}})
    return m


def _applies(path):
    import h5py
    with h5py.File(path, "r") as f:
        return "PartType0" in f and "MagneticField" in f["PartType0"]


plugins.register_product("mri", mri, needs=("Coordinates", "Velocities", "Masses", "Density", "MagneticField", "?InternalEnergy", "?ParticleIDs",
                                                         "PartType5"),
                         version=1, label="mri: butterfly B_phi(z, t) at a few radii, dynamo-period columns, Maxwell alpha per annulus (MHD runs)",
                         applies=_applies,
                         params={"radii": [2.0, 4.0, 8.0], "dr": 0.1, "z_max": 3.0, "nz": 30, "z_unit": "H", "gamma": GAMMA, "exclude_ids": [],
                                 "center": "auto"},
                         defaults=["aM4"], meta=_meta())
