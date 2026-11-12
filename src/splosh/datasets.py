"""Every PartType0 dataset of a snapshot as a plottable field, without writing a plugin (what SPLASH offers by default).

Names are the dataset names of the file:
  Name        a scalar dataset as it is (e.g. ElectronAbundance, TimeStep, a code's own columns); for a dataset of 3 columns, its magnitude |Name|
  Name_k      column k (from 0) of a dataset with several columns (e.g. Metallicity_0, the total metal mass fraction in GIZMO's layout)
Values are those of the file, in its units (the maps carry no unit and are never converted). In a projection they are mass-weighted along the line of sight,
as every field: a mass fraction becomes the mass-weighted mean fraction of the column. The datasets the map pipeline already reads under other names
(positions, velocities, masses, ids, H, u, B) are left out: they have their own fields (rho, vx.., H, u, B, Bx..).

    ensure(name)       register the field `name` when it has the form of a dataset name; True when it is (now) a known field
    from_file(path)    register the fields of one snapshot's datasets; returns their names
"""
import re

import h5py
import numpy as np

from . import plugins

SKIP = {"Coordinates", "Velocities", "Masses", "ParticleIDs", "SmoothingLength", "InternalEnergy", "MagneticField"}
_PAT = re.compile(r"^([A-Z][A-Za-z0-9]*)(?:_(\d+))?$")        # GADGET/GIZMO dataset names are CamelCase; a derived field name starts in lower case


def _fn(base, col):
    def values(ctx):
        a = np.asarray(ctx.snap[base], dtype=np.float64)
        if col is not None:
            if a.ndim != 2 or col >= a.shape[1]:
                raise KeyError(f"dataset {base!r} has no column {col} (shape {a.shape})")
            return a[:, col]
        if a.ndim == 2:
            return np.sqrt((a ** 2).sum(1)) if a.shape[1] == 3 else a[:, 0]
        return a
    return values


def ensure(name):
    if name == "rho" or name in plugins.FIELDS:
        return True
    from .maps import BUILTIN
    if name in BUILTIN:
        return True
    m = _PAT.match(name or "")
    if not m or m.group(1) in SKIP or len(m.group(1)) < 3 or m.group(1).upper() == m.group(1):
        return False                                # short or all-capital names (A2, R, ...) are other quantities (profile modes, geometry), not datasets
    base, col = m.group(1), (int(m.group(2)) if m.group(2) is not None else None)
    what = f"column {col} of the dataset {base}" if col is not None else f"the dataset {base} (magnitude when it is a 3-vector)"
    plugins.register_field(name, _fn(base, col), label=f"{name} (snapshot dataset)", needs=(base,), unit=None, builtin=True,
                           doc=f"{what}, as stored in the snapshot (file units)")
    plugins.FIELD_META[name]["dataset"] = True
    return True


def from_file(path):
    names = []
    try:
        with h5py.File(path, "r") as f:
            g = f.get("PartType0")
            if g is None:
                return names
            for k, ds in g.items():
                if k in SKIP or not isinstance(ds, h5py.Dataset) or ds.dtype.kind not in "fiu" or not _PAT.match(k):
                    continue
                if ds.ndim == 1 or (ds.ndim == 2 and ds.shape[1] == 3):
                    cand = [k]
                elif ds.ndim == 2:
                    cand = [f"{k}_{i}" for i in range(ds.shape[1])]
                else:
                    continue
                names += [n for n in cand if ensure(n)]
    except OSError:
        pass
    return names
