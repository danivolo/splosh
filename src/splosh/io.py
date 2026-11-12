"""Minimal GIZMO snapshot reader (h5py): the arrays a map needs, as float32/float64 numpy arrays in code units.
PartType0 gas: Coordinates, SmoothingLength (= kernel SUPPORT radius H in GIZMO), Masses, Density, InternalEnergy, Velocities.
PartType5 black holes: Coordinates, Masses (BH_Mass if present), Velocities, ParticleIDs for centring and alignment.
Extra, code-specific datasets are read by plugins, not here."""
import h5py
import numpy as np

from . import progress


def load_snapshot(path, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities"), types=(), rows=None):
    """gas_fields: the PartType0 datasets to read (add "ParticleIDs" to centre on gas particles by ID); a name starting with "?" is optional.
    types: other particle types to read for particle overlays, e.g. (1, 4): snap["pt"][t] = dict(pos float64 (N, 3), mass, id, vel when present); a type the file lacks is skipped
    (type 5, the black holes, is always read into the bh_* keys)."""
    out = {}
    with h5py.File(path, "r") as f:
        hd = f["Header"].attrs
        out["time"] = float(hd["Time"])
        out["header"] = {k: (hd[k].tolist() if hasattr(hd[k], "tolist") else hd[k]) for k in
                         ("BoxSize", "Redshift", "UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS", "Kernel_Function_ID", "ComovingIntegrationOn",
                                                   "HubbleParam", "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s") if k in hd}
        g = f["PartType0"]
        out["path"] = str(path)
        todo = []
        for k in gas_fields:
            if k.startswith("?"):                      # optional dataset: read when present, silently skipped otherwise
                if k[1:] in g:
                    todo.append(k[1:])
                continue
            if k not in g:
                raise KeyError(f"{path}: PartType0 has no dataset {k!r} (needed by the requested field or centre rule); datasets present: {sorted(g.keys())}")
            todo.append(k)
        progress.bytes_total(sum(_nbytes(g[k], rows) for k in todo))
        for k in todo:
            out[k] = _read(g[k], rows)
        if rows is not None:
            out["rows"] = [list(r) for r in rows]          # partial load (region.py): only these PartType0 rows, in this order
        if "PartType5" in f:
            b = f["PartType5"]
            out["bh_pos"] = b["Coordinates"][...].astype(np.float64)
            out["bh_mass"] = (b["BH_Mass"][...] if "BH_Mass" in b else b["Masses"][...]).astype(np.float64)
            out["bh_mdyn"] = b["Masses"][...].astype(np.float64)       # dynamical mass (what gravity sees); frames centre on this one
            out["bh_vel"] = b["Velocities"][...].astype(np.float64)
            out["bh_id"] = b["ParticleIDs"][...]
        for t in types:
            if t == 5 or f"PartType{t}" not in f or "Coordinates" not in f[f"PartType{t}"]:
                continue
            g = f[f"PartType{t}"]
            d = {"pos": g["Coordinates"][...].astype(np.float64)}
            for key, name in (("mass", "Masses"), ("id", "ParticleIDs"), ("vel", "Velocities")):
                if name in g:
                    d[key] = g[name][...]
            out.setdefault("pt", {})[int(t)] = d
    return out


BLOCK_BYTES = 64 << 20      # whole datasets are read in blocks of about this size, so the progress bar moves during a large read


def _nbytes(ds, rows):
    row = ds.dtype.itemsize * int(np.prod(ds.shape[1:], dtype=np.int64))
    return row * (ds.shape[0] if rows is None else sum(b - a for a, b in rows))


def _read(ds, rows):
    """a whole dataset, or the concatenation of its row ranges [(start, stop), ...] (contiguous reads: cheap for GIZMO's unchunked datasets).
    Reports the bytes read to progress (the viewer's progress bar)."""
    row = ds.dtype.itemsize * int(np.prod(ds.shape[1:], dtype=np.int64))
    if rows is None:
        n = ds.shape[0]
        step = max(1, BLOCK_BYTES // max(row, 1))
        if n <= step:
            out = ds[...]
            progress.bytes_read(n * row)
            return out
        out = np.empty(ds.shape, ds.dtype)
        for a in range(0, n, step):
            b = min(n, a + step)
            ds.read_direct(out, np.s_[a:b], np.s_[a:b])
            progress.bytes_read((b - a) * row)
        return out
    if not rows:
        return np.empty((0,) + ds.shape[1:], ds.dtype)
    parts = []
    for a, b in rows:
        parts.append(ds[a:b])
        progress.bytes_read((b - a) * row)
    return np.concatenate(parts)


def binary_com(snap):
    """mass-weighted centre of the black holes (the binary centre of mass for a two-BH snapshot), weighted with the dynamical `Masses` (the same rule as the frames' `bh` centre); falls back to the gas density maximum."""
    if "bh_pos" in snap and len(snap["bh_pos"]):
        m = snap["bh_mdyn"] if "bh_mdyn" in snap else snap["bh_mass"]
        return (snap["bh_pos"] * m[:, None]).sum(0) / m.sum()
    i = np.argmax(snap["Density"])
    return snap["Coordinates"][i].astype(np.float64)
