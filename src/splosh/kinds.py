"""Kinds of runs (plugins.register_kind, docs/KINDS.md): which folders and snapshot files belong to a kind that is not GIZMO particles, and opening them.

    kind_of_dir(folder)     the name of the kind that claims the folder (its detect), or None (a particle folder)
    kind_of_file(path)      the kind whose snapshot `path` is (the folder holding it, or that folder's parent, is claimed and lists it), or None
    snapshots(kind, folder) the kind's snapshot files of a folder, in natural order
    open_mesh(path)         the kind's opened object of a snapshot file (a small cache of opened files)
    snapshot_time(path)     the time of a snapshot of a kind (the kind's time(path), else /Header Time)
    field_specs(kind, obj)  the fields a run of this kind offers: the kind's registered fields and the variables of the file
Particle runs never reach this module's answers: with no kind registered every function says None / [] at once."""
import os
import re
from collections import OrderedDict

from . import plugins

_DIR = {}               # (folder, its mtime) -> kind name or None
_SNAPS = {}             # (kind, folder, its mtime) -> [paths]
_OPEN = OrderedDict()   # (path, mtime_ns, size) -> opened object
OPEN_KEEP = 8
_SEEN_VARS = set()      # variable names of opened mesh files (maps.required_gas_fields: not particle datasets)


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _mtime(folder):
    try:
        return os.stat(folder).st_mtime_ns
    except OSError:
        return None


def kind_of_dir(folder):
    if not plugins.KINDS:
        plugins.load_plugins()
        if not plugins.KINDS:
            return None
    folder = os.path.abspath(os.path.expanduser(str(folder)))
    if not os.path.isdir(folder):
        return None
    key = (folder, _mtime(folder))
    if key in _DIR:
        return _DIR[key]
    got = None
    for name, k in plugins.KINDS.items():
        try:
            if k["detect"](folder):
                got = name
                break
        except Exception:                      # a broken detect must not break the picker
            continue
    if len(_DIR) > 20000:
        _DIR.clear()
    _DIR[key] = got
    return got


def snapshots(kind, folder):
    folder = os.path.abspath(os.path.expanduser(str(folder)))
    key = (kind, folder, _mtime(folder))
    if key not in _SNAPS:
        try:
            got = [os.path.abspath(p) for p in plugins.KINDS[kind]["snapshots"](folder)]
        except Exception:
            got = []
        if len(_SNAPS) > 2000:
            _SNAPS.clear()
        _SNAPS[key] = sorted(dict.fromkeys(got), key=_natural)
    return list(_SNAPS[key])


def kind_of_file(path):
    if not plugins.KINDS:
        plugins.load_plugins()
        if not plugins.KINDS:
            return None
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        return None
    d = os.path.dirname(path)
    for folder in (d, os.path.dirname(d)):
        k = kind_of_dir(folder)
        if k is not None and path in snapshots(k, folder):
            return k
    return None


def open_mesh(path, kind=None):
    """the opened object of a kind's snapshot (kept in a small cache keyed by path, mtime and size)."""
    path = os.path.abspath(str(path))
    kind = kind or kind_of_file(path)
    if kind is None:
        raise ValueError(f"{path}: not a snapshot of a registered kind")
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    got = _OPEN.get(key)
    if got is None:
        got = plugins.KINDS[kind]["open"](path)
        _SEEN_VARS.update(str(v) for v in getattr(got, "variables", ()))
        _OPEN[key] = got
        while len(_OPEN) > OPEN_KEEP:
            _, old = _OPEN.popitem(last=False)
            close = getattr(old, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    else:
        _OPEN.move_to_end(key)
    return got


def snapshot_time(path, kind=None):
    kind = kind or kind_of_file(path)
    fn = plugins.KINDS[kind]["time"] if kind else None
    if fn is not None:
        return float(fn(path))
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        return float(np.ravel(f["Header"].attrs["Time"])[0])


def field_specs(kind, obj=None):
    """{name: {"needs", "fn", "label", "unit"}}: the kind's fields, then every variable of `obj` (an opened snapshot) that is not one of them."""
    out = {}
    for v in getattr(obj, "variables", ()) if obj is not None else ():
        out[str(v)] = {"needs": (str(v),), "fn": None, "label": str(v), "unit": None}
    for name, spec in plugins.KINDS[kind]["fields"].items():
        out[name] = {"needs": tuple(spec.get("needs") or (name,)), "fn": spec.get("fn"), "label": spec.get("label") or name, "unit": spec.get("unit")}
    return out


def is_mesh_field(name):
    """a field name of some registered kind (or a variable of an opened mesh file): not a particle dataset."""
    return name in _SEEN_VARS or any(name in k["fields"] for k in plugins.KINDS.values())


def label(kind):
    return plugins.KINDS[kind]["label"] if kind in plugins.KINDS else kind


def default_view(kind):
    return dict(plugins.KINDS[kind]["default_view"]) if kind in plugins.KINDS else {}


def units(kind):
    return dict(plugins.KINDS[kind]["units"]) if kind in plugins.KINDS else {}
