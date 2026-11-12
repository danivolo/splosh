"""`splosh analyze RUN`: the per-run analysis pipeline.

A TOML config declares products; every product is computed per snapshot into a tree, incrementally and cached:

    [[map]]        a View (any `View` field) -> OUT/<name>/<snap>.png (+ .json sidecar) and the float32 map in the mapstore (OUT/.maps);
                   `limits` = global | frame | fixed | viewport (see below)
    [[profile]]    radial profile (quantity, weighting, bins, geometry) -> OUT/<name>/<snap>.npz and .csv
    [[scalar]]     scalar diagnostics (built-ins or plugins.register_scalar) -> one table OUT/scalars.csv
    [[dashboard]]  a layout (layout.py: a declarative grid of map / profile / timeseries / text panels, by name, path or embedded; or the shorthand panels = [...], scalars = [...])
                   drawn per snapshot -> OUT/<name>/<snap>.png
    [[timeseries]] a layout with scope = run (time-series panels only) drawn once for the whole run -> OUT/figures/<name>.png
    [[movie]]      a map or dashboard product -> OUT/movies/<name>.mp4 (ffmpeg; without ffmpeg nothing is written and the movie is retried on the next run)

Also written: OUT/manifest.json (inputs hash, config hash, outputs, wall time, git commit per product and snapshot) and OUT/digest.txt (a short text summary).
Python API:  splosh.analyze.run(run_dir, config=None, out=None, jobs=1, only=None, every=None, force=False, dry_run=False) -> Result;  Pipeline(...) gives plan() / run().

Incremental rule: a (product, snapshot) is recomputed only if the snapshot signature (mapstore.snapshot_hash + header time), the product's config or the splosh version changed
(or an output file is missing). A map is two stages, `render` (the float32 map + colour statistics, keyed by the view) and `png` (keyed by the style and the resolved limits), so a
change of colour map redoes only PNGs. Colour limits:
    global    locked over all maps of the run (limits.Accumulator; `pct` = percentiles of all pixels together, else min / max); they move when the run grows, which redoes the PNGs
    frame     each map on its own (percentiles `pct`, default 1 / 99.9)
    fixed     `vmin` and `vmax`
    viewport  like frame, but the percentiles are taken only over the sub-window `viewport = [x0, y0, x1, y1]` (fractions of the map, default the central half): the colours follow the region of interest
Robustness: a snapshot that cannot be opened (still being written) is skipped and reported, and is picked up on a later run; outputs are never deleted.
Parallel: `jobs > 1` uses spawn worker processes (numba threads divided between them); the results are the same as serial."""
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from . import imaging, limits as limits_mod, plugins, provenance, transforms
from .maps import View, FieldContext, required_gas_fields, _field_values, field_names

MANIFEST_SCHEMA = "splosh-analysis-1"
KINDS = ("map", "profile", "scalar", "dashboard", "timeseries", "movie", "compare")
_VIEW_FIELDS = tuple(View.__dataclass_fields__)
_MAP_STYLE = {"cmap": "viridis", "scale": None, "limits": "global", "pct": None, "vmin": None, "vmax": None, "title": None, "scalebar": True, "colorbar": True, "viewport": None}
_SIGNED = ("vx", "vy", "vz", "vlos", "vR", "vphi", "jz")
_ALLOWED = {
    "map": {"name", *_VIEW_FIELDS, *_MAP_STYLE},
    "profile": {"name", "center", "align", "tilt", "quantities", "quantity", "weighting", "rmin", "rmax", "bins", "spacing", "geometry", "gamma", "xh", "mu", "bunit"},
    "scalar": {"name", "names", "label", "params"},
    "dashboard": {"name", "layout", "panels", "labels", "ncols", "title", "scalars", "series"},
    "timeseries": {"name", "layout", "columns", "xaxis", "xlog", "ylog", "title"},
    "compare": None,                                            # keys checked by compare.py
    "movie": {"name", "source", "fps"},
}
RESERVED = {"movies", "maps"}


# ----------------------------------------------------------------------------- config

def _strip_comment(line):
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).strip()


def _toml_value(text):
    text = text.strip()
    if text[:1] in "\"'" and text[-1:] == text[:1] and len(text) >= 2:
        return text[1:-1] if text[0] == "'" else json.loads(text)
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("["):
        inner = text[1:-1].strip()
        items, depth, cur, q = [], 0, "", None
        for ch in inner:
            if q:
                cur += ch
                q = None if ch == q else q
                continue
            if ch in "\"'":
                q = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == "," and depth == 0:
                items.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            items.append(cur)
        return [_toml_value(i) for i in items]
    try:
        return int(text.replace("_", ""))
    except ValueError:
        return float(text.replace("_", ""))


def parse_toml_subset(text):
    """A reader for the small TOML subset the config uses (used when neither tomllib nor tomli is available): `[table]`, `[[array of tables]]`,
    `key = value` with strings, integers, floats, booleans and (nested, possibly multi-line) arrays of those, `#` comments. No inline tables, dotted keys or dates."""
    root, cur = {}, None
    cur = root
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i])
        i += 1
        if not line:
            continue
        if line.startswith("[["):
            name = line.strip("[] ")
            cur = {}
            root.setdefault(name, []).append(cur)
        elif line.startswith("["):
            cur = root.setdefault(line.strip("[] "), {})
        else:
            key, sep, val = line.partition("=")
            if not sep:
                raise ValueError(f"config line {i}: expected key = value, got {line!r}")
            val = val.strip()
            while val.count("[") > val.count("]") and i < len(lines):
                val += " " + _strip_comment(lines[i])
                i += 1
            cur[key.strip().strip("\"'")] = _toml_value(val)
    return root


def read_toml(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        import tomllib as tl
    except ImportError:
        try:
            import tomli as tl
        except ImportError:
            tl = None
    if tl is not None:
        return tl.loads(raw.decode("utf-8"))
    return parse_toml_subset(raw.decode("utf-8"))


def read_config(path):
    """A config file as a dict: JSON for a .json file (what a GUI writes most easily), TOML otherwise."""
    if str(path).endswith(".json"):
        with open(path) as f:
            return json.load(f)
    return read_toml(path)


def default_config():
    """What runs when no config is given: one projection of the column density about the centre of mass (width = 'auto'), a spherical density profile and the built-in scalars."""
    return {"map": [{"name": "density", "center": "com", "width": "auto", "field": "rho", "mode": "proj", "npix": 512, "limits": "global", "pct": [1, 99.9]}],
            "profile": [{"name": "radial", "center": "com", "quantities": ["u"], "weighting": "mass", "rmin": "auto", "rmax": "auto", "bins": 40, "spacing": "log", "geometry": "spherical"}],
            "scalar": [{"name": "builtin"}]}


def normalize_config(cfg, base_dir=None):
    """Validate and complete a config dict: returns a list of products (dicts with 'kind' and 'name' and every option resolved to its default)."""
    unknown = set(cfg) - set(KINDS) - {"analysis"}
    if unknown:
        raise ValueError(f"config: unknown section(s) {sorted(unknown)}; sections: {list(KINDS)} and [analysis]")
    prods, names = [], set()
    for kind in KINDS:
        for k, raw in enumerate(cfg.get(kind, [])):
            bad = set(raw) - _ALLOWED[kind] if _ALLOWED[kind] is not None else set()
            if bad:
                raise ValueError(f"[[{kind}]] #{k + 1}: unknown key(s) {sorted(bad)}; allowed: {sorted(_ALLOWED[kind])}")
            p = {"kind": kind, **raw}
            if kind == "scalar":
                p.setdefault("names", [p.pop("name")] if "name" in p else ["builtin"])
                p.pop("name", None)
                if isinstance(p["names"], str):
                    p["names"] = [p["names"]]
                p.setdefault("params", {})
                p.setdefault("label", p["names"][0] if len(p["names"]) == 1 else "scalars")
                p["name"] = p["label"]
            if not p.get("name"):
                raise ValueError(f"[[{kind}]] #{k + 1}: needs a `name`")
            n = p["name"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) or n in RESERVED or n in names:
                raise ValueError(f"[[{kind}]] {n!r}: the name must be unique, file-system safe and not one of {sorted(RESERVED)}")
            names.add(n)
            if kind in ("dashboard", "timeseries"):
                done = _complete_layout_product(p, base_dir)
                for g in done.pop("generated"):
                    if g["name"] in names or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", g["name"]):
                        raise ValueError(f"{kind} {n!r}: the inline map panel needs a unique, file-system safe id (would be product {g['name']!r})")
                    names.add(g["name"])
                    prods.append(g)
                prods.append(done)
            elif kind == "compare":
                from .compare import complete_compare
                prods.append(complete_compare(p))
            else:
                prods.append(_complete(p))
    byname = {p["name"]: p for p in prods}
    for p in prods:
        if p["kind"] in ("dashboard", "timeseries"):
            for pan in p["layout"]["panel"]:
                want = {"map": "map", "profile": "profile"}.get(pan["type"])
                if want and byname.get(pan["product"], {}).get("kind") != want:
                    raise ValueError(f"{p['kind']} {p['name']!r}: panel {pan['id']!r}: {pan['product']!r} is not a [[{want}]] product")
        if p["kind"] == "compare":
            from .compare import check_compare
            check_compare(p, byname)
        if p["kind"] == "movie" and byname.get(p["source"], {}).get("kind") not in ("map", "dashboard"):
            raise ValueError(f"movie {p['name']!r}: source {p['source']!r} is not a map or dashboard product")
    return prods


def _check_scale(scale, where):
    sc = imaging.parse_scale(scale)
    if sc == "histeq":
        raise ValueError(f"{where}: histeq ranks the pixels of one picture; use it with `splosh shot`, not in a product or layout")
    return sc


def _symmetric(p):
    """True for a map product whose limits are (-v, v): the transforms that give a contrast."""
    return bool((transforms.style(p.get("post", "none")) or {}).get("symmetric"))


def _complete(p):
    kind = p["kind"]
    if kind == "map":
        view = View(**{k: p[k] for k in _VIEW_FIELDS if k in p and p[k] != "auto"})
        view.validate()
        out = {"kind": kind, "name": p["name"], **{k: p.get(k, d) for k, d in _MAP_STYLE.items()}}
        out.update({k: p[k] if k in p else getattr(view, k) for k in _VIEW_FIELDS})
        st = transforms.style(out["post"])
        if out["scale"] is None:
            out["scale"] = st["scale"] if st else ("asinh" if out["field"] in _SIGNED else "log")
        out["scale"] = str(_check_scale(out["scale"], f"map {p['name']!r}"))
        if out["cmap"] == "viridis" and "cmap" not in p:
            if st:
                out["cmap"] = st["cmap"]
            elif out["field"] in _SIGNED:
                out["cmap"] = "RdBu_r"
        if out["limits"] not in ("global", "frame", "fixed", "viewport"):
            raise ValueError(f"map {p['name']!r}: limits must be global, frame, fixed or viewport")
        if out["limits"] == "fixed" and (out["vmin"] is None or out["vmax"] is None):
            raise ValueError(f"map {p['name']!r}: limits = fixed needs vmin and vmax")
        if out["pct"] is not None:
            out["pct"] = [float(x) for x in out["pct"]]
        if out["viewport"] is not None:
            out["viewport"] = [float(x) for x in out["viewport"]]
            if len(out["viewport"]) != 4:
                raise ValueError("viewport must be [x0, y0, x1, y1]")
        return out
    if kind == "profile":
        q = p.get("quantities", [p["quantity"]] if "quantity" in p else ["u"])
        out = {"kind": kind, "name": p["name"], "center": p.get("center", "com"), "align": p.get("align", "none"), "tilt": p.get("tilt", "none"), "quantities": [q] if isinstance(q, str) else list(q),
               "weighting": p.get("weighting", "mass"), "rmin": p.get("rmin", "auto"), "rmax": p.get("rmax", "auto"), "bins": int(p.get("bins", 40)),
               "spacing": p.get("spacing", "log"), "geometry": p.get("geometry", "spherical"),
               **{k: p[k] for k in ("gamma", "xh", "mu", "bunit") if k in p}}
        if out["weighting"] not in ("mass", "volume", "number") or out["spacing"] not in ("log", "lin") or out["geometry"] not in ("spherical", "cylindrical", "pdf"):
            raise ValueError(f"profile {p['name']!r}: weighting mass|volume|number, spacing log|lin, geometry spherical|cylindrical|pdf")
        if out["geometry"] == "pdf":
            if "auto" in (out["rmin"], out["rmax"]) or not float(out["rmax"]) > float(out["rmin"]) or (out["spacing"] == "log" and not float(out["rmin"]) > 0):
                raise ValueError(f"profile {p['name']!r}: geometry = pdf bins the first quantity ({out['quantities'][0]}) between rmin and rmax: give both (rmin > 0 for log spacing); no `auto`")
            if re.fullmatch(r"(?:A|phi)\d+", out["quantities"][0]):
                raise ValueError(f"profile {p['name']!r}: the first quantity of a pdf is the binned variable, not a Fourier mode")
        return out
    if kind == "scalar":
        return {"kind": kind, "name": p["name"], "label": p["label"], "names": list(p["names"]), "params": dict(p["params"])}
    return {"kind": kind, "name": p["name"], "source": p["source"], "fps": float(p.get("fps", 12))}


def _complete_layout_product(p, base_dir):
    """A [[dashboard]] / [[timeseries]]: the layout (by name or path, embedded, or built from the shorthand keys), validated; inline map panels become generated [[map]] products."""
    from . import layout as lay
    kind, name = p["kind"], p["name"]
    dirs = [os.path.join(base_dir, "layouts"), base_dir] if base_dir else []
    if p.get("layout") is not None:
        L = lay.find_layout(p["layout"], dirs)
    elif kind == "dashboard":
        if not p.get("panels"):
            raise ValueError(f"dashboard {name!r}: needs `layout = name | path` or panels = [map product names]")
        panels = list(p["panels"])
        labels = list(p.get("labels", panels))
        pl = [{"id": m, "type": "map", "product": m, "label": lb} for m, lb in zip(panels, labels)]
        pl += [{"id": c, "type": "timeseries", "columns": [c], "series": p.get("series", "all")} for c in p.get("scalars", [])]
        L = lay.validate_layout({"name": name, "title": p.get("title"), "ncols": int(p.get("ncols", 0)) or min(len(panels), 3 if len(panels) > 4 else 2 if len(panels) > 1 else 1), "panel": pl})
    else:
        cols = p.get("columns")
        if not cols:
            raise ValueError(f"timeseries {name!r}: needs `layout = name | path` or columns = [...]")
        L = lay.validate_layout({"name": name, "title": p.get("title"), "scope": "run", "ncols": 1,
                                 "panel": [{"id": "ts", "type": "timeseries", "columns": cols, "xaxis": p.get("xaxis", "time"), "xlog": bool(p.get("xlog", False)), "ylog": bool(p.get("ylog", False)), "marker": False}]})
    if kind == "dashboard" and L["scope"] != "snapshot":
        raise ValueError(f"dashboard {name!r}: layout {L['name']!r} has scope {L['scope']!r}; a dashboard needs scope = snapshot (use [[timeseries]] for a run-wide figure)")
    if kind == "timeseries":
        if L["scope"] != "run" or any(q["type"] not in ("timeseries", "text") for q in L["panel"]):
            raise ValueError(f"timeseries {name!r}: layout {L['name']!r} must have scope = run and only timeseries / text panels")
        for q in L["panel"]:
            q["marker"] = False
    generated, eff = [], []
    for q in L["panel"]:
        if q["type"] == "map" and not q["product"]:
            raw = {k: q[k] for k in list(lay._view_keys()) + list(lay._MAP_STYLE)}
            raw.update(name=f"{name}.{q['id']}", kind="map")
            generated.append(_complete(raw))
            q = {k: q[k] for k in ("id", "type", "row", "col", "rowspan", "colspan", "label")} | {"product": raw["name"]}
        eff.append(q)
    return {"kind": kind, "name": name, "layout": dict(L, panel=eff), "generated": generated}


def load_config(config=None):
    """config: None (default), a path to a TOML (or .json) file or a dict (the parsed file) -> (normalised products, settings dict from [analysis])."""
    if isinstance(config, list):                                  # already normalised (compare passes the products on)
        return config, {}
    if config is None:
        cfg = default_config()
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = read_config(config)
    base = None if isinstance(config, dict) or config is None else os.path.dirname(os.path.abspath(config))
    return normalize_config(cfg, base), dict(cfg.get("analysis", {}))


# ----------------------------------------------------------------------------- scalars

def _gas_basics(snap):
    m = np.asarray(snap["Masses"], dtype=np.float64)
    return m, np.asarray(snap["Coordinates"], dtype=np.float64)


def _sc_counts(snap, **_):
    return {f"n_{k}": float(v) for k, v in snap["counts"].items()}


def _sc_gas_mass(snap, **_):
    from .units import Units
    m = float(np.asarray(snap["Masses"], dtype=np.float64).sum())
    u = Units.from_header(snap.get("header", {}))
    return {"gas_mass": m, "gas_mass_msun": m * u.mass_g / 1.989e33}


def _sc_com(snap, **_):
    m, x = _gas_basics(snap)
    c = (x * m[:, None]).sum(0) / m.sum()
    return {"com_x": float(c[0]), "com_y": float(c[1]), "com_z": float(c[2])}


def _sc_rho_max(snap, **_):
    return {"rho_max": float(np.max(snap["Density"]))}


def _sc_mean_u(snap, **_):
    m, _x = _gas_basics(snap)
    return {"u_mw": float((m * np.asarray(snap["InternalEnergy"], dtype=np.float64)).sum() / m.sum())}


def _sc_mean_T(snap, **_):
    if "ElectronAbundance" not in snap:
        return {"T_mw": float("nan")}
    from .fields import f_temperature
    T = f_temperature(FieldContext(snap, None, View()))
    m, _x = _gas_basics(snap)
    return {"T_mw": float((m * T).sum() / m.sum())}


def _sc_modes(snap, center="com", rmin=0.0, rmax=0.0, orders=(1, 2, 3), **_):
    """Azimuthal Fourier amplitudes A<m> (and phases phi<m>) of the gas in the annulus rmin..rmax about `center` (a frame centre rule), mass weighted, angle about the frame's z axis: A_m = |sum m_j exp(i m phi_j)| / sum m_j.
    Opt-in (not part of `builtin`): [[scalar]] names = ["modes"], params = {center = "bh", rmin = 0.5, rmax = 5, orders = [1, 2]}. rmin = rmax = 0 (default): the radii enclosing 10% and 90% of the gas mass."""
    from .frames import make_frame
    m, x = _gas_basics(snap)
    pos = make_frame(snap, str(center), "none").pos(x)
    r = np.hypot(pos[:, 0], pos[:, 1])
    if not float(rmax) > float(rmin):
        o = np.argsort(r)
        cum = np.cumsum(m[o])
        rmin, rmax = float(r[o][min(np.searchsorted(cum, 0.1 * cum[-1]), len(r) - 1)]), float(r[o][min(np.searchsorted(cum, 0.9 * cum[-1]), len(r) - 1)])
    sel = (r >= float(rmin)) & (r <= float(rmax))
    out = {"modes_rmin": float(rmin), "modes_rmax": float(rmax)}
    w, phi = m[sel], np.arctan2(pos[sel, 1], pos[sel, 0])
    for k in (orders if isinstance(orders, (list, tuple)) else [orders]):
        k = int(k)
        if w.sum() > 0:
            c, s_ = (w * np.cos(k * phi)).sum(), (w * np.sin(k * phi)).sum()
            out[f"A{k}"], out[f"phi{k}"] = float(np.hypot(c, s_) / w.sum()), float(np.arctan2(s_, c) / k)
        else:
            out[f"A{k}"] = out[f"phi{k}"] = float("nan")
    return out


def _sc_black_holes(snap, **_):
    nan = float("nan")
    out = {"bh_mtot": nan, "bh_m1": nan, "bh_m2": nan, "bh_sep": nan}
    if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
        return out
    order = np.argsort(snap["bh_id"], kind="stable")
    mass, pos = snap["bh_mass"][order], snap["bh_pos"][order]
    out["bh_mtot"] = float(mass.sum())
    out["bh_m1"] = float(mass[0])
    if len(order) >= 2:
        out["bh_m2"] = float(mass[1])
        out["bh_sep"] = float(np.linalg.norm(pos[0] - pos[1]))
    return out


BUILTIN_SCALARS = {"counts": _sc_counts, "gas_mass": _sc_gas_mass, "com": _sc_com, "rho_max": _sc_rho_max, "mean_u": _sc_mean_u, "mean_T": _sc_mean_T, "black_holes": _sc_black_holes, "modes": _sc_modes}
OPT_IN_SCALARS = frozenset({"modes"})          # computed only when named; `builtin` does not include them
BUILTIN_SCALAR_COLUMNS = {"counts": ("n_gas", "n_bh", "(n_<type> for other particle types)"), "gas_mass": ("gas_mass", "gas_mass_msun"), "com": ("com_x", "com_y", "com_z"), "rho_max": ("rho_max",),
                          "mean_u": ("u_mw",), "mean_T": ("T_mw",), "black_holes": ("bh_mtot", "bh_m1", "bh_m2", "bh_sep"), "modes": ("A1", "A2", "A3", "phi1", "phi2", "phi3", "modes_rmin", "modes_rmax")}
_SCALAR_NEEDS = ("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "?ElectronAbundance", "?Metallicity")


def scalar_functions(names):
    """[(name, fn, extra datasets)] for scalar names; 'builtin' expands to every built-in; plugin scalars (plugins.register_scalar) are looked up after load_plugins()."""
    plugins.load_plugins()
    out = []
    for n in names:
        if n == "builtin":
            out += [(b, f, ()) for b, f in BUILTIN_SCALARS.items() if b not in OPT_IN_SCALARS]
        elif n in BUILTIN_SCALARS:
            out.append((n, BUILTIN_SCALARS[n], ()))
        elif n in plugins.SCALARS:
            fn, needs = plugins.SCALARS[n]
            out.append((n, fn, tuple(needs)))
        else:
            raise KeyError(f"unknown scalar {n!r}; built-in: {sorted(BUILTIN_SCALARS)}, plugins: {sorted(plugins.SCALARS)} (plugin errors: {plugins.errors()})")
    return out


def _particle_counts(path):
    import h5py
    with h5py.File(path, "r") as f:
        names = {"PartType0": "gas", "PartType5": "bh"}
        return {names.get(k, k.replace("PartType", "type")): int(f[k]["Coordinates"].shape[0]) for k in sorted(f) if k.startswith("PartType") and "Coordinates" in f[k]}


def compute_scalars(path, snap, names, params):
    row = {}
    snap = dict(snap, counts=_particle_counts(path))
    for n, fn, _needs in scalar_functions(names):
        res = fn(snap, **params) if (n not in BUILTIN_SCALARS or n in OPT_IN_SCALARS) else fn(snap)
        if isinstance(res, dict):
            for k, v in res.items():
                row[k] = float(v)
        else:
            row[n] = float(res)
    return row


# ----------------------------------------------------------------------------- small helpers

def _hash(*objs):
    return hashlib.sha1(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _norm(v):
    return float(f"{v:.6g}") if isinstance(v, float) else v


def _view_of(p):
    return View(**{k: p[k] for k in _VIEW_FIELDS})


def _view_dict(p):
    return {k: _norm(v) for k, v in _view_of(p).as_dict().items()}


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_text(path, text):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _fmt(v):
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else repr(float(v))


def _init_worker(threads):
    try:
        import numba
        numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
    except Exception:                                           # pragma: no cover
        pass


def _stats_path(out, product, stem):
    return os.path.join(out, ".state", product, stem + ".stats.npz")


def _gas_union(views, extra=()):
    """Datasets to load once for a snapshot: everything the views need plus `extra` (a leading '?' = optional)."""
    need = set()
    for v in views:
        need |= required_gas_fields(v)
    for e in extra:
        if e.lstrip("?") not in need and not (e.startswith("?") and e[1:] in need):
            need.add(e)
    return tuple(sorted(need))


# ----------------------------------------------------------------------------- workers (module level: picklable, spawn-safe)

def _auto_radius(snap):
    m, x = _gas_basics(snap)
    c = (x * m[:, None]).sum(0) / m.sum()
    r = np.sqrt(((x - c) ** 2).sum(1))
    o = np.argsort(r)
    cum = np.cumsum(m[o])
    return float(r[o][min(np.searchsorted(cum, 0.95 * cum[-1]), len(r) - 1)])


def _profile(snap, p):
    from .frames import make_frame
    from .profiles import radial_profile
    view = View(center=p["center"], align=p["align"], tilt=p.get("tilt", "none"), **{k: p[k] for k in ("gamma", "xh", "mu", "bunit") if k in p})
    frame = make_frame(snap, view.center, view.align, view.tilt)
    pos = frame.pos(snap["Coordinates"])
    m = np.asarray(snap["Masses"], dtype=np.float64)
    w = {"mass": m, "number": np.ones_like(m), "volume": m / np.asarray(snap["Density"], dtype=np.float64)}[p["weighting"]]
    values, modes = {}, []
    for q in p["quantities"]:
        mm = re.fullmatch(r"(?:A|phi)(\d+)", q)
        if mm:
            modes.append(int(mm.group(1)))              # A<m>, phi<m>: azimuthal Fourier mode m (profiles.radial_profile)
            continue
        v = view.with_(field=q)
        values[q] = np.asarray(snap["Density"] if q == "density" else _field_values(q, FieldContext(snap, frame, v)), dtype=np.float64)
    if p["geometry"] == "pdf":
        from .profiles import value_histogram
        x = values[p["quantities"][0]]
        return value_histogram(x, w, values, lo=p["rmin"], hi=p["rmax"], nbins=p["bins"], spacing=p["spacing"])
    return radial_profile(pos, w, values, rmin=p["rmin"], rmax=p["rmax"], nbins=p["bins"], spacing=p["spacing"], ndim=2 if p["geometry"] == "cylindrical" else 3,
                          modes=sorted(set(modes)))


def _write_profile(prof, base, meta):
    cols = ["rbins", "rgeo", "n", "mass", "sigma"] + [k for k in prof if k not in ("edges", "rbins", "rgeo", "n", "mass", "sigma", "density")]
    np.savez(base + ".npz", meta=json.dumps(meta), **prof)
    lines = [",".join(cols)] + [",".join(_fmt(prof[c][i]) for c in cols) for i in range(len(prof["rbins"]))]
    _atomic_text(base + ".csv", "\n".join(lines) + "\n")
    return [base + ".npz", base + ".csv"]


def _png_limits(p, img, lim):
    """(vmin, vmax) for one map under its product's policy (`lim` = the locked pair for global)."""
    if p["limits"] in ("global", "fixed"):
        return lim
    sym = _symmetric(p)
    pct = tuple(p["pct"]) if p["pct"] else ((1.0, limits_mod.SYM_PCT) if sym else (1.0, 99.9))
    if p["limits"] == "viewport":
        x0, y0, x1, y1 = p["viewport"] or (0.25, 0.25, 0.75, 0.75)
        ny, nx = img.shape
        sub = img[int(y0 * ny):max(int(y1 * ny), int(y0 * ny) + 1), int(x0 * nx):max(int(x1 * nx), int(x0 * nx) + 1)]
        lo, hi = imaging.limits(sub, pclip=pct, scale=p["scale"], symmetric=sym)
        return (lo if p["vmin"] is None else float(p["vmin"])), (hi if p["vmax"] is None else float(p["vmax"]))
    return limits_mod.resolve("frame", p["scale"], pct, p["vmin"], p["vmax"], img=img, symmetric=sym)


def _worker_snapshot(task):
    """Stage A for one snapshot: load once, then render maps (mapstore + colour statistics), profiles and scalar rows as asked. Returns a list of per-item results."""
    from .io import load_snapshot
    from .mapstore import MapStore
    plugins.load_plugins()
    out, path, stem = task["out"], task["path"], task["stem"]
    res = []
    try:
        views = [_view_of(it["p"]) for it in task["items"] if it["stage"] == "render"]
        extra = list(_SCALAR_NEEDS if any(it["stage"] == "scalar" for it in task["items"]) else ())
        for it in task["items"]:
            if it["stage"] == "profile":
                extra += ["Coordinates", "Masses", "Density"] + sorted(set().union(*[required_gas_fields(View(center=it["p"]["center"], align=it["p"]["align"], tilt=it["p"].get("tilt", "none"), field=q)) for q in it["p"]["quantities"] if q != "density"] or [set()]))
            if it["stage"] == "scalar":
                for _n, _f, needs in scalar_functions(it["p"]["names"]):
                    extra += list(needs)
        snap = load_snapshot(path, gas_fields=_gas_union(views, extra))
    except Exception as e:
        if len(task["items"]) > 1:                              # one item's dataset is missing: give every item its own chance
            return [r for it in task["items"] for r in _worker_snapshot(dict(task, items=[it]))]
        return [{"product": it["p"]["name"], "stage": it["stage"], "ok": False, "error": f"{type(e).__name__}: {e}"} for it in task["items"]]
    store = MapStore(task["store"])
    for it in task["items"]:
        p, t0 = it["p"], time.perf_counter()
        r = {"product": p["name"], "stage": it["stage"], "ok": True}
        try:
            if it["stage"] == "render":
                if task.get("force"):
                    from .maps import render_map
                    from .mapstore import snapshot_hash, view_key
                    img, meta = render_map(snap, _view_of(p))
                    meta["key"] = view_key(snapshot_hash(path), _view_of(p))
                    meta["snapshot"] = os.path.basename(path)
                    store.put(meta["key"], img, meta)
                else:
                    img, meta, _hit = store.get_or_render(path, _view_of(p), snap=snap)
                v = limits_mod._usable(img, p["scale"])
                step = -(-v.size // 4000) if v.size else 1
                sp = _stats_path(out, p["name"], stem)
                os.makedirs(os.path.dirname(sp), exist_ok=True)
                tmp = sp + f".tmp{os.getpid()}.npz"
                np.savez(tmp, lo=v.min() if v.size else np.nan, hi=v.max() if v.size else np.nan, n=v.size, sample=v[::step].astype(np.float32))
                os.replace(tmp, sp)
                r["store_key"] = meta.get("key")
                r["time"] = meta["time"]
            elif it["stage"] == "profile":
                d = os.path.join(out, p["name"])
                os.makedirs(d, exist_ok=True)
                prof = _profile(snap, p)
                meta = {"snapshot": stem, "time": snap["time"], "center": p["center"], "weighting": p["weighting"], "geometry": p["geometry"], "quantities": p["quantities"],
                        "note": "code units; rbins = arithmetic bin centres"}
                r["outputs"] = [os.path.relpath(f, out) for f in _write_profile(prof, os.path.join(d, stem), meta)]
            else:
                r["values"] = compute_scalars(path, snap, p["names"], p["params"])
        except Exception as e:
            r.update(ok=False, error=f"{type(e).__name__}: {e}")
        r["wall"] = round(time.perf_counter() - t0, 4)
        res.append(r)
    return res


def _worker_png(task):
    """Stage B for one map of one snapshot: the PNG (+ sidecar) with the resolved limits."""
    from . import figures
    from .mapstore import MapStore
    p, out, stem, path = task["p"], task["out"], task["stem"], task["path"]
    t0 = time.perf_counter()
    r = {"product": p["name"], "stage": "png", "ok": True}
    try:
        store = MapStore(task["store"])
        view = _view_of(p)
        img, _meta, _hit = store.get_or_render(path, view)
        lo, hi = _png_limits(p, img, task["lim"])
        d = os.path.join(out, p["name"])
        os.makedirs(d, exist_ok=True)
        pic = figures.shot(path, view, cmap=p["cmap"], scale=p["scale"], vmin=lo, vmax=hi, title=p["title"] or p["name"], scalebar=p["scalebar"], colorbar=p["colorbar"], store=store)
        files = pic.save(os.path.join(d, stem + ".png"))
        r["outputs"] = [os.path.relpath(f, out) for f in files]
        r["lim"] = [float(lo), float(hi)]
    except Exception as e:
        r.update(ok=False, error=f"{type(e).__name__}: {e}")
    r["wall"] = round(time.perf_counter() - t0, 4)
    return r


def _worker_dashboard(task):
    """One dashboard picture of one snapshot from its layout: maps from the mapstore, profiles from their files, time series from the parent's data."""
    from . import figures, layout as lay
    from .mapstore import MapStore
    p, out, stem, path = task["p"], task["out"], task["stem"], task["path"]
    t0 = time.perf_counter()
    r = {"product": p["name"], "stage": "dashboard", "ok": True}
    try:
        store = MapStore(task["store"])
        tiles, meta0 = {}, None
        for pn in task["maps"]:
            img, meta, _ = store.get_or_render(path, _view_of(pn["p"]))
            meta0 = meta0 or meta
            tiles[pn["id"]] = {"img": img, "meta": meta, "cmap": pn["p"]["cmap"], "scale": pn["p"]["scale"], "lim": pn["lim"], "scalebar": pn["p"]["scalebar"]}
        for pn in task["profiles"]:
            with np.load(os.path.join(out, pn["npz"])) as d:
                qs = pn["quantities"] or [k for k in d.files if k not in ("meta", "edges", "rbins", "rgeo", "n", "mass", "sigma", "density")]
                tiles[pn["id"]] = {"curves": [(q, d["rbins"], d[q]) for q in qs], "xlog": pn["xlog"], "ylog": pn["ylog"]}
        tiles.update(task["series"])
        n = figures._snap_number(path)
        tstr = figures.time_text(meta0["time"], figures.unit_names(meta0.get("units"))) if meta0 else f"t = {task['time']:.4g}"
        header = (f"snap {n}   " if n is not None else "") + tstr
        for pn in task["layout"]["panel"]:
            if pn["type"] == "text":
                tiles[pn["id"]] = {"text": pn["text"].replace("{time}", f"{task['time']:.4g}").replace("{snapshot}", stem)}
        rgb = lay.render_layout(task["layout"], tiles, header)
        d = os.path.join(out, p["name"])
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, stem + ".png")
        tmp = f + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            fh.write(imaging.png_bytes(rgb))
        os.replace(tmp, f)
        side = provenance.write_sidecar(f, {"command": "analyze-dashboard", "layout": task["layout"]["name"], "panels": [{"product": q["p"]["name"], "view": _view_dict(q["p"]), "limits": q["lim"]} for q in task["maps"]],
                                            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])]})
        r["outputs"] = [os.path.relpath(f, out), os.path.relpath(side, out)]
    except Exception as e:
        r.update(ok=False, error=f"{type(e).__name__}: {e}")
    r["wall"] = round(time.perf_counter() - t0, 4)
    return r


# ----------------------------------------------------------------------------- the pipeline

class Result:
    """Outcome of Pipeline.run / plan: `counts` (work units done per stage), `work` [(product, snapshot, stage)], `failures` [(snapshot, product, stage, error)], `skipped` {snapshot: error},
    `plan` (what a dry run would do), `out`, `digest` (text)."""
    def __init__(self, out):
        self.out, self.counts, self.work, self.failures, self.skipped, self.plan, self.digest = out, {}, [], [], {}, None, ""

    def count(self, product, stem, stage):
        self.work.append((product, stem, stage))
        self.counts[stage] = self.counts.get(stage, 0) + 1

    @property
    def total(self):
        return len(self.work)


def run_root(path):
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isdir(path) and os.path.basename(path) == "output":
        return os.path.dirname(path)
    return path if os.path.isdir(path) else os.path.dirname(path)


class Pipeline:
    def __init__(self, run, config=None, out=None, jobs=1, only=None, every=None, force=False, dry_run=False, store=None, log=None, auto_radius=None):
        self.run_path = os.path.expanduser(str(run))
        prods, self.settings = load_config(config)
        prods = copy.deepcopy(prods)                                              # 'auto' values are resolved in place
        self.products = [p for p in prods if p["kind"] != "compare"]            # comparison products belong to `splosh compare`
        self.compare_products = [p for p in prods if p["kind"] == "compare"]
        self.out = os.path.abspath(os.path.expanduser(str(out))) if out else os.path.join(run_root(self.run_path), "analysis")
        self.jobs, self.every, self.force, self.dry = max(1, int(jobs or 1)), every, bool(force), bool(dry_run)
        self.store_dir = os.path.abspath(store or self.settings.get("store") or os.path.join(self.out, ".maps"))
        self.log = log or (lambda m: None)
        self.version = provenance.version()
        self.only = self._select(only)
        self._auto_override = auto_radius
        self._pool = None

    # ---- selection
    def _select(self, only):
        if not only:
            return {p["name"] for p in self.products}
        want = [o for o in ([only] if isinstance(only, str) else only) if o]
        by = {p["name"]: p for p in self.products}
        chosen = set()
        for w in want:
            hit = [p["name"] for p in self.products if p["name"] == w or p["kind"] == w]
            if not hit:
                raise ValueError(f"--only: no product or kind {w!r}; products: {sorted(by)}")
            chosen.update(hit)

        def deps(n):
            p = by[n]
            d = [pan["product"] for pan in p["layout"]["panel"] if pan.get("product")] if "layout" in p else []
            if "layout" in p and any(pan["type"] == "timeseries" for pan in p["layout"]["panel"]):
                d += [q["name"] for q in self.products if q["kind"] == "scalar"]
            if p["kind"] == "movie":
                d += [p["source"]] + deps(p["source"])
            return d
        stack = list(chosen)
        while stack:
            for d in deps(stack.pop()):
                if d not in chosen:
                    chosen.add(d)
                    stack.append(d)
        return chosen

    # ---- manifest
    def _load_manifest(self):
        try:
            with open(os.path.join(self.out, "manifest.json")) as f:
                m = json.load(f)
            if m.get("schema") == MANIFEST_SCHEMA:
                return m
        except (OSError, ValueError):
            pass
        return {"schema": MANIFEST_SCHEMA, "products": {}, "snapshots": {}}

    def _save_manifest(self):
        m = self.manifest
        m.update(splosh_version=self.version, git_commit=provenance.git_commit(), run=os.path.abspath(self.run_path), updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 failures=[list(f) for f in self.result.failures], skipped=self.result.skipped)
        m["config_hash"] = _hash(self.products)
        _atomic_json(os.path.join(self.out, "manifest.json"), m)

    # ---- discovery
    def _discover(self):
        from . import figures
        from .mapstore import snapshot_hash
        import h5py
        from .units import Units
        paths = figures.select_snapshots(self.run_path, every=self.every)
        snaps = []
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                sig = snapshot_hash(p)
                with h5py.File(p, "r") as f:
                    t = float(f["Header"].attrs["Time"])
                    tmyr = t * Units.from_header({k: np.asarray(v).tolist() for k, v in f["Header"].attrs.items()}).time_s / 3.156e13
                    if "PartType0" not in f:
                        raise KeyError("no PartType0 group")
            except Exception as e:
                self.result.skipped[stem] = f"{type(e).__name__}: {e}"
                continue
            snaps.append({"stem": stem, "path": p, "sig": sig, "time": t, "time_myr": tmyr})
        return snaps

    def _resolve_auto(self, snaps):
        """width / rmin / rmax = 'auto' from the first openable snapshot (the 95% mass radius of the gas about its centre of mass), written into the product config."""
        need = [p for p in self.products if (p["kind"] == "map" and p["width"] == "auto") or (p["kind"] == "profile" and "auto" in (p["rmin"], p["rmax"]))]
        if not need or not snaps:
            return
        known = self._auto_override or self.manifest.get("auto_radius")
        if known is None:
            from .io import load_snapshot
            known = _auto_radius(load_snapshot(snaps[0]["path"], gas_fields=("Coordinates", "Masses")))
        self.manifest["auto_radius"] = known
        width = float(f"{2.2 * known:.2g}")
        for p in need:
            if p["kind"] == "map" and p["width"] == "auto":
                p["width"] = width
            if p["kind"] == "profile":
                if p["rmax"] == "auto":
                    p["rmax"] = width / 2
                if p["rmin"] == "auto":
                    p["rmin"] = float(p["rmax"]) * 1e-3

    # ---- planning helpers
    def _unit(self, product, stem):
        return self.manifest["products"].get(product, {}).get("units", {}).get(stem)

    def _set_unit(self, product, kind, stem, **kw):
        prod = self.manifest["products"].setdefault(product, {"kind": kind, "units": {}})
        prod["kind"] = kind
        prod["config_hash"] = _hash(self._cfg(product))
        prod["units"].setdefault(stem, {}).update(kw)

    def _cfg(self, name):
        return next(p for p in self.products if p["name"] == name)

    def _have(self, u, key_field="key"):
        return u is not None and all(os.path.exists(os.path.join(self.out, o)) for o in u.get("outputs", []))

    def _map_keys(self, p, snap):
        return _hash("render", snap["sig"], _view_dict(p), p["scale"], self.version)

    def _png_key(self, p, rk, lim):
        return _hash("png", rk, {k: p[k] for k in _MAP_STYLE}, lim, self.version)

    def _executor(self):
        if self.jobs <= 1:
            return None
        if self._pool is None:
            import numba
            threads = max(1, numba.config.NUMBA_NUM_THREADS // self.jobs)
            self._pool = ProcessPoolExecutor(max_workers=self.jobs, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker, initargs=(threads,))
        return self._pool

    def _map(self, fn, tasks):
        ex = self._executor()
        return [fn(t) for t in tasks] if ex is None else list(ex.map(fn, tasks))

    # ---- the run
    def plan(self):
        self.dry = True
        return self.run()

    def run(self):
        self.result = res = Result(self.out)
        t_start = time.perf_counter()
        if self.dry:
            self.manifest = self._load_manifest()
        else:
            os.makedirs(self.out, exist_ok=True)
            self.manifest = self._load_manifest()
        snaps = self._discover()
        self.snaps = snaps
        if not snaps:
            res.digest = f"no readable snapshots in {self.run_path} ({len(res.skipped)} skipped)"
            if not self.dry:
                self._write_digest(t_start)
            return res
        self._resolve_auto(snaps)
        self.manifest["snapshots"].update({s["stem"]: {"sig": s["sig"], "time": s["time"], "time_myr": s["time_myr"], "path": s["path"]} for s in snaps})
        prods = [p for p in self.products if p["name"] in self.only]
        try:
            self._stage_a(prods, snaps)
            if not self.dry:
                self._stage_b(prods, snaps)
                self._write_scalars_table(snaps)
                self._stage_dashboards(prods, snaps)
                self._stage_timeseries(prods, snaps)
                self._stage_movies(prods, snaps)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
            if not self.dry:
                self._write_scalars_table(snaps)
                self._save_manifest()
                self._write_digest(t_start)
        if self.dry:
            res.plan = self._plan_text(prods, snaps)
        return res

    # stage A: per snapshot, everything that needs the snapshot data
    def _stage_a(self, prods, snaps):
        tasks, planned = [], {}
        for s in snaps:
            items = []
            for p in prods:
                u = self._unit(p["name"], s["stem"])
                if p["kind"] == "map":
                    rk = self._map_keys(p, s)
                    if self.force or u is None or u.get("render_key") != rk or not os.path.exists(_stats_path(self.out, p["name"], s["stem"])):
                        items.append({"stage": "render", "p": p, "key": rk})
                elif p["kind"] == "profile":
                    k = _hash("profile", s["sig"], p, self.version)
                    if self.force or u is None or u.get("key") != k or not self._have(u):
                        items.append({"stage": "profile", "p": p, "key": k})
                elif p["kind"] == "scalar":
                    k = _hash("scalar", s["sig"], p, self.version)
                    if self.force or u is None or u.get("key") != k:
                        items.append({"stage": "scalar", "p": p, "key": k})
            if items:
                planned[s["stem"]] = items
                tasks.append({"out": self.out, "store": self.store_dir, "path": s["path"], "stem": s["stem"], "items": items, "force": self.force})
        self._pending_a = planned
        if self.dry:
            return
        if tasks:
            self.log(f"stage A: {len(tasks)} snapshot(s)")
        for k, (task, results) in enumerate(zip(tasks, self._map(_worker_snapshot, tasks)), 1):
            stem = task["stem"]
            self.log(f"stage A {k}/{len(tasks)}  {stem}")
            for it, r in zip(task["items"], results):
                p = it["p"]
                if not r["ok"]:
                    self.result.failures.append((stem, p["name"], r["stage"], r["error"]))
                    continue
                self.result.count(p["name"], stem, r["stage"])
                if r["stage"] == "render":
                    self._set_unit(p["name"], "map", stem, render_key=it["key"], store_key=r.get("store_key"), key=None, wall_render=r["wall"], time=r["time"])
                elif r["stage"] == "profile":
                    self._set_unit(p["name"], "profile", stem, key=it["key"], outputs=r["outputs"], wall=r["wall"])
                else:
                    self._set_unit(p["name"], "scalar", stem, key=it["key"], values=r["values"], wall=r["wall"], outputs=[])
        self._save_manifest()

    # stage B: map PNGs (limits known after A)
    def _global_limits(self, p, snaps):
        acc = limits_mod.Accumulator(p["scale"], symmetric=_symmetric(p))
        self._add_stats(acc, p, snaps)
        return limits_mod.resolve("global", p["scale"], tuple(p["pct"]) if p["pct"] else None, p["vmin"], p["vmax"], accumulator=acc)

    def _add_stats(self, acc, p, snaps):
        for s in snaps:
            sp = _stats_path(self.out, p["name"], s["stem"])
            if not os.path.exists(sp) or (self._unit(p["name"], s["stem"]) or {}).get("render_key") != self._map_keys(p, s):
                continue
            with np.load(sp) as d:
                if int(d["n"]) == 0:
                    continue
                big = max(abs(float(d["lo"])), abs(float(d["hi"]))) if acc.symmetric else float(d["hi"])
                acc.lo, acc.hi, acc.n = min(acc.lo, float(d["lo"])), max(acc.hi, big), acc.n + int(d["n"])
                acc._samples.append(np.abs(d["sample"]) if acc.symmetric else d["sample"])

    def _stage_b(self, prods, snaps):
        tasks = []
        for p in [q for q in prods if q["kind"] == "map"]:
            if p["limits"] == "global":
                lim = list(self._global_limits(p, snaps))
            elif p["limits"] == "fixed":
                lim = [float(p["vmin"]), float(p["vmax"])]
            else:
                lim = None
            for s in snaps:
                u = self._unit(p["name"], s["stem"])
                if u is None or u.get("render_key") != self._map_keys(p, s):
                    continue                                    # the render failed
                k = self._png_key(p, u["render_key"], lim)
                if self.force or u.get("key") != k or not self._have(u):
                    tasks.append({"p": p, "out": self.out, "store": self.store_dir, "stem": s["stem"], "path": s["path"], "lim": lim, "key": k})
        if tasks:
            self.log(f"stage B: {len(tasks)} map picture(s)")
        for k, (task, r) in enumerate(zip(tasks, self._map(_worker_png, tasks)), 1):
            p = task["p"]
            if k % 10 == 0 or k == len(tasks):
                self.log(f"stage B {k}/{len(tasks)}")
            if not r["ok"]:
                self.result.failures.append((task["stem"], p["name"], "png", r["error"]))
                continue
            self.result.count(p["name"], task["stem"], "png")
            self._set_unit(p["name"], "map", task["stem"], key=task["key"], outputs=r["outputs"], lim=r["lim"], wall=round(r["wall"] + self._unit(p["name"], task["stem"]).get("wall_render", 0), 4))
        self._save_manifest()

    # scalars table (also what dashboards plot)
    def _table(self, snaps):
        """(columns, rows): one row per readable snapshot, the values of every scalar product whose entry is current."""
        scal = [p for p in self.products if p["kind"] == "scalar"]
        cols, rows = ["snapshot", "time", "time_myr"], []
        for s in snaps:
            row = {"snapshot": s["stem"], "time": s["time"], "time_myr": s["time_myr"]}
            for p in scal:
                u = self._unit(p["name"], s["stem"])
                if u and u.get("values") is not None:
                    row.update(u["values"])
                    cols += [c for c in u["values"] if c not in cols]
            rows.append(row)
        return cols, rows

    def _arrays(self, snaps):
        cols, rows = self._table(snaps)
        t = {c: np.array([r.get(c, np.nan) for r in rows], float) for c in cols if c != "snapshot"}
        t["index"] = np.arange(len(rows), dtype=float)
        return t

    def _write_scalars_table(self, snaps):
        if not any(p["kind"] == "scalar" for p in self.products):
            return
        cols, rows = self._table(snaps)
        lines = [",".join(cols)] + [",".join(r["snapshot"] if c == "snapshot" else _fmt(r.get(c)) for c in cols) for r in rows]
        _atomic_text(os.path.join(self.out, "scalars.csv"), "\n".join(lines) + "\n")

    @staticmethod
    def _series_key(tiles):
        return [(i, t["xnow"], [(n, [_fmt(v) for v in x], [_fmt(v) for v in y]) for n, x, y in t["curves"]]) for i, t in sorted(tiles.items())]

    # stage D: dashboards (a layout per snapshot)
    def _stage_dashboards(self, prods, snaps):
        from . import layout as lay
        tasks = []
        base = self._arrays(snaps)
        for p in [q for q in prods if q["kind"] == "dashboard"]:
            L = p["layout"]
            try:
                table = lay.with_derived(L, base)
                for pan in L["panel"]:
                    if pan["type"] == "timeseries":
                        lay.timeseries_data(pan, table, 0)
            except (ValueError, KeyError) as e:
                self.result.failures += [(s["stem"], p["name"], "dashboard", str(e)) for s in snaps]
                continue
            for idx, s in enumerate(snaps):
                maps, profs, series, keys, missing = [], [], {}, [], None
                for pan in L["panel"]:
                    if pan["type"] in ("map", "profile"):
                        u = self._unit(pan["product"], s["stem"])
                        if not u or u.get("key") is None or (pan["type"] == "map" and not u.get("lim")):
                            missing = pan["product"]
                            break
                        keys.append(u["key"])
                        if pan["type"] == "map":
                            maps.append({"id": pan["id"], "p": self._cfg(pan["product"]), "lim": u["lim"]})
                        else:
                            profs.append({"id": pan["id"], "npz": next(o for o in u["outputs"] if o.endswith(".npz")), "quantities": pan["quantities"], "xlog": pan["xlog"], "ylog": pan["ylog"]})
                    elif pan["type"] == "timeseries":
                        series[pan["id"]] = lay.timeseries_data(pan, table, idx)
                if missing:
                    self.result.failures.append((s["stem"], p["name"], "dashboard", f"the product {missing!r} is not available for this snapshot"))
                    continue
                key = _hash("dash", s["sig"], L, keys, self._series_key(series), self.version)
                u = self._unit(p["name"], s["stem"])
                if self.force or u is None or u.get("key") != key or not self._have(u):
                    tasks.append({"p": p, "layout": L, "out": self.out, "store": self.store_dir, "stem": s["stem"], "path": s["path"], "time": s["time"], "maps": maps, "profiles": profs, "series": series, "key": key})
        if tasks:
            self.log(f"dashboards: {len(tasks)} picture(s)")
        for k, (task, r) in enumerate(zip(tasks, self._map(_worker_dashboard, tasks)), 1):
            p = task["p"]
            if k % 10 == 0 or k == len(tasks):
                self.log(f"dashboards {k}/{len(tasks)}")
            if not r["ok"]:
                self.result.failures.append((task["stem"], p["name"], "dashboard", r["error"]))
                continue
            self.result.count(p["name"], task["stem"], "dashboard")
            self._set_unit(p["name"], "dashboard", task["stem"], key=task["key"], outputs=r["outputs"], wall=r["wall"])
        self._save_manifest()

    # stage D2: run-wide time-series figures (OUT/figures/<name>.png), drawn here (cheap)
    def _stage_timeseries(self, prods, snaps):
        from . import layout as lay, figures
        figs = [q for q in prods if q["kind"] == "timeseries"]
        if not figs or not snaps:
            return
        base = self._arrays(snaps)
        for p in figs:
            t0 = time.perf_counter()
            L = p["layout"]
            try:
                table = lay.with_derived(L, base)
                tiles = {pan["id"]: (lay.timeseries_data(pan, table, None) if pan["type"] == "timeseries" else {"text": pan["text"]}) for pan in L["panel"]}
            except (ValueError, KeyError) as e:
                self.result.failures.append(("(run)", p["name"], "timeseries", str(e)))
                continue
            key = _hash("ts", L, self._series_key({k: v for k, v in tiles.items() if "curves" in v}), self.version)
            out = os.path.join(self.out, "figures", p["name"] + ".png")
            u = self._unit(p["name"], "run")
            if not (self.force or u is None or u.get("key") != key or not os.path.exists(out)):
                continue
            rgb = lay.render_layout(L, tiles, f"{len(snaps)} snapshots, t = {snaps[0]['time']:.4g} .. {snaps[-1]['time']:.4g}")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + f".tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(imaging.png_bytes(rgb))
            os.replace(tmp, out)
            side = provenance.write_sidecar(out, {"command": "analyze-timeseries", "layout": L["name"], "image_size": [int(rgb.shape[1]), int(rgb.shape[0])]})
            self.result.count(p["name"], "run", "timeseries")
            self._set_unit(p["name"], "timeseries", "run", key=key, outputs=[os.path.relpath(out, self.out), os.path.relpath(side, self.out)], wall=round(time.perf_counter() - t0, 4))
        self._save_manifest()

    # stage E: movies
    def _stage_movies(self, prods, snaps):
        from . import figures
        for p in [q for q in prods if q["kind"] == "movie"]:
            frames = []
            for s in snaps:
                u = self._unit(p["source"], s["stem"])
                if u and u.get("key") and self._have(u):
                    frames.append((s["stem"], u["key"], os.path.join(self.out, u["outputs"][0])))
            if len(frames) < 2:
                continue
            key = _hash("movie", p, [(a, b) for a, b, _ in frames], self.version)
            out = os.path.join(self.out, "movies", p["name"] + ".mp4")
            u = self._unit(p["name"], "movie")
            if not (self.force or u is None or u.get("key") != key or not os.path.exists(out)):
                continue
            exe = shutil.which("ffmpeg")
            if not exe:
                self.log(f"movie {p['name']}: ffmpeg not found, retried on the next run")
                continue
            t0 = time.perf_counter()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fdir = os.path.join(self.out, ".tmp", p["name"])
            shutil.rmtree(fdir, ignore_errors=True)
            os.makedirs(fdir)
            try:
                for k, (_, _, f) in enumerate(frames):
                    shutil.copyfile(f, os.path.join(fdir, f"frame_{k:05d}.png"))
                with open(frames[0][2], "rb") as fh:
                    w, h = np.frombuffer(fh.read(24)[16:24], ">u4")
                w, h = int(w) + int(w) % 2, int(h) + int(h) % 2
                tmp = os.path.join(os.path.dirname(out), "." + p["name"] + ".tmp.mp4")
                cmd = figures.ffmpeg_command(exe, p["fps"], os.path.join(fdir, "frame_%05d.png"), tmp)
                cmd = cmd[:-1] + ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x121216"] + cmd[-1:]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")
                os.replace(tmp, out)
            except Exception as e:
                self.result.failures.append(("(run)", p["name"], "movie", f"{type(e).__name__}: {e}"))
                continue
            finally:
                shutil.rmtree(fdir, ignore_errors=True)
            self.result.count(p["name"], "movie", "movie")
            self._set_unit(p["name"], "movie", "movie", key=key, outputs=[os.path.relpath(out, self.out)], wall=round(time.perf_counter() - t0, 3), frames=len(frames))
        self._save_manifest()

    # ---- reports
    def _plan_text(self, prods, snaps):
        lines = [f"run {self.run_path}: {len(snaps)} readable snapshot(s), {len(self.result.skipped)} unreadable; output {self.out}"]
        for stem, err in self.result.skipped.items():
            lines.append(f"  unreadable {stem}: {err}")
        n = {}
        for stem, items in self._pending_a.items():
            for it in items:
                n.setdefault((it["p"]["name"], it["stage"]), []).append(stem)
        for (name, stage), stems in sorted(n.items()):
            lines.append(f"  {name}: {stage} for {len(stems)} snapshot(s): {', '.join(stems[:4])}{' ...' if len(stems) > 4 else ''}")
        later = [p["name"] for p in prods if p["kind"] in ("dashboard", "timeseries", "movie")]
        pend_maps = {k[0] for k in n if k[1] == "render"}
        for p in prods:
            if p["kind"] == "map":
                stale = [s["stem"] for s in snaps if (self._unit(p["name"], s["stem"]) or {}).get("key") is None]
                if p["name"] in pend_maps and p["limits"] == "global":
                    lines.append(f"  {p['name']}: png for all snapshots probably (global colour limits will move)")
                elif stale and p["name"] not in pend_maps:
                    lines.append(f"  {p['name']}: png for {len(stale)} snapshot(s)")
        if later:
            lines.append(f"  dashboards / movies ({', '.join(later)}): redone where their panels or series changed")
        if not n:
            lines.append("  nothing to do for profiles, scalars or map data (maybe pictures, see above)")
        return "\n".join(lines)

    def _write_digest(self, t_start):
        res, lines = self.result, []
        lines.append(f"splosh analyze digest  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"run: {os.path.abspath(self.run_path)}")
        lines.append(f"splosh {self.version}  commit {provenance.git_commit() or '-'}  config {_hash(self.products)[:10]}")
        ok = len(self.snaps) if hasattr(self, "snaps") else 0
        lines.append(f"snapshots: {ok} readable, {len(res.skipped)} unreadable (skipped, retried next run)")
        if ok:
            last = self.snaps[-1]
            lines.append(f"last snapshot: {last['stem']}  time {last['time']:.6g}   first: {self.snaps[0]['stem']}  time {self.snaps[0]['time']:.6g}")
        lines.append(f"this run: {res.total} work units " + ("(" + ", ".join(f"{k} {v}" for k, v in sorted(res.counts.items())) + ")" if res.counts else "(nothing was out of date)") + f", {time.perf_counter() - t_start:.1f} s")
        lines.append("products:")
        for p in self.products:
            units = self.manifest["products"].get(p["name"], {}).get("units", {})
            done = sum(1 for u in units.values() if u.get("key"))
            lines.append(f"  {p['kind']:9s} {p['name']:20s} {done}/{ok if p['kind'] != 'movie' else 1} done" + ("" if p["name"] in self.only else "  (not selected)"))
        if ok and any(p["kind"] == "scalar" for p in self.products):
            cols, rows = self._table(self.snaps)
            lines.append("scalars (min .. max, last):")
            for c in cols[3:][:24]:
                v = np.array([r.get(c, np.nan) for r in rows], float)
                v = v[np.isfinite(v)]
                if v.size:
                    lines.append(f"  {c:14s} {v.min():.5g} .. {v.max():.5g}   last {v[-1]:.5g}")
                else:
                    lines.append(f"  {c:14s} (no data)")
        errs = [(s, "-", "open", e) for s, e in res.skipped.items()] + list(res.failures)
        if errs:
            lines.append(f"failures: {len(errs)}")
            for s, pn, st, e in errs[:12]:
                lines.append(f"  {s} {pn} {st}: {e[:110]}")
            if len(errs) > 12:
                lines.append(f"  ... and {len(errs) - 12} more (see manifest.json)")
        else:
            lines.append("failures: none")
        lines = lines[:58] + ([f"... truncated"] if len(lines) > 58 else [])
        res.digest = "\n".join(lines) + "\n"
        _atomic_text(os.path.join(self.out, "digest.txt"), res.digest.encode("ascii", "replace").decode())


def run(run, config=None, out=None, jobs=1, only=None, every=None, force=False, dry_run=False, store=None, log=None):
    """Run the pipeline (see the module doc). Returns a Result (`counts`, `work`, `failures`, `skipped`, `plan` for a dry run, `digest`, `out`)."""
    return Pipeline(run, config, out, jobs, only, every, force, dry_run, store, log).run()
