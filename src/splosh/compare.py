"""`splosh compare RUN1 RUN2 ... [label=path ...]`: comparison products of several runs.

Every run is analysed by its own `analyze.Pipeline` (own cache, by default RUN/analysis, or --run-out DIR/<label>); the comparison figures are then drawn from those caches into a separate tree OUT:

    [[compare]]  type = "timeseries"   columns = ["gas_mass", "bh_sep / bh_mtot"]      one panel per column (names of scalars.csv columns or safe expressions), one curve per run
    [[compare]]  type = "profile"      product = "radial"  quantities = ["u"]  at = [0.5, 1.0]   one panel per aligned time, one curve per run (the run's nearest snapshot)
    [[compare]]  type = "maps"         maps = ["density", "temperature"]  at = [0.5, 1.0]        one picture per aligned time: a row per map product, a column per run

Common keys: `name`, `align` = time | time_myr | index | offset, `offsets` = [shift of run 1, run 2, ...] (subtracted from each run's time / index; align = "offset" is time with offsets required),
`at` = list of aligned values or `n` = how many (evenly spaced over the range all runs cover; default 4), `tol` = the largest distance from the wanted value to a run's nearest snapshot in
aligned units (default: half of the run's median snapshot spacing). A run without a snapshot within `tol` gets a blank tile with a note. Maps of one product use the same View (an `auto` width is taken from the
first run) and the same colour-limit policy; `global` limits are locked over all snapshots of all runs. Outputs: OUT/<name>.png (timeseries, profile) or OUT/<name>/at_NNN.png (maps), each with a .json
sidecar listing the snapshot chosen for every run; OUT/manifest.json and OUT/digest.txt. A figure is redone only if its inputs (cache keys of the runs' products, config, version) changed."""
import json
import os
import time

import numpy as np

from . import analyze, imaging, layout as lay, limits as limits_mod, provenance
from .analyze import _hash, _fmt, _atomic_json, _atomic_text, _stats_path, _png_limits, _view_of

TYPES = ("timeseries", "profile", "maps")
ALIGNS = ("time", "time_myr", "index", "offset")
_KEYS = {"name", "type", "columns", "product", "quantities", "maps", "align", "offsets", "at", "n", "tol", "xlog", "ylog", "ncols", "title"}


def complete_compare(p):
    bad = set(p) - _KEYS - {"kind"}
    if bad:
        raise ValueError(f"[[compare]] {p['name']!r}: unknown key(s) {sorted(bad)}; allowed: {sorted(_KEYS)}")
    t = p.get("type")
    if t not in TYPES:
        raise ValueError(f"[[compare]] {p['name']!r}: type must be one of {TYPES}")
    out = {"kind": "compare", "name": p["name"], "type": t, "align": p.get("align", "time"), "offsets": [float(x) for x in p.get("offsets", [])], "at": p.get("at"), "n": int(p.get("n", 4)),
           "tol": None if p.get("tol") is None else float(p["tol"]), "ncols": int(p.get("ncols", 0)) or None, "title": p.get("title"), "xlog": bool(p.get("xlog", False)), "ylog": bool(p.get("ylog", False))}
    if out["align"] not in ALIGNS:
        raise ValueError(f"[[compare]] {p['name']!r}: align must be one of {ALIGNS}")
    if out["align"] == "offset" and not out["offsets"]:
        raise ValueError(f"[[compare]] {p['name']!r}: align = offset needs offsets = [...]")
    if out["at"] is not None:
        out["at"] = [float(x) for x in ([p["at"]] if isinstance(p["at"], (int, float)) else p["at"])]
    if t == "timeseries":
        cols = [p["columns"]] if isinstance(p.get("columns"), str) else list(p.get("columns") or [])
        if not cols:
            raise ValueError(f"[[compare]] {p['name']!r}: a timeseries comparison needs columns")
        for c in cols:
            lay.parse_expr(c)
        out["columns"] = cols
    elif t == "profile":
        if not p.get("product"):
            raise ValueError(f"[[compare]] {p['name']!r}: a profile comparison needs product = a [[profile]] name")
        out["product"] = p["product"]
        out["quantities"] = None if p.get("quantities") is None else ([p["quantities"]] if isinstance(p["quantities"], str) else list(p["quantities"]))
    else:
        if not p.get("maps"):
            raise ValueError(f"[[compare]] {p['name']!r}: a maps comparison needs maps = [map product names]")
        out["maps"] = list(p["maps"])
    return out


def check_compare(p, byname):
    refs = [(p.get("product"), "profile")] if p["type"] == "profile" else [(m, "map") for m in p.get("maps", [])]
    for name, kind in refs:
        if byname.get(name, {}).get("kind") != kind:
            raise ValueError(f"[[compare]] {p['name']!r}: {name!r} is not a [[{kind}]] product")


def default_compare_config():
    cfg = analyze.default_config()
    cfg["compare"] = [{"name": "scalars", "type": "timeseries", "columns": ["gas_mass", "rho_max"]}, {"name": "grid", "type": "maps", "maps": ["density"], "n": 3}]
    return cfg


def parse_runs(runs):
    """[(label, path)] from 'label=path' strings, plain paths (label = the run directory's name), or (label, path) pairs."""
    out = []
    for r in runs:
        if isinstance(r, (tuple, list)):
            label, path = r
        elif "=" in str(r) and not os.path.exists(str(r)):
            label, _, path = str(r).partition("=")
        else:
            path = str(r)
            label = os.path.basename(analyze.run_root(path).rstrip("/")) or "run"
        out.append((label, os.path.expanduser(str(path))))
    labels = [l for l, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"run labels must be unique, got {labels}; give them as label=path")
    for l in labels:
        if not l or any(c in l for c in "/\\ "):
            raise ValueError(f"run label {l!r} must be non-empty without spaces or slashes")
    return out


class CompareResult:
    """`runs` {label: analyze.Result}, `counts` / `work` of the comparison figures, `failures`, `digest`, `out`, `plan` (dry run)."""
    def __init__(self, out):
        self.out, self.runs, self.counts, self.work, self.failures, self.digest, self.plan = out, {}, {}, [], [], "", None

    @property
    def total(self):
        return len(self.work)


class _Run:
    def __init__(self, label, pipe):
        self.label, self.pipe = label, pipe
        self.snaps = pipe.snaps
        self.table = pipe._arrays(pipe.snaps) if pipe.snaps else {}

    def coord(self, align, offset):
        base = {"time": "time", "time_myr": "time_myr", "index": "index", "offset": "time"}[align]
        return np.asarray(self.table[base], float) - offset if self.table else np.array([])


def _nearest(coord, at, tol, spacing):
    if coord.size == 0:
        return None, "no snapshots"
    j = int(np.argmin(np.abs(coord - at)))
    lim = tol if tol is not None else 0.5 * spacing
    if abs(coord[j] - at) > lim * (1 + 1e-9):
        return None, f"no snapshot within {lim:.3g} of {at:.4g} (nearest {coord[j]:.4g})"
    return j, ""


def _spacing(coord):
    return float(np.median(np.diff(coord))) if coord.size > 1 else float("inf")


def compare(runs, config=None, out=None, jobs=1, only=None, every=None, force=False, dry_run=False, run_out=None, align=None, offsets=None, log=None):
    """Analyse every run (own cache) and draw the comparison products into `out`. runs: see parse_runs; `align` / `offsets` ({label: shift}) override the config's for every product."""
    log = log or (lambda m: None)
    runs = parse_runs(runs)
    if len(runs) < 2:
        raise ValueError("compare needs at least two runs")
    if config is None:
        config = default_compare_config()
    prods, _ = analyze.load_config(config)
    cmps = [p for p in prods if p["kind"] == "compare"]
    if only:
        want = [only] if isinstance(only, str) else list(only)
        cmps = [p for p in cmps if p["name"] in want or p["type"] in want]
    if not cmps:
        raise ValueError("the config has no [[compare]] product (selected)")
    out = os.path.abspath(os.path.expanduser(str(out or "compare")))
    needed = set()
    byname = {p["name"]: p for p in prods}
    for c in cmps:
        needed |= set(c.get("maps", [])) | ({c["product"]} if c["type"] == "profile" else set())
        needed |= {p["name"] for p in prods if p["kind"] == "scalar"}
    res = CompareResult(out)
    data, auto = [], None
    for label, path in runs:
        ro = os.path.join(os.path.abspath(run_out), label) if run_out else None
        pipe = analyze.Pipeline(path, [p for p in prods if p["kind"] != "compare"], out=ro, jobs=jobs, only=sorted(needed), every=every, force=force, dry_run=dry_run, log=log, auto_radius=auto)
        r = pipe.run()
        res.runs[label] = r
        auto = auto or pipe.manifest.get("auto_radius")
        if not pipe.snaps:
            res.failures.append((label, "-", "run", f"no readable snapshots ({len(r.skipped)} skipped)"))
        data.append(_Run(label, pipe))
    if dry_run:
        res.plan = "\n".join(f"run {d.label}: " + (res.runs[d.label].plan or "") for d in data) + "\ncomparison figures: " + ", ".join(c["name"] for c in cmps) + " (redone where their inputs changed)"
        return res
    if len([d for d in data if d.snaps]) < 1:
        return res
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, "manifest.json")
    try:
        with open(man_path) as f:
            man = json.load(f)
    except (OSError, ValueError):
        man = {}
    man.setdefault("figures", {})
    t0 = time.perf_counter()
    for c in cmps:
        try:
            _make(c, data, out, man, res, force, align, offsets)
        except (ValueError, KeyError) as e:
            res.failures.append(("-", c["name"], c["type"], f"{type(e).__name__}: {e}"))
    man.update(schema="splosh-compare-1", splosh_version=provenance.version(), git_commit=provenance.git_commit(), updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               runs={d.label: {"path": os.path.abspath(d.pipe.run_path), "cache": d.pipe.out, "snapshots": len(d.snaps)} for d in data}, failures=[list(f) for f in res.failures])
    _atomic_json(man_path, man)
    _digest(res, data, cmps, t0)
    return res


def _offsets(c, data, override):
    offs = list(c["offsets"]) + [0.0] * (len(data) - len(c["offsets"]))
    offs = offs[:len(data)]
    if override:
        for k, d in enumerate(data):
            if d.label in override:
                offs[k] = float(override[d.label])
    return offs


def _at_values(c, coords, spacings):
    if c["at"]:
        return list(c["at"])
    cov = [(x.min(), x.max()) for x in coords if x.size]
    lo, hi = max(a for a, _ in cov), min(b for _, b in cov)
    if hi < lo:
        lo, hi = min(a for a, _ in cov), max(b for _, b in cov)
    n = max(1, c["n"])
    return [float(lo)] if n == 1 or hi == lo else [float(v) for v in np.linspace(lo, hi, n)]


def _save_png(path, rgb, side):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(imaging.png_bytes(rgb))
    os.replace(tmp, path)
    return [path, provenance.write_sidecar(path, side)]


def _done(man, res, name, tag, key, path, force):
    old = man["figures"].get(name, {}).get(tag)
    return not (force or old is None or old["key"] != key or not os.path.exists(path))


def _finish(man, res, c, tag, key, files, out):
    man["figures"].setdefault(c["name"], {})[tag] = {"key": key, "outputs": [os.path.relpath(f, out) for f in files]}
    res.work.append((c["name"], tag, c["type"]))
    res.counts[c["type"]] = res.counts.get(c["type"], 0) + 1


def _make(c, data, out, man, res, force, align_override, off_override):
    align = align_override or c["align"]
    offs = _offsets(c, data, off_override)
    live = [(d, o) for d, o in zip(data, offs) if d.snaps]
    coords = [d.coord(align, o) for d, o in live]
    labels = [d.label for d, _ in live]
    if c["type"] == "timeseries":
        panels = [{"id": f"c{k}", "type": "timeseries", "columns": [col], "label": col, "xaxis": "time", "marker": False} for k, col in enumerate(c["columns"])]
        L = lay.validate_layout({"name": c["name"], "title": c["title"], "ncols": c["ncols"] or min(2, len(panels)), "panel": panels})
        tiles = {}
        for k, col in enumerate(c["columns"]):
            curves = [(d.label, x, lay.curve_values(col, d.table)) for (d, _), x in zip(live, coords)]
            tiles[f"c{k}"] = {"curves": curves, "xnow": None, "xlabel": align, "xlog": c["xlog"], "ylog": c["ylog"]}
        key = _hash("cmp-ts", c, align, offs, [(lb, [_fmt(v) for v in x], [[_fmt(v) for v in t["curves"][i][2]] for t in tiles.values()]) for i, (lb, x) in enumerate(zip(labels, coords))], provenance.version())
        path = os.path.join(out, c["name"] + ".png")
        if _done(man, res, c["name"], "figure", key, path, force):
            return
        rgb = lay.render_layout(L, tiles, "runs: " + ", ".join(labels))
        _finish(man, res, c, "figure", key, _save_png(path, rgb, {"command": "compare-timeseries", "runs": labels, "align": align, "offsets": offs, "columns": c["columns"]}), out)
        return
    ats = _at_values(c, coords, None)
    sp = [_spacing(x) for x in coords]
    if c["type"] == "profile":
        panels, tiles, picks = [], {}, []
        for k, at in enumerate(ats):
            curves, notes = [], []
            for (d, _), x, lb, spc in zip(live, coords, labels, sp):
                j, why = _nearest(x, at, c["tol"], spc)
                if j is None:
                    notes.append(f"{lb}: none")
                    continue
                u = d.pipe._unit(c["product"], d.snaps[j]["stem"])
                if not u or not u.get("outputs"):
                    notes.append(f"{lb}: no profile")
                    continue
                npz = next(o for o in u["outputs"] if o.endswith(".npz"))
                with np.load(os.path.join(d.pipe.out, npz)) as f:
                    qs = c["quantities"] or [q for q in f.files if q not in ("meta", "edges", "rbins", "rgeo", "n", "mass", "sigma", "density")][:1]
                    for q in qs:
                        curves.append((lb if len(qs) == 1 else f"{lb}:{q}", f["rbins"], f[q]))
                picks.append((k, lb, d.snaps[j]["stem"], u["key"]))
            pid = f"p{k}"
            panels.append({"id": pid, "type": "profile", "product": c["product"], "label": f"{align} = {at:.4g}" + (f"  ({'; '.join(notes)})" if notes else "")})
            tiles[pid] = {"curves": curves, "xlog": True, "ylog": True} if curves else {"blank": "; ".join(notes) or "no data"}
        L = lay.validate_layout({"name": c["name"], "title": c["title"], "ncols": c["ncols"] or min(2, len(panels)), "panel": panels})
        key = _hash("cmp-prof", c, align, offs, picks, provenance.version())
        path = os.path.join(out, c["name"] + ".png")
        if _done(man, res, c["name"], "figure", key, path, force):
            return
        rgb = lay.render_layout(L, tiles, "runs: " + ", ".join(labels))
        _finish(man, res, c, "figure", key, _save_png(path, rgb, {"command": "compare-profile", "runs": labels, "align": align, "offsets": offs, "at": ats, "snapshots": [list(p) for p in picks]}), out)
        return
    # maps: same limits across runs
    from .mapstore import MapStore
    cfgs = {m: next(q for q in d_products(data) if q["name"] == m) for m in c["maps"]}
    glob = {}
    for m, p in cfgs.items():
        if p["limits"] == "global":
            acc = limits_mod.Accumulator(p["scale"])
            for d, _ in live:
                d.pipe._add_stats(acc, p, d.snaps)
            glob[m] = limits_mod.resolve("global", p["scale"], tuple(p["pct"]) if p["pct"] else None, p["vmin"], p["vmax"], accumulator=acc)
        elif p["limits"] == "fixed":
            glob[m] = (float(p["vmin"]), float(p["vmax"]))
    for k, at in enumerate(ats):
        tag = f"at_{k:03d}"
        panels, tiles, picks = [], {}, []
        for r_i, m in enumerate(c["maps"]):
            p = cfgs[m]
            for c_i, ((d, _), x, lb, spc) in enumerate(zip(live, coords, labels, sp)):
                pid = f"{m}@{lb}"
                panels.append({"id": pid, "type": "map", "product": m, "row": r_i, "col": c_i, "label": f"{lb}  {m}"})
                j, why = _nearest(x, at, c["tol"], spc)
                if j is None:
                    tiles[pid] = {"blank": why}
                    continue
                s = d.snaps[j]
                store = MapStore(d.pipe.store_dir)
                img, meta, _ = store.get_or_render(s["path"], _view_of(p))
                lim = glob.get(m) or _png_limits(p, img, None)
                tiles[pid] = {"img": img, "meta": meta, "cmap": p["cmap"], "scale": p["scale"], "lim": lim, "scalebar": p["scalebar"]}
                picks.append((m, lb, s["stem"], s["sig"], list(map(float, lim))))
        L = lay.validate_layout({"name": c["name"], "title": c["title"], "ncols": len(live), "panel": panels})
        key = _hash("cmp-maps", c, align, offs, at, picks, [_view_dict_of(cfgs[m]) for m in c["maps"]], provenance.version())
        path = os.path.join(out, c["name"], tag + ".png")
        if _done(man, res, c["name"], tag, key, path, force):
            continue
        rgb = lay.render_layout(L, tiles, f"{align} = {at:.4g}")
        _finish(man, res, c, tag, key, _save_png(path, rgb, {"command": "compare-maps", "runs": labels, "align": align, "offsets": offs, "at": at, "snapshots": [list(p[:4]) for p in picks]}), out)


def d_products(data):
    return data[0].pipe.products


def _view_dict_of(p):
    return analyze._view_dict(p)


def _digest(res, data, cmps, t0):
    lines = [f"splosh compare digest  {time.strftime('%Y-%m-%d %H:%M:%S')}", f"output: {res.out}", f"splosh {provenance.version()}  commit {provenance.git_commit() or '-'}"]
    for d in data:
        r = res.runs[d.label]
        last = f"last {d.snaps[-1]['stem']} t={d.snaps[-1]['time']:.5g}" if d.snaps else "no readable snapshots"
        lines.append(f"run {d.label}: {len(d.snaps)} snapshots, {len(r.skipped)} unreadable, {last}; this run {r.total} work units; cache {d.pipe.out}")
    lines.append(f"figures this run: {res.total} " + ("(" + ", ".join(f"{k} {v}" for k, v in sorted(res.counts.items())) + ")" if res.counts else "(nothing was out of date)") + f", {time.perf_counter() - t0:.1f} s")
    for c in cmps:
        lines.append(f"  {c['type']:10s} {c['name']}")
    errs = list(res.failures) + [(lb, f[1], f[2], f[3]) for lb, r in res.runs.items() for f in r.failures]
    lines.append(f"failures: {len(errs) or 'none'}")
    for e in errs[:12]:
        lines.append(f"  {e[0]} {e[1]} {e[2]}: {str(e[3])[:110]}")
    res.digest = "\n".join(lines[:58]) + "\n"
    _atomic_text(os.path.join(res.out, "digest.txt"), res.digest.encode("ascii", "replace").decode())
