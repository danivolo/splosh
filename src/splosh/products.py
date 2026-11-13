"""Snapshot products: quantities computed once per snapshot (stresses, torques by region, profiles, histograms, ...), kept next to the run, read as time series.

A product is registered by a plugin (or the core):
    plugins.register_product(name, fn, needs, version=1, label=None, applies=None, params=None)
        fn(path, params) -> {"scalars": {column: float}, "arrays": {name: ndarray}}       (one snapshot; read what it needs itself)
        needs: datasets of PartType0 (and "PartType5" for black holes); applies(path) -> bool: whether the product makes sense for this run (e.g. two BHs)
        params: the defaults of its parameters (the stored results are keyed by them)

Results: <run>/analysis_viewer/products/<name>/<snapshot stem>.npz with a key (the snapshot's signature, the product's version, its parameters): a
changed snapshot, product or parameter is computed again. `missing(run, names)` lists what is not computed yet; `compute(...)` does it (a background job
from the viewer: `splosh products RUN`, parallel over snapshots, the number of workers set by the physical cores and the free memory).

Time series: every product is a series source `prod_<name>` (series.py): its scalars over the run's snapshots, rows sorted by time. A family of scalars
<prefix><k> (e.g. a histogram per snapshot) is a heatmap against time."""
import hashlib
import json
import os
import time as _time

import numpy as np

from . import plugins

PRODUCTS = plugins.PRODUCTS


def product_dir(run_dir, name):
    return os.path.join(os.path.abspath(run_dir), "analysis_viewer", "products", name)


def _key(path, name, params):
    from .mapstore import snapshot_hash
    p = PRODUCTS[name]
    return hashlib.sha1(json.dumps([snapshot_hash(path), name, p["version"], params], sort_keys=True, default=str).encode()).hexdigest()


def params_of(name, overrides=None):
    return {**PRODUCTS[name]["params"], **(overrides or {})}


def result_path(run_dir, name, path):
    return os.path.join(product_dir(run_dir, name), os.path.splitext(os.path.basename(path))[0] + ".npz")


def load_result(run_dir, name, path, params=None):
    """the stored result of a product for one snapshot ({"scalars", "arrays", "time"}) when it is current, else None."""
    params = params_of(name, params)
    f = result_path(run_dir, name, path)
    try:
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("key") != _key(path, name, params):
                return None
            return {"scalars": meta["scalars"], "time": meta["time"], "arrays": {k[2:]: z[k] for k in z.files if k.startswith("a_")}}
    except (OSError, KeyError, ValueError):
        return None


def compute_one(run_dir, name, path, params=None):
    """compute a product for one snapshot and store it (atomically). Returns the result."""
    from .runs import snapshot_times
    params = params_of(name, params)
    t0 = _time.perf_counter()
    res = PRODUCTS[name]["fn"](path, params) or {}
    scalars = {k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in (res.get("scalars") or {}).items()}
    arrays = {k: np.asarray(v) for k, v in (res.get("arrays") or {}).items()}
    t = snapshot_times(path)[0]
    d = product_dir(run_dir, name)
    os.makedirs(d, exist_ok=True)
    f = result_path(run_dir, name, path)
    tmp = f + ".tmp.npz"
    meta = {"key": _key(path, name, params), "scalars": scalars, "time": float(t), "params": params, "seconds": round(_time.perf_counter() - t0, 3),
            "snapshot": os.path.basename(path)}
    np.savez(tmp, meta=json.dumps(meta), **{"a_" + k: v for k, v in arrays.items()})
    os.replace(tmp, f)
    return {"scalars": scalars, "time": float(t), "arrays": arrays}


def applicable(name, paths):
    """whether a product applies to a run (its `applies` on the first snapshot; True without one)."""
    plugins.load_plugins()
    fn = PRODUCTS[name].get("applies")
    if not fn or not paths:
        return True
    try:
        return bool(fn(paths[0]))
    except Exception:
        return False


def missing(run_dir, names, paths, params=None):
    """[(name, snapshot path)] not computed yet (or out of date)."""
    out = []
    for n in names:
        for p in paths:
            if load_result(run_dir, n, p, (params or {}).get(n)) is None:
                out.append((n, p))
    return out


def _mem_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def workers_for(paths, max_workers=None):
    """processes to compute products in parallel: at most the physical cores, and few enough that each has ~4x its snapshot's size of memory
    (a product holds a few fields of all particles; four parallel full reads of a large snapshot once took the laptop down)."""
    from .cpus import physical_cores
    n = physical_cores() if max_workers is None else max(1, int(max_workers))
    mem = _mem_available()
    if mem and paths:
        size = max(os.path.getsize(p) for p in paths[:20])
        n = min(n, max(1, int(0.5 * mem // max(4 * size, 1))))
    return max(1, n)


def _work(task):
    run_dir, name, path, params = task
    plugins.load_plugins()
    t0 = _time.perf_counter()
    compute_one(run_dir, name, path, params)
    return name, path, _time.perf_counter() - t0


def compute(run_dir, names, paths, params=None, workers=None, log=print):
    """compute what is missing, in parallel processes (one snapshot per task); prints "k/n name snapshot seconds" (the job list's progress)."""
    todo = missing(run_dir, names, paths, params)
    if not todo:
        log("nothing to do: every product is current")
        return 0
    workers = workers_for(paths, workers)
    tasks = [(run_dir, n, p, (params or {}).get(n)) for n, p in todo]
    log(f"{len(tasks)} product(s) to compute with {workers} process(es)")
    if workers == 1:
        it = map(_work, tasks)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        it = (f.result() for f in cf.as_completed([ex.submit(_work, t) for t in tasks]))
    for k, (n, p, dt) in enumerate(it, start=1):
        log(f"{k}/{len(tasks)}  {n}  {os.path.basename(p)}  {dt:.1f} s")
    return len(tasks)


# ----------------------------------------------------------------------------- products as series sources

def _source_files(name):
    def files(ctx):
        d = product_dir(ctx["run_dir"], name)
        if not os.path.isdir(d):
            return []
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz"))
    return files


def _source(name):
    def fn(ctx):
        from .series import Table
        rows = []
        for p in ctx.get("snapshots") or []:
            r = load_result(ctx["run_dir"], name, p)
            if r is not None:
                rows.append(r)
        rows.sort(key=lambda r: r["time"])
        cols = {}
        for k, r in enumerate(rows):
            for c, v in r["scalars"].items():
                cols.setdefault(c, [np.nan] * k)
                cols[c].append(np.nan if v is None else v)
            for c in cols:
                if len(cols[c]) < k + 1:
                    cols[c].append(np.nan)
        pm = PRODUCTS[name].get("meta") or {}
        fams = sorted((k for k in pm if k.endswith("_")), key=len, reverse=True)       # "sigma_": the label of the family sigma_<R>
        def _m(c):
            if c in pm:
                return pm[c]
            for f in fams:
                if c.startswith(f):
                    try:
                        float(c[len(f):])
                        return pm[f]
                    except ValueError:
                        pass
            return {}
        meta = {c: {"style": "linepoints", **_m(c)} for c in cols}
        return Table([r["time"] for r in rows], cols, meta=meta,
                     note=None if len(rows) == len(ctx.get("snapshots") or []) else f"{len(rows)} of {len(ctx.get('snapshots') or [])} snapshots computed so far")
    return fn


def register_sources():
    """every product is the series source prod_<name> (called after the plugins are loaded)."""
    for name, p in PRODUCTS.items():
        sname = "prod_" + name
        if sname not in plugins.SERIES:
            import zlib
            mh = zlib.crc32(json.dumps(p.get("meta") or {}, sort_keys=True, default=str).encode()) % 1000      # new labels: the cached table is rebuilt
            plugins.register_series(sname, _source(name), _source_files(name), label=p["label"] + " (per snapshot)", defaults=p.get("defaults") or [],
                                    version=p["version"] * 1000 + mh, priority=3)


def main(a):
    """`splosh products RUN [--names a,b] [--workers N]`"""
    import sys
    from .runs import run_snapshots
    plugins.load_plugins()
    paths = run_snapshots(a.run)
    if not paths:
        raise SystemExit(f"no snapshots in {a.run}")
    from .analyze import run_root
    run_dir = run_root(os.path.dirname(os.path.abspath(paths[0])))
    names = [n for n in (a.names.split(",") if a.names else sorted(PRODUCTS)) if n]
    unknown = [n for n in names if n not in PRODUCTS]
    if unknown:
        raise SystemExit(f"unknown product(s) {unknown}; registered: {sorted(PRODUCTS)} (plugin errors: {plugins.errors()})")
    names = [n for n in names if applicable(n, paths)]
    compute(run_dir, names, paths, workers=a.workers, log=lambda m: print(m, file=sys.stderr, flush=True))


from . import disc  # noqa: E402,F401  (the core product `disc`: disc profiles, angular-momentum budget, eccentricity, minidiscs)
