"""`splosh prefetch --plan plan.json`: render every (snapshot, view) of a plan into the map store, so that the viewer finds them cached.

plan.json: [{"files": [snapshot paths], "view": {View fields}}, ...] (the viewer writes one per pane: its run's snapshots and its current view).
Prints `k/n  name  seconds` per map (the progress the job manager reads). --procs N renders N snapshots at a time, each with NUMBA_NUM_THREADS / N threads
(rendering stops scaling at about 5-8 threads, bench/scaling.py, so a 20-core node is best used as 4 x 5)."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _one(task):
    path, views, store_root, threads = task
    import numba
    from .io import load_snapshot
    from .maps import View, required_gas_fields
    from .mapstore import MapStore, snapshot_hash, view_key
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    store = MapStore(store_root)
    t0 = time.perf_counter()
    vs = [View(**v).validate() for v in views]
    snap, hits, sig = None, 0, snapshot_hash(path)
    for v in vs:
        if store.get(view_key(sig, v)) is not None:
            hits += 1
            continue
        if snap is None:
            fields = sorted(set().union(*[required_gas_fields(x) for x in vs]) | {"Coordinates", "SmoothingLength", "Masses", "Density"})
            snap = load_snapshot(path, gas_fields=tuple(fields))
        store.get_or_render(path, v, snap=snap)
    from . import region
    if not os.path.exists(region.index_file(path, store.root)):          # the chunk index for partial loading (serve --partial): ready before the first zoom
        region.get_index(path, store.root)
    return path, len(vs), hits, time.perf_counter() - t0


def run(plan, store_root=None, procs=1, log=print):
    from .mapstore import MapStore
    store_root = store_root or MapStore().root
    by_file = {}
    for item in plan:
        for f in item["files"]:
            by_file.setdefault(f, []).append(item["view"])
    tasks = list(by_file.items())
    import numba
    from .cpus import worker_threads
    total = worker_threads(numba.config.NUMBA_NUM_THREADS)                  # physical cores: hyperthreads do not speed up the kernels
    procs = max(1, total // 5) if procs == "auto" else int(procs)          # auto: about 5 threads per snapshot
    procs = max(1, min(procs, len(tasks) or 1))
    per = max(1, total // procs)
    n, done, maps, hits = len(tasks), 0, 0, 0
    work = [(f, vs, store_root, per) for f, vs in tasks]
    if procs == 1:
        results = map(_one, work)
    else:
        ex = ProcessPoolExecutor(procs, mp_context=mp.get_context("spawn"))
        results = ex.map(_one, work)
    for path, nv, h, dt in results:
        done += 1
        maps += nv
        hits += h
        log(f"{done}/{n}  {os.path.basename(path)}  {nv} view(s), {h} cached  {dt:.1f} s")
    if procs > 1:
        ex.shutdown()
    log(f"prefetched {maps} map(s) of {n} snapshot(s), {hits} were cached; store {store_root}")
    return {"snapshots": n, "maps": maps, "cached": hits}


def main(a):
    with open(a.plan) as f:
        plan = json.load(f)
    run(plan, a.store, a.procs or 1, log=lambda m: print(m, flush=True))


def add_parser(sub):
    p = sub.add_parser("prefetch", help="render the (snapshot, view) pairs of a plan into the map store (the viewer's 'prefetch all' job)")
    p.add_argument("--plan", required=True, help='JSON: [{"files": [...], "view": {...}}, ...]'); p.add_argument("--store")
    p.add_argument("--procs", default="1", help="snapshots rendered at once (auto: NUMBA_NUM_THREADS / 5)")
    p.set_defaults(fn=lambda a: main(a))
