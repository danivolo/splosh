"""A movie of a layout over a run's snapshots: one frame per snapshot, drawn by the publication renderer (layout_mpl, the default) or the bitmap one,
encoded with ffmpeg as `splosh movie` does.

    splosh layout-movie RUN --layout FILE [--out movie.mp4] [--from I] [--to J] [--every N] [--fps F] [--backend mpl|bitmap]
                           [--analysis DIR] [--config FILE] [--store DIR] [--dpi N] [--keep-frames]

Colour limits: a map panel with limits = global gets ONE pair for the whole movie: a first pass takes each frame's percentiles (the maps go to the map
store, so the second pass that draws the frames does not render again) and keeps the widest range; frame / viewport / fixed act per frame as in a
dashboard. Progress lines "pass k/n" and "frame k/n" (the viewer's job list shows them). Frames (and pass 1) are made in parallel processes:
--workers, default about one per 4 numba threads (figures.movie_workers), each with its share of the threads.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from . import layout as lay, layout_preview as lp
from .figures import ffmpeg_command


def _loader(path):
    from .io import load_snapshot
    cache = {}

    def load(fields):
        key = tuple(sorted(set(fields)))
        got = next((s for k, s in cache.items() if set(key) <= set(k)), None)
        if got is None:
            cache.clear()
            got = cache[key] = load_snapshot(path, gas_fields=key)
        return got
    return load


def _pass1(a):
    """the frame's own limits of the global panels (pass 1, in a worker)."""
    from .mapstore import MapStore
    _, tiles, _, _ = lp.tiles_for(a["L"], a["f"], _loader(a["f"]), MapStore(a["root"]), a["products"], a["table"])
    out = {}
    for pid in a["ids"]:
        lim = tiles.get(pid, {}).get("lim")
        if lim and np.all(np.isfinite(lim)):
            out[pid] = tuple(float(v) for v in lim)
    return out


def _frame(a):
    """one frame of the movie written as frame_k.png (in a worker); size None: this frame's (even) size, returned."""
    from .mapstore import MapStore
    lo, tiles, notes, header = lp.tiles_for(a["L"], a["f"], _loader(a["f"]), MapStore(a["root"]), a["products"], a["table"], limits=a["limits"])
    fn = os.path.join(a["fdir"], f"frame_{a['k']:05d}.png")
    size = a["size"]
    if a["backend"] == "mpl":
        import io
        from PIL import Image
        from . import layout_mpl
        im = Image.open(io.BytesIO(layout_mpl.render_bytes(lo, tiles, header, fmt="png"))).convert("RGB")
        size = size or (im.width - im.width % 2, im.height - im.height % 2)     # libx264 wants even sizes; every frame the size of the first
        im.crop((0, 0) + tuple(size)).save(fn)
    else:
        from .imaging import png_bytes
        rgb = lay.render_layout(lo, tiles, header)
        size = size or (rgb.shape[1] - rgb.shape[1] % 2, rgb.shape[0] - rgb.shape[0] % 2)
        with open(fn, "wb") as fh:
            fh.write(png_bytes(np.ascontiguousarray(rgb[:size[1], :size[0]])))
    return {"k": a["k"], "size": size, "notes": notes}


def make(run, layout_file, out, *, start=0, stop=None, every=1, fps=12, backend="mpl", analysis=None, config=None, store=None, dpi=None,
         keep_frames=False, ffmpeg=None, log=print, workers=None):
    from .mapstore import MapStore
    from .server import find_snapshots
    from . import analyze
    from .runs import run_snapshots
    paths = run_snapshots(run)
    if not paths:
        raise SystemExit(f"no snapshots in {run}")
    stop = len(paths) - 1 if stop is None else min(stop, len(paths) - 1)
    files = paths[max(0, start):stop + 1:max(1, every)]
    if len(files) < 2:
        raise SystemExit("a movie needs at least two snapshots")
    L = lay.validate_layout(lay.load_layout(layout_file))
    if dpi:
        L["figure"] = dict(L.get("figure") or lay.figure_options(L), dpi=int(dpi))
    products = analyze.normalize_config(analyze.read_config(config), os.path.dirname(os.path.abspath(config))) if config else None
    table = lp.read_scalars_table(analysis) if analysis else None
    store = store if store is not None else MapStore()
    if backend == "mpl":
        from . import layout_mpl
        prob = layout_mpl.style_problem((L.get("figure") or lay.figure_options(L))["style"])
        if prob:
            log("WARNING: " + prob)
    glob_ids = [p["id"] for p in L["panel"] if p["type"] == "map" and not p.get("product") and p.get("limits") == "global"]
    limits = {}
    from .figures import _pool_map, movie_workers
    workers = movie_workers() if workers is None else max(1, int(workers))
    common = {"L": L, "products": products, "table": table, "root": store.root}
    if glob_ids:                                         # pass 1: the widest range of the frames' percentiles, per panel (the maps go to the store)
        jobs = [dict(common, f=f, ids=glob_ids) for f in files]
        it = _pool_map(_pass1, jobs, workers) if workers > 1 and len(files) > 1 else map(_pass1, jobs)
        for k, lims in enumerate(it, start=1):
            for pid, lim in lims.items():
                lo, hi = limits.get(pid, (np.inf, -np.inf))
                limits[pid] = (min(lo, lim[0]), max(hi, lim[1]))
            log(f"pass {k}/{len(files)}")
    fdir = tempfile.mkdtemp(prefix=".frames_", dir=os.path.dirname(os.path.abspath(out)) or ".")
    first = _frame(dict(common, k=0, f=files[0], limits=limits, backend=backend, fdir=fdir, size=None))     # its size is every frame's
    size = first["size"]
    log("frame 1/%d" % len(files) + (f"  notes: {first['notes']}" if first["notes"] else ""))
    jobs = [dict(common, k=k, f=f, limits=limits, backend=backend, fdir=fdir, size=size) for k, f in enumerate(files) if k > 0]
    it = _pool_map(_frame, jobs, workers) if workers > 1 and len(jobs) > 1 else map(_frame, jobs)
    for k, _ in enumerate(it, start=2):
        log(f"frame {k}/{len(files)}")
    exe = ffmpeg or os.environ.get("SPLOSH_FFMPEG") or shutil.which("ffmpeg")
    if exe and not (os.path.isfile(exe) or shutil.which(exe)):
        exe = None                                        # a given ffmpeg that does not exist: as without one
    cmd = ffmpeg_command(exe or "ffmpeg", fps, os.path.join(fdir, "frame_%05d.png"), out)
    if not exe:
        keep = os.path.splitext(out)[0] + "_frames"
        shutil.rmtree(keep, ignore_errors=True)
        os.replace(fdir, keep)
        log(f"ffmpeg not found: {len(files)} frames in {keep}; encode with: " + " ".join(cmd).replace(fdir, keep))
        return {"encoded": False, "frames_dir": keep}
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        keep = os.path.splitext(out)[0] + "_frames"
        shutil.rmtree(keep, ignore_errors=True)
        os.replace(fdir, keep)
        raise SystemExit(f"ffmpeg failed (exit {r.returncode}): {r.stderr.strip()[:300]}; frames kept in {keep}")
    if keep_frames:
        keep = os.path.splitext(out)[0] + "_frames"
        shutil.rmtree(keep, ignore_errors=True)
        os.replace(fdir, keep)
    else:
        shutil.rmtree(fdir, ignore_errors=True)
    log(f"encoded {len(files)} frames to {out}")
    return {"encoded": True, "out": out, "frames": len(files), "limits": {k: list(v) for k, v in limits.items()}}


def main(a):
    from .mapstore import MapStore
    make(a.run, a.layout, a.out, start=a.start, stop=a.stop, every=a.every, fps=a.fps, backend=a.backend, analysis=a.analysis, config=a.config,
         store=MapStore(a.store) if a.store else None, dpi=a.dpi, keep_frames=a.keep_frames, log=lambda m: print(m, file=sys.stderr, flush=True),
         workers=getattr(a, "workers", None))
