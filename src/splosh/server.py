"""A small local server for the browser viewer: stdlib http.server, bound to localhost only, every request needs a random token (query `token=` or header
`X-Splosh-Token`), Host header restricted to localhost names. Designed for `ssh -L PORT:localhost:PORT login-node`: the browser talks to its own localhost.

    GET /?token=T                 the single-page viewer (viewer.html)
    GET /api/progress             the render in flight: stage (reading | rendering), fraction (bytes read; rendering: time over the estimate from the last renders), elapsed (the viewer's progress bar)
    GET /api/state                snapshots of the first run (index, name, time), run labels, fields, colour maps, default view, tile grid, store statistics
    GET /api/runs                 every run: label and snapshots (index, name, time, time_myr) (see runs.py)
    GET /api/series?run=R&radial=<panel JSON> | &spectrum=<panel JSON>   a layout radial / spectrum panel drawn as data (series.radial_tile / spectrum_tile; see ViewerState.series_tile)
    GET /api/align?mode=time&at=T[&tol=..&offset=a,b&prefer=i,j]   the snapshot of every run nearest to the aligned value T (time | time_myr | index | offset), within tol
    every snapshot endpoint below takes run=LABEL (or the run's number); the default is the first run, so single-run URLs are unchanged
    GET /api/map?snap=I&field=..  float32 map (little endian, npix*npix, origin lower-left), metadata in the X-Splosh-Meta header (JSON); gzip if accepted
    GET /api/tile?snap=I&level=L&ix=X&iy=Y[&tile=256&base=B&field=..]   one tile of the deep-zoom grid, same format as /api/map (see tile_response)
    GET /api/bh?snap=I&...        black-hole markers in the view coordinates of that map request
    GET /api/png?snap=I&...       the same map as a PNG (cmap, scale, vmin, vmax)
    GET /api/lut?name=viridis     256x3 colour table
    GET /api/browse?dir=REL[&deep=1]   folders under the browsable root (--root, default home) with their snapshot counts; deep: every folder below with snapshots
    GET /api/jobs, POST /api/jobs {kind: movie | prefetch | products {run, names?}, ...}, POST /api/jobs/cancel {id}, GET /api/jobs/file?id=&name=   long jobs (jobs.py): a free node at once
                                  (serve --slurm "srun options"), else here; their folders under --jobs-dir
    POST /api/open {paths: [REL, ...], add: false}   serve these folders as the runs (replacing the current ones, or adding); `serve` without paths starts empty
    GET /layout                   the layout editor page (layout.html); GET /api/layout/registry (what a panel can use, layouts in the directory), GET /api/layout?name=N (file as written)
    POST /api/threads {threads: N}   cores used for rendering from now on (1 .. the pool size NUMBA_NUM_THREADS fixed at start); /api/state reports threads and max_threads
    POST /api/layout/validate | /api/layout/save | /api/layout/preview   JSON bodies {layout} | {name, layout} | {layout, snap, run}: validated copy or error, write <layouts dir>/N.toml, PNG of the layout
                                  for that snapshot (X-Splosh-Notes: what the preview could not show); POST needs Content-Type: application/json and the token
    GET /overlay?snap=I&spec=vectors:v&spec=...&<view>   overlay items (plain data in window coordinates, see overlays.py) for the view
    GET /inspect?snap=I&x=..&y=..&limit=N&<view>        the gas particles contributing at the map position x, y (window coordinates), with their smoothing circles
    GET /api/phase?snap=I&x=Q&y=Q[&bins=256&xlog=1&ylog=1&xr=a,b&yr=c,d&weight=mass|count|volume]&<view>   2D histogram of the gas in the view's window
                                  (phase.py: quantities density, the per-particle fields, x y z R r h v los), same binary format as /api/map
    /api/map?...&sel={"x","y","xlog","ylog","poly":[[x,y],..]}   the map of only the gas inside that polygon of a phase diagram (not stored)
    GET /api/selection?snap=I&sel=..&<view>    the selected particles in the window (count, mass, up to 20000 points for an overlay)
Maps are rendered on request (one at a time), cached in the mapstore (disk) and the last few snapshots are kept in memory. Requests are quantised
(width to 4 significant digits, the window centre to a quarter pixel) so that panning reuses cache entries."""
import glob
import contextlib
import gzip
import hmac
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import h5py
import numpy as np

from . import colormaps, imaging, plugins, progress
from .io import load_snapshot
from .mapstore import MapStore, snapshot_hash, view_key
from .maps import View, bh_markers, field_names, render_map, required_gas_fields
from . import phase as _phase
from .frames import make_frame

VIEWER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
LAYOUT_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout.html")
BASE_FIELDS = ("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities")
PARTIAL_MIN = int(os.environ.get("SPLOSH_PARTIAL_MIN", "2000000"))     # serve --partial auto: region loads for snapshots with at least this many gas particles
_LOCAL_HOSTS = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find_snapshots(paths):
    """sorted list of snapshot files from files and directories (*.hdf5 / *.h5)."""
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "*.hdf5")) + glob.glob(os.path.join(p, "*.h5"))
        else:
            out.append(p)
    return sorted(dict.fromkeys(os.path.abspath(o) for o in out), key=natural_key)


class ViewerState:
    """`paths`: directories, files or `label=path` (grouped into runs by runs.group_runs: each directory / label is a run, loose files form one). `self.paths` / `self.times` are those of the
    first run (the single-run view of earlier versions); `self.runs` lists all runs. No paths: the server starts without runs and the page asks for a folder
    (`browse` / `open_runs`, limited to the directory tree under `root`, default the home directory)."""
    def __init__(self, paths, store=None, defaults=None, keep=3, token=None, layouts=None, analysis=None, config=None, threads=None, root=None, slurm=None, node_wait=10, jobs_dir=None, ffmpeg=None, partial="auto", auto_products=False):
        if partial not in ("auto", "on", "off"):
            raise ValueError("partial must be auto, on or off")
        self.partial = partial                   # region.py: read only the gas a zoomed view needs (auto: snapshots of >= PARTIAL_MIN gas particles)
        self._regions = OrderedDict()            # loaded regions: (path, n) -> (fields, snapshot dict with snap["region"])
        self.root = os.path.realpath(os.path.expanduser(root or "~"))
        from .jobs import JobManager
        self.ffmpeg = ffmpeg or os.environ.get("SPLOSH_FFMPEG") or shutil.which("ffmpeg")
        self.jobs = JobManager(jobs_dir or os.path.join(os.path.expanduser("~"), ".cache", "splosh", "jobs"), slurm=slurm, wait=node_wait,
                               local_threads=lambda: self.threads, env={"SPLOSH_FFMPEG": self.ffmpeg} if self.ffmpeg else {})
        self.runs = []
        self.auto_products = bool(auto_products)  # start a background `products` job for every opened run with snapshot products not computed yet
        paths = [paths] if isinstance(paths, str) else list(paths or [])
        if paths:
            self.runs = self._make_runs(paths)
            if not self.runs:
                raise SystemExit("no snapshots found")
        self.store = store or MapStore()
        self.defaults = defaults or View()
        self.token = token or secrets.token_urlsafe(18)
        self.keep = keep
        self._snaps = OrderedDict()
        self._selmaps = OrderedDict()          # maps of phase-diagram selections (not in the store)
        self._lock = threading.Lock()            # one render / load at a time (taken through work())
        self.layouts = os.path.abspath(os.path.expanduser(layouts or "layouts"))      # where the layout editor reads and writes layout files (~ expanded: the cockpit passes it quoted)
        self.analysis, self.config = analysis, config             # optional: an analysis directory (scalars.csv) and an analyze config (products) for the editor's preview
        self._preview_cache = {}                                  # profiles and auto radii of the editor's previews (layout_preview.preview)
        import numba
        self.max_threads = int(numba.config.NUMBA_NUM_THREADS)    # the pool size, fixed at start (NUMBA_NUM_THREADS); set_threads can go up to it
        self.threads = self.max_threads
        if threads:
            self.set_threads(threads)
        self.last_request = time.time()

    def set_threads(self, n):
        """cores used for rendering from the next request on (1 .. max_threads; numba cannot grow its pool after start)."""
        self.threads = max(1, min(int(n), self.max_threads))
        return self.threads

    @contextlib.contextmanager
    def work(self):
        """the render lock, with numba's thread count set for this request's thread (numba's setting is per thread, and every request has its own)."""
        import numba
        with self._lock:
            numba.set_num_threads(self.threads)
            yield

    @property
    def paths(self):
        return self.runs[0].paths if self.runs else []

    @property
    def times(self):
        return self.runs[0].times if self.runs else []

    @staticmethod
    def _make_runs(paths):
        from . import runs as R
        try:
            return [R.Run(label, ps) for label, ps in R.group_runs(paths)]
        except SystemExit as e:                                 # group_runs exits for a run without snapshots (right for the command line)
            raise ValueError(str(e)) from None

    # ---- choosing folders from the page
    def _inside(self, rel):
        """the real path of `rel` (relative to root, or absolute) if it lies inside root, else ValueError."""
        p = os.path.realpath(os.path.join(self.root, os.path.expanduser(rel or "")))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"{rel!r} is outside the browsable root {self.root}")
        return p

    def browse(self, rel="", deep=False, max_dirs=4000, seconds=4.0):
        """the folders under root/rel: {'dir', 'parent', 'n' (snapshots in dir itself), 'entries': [{'path', 'name', 'n', 'sub'}]}. Not deep: the direct
        subfolders (n = snapshot files directly inside, sub = has subfolders); deep: every folder below that holds snapshot files (hidden folders skipped,
        symlinks not followed; at most max_dirs folders or `seconds`, then 'truncated')."""
        d = self._inside(rel)
        if not os.path.isdir(d):
            raise ValueError(f"not a folder: {rel!r}")

        def count(path):
            try:
                return sum(1 for e in os.scandir(path) if e.is_file() and e.name.endswith((".hdf5", ".h5")))
            except OSError:
                return 0
        rel_of = lambda p: os.path.relpath(p, self.root) if p != self.root else ""
        out, truncated = [], False
        if not deep:
            try:
                subs = sorted((e for e in os.scandir(d) if e.is_dir() and not e.name.startswith(".")), key=lambda e: natural_key(e.name))
            except OSError as e:
                raise ValueError(f"cannot read {rel!r}: {e}") from None
            for e in subs:
                try:
                    sub = any(x.is_dir() and not x.name.startswith(".") for x in os.scandir(e.path))
                except OSError:
                    sub = False
                out.append({"path": rel_of(e.path), "name": e.name, "n": count(e.path), "sub": sub})
        else:
            t0, seen = time.time(), 0
            for top, dirs, files in os.walk(d, followlinks=False):
                dirs[:] = sorted((x for x in dirs if not x.startswith(".")), key=natural_key)
                seen += 1
                if seen > max_dirs or time.time() - t0 > seconds:
                    truncated = True
                    break
                n = sum(1 for f in files if f.endswith((".hdf5", ".h5")))
                if n and top != d:
                    out.append({"path": rel_of(top), "name": os.path.relpath(top, d), "n": n, "sub": bool(dirs)})
        return {"root": self.root, "dir": rel_of(d), "parent": None if d == self.root else rel_of(os.path.dirname(d)), "n": count(d), "entries": out, "truncated": truncated}

    def open_runs(self, rels, add=False):
        """replace (or, add=True, extend) the runs by these folders / files (relative to root): every folder is a run, as on the command line."""
        paths = [self._inside(r) for r in rels]
        if not paths:
            raise ValueError("nothing to open")
        new = self._make_runs(paths)
        remember_recent([os.path.realpath(p) for p in paths], add)
        with self.work():
            self.runs = (self.runs + new) if add else new
            self._snaps.clear()
            self.__dict__.pop("_tile_renderers", None)
            self._preview_cache.clear()
        self.start_auto_products()
        return [{"label": r.label, "n": len(r.paths)} for r in self.runs]

    # ---- snapshot products (products.py): computed in the background when a run is opened
    def products_status(self, run, names=None):
        """(names that apply to the run, number of (product, snapshot) results missing)."""
        from . import products as PR
        plugins.load_plugins()
        paths = self.runs[run].paths
        names = [n for n in (names or PR.default_names(self._rundir(run))) if n in PR.PRODUCTS and PR.applicable(n, paths)]
        return names, len(PR.missing(self._rundir(run), names, paths))

    def job_products(self, b):
        """compute a run's snapshot products in the background (`splosh products RUN`; the series sources prod_<name> fill in as it goes). One job per run at
        a time: a second request while one is queued or running returns that one."""
        run = self.run_index({"run": [str(b.get("run", ""))]})
        R, rundir = self.runs[run], self._rundir(run)
        for j in self.jobs.list():
            if j.get("kind") == "products" and j.get("rundir") == rundir and j.get("state") in ("starting", "queued", "running"):
                return {"jobs": [j], "note": "already computing"}
        names, n = self.products_status(run, b.get("names"))
        if not n:
            return {"jobs": [], "note": "every product is current" if names else "no snapshot product applies to this run"}
        job = self.jobs.submit("products", ["products", rundir, "--names", ",".join(names)],
                               label=f"{R.label}: {', '.join(names)} ({n} result(s) to compute)" + (" [automatic]" if b.get("auto") else ""))
        self.jobs.jobs[job["id"]]["rundir"] = rundir
        return {"jobs": [self.jobs.public(self.jobs.jobs[job["id"]])]}

    def start_auto_products(self):
        """with auto_products: a background thread checks every open run and starts the products job where results are missing (checking reads every stored
        result's key, a few seconds for a long run, so it does not hold up the request that opened the runs)."""
        if not self.auto_products:
            return None

        def go():
            for k in range(len(self.runs)):
                try:
                    self.job_products({"run": str(k), "auto": True})
                except Exception as e:                       # a run whose products cannot be checked must not stop the others
                    print(f"splosh: automatic products for run {k}: {type(e).__name__}: {e}", flush=True)
        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t

    # ---- long jobs (jobs.py): a movie of a pane, every snapshot of the panes into the store
    def _view_args(self, pane):
        view = self.view_from_query({k: [str(v)] for k, v in pane.items()})
        out = []
        for k, v in view.as_dict().items():
            if k == "exact":
                out += ["--exact"] if v else []
            elif v is not None:
                out += [f"--{k}", str(v)]
        return view, out

    def job_movie(self, b):
        pane = dict(b["pane"])
        run = self.run_index({"run": [str(pane.pop("run", ""))]})
        pane.pop("snap", None)
        paths = self.runs[run].paths
        i0, i1, every = int(b.get("from", 0)), int(b.get("to", len(paths) - 1)), max(1, int(b.get("every", 1)))
        files = paths[max(0, i0):min(len(paths) - 1, i1) + 1:every]
        if len(files) < 2:
            raise ValueError("a movie needs at least two snapshots")
        view, vargs = self._view_args(pane)
        lim = b.get("limits", "global")
        if lim not in ("global", "frame", "fixed"):
            raise ValueError("limits: global, frame or fixed")
        style = []
        for k in ("cmap", "scale"):
            if b.get(k):
                style += [f"--{k}", str(b[k])]
        if lim == "fixed":
            style += ["--vmin", str(float(b["vmin"])), "--vmax", str(float(b["vmax"]))]
        if b.get("orbits"):
            style += ["--orbits"]
        for spec in b.get("overlays") or []:
            style += ["--overlay", str(spec)]
        args = ["movie", *files, "--out", "{dir}/movie.mp4", "--fps", str(float(b.get("fps", 12))), "--limits", lim, "--verbose", "--store", self.store.root, *vargs, *style]
        label = f"{self.runs[run].label}: {view.field} {view.mode}, {len(files)} frames"
        return self.jobs.submit("movie", args, outputs=["movie.mp4", "movie.json"], label=label)

    def job_layout_movie(self, b):
        """a movie of a saved layout over a snapshot range of a run (layout_movie.py; publication renderer by default) as a long job."""
        f = self.layout_path(b.get("name") or "")
        if not os.path.isfile(f):
            raise ValueError(f"layout {b.get('name')!r} is not saved yet: save it first")
        run = self.run_index({"run": [str(b.get("run", ""))]})
        paths = self.runs[run].paths
        rundir = os.path.dirname(os.path.abspath(paths[0]))
        i0, i1, every = int(b.get("from", 0)), int(b.get("to", len(paths) - 1)), max(1, int(b.get("every", 1)))
        if len(paths[max(0, i0):min(len(paths) - 1, i1) + 1:every]) < 2:
            raise ValueError("a movie needs at least two snapshots")
        backend = b.get("backend", "mpl")
        if backend not in ("mpl", "bitmap"):
            raise ValueError("backend: mpl or bitmap")
        args = ["layout-movie", rundir, "--layout", f, "--out", "{dir}/movie.mp4", "--from", str(i0), "--to", str(i1), "--every", str(every),
                "--fps", str(float(b.get("fps", 12))), "--backend", backend, "--store", self.store.root]
        analysis = self.analysis or self.run_analysis(run)
        if analysis:
            args += ["--analysis", analysis]
        if self.config:
            args += ["--config", self.config]
        label = f"{self.runs[run].label}: layout {b.get('name')}, frames {i0}..{i1} every {every}"
        return self.jobs.submit("layout_movie", args, outputs=["movie.mp4"], label=label)

    def job_analyze(self, b):
        """an analyze pipeline of the panes' views, one job per run: a [[map]] per pane (its view and colours, global limits), a radial profile about the first pane's
        centre, the built-in scalars, and a dashboard layout (the maps, the profile, a time series), written to <run>/analysis_viewer/<name>/ (incremental: a second job only
        adds what is new). The layout is also put in the editor's layouts directory as <run>-<name>.json."""
        name = str(b.get("name") or "viewer")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", name):
            raise ValueError("name: letters, digits, _ and - only")
        byrun = {}
        for pane in b["panes"]:
            pane = dict(pane)
            run = self.run_index({"run": [str(pane.pop("run", ""))]})
            style = {k: pane.pop(k) for k in ("cmap", "scale") if k in pane}
            pane.pop("snap", None)
            view, _ = self._view_args(pane)
            byrun.setdefault(run, []).append((view, style))
        jobs = []
        for run, panes in byrun.items():
            R = self.runs[run]
            folder = os.path.dirname(R.paths[0])
            rundir = os.path.dirname(folder) if os.path.basename(folder) == "output" else folder
            out = os.path.join(rundir, "analysis_viewer", name)
            maps = []
            for k, (v, st) in enumerate(panes):
                d = {kk: vv for kk, vv in v.as_dict().items() if vv is not None}
                maps.append({"name": f"p{k + 1}_{v.field}", **d, **{kk: vv for kk, vv in st.items() if vv}, "limits": "global"})
            nc = min(3, max(1, len(maps)))
            layout = {"schema": "splosh-layout-1", "name": name, "title": f"{R.label}: viewer panes", "ncols": nc, "x_viewer_job": True,
                      "panel": [{"id": m["name"], "type": "map", "product": m["name"]} for m in maps] +
                               [{"id": "profile", "type": "profile", "product": "radial", "quantities": ["u"]},
                                {"id": "series", "type": "timeseries", "columns": ["gas_mass", "rho_max"], "series": "past", "colspan": max(1, nc - 1) if nc > 1 else 1, "ylog": True}]}
            cfg = {"map": maps, "profile": [{"name": "radial", "center": panes[0][0].center, "align": panes[0][0].align, "tilt": panes[0][0].tilt, "quantities": ["u"]}],
                   "scalar": [{"name": "builtin"}], "dashboard": [{"name": name, "layout": name}]}
            from . import analyze, layout as lay
            lay.validate_layout(layout)
            ed_name = f"{R.label}-{name}"
            ed_file = os.path.join(self.layouts, ed_name + ".json")
            note = ""
            if os.path.exists(ed_file) and not json.load(open(ed_file)).get("x_viewer_job"):
                note = f"{ed_file} exists and was not written by a viewer job: left alone"
            else:
                os.makedirs(self.layouts, exist_ok=True)
                with open(ed_file, "w") as f:
                    json.dump(layout, f, indent=1)
            os.makedirs(out, exist_ok=True)
            stem = os.path.splitext(os.path.basename(R.paths[-1]))[0]
            job = self.jobs.submit("analyze", ["analyze", rundir, "--config", "{dir}/config.json", "--out", out, "--jobs", "auto", "--store", self.store.root],
                                   outputs=["out/scalars.csv", f"out/{name}/{stem}.png", "out/digest.txt", "config.json"], label=f"{R.label}: {len(maps)} pane(s) -> {out}",
                                   write={"config.json": json.dumps(cfg, indent=1), f"layouts/{name}.json": json.dumps(layout, indent=1)})
            os.symlink(out, os.path.join(job["dir"], "out"))
            self.jobs.jobs[job["id"]]["warning"] = note
            job.update(warning=note, editor={"open": ed_name, "analysis": out, "config": os.path.join(job["dir"], "config.json"), "run": R.label})
            self.jobs.jobs[job["id"]]["editor"] = job["editor"]
            jobs.append(job)
        return {"jobs": jobs}

    def _rundir(self, run):
        folder = os.path.dirname(self.runs[run].paths[0])
        return os.path.dirname(folder) if os.path.basename(folder) == "output" else folder

    def series_context(self, run):
        from . import series as SR
        R = self.runs[run]
        try:
            period = R.period
        except Exception:                                    # no BH pair, unreadable header: no orbit time axis
            period = None
        return SR.context(self._rundir(run), R.paths, self.run_analysis(run), period)

    def series(self, run, q=None):
        """the viewer's time-series panes (GET /api/series): without `source`, the run's sources and its binary period P0 (code time; the t/P_orb axis);
        with `source`, its columns and ids (describe); with `columns` too, the curves (thinned to `n` buckets, `smooth` = running-mean window in code time,
        `ids`, `tmin`, `tmax`). The snapshot scalars of a run without an analysis table: {"missing": True} (the pane offers the background "series" job)."""
        from . import series as SR
        q = q or {}
        one = lambda k, d=None: q.get(k, [d])[0]
        ctx = self.series_context(run)
        label = self.runs[run].label
        for kind in ("radial", "spectrum"):
            if one(kind):
                with self.work():
                    return {"run": label, "kind": kind, "period": ctx["period"], **self.series_tile(kind, one(kind), ctx)}
        src = one("source")
        if not src:
            return {"run": label, "sources": SR.available(ctx), "period": ctx["period"]}
        if src == "snapshots" and not SR.plugins.SERIES["snapshots"]["files"](ctx):
            return {"missing": True, "run": label, "source": src, "n_snapshots": len(self.runs[run].paths)}
        with self.work():
            tab = SR.table(src, ctx)
            if one("heat"):                                   # a heatmap: the columns <heat><number> against time (series.heat_grid)
                ids = [int(x) for x in one("ids").split(",") if x] if one("ids") else None
                fl = lambda k: float(one(k)) if one(k) not in (None, "") else None
                edges, Y, Z = SR.heat_grid(tab, one("heat"), ids=ids, nx=max(10, min(2000, int(one("n", "600")))), tmin=fl("tmin"), tmax=fl("tmax"), norm=one("norm", "none"))
                Y, Z = SR.heat_rows(Y, Z)
                fin = lambda a: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]
                return {"run": label, "source": src, "x": [float(v) for v in edges], "y": [float(v) for v in Y], "z": fin(Z), "period": ctx["period"]}
            cols = [c for c in (one("columns") or "").split(",") if c]
            if not cols:
                return {"run": label, "source": src, "period": ctx["period"], **tab.describe(),
                        "dir": ctx["analysis"] if src == "snapshots" else ctx["run_dir"]}
            ids = [int(x) for x in one("ids").split(",") if x] if one("ids") else None
            fl = lambda k: float(one(k)) if one(k) not in (None, "") else None
            cv = SR.curves(tab, cols, ids=ids, n=max(50, min(20000, int(one("n", "4000")))), smooth=fl("smooth"), tmin=fl("tmin"), tmax=fl("tmax"))
        return {"run": label, "source": src, "curves": SR.jsonable(cv)}

    @staticmethod
    def series_tile(kind, text, ctx):
        """the viewer's `radial` / `spectrum` panes (GET /api/series?radial=<panel JSON> | spectrum=<panel JSON>): the panel's keys as in a layout (validated and
        completed by layout.validate_layout), drawn by series.radial_tile / spectrum_tile; JSON: curves [{label, x, y}], styles (band [lo, hi] as lists), xlog,
        ylog, yscale ("lin" | "log" | ["symlog", linthresh]), xlabel, ylabel, hlines, vlines, note; NaN as null."""
        from . import layout as LY, series as SR
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise ValueError(f"{kind}: the panel is not JSON ({e})") from None
        if not isinstance(raw, dict):
            raise ValueError(f"{kind}: the panel is a JSON object of layout keys")
        drop = {"id", "type", "row", "col", "rowspan", "colspan"}
        p = LY.validate_layout({"name": "viewer", "panel": [{"id": "p", "type": kind, **{k: v for k, v in raw.items() if k not in drop}}]})["panel"][0]
        t = (SR.radial_tile if kind == "radial" else SR.spectrum_tile)(p, ctx)
        fin = lambda a: [None if v is None or not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float).ravel()]
        styles = []
        for st in t.get("styles", []):
            s = dict(st)
            if s.get("band") is not None:
                s["band"] = [fin(s["band"][0]), fin(s["band"][1])]
            styles.append(s)
        ys = t.get("yscale")
        return {"curves": [{"label": str(lab), "x": fin(x), "y": fin(y)} for lab, x, y in t["curves"]], "styles": styles,
                "xlog": bool(t.get("xlog")), "ylog": bool(t.get("ylog")), "yscale": list(ys) if isinstance(ys, tuple) else (ys or ("log" if t.get("ylog") else "lin")),
                "xlabel": t.get("xlabel"), "ylabel": t.get("ylabel"), "hlines": t.get("hlines") or [], "vlines": t.get("vlines") or [], "note": t.get("note")}

    def job_series(self, b):
        """the built-in scalars of every snapshot of a run (a background analyze with only [[scalar]] builtin), to <run>/analysis_viewer/series/: what the
        viewer's time-series panes plot when the run has no analysis table yet (incremental: a second job only adds the new snapshots)."""
        run = self.run_index({"run": [str(b.get("run", ""))]})
        R = self.runs[run]
        out = os.path.join(self._rundir(run), "analysis_viewer", "series")
        os.makedirs(out, exist_ok=True)
        cfg = {"scalar": [{"name": "builtin"}]}
        job = self.jobs.submit("analyze", ["analyze", self._rundir(run), "--config", "{dir}/config.json", "--out", out, "--jobs", "auto", "--store", self.store.root],
                               outputs=["out/scalars.csv", "out/digest.txt"], label=f"{R.label}: time series -> {out}", write={"config.json": json.dumps(cfg, indent=1)})
        os.symlink(out, os.path.join(job["dir"], "out"))
        return {"jobs": [job]}

    def job_prefetch(self, b):
        plan = []
        for pane in b["panes"]:
            pane = dict(pane)
            run = self.run_index({"run": [str(pane.pop("run", ""))]})
            pane.pop("snap", None)
            view, _ = self._view_args(pane)
            plan.append({"files": self.runs[run].paths, "view": view.as_dict()})
        n = sum(len(x["files"]) for x in plan)
        return self.jobs.submit("prefetch", ["prefetch", "--plan", "{dir}/plan.json", "--store", self.store.root, "--procs", "auto"], label=f"{len(plan)} pane(s), {n} maps",
                                write={"plan.json": json.dumps(plan)})

    def run_index(self, q):
        """index of the run named by the query's `run=` (label or number); default 0. LookupError for an unknown run."""
        if not self.runs:
            raise LookupError("no runs open: choose a folder in the page (or give paths to `splosh serve`)")
        v = (q.get("run") or [None])[0]
        if v is None or v == "":
            return 0
        for r, run in enumerate(self.runs):
            if run.label == v:
                return r
        if v.isdigit() and int(v) < len(self.runs):
            return int(v)
        raise LookupError(f"no such run: {v!r}; runs: {[r.label for r in self.runs]}")

    def path(self, run, i):
        return self.runs[run].paths[i]

    def snap(self, i, view, run=0):
        """loaded snapshot dict for index i of a run (LRU); reloads with more fields if the view needs a dataset that was not loaded."""
        return self.snap_fields(i, required_gas_fields(view), run)

    def snap_fields(self, i, fields, run=0):
        need = set(BASE_FIELDS) | set(fields)
        key = self.path(run, i)
        got = self._snaps.get(key)
        if got is not None and need <= got[0]:
            self._snaps.move_to_end(key)
            return got[1]
        s = load_snapshot(key, gas_fields=tuple(sorted(need)))
        self._snaps[key] = (need, s)
        while len(self._snaps) > self.keep:
            self._snaps.popitem(last=False)
        return s

    def snap_for(self, i, view, run=0, margin=0.5):
        """the snapshot dict to render `view` from: a loaded region (region.py) when partial loading applies and pays off, else the whole snapshot."""
        if self.partial == "off":
            return self.snap(i, view, run)
        from . import region
        path = self.path(run, i)
        need = set(BASE_FIELDS) | set(required_gas_fields(view))
        got = self._snaps.get(path)
        if got is not None and need <= got[0]:                  # the whole snapshot is already in memory: nothing to gain
            self._snaps.move_to_end(path)
            return got[1]
        for k, (fields, snap) in self._regions.items():
            if k[0] == path and need <= fields and region.covers(snap["region"], view):
                self._regions.move_to_end(k)
                return snap
        snap = region.load_region(path, view, tuple(sorted(need)), root=self.store.root, margin=margin, max_fraction=0.5,
                                  min_particles=PARTIAL_MIN if self.partial == "auto" else 0)
        if snap is None:
            return self.snap(i, view, run)
        self._nregion = getattr(self, "_nregion", 0) + 1
        self._regions[(path, self._nregion)] = (need, snap)
        while len(self._regions) > 4:
            self._regions.popitem(last=False)
        return snap

    def view_from_query(self, q):
        d = self.defaults.as_dict()
        for k, typ in (("center", str), ("align", str), ("tilt", str), ("axes", str), ("mode", str), ("field", str), ("weight", str), ("npix", int), ("width", float),
                       ("cx", float), ("cy", float), ("z0", float), ("hscale", float), ("units", str), ("gamma", float), ("xh", float), ("mu", float), ("bunit", float), ("post", str), ("kappa", float), ("exact", lambda v: v in ("1", "true", "True"))):
            if k in q:
                d[k] = typ(q[k][0])
        # quantise so that panning and zooming revisit cache entries
        d["width"] = float(f"{d['width']:.4g}")
        pix = d["width"] / d["npix"]
        step = pix / 4
        d["cx"] = round(d["cx"] / step) * step
        d["cy"] = round(d["cy"] / step) * step
        d["z0"] = round(d["z0"] / step) * step
        return View(**d).validate()

    def get_map(self, i, view, run=0, sel=None):
        if sel:
            return self.selection_map(i, view, sel, run)
        with self.work():
            path = self.path(run, i)
            sig = snapshot_hash(path)
            key = view_key(sig, view)
            got = self.store.get(key)
            if got is not None:
                self.store.hits += 1
                return got[0], dict(got[1], cached=True)
            t0 = time.perf_counter()
            progress.begin(f"{os.path.basename(path)} {view.field} {view.mode} {view.npix}^2")
            try:
                progress.stage("reading")
                snap = self.snap_for(i, view, run)
                n = len(snap["Masses"]) if "Masses" in snap else 0
                t1 = time.perf_counter()
                progress.stage("rendering", progress.expect("render", n))
                img, meta, _ = self.store.get_or_render(path, view, snap=snap)
                progress.learn("render", n, time.perf_counter() - t1)
            finally:
                progress.end()
            if "region" in snap:
                meta = dict(meta, region={k: snap["region"][k] for k in ("rows", "n", "fraction", "runs")})
            meta = dict(meta, cached=False, total_seconds=round(time.perf_counter() - t0, 4))
            return img, meta

    def _selection(self, i, view, sel, run, window):
        from . import phase
        crit = phase.parse_selection(sel)
        snap = self.snap_fields(i, phase.needs(view, crit["x"], crit["y"]), run)
        return snap, phase.select(snap, view, crit["x"], crit["y"], crit["poly"], crit["xlog"], crit["ylog"], window=window), crit

    def selection_map(self, i, view, sel, run=0):
        """the map of only the gas selected on a phase diagram (not stored: a selection is a question of the moment)."""
        from . import phase
        key = (self.path(run, i), json.dumps(view.as_dict(), sort_keys=True), sel)
        if key in self._selmaps:
            self._selmaps.move_to_end(key)
            return self._selmaps[key]
        with self.work():
            t0 = time.perf_counter()
            snap, mask, crit = self._selection(i, view, sel, run, window=False)
            img, meta = render_map(phase.subset(snap, mask), view)
            meta = dict(meta, cached=False, render_seconds=round(time.perf_counter() - t0, 4), selection={"n_selected": int(mask.sum()), **crit})
        self._selmaps[key] = (img, meta)
        while len(self._selmaps) > 16:
            self._selmaps.popitem(last=False)
        return img, meta

    def phase(self, i, view, q, run=0):
        from . import phase
        xq, yq = q.get("x", ["density"])[0], q.get("y", ["T"])[0]
        rng = lambda k: [float(t) for t in q[k][0].split(",")] if k in q and q[k][0] else None
        flag = lambda k: q.get(k, ["1"])[0] not in ("0", "false", "False")
        with self.work():
            snap = self.snap_fields(i, phase.needs(view, xq, yq), run)
            return phase.histogram(snap, view, xq, yq, int(q.get("bins", ["256"])[0]), flag("xlog"), flag("ylog"), rng("xr"), rng("yr"), q.get("weight", ["mass"])[0])

    def selection_points(self, i, view, sel, run=0):
        from . import phase
        with self.work():
            snap, mask, crit = self._selection(i, view, sel, run, window=False)
            return dict(phase.selected_points(snap, view, mask), selection=crit)

    def overlay(self, i, view, specs, run=0):
        with self.work():
            from . import overlays
            gas, types = overlays.required_data(specs, view)
            path = self.path(run, i)
            if types:                                   # particle types other than the black holes: the overlay loader reads them
                snap = overlays.load_for(path, view, specs)
            else:                                       # the snapshot the maps use (already in memory when the pane was just drawn), plus what the overlays need
                snap = self.snap_fields(i, set(required_gas_fields(view)) | {"?" + k for k in gas - set(required_gas_fields(view))}, run)
            get_map = lambda v: self.store.get_or_render(path, v, snap=snap)[:2]      # contours: the pane's own map from the store, not a second render
            return dict(overlays.compute(snap, view, specs, get_map=get_map), time=self.runs[run].times[i])

    def inspect(self, i, view, x, y, limit=20, run=0):
        with self.work():
            from . import overlays
            return overlays.inspect(overlays.load_for(self.path(run, i), view), view, x, y, limit=limit)

    # ---- layout editor
    def layout_path(self, name):
        """The file of layout `name` in the layouts directory (.toml, else .json; a new name gives .toml). ValueError for a name that is not a plain file stem."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", name or "") or name.endswith((".toml", ".json")):
            raise ValueError(f"layout name {name!r}: letters, digits, _ . - only (no extension)")
        for ext in (".toml", ".json"):
            f = os.path.join(self.layouts, name + ext)
            if os.path.isfile(f):
                return f
        return os.path.join(self.layouts, name + ".toml")

    def layout_list(self):
        out = []
        for ext in (".toml", ".json"):
            for f in sorted(glob.glob(os.path.join(self.layouts, "*" + ext))):
                out.append({"name": os.path.basename(f)[:-len(ext)], "file": os.path.basename(f)})
        return out

    def _readable(self, path):
        """a file or folder the page may name for the editor's preview: inside the browsable root or the jobs folder."""
        p = os.path.realpath(os.path.expanduser(path))
        if not any(p == r or p.startswith(r + os.sep) for r in (self.root, os.path.realpath(self.jobs.root))):
            raise ValueError(f"{path!r} is outside {self.root} and the jobs folder")
        return p

    def run_analysis(self, run):
        """the newest analysis table of a run opened from the page (no --analysis): <run>/analysis/scalars.csv (splosh analyze) or
        <run>/analysis_viewer/<name>/scalars.csv ("analyze these panes"); None when the run has none."""
        from .analyze import run_root
        try:
            root = run_root(os.path.dirname(os.path.abspath(self.runs[run].paths[0])))      # <run>/output/snap -> <run>
        except (IndexError, OSError, ValueError):
            return None
        cands = [f for f in [os.path.join(root, "analysis", "scalars.csv")] + glob.glob(os.path.join(root, "analysis_viewer", "*", "scalars.csv")) if os.path.isfile(f)]
        return os.path.dirname(max(cands, key=os.path.getmtime)) if cands else None

    def layout_preview(self, layout, run, i, analysis=None, config=None, backend="bitmap", fmt="png", dpi=None):
        """(rgb, notes) of a layout for snapshot i of a run: the analyze config and scalars.csv are read again on every call (edits show at the next preview); a broken config is a note, not a failure."""
        from . import layout as lay, layout_preview as lp
        products, table, notes = None, None, {}
        config = self._readable(config) if config else self.config
        analysis = self._readable(analysis) if analysis else self.analysis or self.run_analysis(run)
        if config:
            from . import analyze
            try:
                products = analyze.normalize_config(analyze.read_config(config), os.path.dirname(os.path.abspath(config)))
            except (OSError, ValueError) as e:
                notes["(config)"] = f"{config}: {e}"
        if analysis:
            table = lp.read_scalars_table(analysis)
        layout = lay.apply_format(layout, lay.format_dirs(self.layouts))      # a layout's format (formats/NAME.toml) under its own [figure]
        with self.work():
            args = (layout, self.path(run, i), lambda fields: self.snap_fields(i, fields, run), self.store, products, table, self.runs[run].times[i], self._preview_cache)
            if backend == "mpl":                         # the publication renderer: PNG for the preview, or the exported PDF / SVG / PNG
                from . import layout_mpl
                lo, tiles, more, header = lp.tiles_for(*args)
                prob = layout_mpl.style_problem((lo.get("figure") or lay.figure_options(lo))["style"])
                if prob:
                    notes["(figure)"] = prob                  # shown in the editor, not a silent fallback
                return layout_mpl.render_bytes(lo, tiles, header, fmt=fmt, dpi=dpi), {**notes, **more}
            rgb, more = lp.preview(*args)
        return rgb, {**notes, **more}

    def markers(self, i, view, run=0):
        with self.work():
            snap = self.snap(i, view, run)
            return {"time": snap["time"], "markers": bh_markers(snap, view)}


RECENT_FILE = os.environ.get("SPLOSH_RECENT") or os.path.join(os.path.expanduser("~"), ".local", "state", "splosh", "recent.json")
RECENT_MAX = 12


def recent_sets():
    """the folder sets opened most recently (newest first): [{"paths": [...], "when": "YYYY-mm-dd HH:MM"}], kept on the server's machine so that
    they survive the page's origin changing (a new port every start)."""
    try:
        with open(RECENT_FILE) as f:
            out = json.load(f)
        return [r for r in out if isinstance(r, dict) and r.get("paths")]
    except (OSError, ValueError):
        return []


def remember_recent(paths, add=False):
    paths = [p for p in paths if p]
    if not paths:
        return
    cur = recent_sets()
    if add and cur:                                   # "add as runs": the newest set grows
        paths = cur[0]["paths"] + [p for p in paths if p not in cur[0]["paths"]]
        cur = cur[1:]
    cur = [r for r in cur if r["paths"] != paths]
    cur.insert(0, {"paths": paths, "when": time.strftime("%Y-%m-%d %H:%M")})
    try:
        os.makedirs(os.path.dirname(RECENT_FILE), exist_ok=True)
        tmp = RECENT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur[:RECENT_MAX], f, indent=1)
        os.replace(tmp, RECENT_FILE)
    except OSError:
        pass


def available_fields(state):
    """the fields the open runs can show: those whose PartType0 datasets are all in the first snapshot of at least one run (B and beta need
    MagneticField, divB needs DivergenceOfMagneticField, ...). Every field when no run is open or a snapshot cannot be read."""
    from . import datasets
    for r in state.runs:                                    # the runs' own datasets become fields too (WindFraction, TimeStep, Metallicity_0, ...)
        if r.paths:
            datasets.from_file(r.paths[0])
    names = field_names()
    present = set()
    for r in state.runs:
        if not r.paths:
            continue
        try:
            with h5py.File(r.paths[0], "r") as f:
                present |= set(f["PartType0"].keys()) if "PartType0" in f else set()
        except (OSError, KeyError):
            return names
    if not present:
        return names
    out = []
    for n in names:
        try:
            need = {k for k in required_gas_fields(View(field=n, center="xyz:0,0,0")) if not k.startswith("?")}
        except (ValueError, KeyError):
            need = set()
        if need <= present:
            out.append(n)
    return out


def _builtin_overlays():
    from .overlays import BUILTIN_OVERLAYS      # (the page's own checkboxes: not listed as plugin overlays)
    return BUILTIN_OVERLAYS


def _finite(x):
    """None for NaN/inf (a snapshot whose header could not be read): json.dumps would write NaN, which browsers reject as invalid JSON."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None if isinstance(x, float) else x


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "splosh"

        def log_message(self, fmt, *a):
            if os.environ.get("SPLOSH_VERBOSE"):
                super().log_message(fmt, *a)

        def _send(self, code, body, ctype="text/plain; charset=utf-8", headers=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            if "gzip" in self.headers.get("Accept-Encoding", "") and ctype.startswith("application/octet") and len(body) > 4096:
                body = gzip.compress(body, 1)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self, q):
            if not _LOCAL_HOSTS.match(self.headers.get("Host", "")):
                return False
            tok = (q.get("token") or [self.headers.get("X-Splosh-Token", "") or self.headers.get("X-Astroview-Token", "")])[0]     # the old header name too
            return hmac.compare_digest(tok.encode(), state.token.encode())

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            state.last_request = time.time()
            if not self._authorised(q):
                return self._send(403, "forbidden: token required (open the URL printed by `splosh serve`)")
            try:
                if u.path == "/":
                    with open(VIEWER, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if u.path == "/layout":
                    with open(LAYOUT_PAGE, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if u.path == "/api/layout/registry":
                    from . import layout as lay
                    return self._send(200, json.dumps(dict(lay.registry(), layouts=state.layout_list(), formats=lay.list_formats(state.layouts), dir=state.layouts, has_table=bool(state.analysis) or any(state.run_analysis(r) for r in range(len(state.runs))), has_config=bool(state.config))), "application/json")
                if u.path == "/api/layout":
                    from . import layout as lay
                    f = state.layout_path(q.get("name", [""])[0])
                    if not os.path.isfile(f):
                        return self._send(404, "no such layout")
                    return self._send(200, json.dumps({"raw": lay.migrate(lay._read(f)), "file": os.path.basename(f)}), "application/json")
                if u.path == "/api/recent":                   # the recently opened folder sets (relative to the root when inside it)
                    out = []
                    for r in recent_sets():
                        rel = [os.path.relpath(p, state.root) for p in r["paths"] if p == state.root or p.startswith(state.root + os.sep)]
                        if len(rel) == len(r["paths"]):
                            out.append({"paths": rel, "when": r.get("when", ""), "exists": all(os.path.exists(p) for p in r["paths"])})
                    return self._send(200, json.dumps({"recent": out, "root": state.root}), "application/json")
                if u.path == "/api/progress":                 # what the render in flight is doing (the viewer's progress bar); answers while a render runs
                    return self._send(200, json.dumps(progress.current()), "application/json")
                if u.path == "/api/state":
                    plugins.load_plugins()
                    return self._send(200, json.dumps({
                        "snapshots": [{"i": i, "name": os.path.basename(p), "time": _finite(t)} for i, (p, t) in enumerate(zip(state.paths, state.times))],
                        "runs": [{"run": r, "label": run.label, "n": len(run.paths)} for r, run in enumerate(state.runs)],
                        "fields": available_fields(state), "phase_quantities": _phase.quantities(), "colormaps": colormaps.names(), "defaults": state.defaults.as_dict(), "store": state.store.stats(),
                        "presets": sorted(plugins.PRESETS), "tile": tile_info(state),
                        "plugin_overlays": [{"name": n, "label": ((fn.__doc__ or n).strip().splitlines() or [n])[0][:80],
                                             "options": [{"key": k, "label": lab, "default": d} for k, lab, d in plugins.OVERLAY_OPTIONS.get(n, [])]}
                                            for n, fn in sorted(plugins.OVERLAYS.items()) if n not in _builtin_overlays()],
                        "threads": state.threads, "max_threads": state.max_threads, "root": state.root}), "application/json")
                if u.path == "/api/jobs":
                    return self._send(200, json.dumps({"jobs": state.jobs.list(), "slurm": " ".join(state.jobs.slurm), "ffmpeg": bool(state.ffmpeg)}), "application/json")
                if u.path == "/api/jobs/file":
                    f = state.jobs.path(q.get("id", [""])[0], q.get("name", [""])[0])
                    ctype = {".mp4": "video/mp4", ".json": "application/json", ".png": "image/png"}.get(os.path.splitext(f)[1], "text/plain; charset=utf-8")
                    name = os.path.basename(f)
                    want = re.sub(r"[^A-Za-z0-9_.-]+", "_", q.get("as", [""])[0])[:120]      # the name the user chose for the download (same extension only)
                    if want and os.path.splitext(want)[1].lower() == os.path.splitext(name)[1].lower():
                        name = want
                    elif want and not os.path.splitext(want)[1]:
                        name = want + os.path.splitext(name)[1]
                    with open(f, "rb") as fh:
                        return self._send(200, fh.read(), ctype, {"Content-Disposition": f'attachment; filename="{name}"'})
                if u.path == "/api/browse":
                    return self._send(200, json.dumps(state.browse(q.get("dir", [""])[0], q.get("deep", ["0"])[0] in ("1", "true"))), "application/json")
                if u.path == "/api/runs":
                    from . import runs as R
                    return self._send(200, json.dumps({"runs": [{"run": r, "label": run.label, "n": len(run.paths), "time_myr": run.has_myr, "period": run.period,
                                                                  "snapshots": [run.describe(i) for i in range(len(run.paths))]} for r, run in enumerate(state.runs)],
                                                       "aligns": list(R.ALIGNS)}), "application/json")
                if u.path == "/api/series":
                    return self._send(200, json.dumps(state.series(state.run_index(q), q)), "application/json")
                if u.path == "/api/align":
                    return self._send(200, json.dumps(align_response(state, q)), "application/json")
                if u.path == "/api/lut":
                    return self._send(200, json.dumps(colormaps.lut(q.get("name", ["viridis"])[0]).tolist()), "application/json")
                if u.path in ("/api/map", "/api/png", "/api/bh", "/overlay", "/inspect", "/api/tile", "/api/phase", "/api/selection"):
                    try:
                        run = state.run_index(q)
                    except LookupError as e:
                        return self._send(404, str(e))
                if u.path in ("/api/map", "/api/png", "/api/bh"):
                    i = int(q.get("snap", ["0"])[0])
                    if not 0 <= i < len(state.runs[run].paths):
                        return self._send(404, "no such snapshot")
                    view = state.view_from_query(q)
                    if u.path == "/api/bh":
                        return self._send(200, json.dumps(state.markers(i, view, run)), "application/json")
                    img, meta = state.get_map(i, view, run, sel=q.get("sel", [None])[0])
                    meta = {k: v for k, v in meta.items() if k not in ("units",)}
                    if u.path == "/api/png":
                        from .figures import resolve_style
                        cmap, scale, sym = resolve_style(view, q.get("cmap", [None])[0], q.get("scale", [None])[0])      # a transform has its own default colours
                        lo, hi = imaging.limits(img, scale=scale, symmetric=sym)
                        vmin = float(q["vmin"][0]) if "vmin" in q else lo
                        vmax = float(q["vmax"][0]) if "vmax" in q else hi
                        rgb, _ = imaging.colorize(img, cmap, scale, vmin, vmax)
                        return self._send(200, imaging.png_bytes(rgb), "image/png")
                    fin = img[np.isfinite(img)]
                    meta["n"] = int(img.shape[0])
                    meta["stats"] = {"min": float(fin.min()), "max": float(fin.max())} if fin.size else {}
                    return self._send(200, np.ascontiguousarray(img, dtype="<f4").tobytes(), "application/octet-stream",
                                      {"X-Splosh-Meta": json.dumps(meta, ensure_ascii=True), "Access-Control-Expose-Headers": "X-Splosh-Meta"})
                if u.path in ("/api/phase", "/api/selection"):
                    i = int(q.get("snap", ["0"])[0])
                    if not 0 <= i < len(state.runs[run].paths):
                        return self._send(404, "no such snapshot")
                    view = state.view_from_query(q)
                    if u.path == "/api/selection":
                        return self._send(200, json.dumps(state.selection_points(i, view, q["sel"][0], run)), "application/json")
                    grid, meta = state.phase(i, view, q, run)
                    return self._send(200, np.ascontiguousarray(grid, dtype="<f4").tobytes(), "application/octet-stream",
                                      {"X-Splosh-Meta": json.dumps(meta, ensure_ascii=True), "Access-Control-Expose-Headers": "X-Splosh-Meta"})
                if u.path in ("/overlay", "/inspect"):
                    i = int(q.get("snap", ["0"])[0])
                    if not 0 <= i < len(state.runs[run].paths):
                        return self._send(404, "no such snapshot")
                    view = state.view_from_query(q)
                    if u.path == "/overlay":
                        return self._send(200, json.dumps(state.overlay(i, view, q.get("spec", []), run)), "application/json")
                    return self._send(200, json.dumps(state.inspect(i, view, float(q["x"][0]), float(q["y"][0]), int(q.get("limit", ["20"])[0]), run)), "application/json")
                if u.path == "/api/tile":
                    body, hdr = tile_response(state, q, run)
                    return self._send(200, body, "application/octet-stream", hdr)
                return self._send(404, "not found")
            except (ValueError, KeyError) as e:
                return self._send(400, f"bad request: {e}")
            except LookupError as e:
                return self._send(404, str(e))
            except Exception as e:                                  # pragma: no cover
                return self._send(500, f"error: {e!r}")

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            state.last_request = time.time()
            if not self._authorised(q):
                return self._send(403, "forbidden: token required (open the URL printed by `splosh serve`)")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, "POST needs Content-Type: application/json")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 2_000_000:
                    return self._send(413, "body too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/jobs":
                    kind = body.get("kind")
                    if kind == "movie":
                        return self._send(200, json.dumps(state.job_movie(body)), "application/json")
                    if kind == "analyze":
                        return self._send(200, json.dumps(state.job_analyze(body)), "application/json")
                    if kind == "prefetch":
                        return self._send(200, json.dumps(state.job_prefetch(body)), "application/json")
                    if kind == "series":
                        return self._send(200, json.dumps(state.job_series(body)), "application/json")
                    if kind == "layout_movie":
                        return self._send(200, json.dumps(state.job_layout_movie(body)), "application/json")
                    if kind == "products":
                        return self._send(200, json.dumps(state.job_products(body)), "application/json")
                    raise ValueError("kind: movie, layout_movie, analyze, series, products or prefetch")
                if u.path == "/api/jobs/cancel":
                    return self._send(200, json.dumps(state.jobs.cancel(body["id"])), "application/json")
                if u.path == "/api/open":
                    return self._send(200, json.dumps({"runs": state.open_runs(list(body["paths"]), bool(body.get("add")))}), "application/json")
                if u.path == "/api/threads":
                    return self._send(200, json.dumps({"threads": state.set_threads(body["threads"]), "max_threads": state.max_threads}), "application/json")
                from . import layout as lay
                if u.path == "/api/layout/validate":
                    try:
                        lay.validate_layout(lay.apply_format(body["layout"], lay.format_dirs(state.layouts)))      # its format exists and fits
                        return self._send(200, json.dumps({"ok": True, "layout": lay.validate_layout(body["layout"])}), "application/json")
                    except lay.LayoutError as e:
                        return self._send(200, json.dumps({"ok": False, "error": str(e)}), "application/json")
                if u.path == "/api/layout/save":
                    f = state.layout_path(body.get("name"))
                    layout = body["layout"] if body["layout"].get("name") else dict(body["layout"], name=body["name"])
                    lay.save_layout(layout, f, completed=False)
                    return self._send(200, json.dumps({"ok": True, "file": os.path.basename(f), "dir": state.layouts}), "application/json")
                if u.path == "/api/layout/addpane":           # the viewer's "send this pane to a layout": a map panel with the pane's view, colours and overlays
                    name = body.get("name") or ""
                    f = state.layout_path(name)
                    L = lay.load_layout(f) if os.path.isfile(f) else {"schema": lay.SCHEMA, "name": name, "ncols": 2, "panel": []}
                    pane = {k: v for k, v in (body.get("pane") or {}).items() if k not in ("snap", "run", "sel", "token")}
                    view = state.view_from_query({k: [str(v)] for k, v in pane.items()}).as_dict()
                    ids = {p["id"] for p in L.get("panel", [])}
                    pid = re.sub(r"[^A-Za-z0-9_]", "_", str(body.get("label") or view["field"]))[:30] or "map"
                    k, base = 1, pid
                    while pid in ids:
                        k += 1
                        pid = f"{base}_{k}"
                    panel = {"id": pid, "type": "map", "label": body.get("label") or view["field"], **{k2: v for k2, v in view.items() if v is not None},
                             "cmap": body.get("cmap") or "viridis", "scale": str(body.get("scale") or "log"), "overlays": list(body.get("overlays") or [])}
                    if body.get("vmin") is not None and body.get("vmax") is not None:
                        panel.update(limits="fixed", vmin=float(body["vmin"]), vmax=float(body["vmax"]))
                    else:
                        panel["limits"] = "frame"
                    L = dict(L, panel=list(L.get("panel", [])) + [panel])
                    lay.validate_layout(L)                       # nothing invalid is written
                    lay.save_layout(L, f, completed=False)
                    return self._send(200, json.dumps({"ok": True, "name": name, "panel": pid, "file": os.path.basename(f)}), "application/json")
                if u.path == "/api/layout/preview":
                    from . import imaging
                    run = state.run_index({"run": [str(body.get("run", ""))]})
                    i = int(body.get("snap", 0))
                    if not 0 <= i < len(state.runs[run].paths):
                        return self._send(404, "no such snapshot")
                    hdr = lambda notes: {"X-Splosh-Notes": json.dumps(notes, ensure_ascii=True), "Access-Control-Expose-Headers": "X-Splosh-Notes"}
                    export = body.get("export")
                    if export:                                  # the publication figure as a file (pdf | svg | png at the layout's dpi)
                        if export not in ("pdf", "svg", "png"):
                            return self._send(400, "export: pdf, svg or png")
                        data, notes = state.layout_preview(body["layout"], run, i, analysis=body.get("analysis"), config=body.get("config"), backend="mpl", fmt=export)
                        stem = os.path.splitext(os.path.basename(state.path(run, i)))[0]
                        name = f"{(body['layout'].get('name') or 'layout')}_{stem}.{export}"
                        ctype = {"pdf": "application/pdf", "svg": "image/svg+xml", "png": "image/png"}[export]
                        return self._send(200, data, ctype, {**hdr(notes), "Content-Disposition": f'attachment; filename="{name}"'})
                    if body.get("backend") == "mpl":
                        data, notes = state.layout_preview(body["layout"], run, i, analysis=body.get("analysis"), config=body.get("config"), backend="mpl", dpi=90)
                        return self._send(200, data, "image/png", hdr(notes))
                    rgb, notes = state.layout_preview(body["layout"], run, i, analysis=body.get("analysis"), config=body.get("config"))
                    return self._send(200, imaging.png_bytes(rgb), "image/png", hdr(notes))
                return self._send(404, "not found")
            except (ValueError, KeyError, TypeError, LookupError) as e:          # LayoutError is a ValueError
                return self._send(400, f"bad request: {e}" if isinstance(e, ValueError) else f"bad request: {type(e).__name__}: {e}")
            except Exception as e:                                  # pragma: no cover
                return self._send(500, f"error: {e!r}")

    return Handler


_TILE_RENDERERS = 4          # TileRenderers kept per server (frame + particle binning of one snapshot / view / level)


def tile_info(state):
    """what a client needs to address the tile grid: the base width the server uses by default (the default width rounded up to a power of two) and the default tile size."""
    from . import tiles as T
    return {"base": T.default_base_width(state.defaults.width), "size": T.DEFAULT_TILE}


def align_response(state, q):
    """GET /api/align?mode=time|time_myr|index|offset&at=T[&tol=X][&offset=o1,o2,..][&prefer=i1,i2,..]: for every run the snapshot nearest to the aligned value T (see runs.align)."""
    from . import runs as R
    mode = q.get("mode", ["time"])[0]
    at = float(q["at"][0])
    tol = float(q["tol"][0]) if q.get("tol", [""])[0] != "" else None
    offsets = [float(x) for x in q.get("offset", [""])[0].split(",") if x != ""] or None
    prefer = [int(x) if x != "" else None for x in q.get("prefer", [""])[0].split(",")] if q.get("prefer", [""])[0] != "" else None
    return {"mode": mode, "at": at, "runs": R.align(state.runs, mode, at, offsets, tol, prefer)}


def _render_tile(state, T, paths, i, view, grid, tile, level, ix, iy, base, run):
    """a tile that is not in the store: load (or reuse) the region around it and render it (under the render lock)."""
    progress.stage("reading")
    x0, x1, y0, y1 = grid.extent(ix, iy)
    snap = state.snap_for(i, view.with_(cx=0.5 * (x0 + x1), cy=0.5 * (y0 + y1), width=float(f"{x1 - x0:.12g}")), run, margin=2.0)   # a region holding this tile and two rings of neighbours
    cache = state.__dict__.setdefault("_tile_renderers", OrderedDict())
    rk = (paths[i], id(snap), tuple(sorted((k, v) for k, v in view.as_dict().items() if k not in ("cx", "cy", "width", "npix"))), tile, level, base)
    rend = cache.get(rk)
    if rend is None:
        rend = cache[rk] = T.TileRenderer(snap, view, grid)
        while len(cache) > _TILE_RENDERERS:
            cache.popitem(last=False)
    cache.move_to_end(rk)
    n = len(snap["Masses"]) if "Masses" in snap else 0
    t1 = time.perf_counter()
    progress.stage("rendering", progress.expect("tile", n))
    img, meta = T.render_tile(snap, view, level, ix, iy, tile=tile, base_width=base, store=state.store, renderer=rend)
    progress.learn("tile", n, time.perf_counter() - t1)
    return img, meta


def tile_response(state, q, run=0):
    """(body, headers) of GET /api/tile?snap=I&level=L&ix=X&iy=Y[&tile=256][&base=B] plus the usual view options (field, mode, axes, center, ...; the window options
    cx, cy, width, npix are ignored: the tile grid replaces them). A tile is tile x tile float32 values, little endian, row 0 = the LOWEST y (as /api/map), in
    the global grid of tiles.py: tile (ix, iy) at level L covers [ix tw, (ix + 1) tw) x [iy tw, (iy + 1) tw) in the frame coordinates of the view, tw = base / 2^L,
    base = `base` (default: the server's default width rounded up to a power of two). X-Splosh-Meta is the map metadata plus tile = {tile, level, ix, iy, extent,
    tile_width, base_width}. Raises ValueError / KeyError (400) for a missing or invalid parameter. The token rules are those of every other endpoint."""
    from . import tiles as T
    try:
        i, level, ix, iy = (int(q[k][0]) for k in ("snap", "level", "ix", "iy"))
    except KeyError as e:
        raise KeyError(f"missing parameter {e}") from None
    tile = int(q.get("tile", [T.DEFAULT_TILE])[0])
    paths = state.runs[run].paths
    if not 0 <= i < len(paths):
        raise ValueError("no such snapshot")
    if abs(level) > 40 or abs(ix) > 2 ** 31 or abs(iy) > 2 ** 31:
        raise ValueError("level / ix / iy out of range")
    view = state.view_from_query(q)
    base = float(q["base"][0]) if "base" in q else T.default_base_width(state.defaults.width)
    if not base > 0:
        raise ValueError("base must be > 0")
    T._check_tile(tile)
    t0 = time.perf_counter()
    with state.work():
        sig = snapshot_hash(paths[i])
        grid = T.TileGrid(tile, base / 2.0 ** level, 0.0, 0.0, level, base)
        key = T.tile_key(sig, view, tile, grid.tw, 0.0, 0.0, ix, iy, level)
        got = state.store.get(key)
        if got is not None:
            state.store.hits += 1
            img, meta = got
            meta = dict(meta, cached=True)
        else:
            progress.begin(f"{os.path.basename(paths[i])} {view.field} tile {level}:{ix},{iy}")
            try:
                img, meta = _render_tile(state, T, paths, i, view, grid, tile, level, ix, iy, base, run)
            finally:
                progress.end()
    meta = {k: v for k, v in meta.items() if k not in ("units",)}
    meta["snapshot"] = os.path.basename(paths[i])
    meta["tile"] = dict(meta.get("tile") or {}, tile=tile, level=level, ix=ix, iy=iy, extent=grid.extent(ix, iy), tile_width=grid.tw, base_width=base)
    meta["total_seconds"] = round(time.perf_counter() - t0, 4)
    fin = img[np.isfinite(img)]
    meta["n"] = int(img.shape[0])
    meta["stats"] = {"min": float(fin.min()), "max": float(fin.max())} if fin.size else {}
    return (np.ascontiguousarray(img, dtype="<f4").tobytes(),
            {"X-Splosh-Meta": json.dumps(meta, ensure_ascii=True), "Access-Control-Expose-Headers": "X-Splosh-Meta"})


def make_server(paths, host="127.0.0.1", port=0, token=None, store=None, defaults=None, **kw):
    """(ThreadingHTTPServer, ViewerState). host must be a loopback address."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("splosh serve binds to localhost only (use ssh -L to reach it)")
    state = ViewerState(paths, store=store, defaults=defaults, token=token, **kw)
    state.start_auto_products()
    srv = ThreadingHTTPServer((host, port), make_handler(state))
    srv.daemon_threads = True
    return srv, state


def watchdog(srv, state, orphan=False, idle_minutes=None, every=5.0):
    """Stop the server (srv.shutdown) when its parent process is gone (orphan: the ssh session that started it closed) or after `idle_minutes` without a request.
    Returns the thread (a daemon). The reason is printed."""
    parent = os.getppid()

    def loop():
        while True:
            time.sleep(every)
            why = None
            if orphan and os.getppid() != parent:
                why = "the parent process is gone (ssh session closed)"
            elif idle_minutes and time.time() - state.last_request > 60.0 * idle_minutes:
                why = f"no request for {idle_minutes:g} min"
            if why:
                print(f"splosh: stopping: {why}", flush=True)
                srv.shutdown()
                return
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def serve(paths, host="127.0.0.1", port=0, token=None, store=None, defaults=None, exit_when_orphaned=False, idle_exit=None, **kw):
    srv, state = make_server(paths, host, port, token, store, defaults, **kw)
    if exit_when_orphaned or idle_exit:
        watchdog(srv, state, exit_when_orphaned, idle_exit)
    import signal

    def _stop(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):                 # the ssh session closed / a kill: leave through `finally` (cancels the jobs this server started)
        signal.signal(sig, _stop)
    port = srv.server_address[1]
    print(f"splosh: {sum(len(r.paths) for r in state.runs)} snapshot(s) in {len(state.runs)} run(s)" + (": " + ", ".join(f"{r.label} ({len(r.paths)})" for r in state.runs) if len(state.runs) > 1 else "") + f"; store {state.store.root}")
    print(f"rendering threads {state.threads} of {state.max_threads} (change in the page, or POST /api/threads)")
    print(f"open  http://127.0.0.1:{port}/?token={state.token}")
    print(f"layout editor  http://127.0.0.1:{port}/layout?token={state.token}   (layouts in {state.layouts})")
    print(f"remote use:  ssh -L {port}:localhost:{port} <login-node>   (then open the URL above in the local browser)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.jobs.cancel_all()
        srv.server_close()
