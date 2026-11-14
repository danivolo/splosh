"""splosh command line:  info | render | tiles | shot | sheet | movie | overlays | inspect | analyze | compare | serve | fields | store | products | figure | dmd."""
import argparse
import json
import os
import sys
import time
from dataclasses import fields as dc_fields

import numpy as np

from . import plugins
from .maps import View, field_names

_VIEW_TYPES = {f.name: f.type for f in dc_fields(View)}


def _coerce(name, text):
    t = _VIEW_TYPES[name]
    t = t if isinstance(t, type) else {"str": str, "int": int, "float": float, "bool": bool}[t]
    if t is bool:
        return text.lower() in ("1", "true", "yes")
    return t(text)


def view_from_args(a):
    """View from --preset, the individual options and --view k=v,k=v (later wins)."""
    kw = {}
    if getattr(a, "preset", None):
        plugins.load_plugins()
        if a.preset not in plugins.PRESETS:
            raise SystemExit(f"unknown preset {a.preset!r}; known: {sorted(plugins.PRESETS)} (plugins loaded: {plugins.load_plugins()}, errors: {plugins.errors()})")
        p = dict(plugins.PRESETS[a.preset])
        if "fov" in p:
            p["width"] = p.pop("fov")
        kw.update({k: v for k, v in p.items() if k in _VIEW_TYPES})
    for k in _VIEW_TYPES:
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    for item in filter(None, (a.view or "").split(",")):
        k, _, v = item.partition("=")
        if k not in _VIEW_TYPES:
            raise SystemExit(f"--view: unknown key {k!r}; keys: {sorted(_VIEW_TYPES)}")
        kw[k] = _coerce(k, v)
    return View(**kw)


def add_view_options(p):
    p.add_argument("--preset", help="named bundle of settings from a plugin")
    p.add_argument("--view", help="k=v,k=v overrides, e.g. width=5,npix=1024,mode=slice")
    p.add_argument("--center", help="com | bh | mass-weighted | peak | ids:A,B | xyz:x,y,z | origin | box   (default bh)")
    p.add_argument("--align", help="none | bh | pair:A,B | vec:x,y,z   rotation about z so that the vector points along +x;  orbit[:ID]   orbital frame of one particle (default the only / most massive black hole): its L about the centre along +z, the particle on +x")
    p.add_argument("--tilt", help="none | incl:I[,PA] | euler:A,B,C | los:x,y,z | L[:R] | Lbh, joined by '+' (degrees): any 3D rotation after --align, e.g. L:2+incl:60")
    p.add_argument("--axes", choices=["xy", "xz", "yz"], help="third letter is the line of sight")
    p.add_argument("--mode", choices=["proj", "slice", "volume"], help="volume: emission + absorption front to back (rho: the opacity; a field: the value of the gas you see)")
    p.add_argument("--kappa", type=float, help="volume: opacity per unit mass (code length^2 / mass); 0 = automatic (tau = 1 at the window's mean column density)")
    p.add_argument("--field", help="rho, u, vx, vy, vz, vlos, speed, H (+ plugin fields); see `splosh fields`")
    p.add_argument("--width", type=float, help="window width in code length units")
    p.add_argument("--npix", type=int)
    p.add_argument("--cx", type=float); p.add_argument("--cy", type=float); p.add_argument("--z0", type=float, help="slice plane along the line of sight")
    p.add_argument("--weight", choices=["auto", "mass", "volume", "column"])
    p.add_argument("--hscale", type=float, help="multiply H before splatting (1 = the code's kernel; 2 reproduces readers that treat H as h with support 2h)")
    p.add_argument("--units", choices=["code", "cgs", "astro"], help="unit system of the map values: code (default), cgs, astro (Msun/pc^2, K, km/s, Myr, pc)")
    p.add_argument("--gamma", type=float, help="adiabatic index of the derived fields (default 5/3)")
    p.add_argument("--xh", type=float, help="hydrogen mass fraction for mu of T (default: from the snapshot's Metallicity when present, else 0.76)")
    p.add_argument("--bunit", type=float, help="UnitMagneticField_in_gauss of the run, for beta (default: read from the parameter file next to the snapshot, else sqrt(4 pi))")
    p.add_argument("--mu", type=float, help="mean molecular weight for T (default: from ElectronAbundance)")
    p.add_argument("--post", help="transform of the map: resid (contrast against the azimuthal mean about the centre) | resid:mean | resid:smooth:S | grad | grad:S (see docs/USAGE.md, 'Residual and gradient maps')")
    p.add_argument("--exact", action="store_true", default=None, help="exact (slower) engine instead of the multi-resolution one")


def add_style_options(p):
    p.add_argument("--cmap", help="colour map (default viridis; RdBu_r for a contrast map, magma for a gradient map)")
    p.add_argument("--scale", help="log (default) | linear | asinh[:F] (F = softening, default 0.1; smaller is more logarithmic) | histeq (one picture only); contrast and gradient maps default to linear")
    p.add_argument("--vmin", type=float); p.add_argument("--vmax", type=float); p.add_argument("--pct", help="lo,hi percentiles for the limits (default 1,99.9 per frame; global: min/max unless given)")
    p.add_argument("--title"); p.add_argument("--no-scalebar", action="store_true"); p.add_argument("--font-scale", type=int, help="integer size of the built-in font (default from the map size)")
    p.add_argument("--overlay", action="append", metavar="SPEC", help="overlay drawn on the map, repeatable: vectors:v | vectors:B | streamlines:v | streamlines:B | particles:bh,star,dm | a plugin overlay; "
                   "options after the argument, e.g. vectors:v,grid=32,color=cyan (see `splosh overlays`)")


def add_selection_options(p):
    p.add_argument("--every", type=int, help="every Nth snapshot"); p.add_argument("--snaps", help="snapshot numbers a,b,c (the number in the file name)")
    p.add_argument("--limits", choices=["global", "frame", "fixed"], default="global", help="colour limits: locked over the selection, per frame, or --vmin/--vmax")


def cmd_info(a):
    from .info import snapshot_info, format_info
    info = snapshot_info(a.snapshot, units=a.units or "code")
    print(json.dumps(info, indent=1) if a.json else format_info(info))


def cmd_render(a):
    if getattr(a, "tiles", None):
        return _render_tiles(a)
    from .mapstore import MapStore
    from . import imaging
    from .figures import resolve_style
    view = view_from_args(a)
    cmap, scale, sym = resolve_style(view, a.cmap, a.scale)
    store = MapStore(a.store)
    if a.no_cache:
        from .io import load_snapshot
        from .maps import render_map, required_gas_fields
        t0 = time.perf_counter()
        img, meta = render_map(load_snapshot(a.snapshot, gas_fields=tuple(sorted(required_gas_fields(view)))), view)
        hit, dt = False, time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        img, meta, hit = store.get_or_render(a.snapshot, view)
        dt = time.perf_counter() - t0
    out = a.out
    if out:
        ext = os.path.splitext(out)[1].lower()
        if ext == ".png":
            lim = imaging.write_png(out, img, cmap=cmap, scale=scale, vmin=a.vmin, vmax=a.vmax, symmetric=sym)
            meta["png_limits"] = list(lim)
        elif ext == ".npz":
            np.savez(out, img=img, meta=json.dumps(meta))
        elif ext == ".npy":
            np.save(out, img)
        else:
            raise SystemExit("--out must end in .png, .npz or .npy")
    fin = img[np.isfinite(img)]
    print(f"{'cache hit' if hit else 'rendered'} in {dt:.3f} s  {view.mode} {view.field} {view.axes} {img.shape[1]}x{img.shape[0]}  t={meta['time']:.6g}  "
          f"range {fin.min():.4g} .. {fin.max():.4g}" + (f"  -> {out}" if out else ""))


def _peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _render_tiles(a):
    """render --tiles N: the image assembled from N x N pixel tiles (memory bounded: .npy is a memory-mapped file, .png is coloured and compressed band by band)."""
    from . import imaging, tiles
    from .mapstore import MapStore
    view = view_from_args(a)
    if view.post != "none":
        raise SystemExit("--tiles assembles raw maps; a --post transform needs the whole window (drop --tiles)")
    store = None if a.no_cache else MapStore(a.store)
    out = a.out
    ext = os.path.splitext(out)[1].lower() if out else ".npy"
    if ext not in (".png", ".npz", ".npy"):
        raise SystemExit("--out must end in .png, .npz or .npy")
    tmp = None
    if out and ext == ".npy":
        target = out
    elif view.npix <= 4096:
        target = None
    else:
        d = os.environ.get("SPLOSH_TMP") or os.path.dirname(os.path.abspath(out or "."))
        tmp = target = os.path.join(d, f".splosh_tiles_{os.getpid()}.npy")
    t0 = time.perf_counter()
    done = [0]
    def prog(k, n):
        if k == n or k % max(1, n // 10) == 0:
            print(f"  tiles {k}/{n}", file=sys.stderr, flush=True)
    img, meta = tiles.render_tiled(a.snapshot, view, tile=a.tiles, out=target, store=store, workers=a.workers, progress=prog)
    dt = time.perf_counter() - t0
    try:
        if out and ext == ".png":
            meta["png_limits"] = list(tiles.write_png_streaming(out, img, cmap=a.cmap or "viridis", scale=a.scale or "log", vmin=a.vmin, vmax=a.vmax))
        elif out and ext == ".npz":
            np.savez(out, img=np.asarray(img), meta=json.dumps(meta))
    finally:
        del img
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    t = meta["tiles"]
    print(f"tiled render in {dt:.2f} s  {view.mode} {view.field} {view.axes} {view.npix}x{view.npix} from {t['n_tiles']} tiles of {t['tile']}  ({t['cache_hits']} from the cache)  "
          f"peak memory {_peak_rss_mb():.0f} MB" + (f"  -> {out}" if out else ""))


def cmd_tiles(a):
    """tiles SNAP: list (--list) or pre-render into a tile store (--out DIR) the tiles covering the view window at one or several zoom levels."""
    from . import tiles
    from .mapstore import MapStore, snapshot_hash
    view = view_from_args(a)
    if not a.list and not a.out:
        raise SystemExit("tiles needs --out DIR (render into that tile store) or --list")
    store = MapStore(a.out) if a.out else MapStore(a.store)
    bw = a.base_width or tiles.default_base_width(view.width)
    if a.level is None:
        levels = [tiles.level_for(view, a.tile, bw)]
    else:
        lo, _, hi = a.level.partition(":")
        levels = list(range(int(lo), int(hi or lo) + 1))
    grids = [tiles.tile_grid(view, a.tile, bw, level=L) for L in levels]
    total = sum(g.nx * g.ny for g in grids)
    if total > a.max_tiles:
        raise SystemExit(f"{total} tiles requested, above --max-tiles {a.max_tiles} (narrow --width or the level range)")
    index = {"snapshot": os.path.abspath(a.snapshot), "view": view.as_dict(), "tile": a.tile, "base_width": bw, "levels": []}
    for g in grids:
        entries = tiles.tile_index(g, store, view, a.snapshot)
        done = 0
        if not a.list:
            t0 = time.perf_counter()
            for ix, iy, img, meta, hit in tiles.iter_tiles(a.snapshot, view, g, store=store, workers=a.workers):
                done += 1
            for e in entries:
                e["cached"] = True
            print(f"level {g.level}: {len(entries)} tiles rendered/checked in {time.perf_counter() - t0:.2f} s")
        else:
            n = sum(e["cached"] for e in entries)
            print(f"level {g.level}: pixel {g.pixel:.6g}, tile width {g.tw:.6g}, ix {g.ix0}..{g.ix1}, iy {g.iy0}..{g.iy1}: {len(entries)} tiles, {n} cached")
            if a.verbose:
                for e in entries:
                    print(f"  ({e['ix']},{e['iy']}) {'cached' if e['cached'] else '-'}  {e['key']}")
        index["levels"].append(dict(g.describe(), tiles=entries))
    if a.out:
        with open(os.path.join(a.out, "tiles_index.json"), "w") as f:
            json.dump(index, f, indent=1)
        print(f"tile store {a.out}: {json.dumps(store.stats())}")


def _style(a):
    from .limits import parse_pct
    return dict(cmap=a.cmap, scale=a.scale, vmin=a.vmin, vmax=a.vmax, pct=parse_pct(a.pct), title=a.title, scalebar=not a.no_scalebar, font_scale=a.font_scale, overlays=a.overlay)


def _store(a):
    from .mapstore import MapStore
    return MapStore(a.store)


def cmd_shot(a):
    from . import figures
    if a.from_sidecar:
        _, out = figures.replay(a.from_sidecar, out=a.out, store=_store(a), use_cache=not a.no_cache)
        print(f"reproduced -> {out}")
        return
    if not a.snapshot or not a.out:
        raise SystemExit("shot needs SNAP and --out (or --from-sidecar FILE.json)")
    ext = os.path.splitext(a.out)[1].lower()
    if ext not in (".png", ".pdf", ".svg"):
        raise SystemExit("--out must end in .png (or .pdf / .svg with matplotlib)")
    backend = a.backend or ("mpl" if ext in (".pdf", ".svg") else "png")
    pic = figures.shot(a.snapshot, view_from_args(a), backend=backend, colorbar=not a.no_colorbar, store=_store(a), use_cache=not a.no_cache, orbits=a.orbits, **_style(a))
    files = pic.save(a.out)
    print(f"shot t={pic.meta['time']:.6g}  limits {pic.limits[0]:.4g} .. {pic.limits[1]:.4g}  {pic.rgba.shape[1]}x{pic.rgba.shape[0]}  -> {', '.join(files)}")


def cmd_explore(a):
    from . import figures
    if a.from_sidecar:
        _, out = figures.replay(a.from_sidecar, out=a.out, store=_store(a), use_cache=not a.no_cache)
        print(f"reproduced -> {out}")
        return
    if not a.snapshot or not a.out:
        raise SystemExit("explore needs SNAP and --out (or --from-sidecar FILE.json)")
    pic = figures.explore(a.snapshot, view_from_args(a), ncols=a.ncols, title=a.title, store=_store(a), use_cache=not a.no_cache, font_scale=a.font_scale)
    files = pic.save(a.out)
    print(f"explore {len(pic.sidecar['panels'])} panels  t={pic.meta['time']:.6g}  {pic.rgba.shape[1]}x{pic.rgba.shape[0]}  -> {', '.join(files)}")


def cmd_sheet(a):
    from . import figures
    if a.from_sidecar:
        _, out = figures.replay(a.from_sidecar, out=a.out, store=_store(a))
        print(f"reproduced -> {out}")
        return
    if not a.paths or not a.out:
        raise SystemExit("sheet needs RUNDIR_or_SNAPS and --out (or --from-sidecar FILE.json)")
    pic = figures.contact_sheet(a.paths, view_from_args(a), ncols=a.ncols, store=_store(a), every=a.every, snaps=a.snaps, limits=a.limits, **_style(a))
    files = pic.save(a.out)
    print(f"sheet of {len(pic.sidecar['snapshots'])} snapshots ({pic.sidecar['map_cache_hits']} maps from the cache)  {pic.rgba.shape[1]}x{pic.rgba.shape[0]}  -> {', '.join(files)}")


def cmd_movie(a):
    from . import figures
    log = (lambda m: print(m, file=sys.stderr)) if a.verbose else None
    if a.from_sidecar:
        figures.replay(a.from_sidecar, out=a.out, store=_store(a), log=log)
        print("reproduced")
        return
    if not a.paths or not a.out:
        raise SystemExit("movie needs RUNDIR_or_SNAPS and --out (or --from-sidecar FILE.json)")
    r = figures.movie(a.paths, a.out, view_from_args(a), fps=a.fps, store=_store(a), keep_frames=a.keep_frames, log=log, every=a.every, snaps=a.snaps, limits=a.limits, orbits=a.orbits, workers=getattr(a, "workers", None), **_style(a))
    print(r["message"])
    print("sidecar:", r["sidecar"])


def cmd_analyze(a):
    from . import analyze
    only = [x for x in (a.only or "").split(",") if x]
    if a.jobs == "auto":
        import numba
        a.jobs = max(1, numba.config.NUMBA_NUM_THREADS // 5)        # rendering stops scaling at ~5-8 threads (bench/scaling.py)
    r = analyze.run(a.run, a.config, out=a.out, jobs=int(a.jobs), only=only or None, every=a.every, force=a.force, dry_run=a.dry_run, store=a.store, log=lambda m: print(m, file=sys.stderr))
    if a.dry_run:
        print(r.plan)
    else:
        print(r.digest, end="")


def cmd_compare(a):
    from . import compare
    only = [x for x in (a.only or "").split(",") if x]
    off = dict(kv.split("=", 1) for kv in (a.offset or []))
    r = compare.compare(a.runs, a.config, out=a.out, jobs=a.jobs, only=only or None, every=a.every, force=a.force, dry_run=a.dry_run, run_out=a.run_out, align=a.align, offsets=off, log=lambda m: print(m, file=sys.stderr))
    print(r.plan if a.dry_run else r.digest, end="\n" if a.dry_run else "")


def cmd_serve(a):
    from .mapstore import MapStore
    from .server import serve
    serve(a.paths, host=a.host, port=a.port, token=a.token, store=MapStore(a.store), defaults=view_from_args(a), layouts=a.layouts, analysis=a.analysis, config=a.config,
          threads=a.threads, root=a.root, slurm=a.slurm, node_wait=a.node_wait, jobs_dir=a.jobs_dir, ffmpeg=a.ffmpeg, exit_when_orphaned=a.exit_when_orphaned, idle_exit=a.idle_exit, partial=a.partial,
          auto_products=not a.no_auto_products)


def cmd_overlays(a):
    from . import overlays
    plugins.load_plugins()
    print(overlays.describe())
    print("registered:", ", ".join(sorted(plugins.OVERLAYS)))


def cmd_inspect(a):
    from . import overlays
    view = view_from_args(a)
    snap = overlays.load_for(a.snapshot, view)
    res = overlays.inspect(snap, view, a.x, a.y, limit=a.limit)
    print(json.dumps(res, indent=1))


def cmd_fields(a):
    from . import overlays          # noqa: F401  (registers the Bx, By, Bz fields)
    from .fields import format_fields
    plugins.load_plugins()
    print(format_fields())
    plugins.load_plugins()
    print("presets:", ", ".join(sorted(plugins.PRESETS)) or "(none)")
    print("plugins:", plugins.load_plugins(), plugins.errors() or "")


def cmd_store(a):
    from .mapstore import MapStore
    s = MapStore(a.store)
    if a.clear:
        s.clear()
    print(json.dumps(s.stats()))


def cmd_figure(a):
    """`splosh figure RUN --layout F --out fig.pdf [--snap I]`: what the layout editor's export does, from the command line."""
    import os
    from . import layout as lay, layout_preview as lp, analyze
    from .io import load_snapshot
    from .mapstore import MapStore
    from .runs import run_snapshots
    paths = [a.run] if os.path.isfile(a.run) else run_snapshots(a.run)
    if not paths:
        raise SystemExit(f"no snapshots in {a.run}")
    path = paths[a.snap]
    L = lay.load_layout(a.layout)
    products = analyze.normalize_config(analyze.read_config(a.config), os.path.dirname(os.path.abspath(a.config))) if a.config else None
    table = lp.read_scalars_table(a.analysis) if a.analysis else None
    cache = {}
    lo, tiles, notes, header = lp.tiles_for(L, path, lambda f: load_snapshot(path, gas_fields=f), MapStore(a.store) if a.store else MapStore(), products, table, cache=cache)
    for k, v in notes.items():
        print(f"note {k}: {v}", file=sys.stderr)
    if a.backend == "mpl":
        from . import layout_mpl
        prob = layout_mpl.style_problem((lo.get("figure") or lay.figure_options(lo))["style"])
        if prob:
            print("WARNING: " + prob, file=sys.stderr)
        layout_mpl.save(lo, tiles, a.out, header)
    else:
        from .imaging import png_bytes
        with open(a.out, "wb") as f:
            f.write(png_bytes(lay.render_layout(lo, tiles, header)))
    print(f"wrote {a.out}  ({os.path.basename(path)})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="splosh", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="summary of a snapshot"); p.add_argument("snapshot"); p.add_argument("--json", action="store_true"); p.add_argument("--units", choices=["code", "cgs", "astro"], help="also give gas totals / percentiles in this system"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("render", help="render one map (cached) to png/npz/npy"); p.add_argument("snapshot"); add_view_options(p)
    p.add_argument("--out"); p.add_argument("--cmap"); p.add_argument("--scale", help="log (default) | linear | asinh[:F] | histeq")
    p.add_argument("--vmin", type=float); p.add_argument("--vmax", type=float)
    p.add_argument("--store", help="mapstore directory (default $SPLOSH_STORE or ~/.cache/splosh/maps)"); p.add_argument("--no-cache", action="store_true"); p.set_defaults(fn=cmd_render)
    p.add_argument("--tiles", type=int, choices=[64, 128, 256, 512, 1024], help="assemble the image from tiles of this many pixels (bounded memory; npix may exceed 8192)"); p.add_argument("--workers", type=int, help="tiles rendered in parallel (default: one per CPU)")
    p = sub.add_parser("tiles", help="list / pre-render the tiles of a zoom range (deep-zoom cache warm-up)"); p.add_argument("snapshot"); add_view_options(p)
    p.add_argument("--tile", type=int, default=256, choices=[64, 128, 256, 512, 1024]); p.add_argument("--level", help="zoom level L or A:B (default: the level matching --npix)"); p.add_argument("--base-width", type=float, help="width of a level-0 tile (default: --width rounded up to a power of two)")
    p.add_argument("--out", help="tile store directory to render into (also gets tiles_index.json)"); p.add_argument("--store", help="with --list: the store to look in"); p.add_argument("--list", action="store_true", help="only list the tiles and whether they are cached")
    p.add_argument("--verbose", action="store_true"); p.add_argument("--workers", type=int); p.add_argument("--max-tiles", type=int, default=4096); p.set_defaults(fn=cmd_tiles)
    p = sub.add_parser("shot", help="one finished picture: map + colour bar, scale bar, time, title (+ .json sidecar)"); p.add_argument("snapshot", nargs="?"); add_view_options(p); add_style_options(p)
    p.add_argument("--orbits", action="store_true", help="time label also in units of the binary's initial period P0 (first snapshot of the folder)"); p.add_argument("--out"); p.add_argument("--backend", choices=["png", "mpl"], help="mpl = matplotlib, if installed (default for .pdf / .svg)"); p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--no-cache", action="store_true"); p.add_argument("--from-sidecar", metavar="JSON", help="redo the picture of a sidecar (limits as used then); --out defaults to <stem>_repro.<ext>")
    p.add_argument("--store"); p.set_defaults(fn=cmd_shot)
    p = sub.add_parser("explore", help="discovery sheet of one snapshot: log, asinh, histogram-equalised, contrast and gradient maps side by side (+ .json sidecar)"); p.add_argument("snapshot", nargs="?"); add_view_options(p)
    p.add_argument("--out"); p.add_argument("--ncols", type=int, default=3); p.add_argument("--title"); p.add_argument("--font-scale", type=int); p.add_argument("--no-cache", action="store_true")
    p.add_argument("--from-sidecar", metavar="JSON"); p.add_argument("--store"); p.set_defaults(fn=cmd_explore)
    p = sub.add_parser("sheet", help="contact sheet of many snapshots with shared colour limits"); p.add_argument("paths", nargs="*", help="run directory (or its output/) and/or snapshot files"); add_view_options(p); add_style_options(p)
    add_selection_options(p); p.add_argument("--out"); p.add_argument("--ncols", type=int); p.add_argument("--from-sidecar", metavar="JSON"); p.add_argument("--store"); p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("movie", help="movie of many snapshots (ffmpeg if available, else numbered PNGs)"); p.add_argument("paths", nargs="*"); add_view_options(p); add_style_options(p)
    add_selection_options(p); p.add_argument("--orbits", action="store_true", help="time label also in units of the binary's initial period P0"); p.add_argument("--out"); p.add_argument("--fps", type=float, default=12); p.add_argument("--keep-frames", action="store_true"); p.add_argument("--verbose", action="store_true"); p.add_argument("--workers", type=int, help="frames at the same time (each its share of NUMBA_NUM_THREADS); default about one per 4 threads")
    p.add_argument("--from-sidecar", metavar="JSON"); p.add_argument("--store"); p.set_defaults(fn=cmd_movie)
    p = sub.add_parser("products", help="compute the snapshot products (plugins.register_product) of a run that are missing or out of date, in parallel")
    p.add_argument("run", help="the run directory (or its output/)"); p.add_argument("--names", help="comma-separated products (default: all that apply)")
    p.add_argument("--workers", type=int, help="processes (default: physical cores, limited by the free memory)")
    p.set_defaults(fn=lambda a: __import__("splosh.products", fromlist=["main"]).main(a))
    p = sub.add_parser("figure", help="one figure of a layout for one snapshot of a run (publication renderer): .pdf / .png / .svg")
    p.add_argument("run", help="a run directory (or its output/) or a snapshot file"); p.add_argument("--layout", required=True, help="the layout file (.toml / .json)")
    p.add_argument("--out", default="figure.pdf"); p.add_argument("--snap", type=int, default=-1, help="snapshot index in the run (default: the last)")
    p.add_argument("--analysis", help="an analysis folder (its scalars.csv feeds snapshot time-series panels)"); p.add_argument("--config", help="an analyze config (products named by panels)")
    p.add_argument("--store"); p.add_argument("--backend", choices=["mpl", "bitmap"], default="mpl")
    p.set_defaults(fn=cmd_figure)
    p = sub.add_parser("layout-movie", help="a movie of a layout over a run's snapshots (publication renderer by default)")
    p.add_argument("run"); p.add_argument("--layout", required=True, help="the layout file (.toml / .json)"); p.add_argument("--out", default="layout_movie.mp4")
    p.add_argument("--from", dest="start", type=int, default=0); p.add_argument("--to", dest="stop", type=int); p.add_argument("--every", type=int, default=1)
    p.add_argument("--fps", type=float, default=12); p.add_argument("--backend", choices=["mpl", "bitmap"], default="mpl")
    p.add_argument("--analysis", help="an analysis folder (its scalars.csv feeds time-series panels)"); p.add_argument("--config", help="an analyze config (products named by panels)")
    p.add_argument("--store"); p.add_argument("--dpi", type=int, help="frame resolution (default: the layout's figure dpi)"); p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--workers", type=int, help="frames at the same time (each its share of NUMBA_NUM_THREADS); default about one per 4 threads")
    p.set_defaults(fn=lambda a: __import__("splosh.layout_movie", fromlist=["main"]).main(a))
    p = sub.add_parser("serve", help="local viewer server (localhost + token)"); p.add_argument("paths", nargs="*", help="snapshot files and/or directories (none: choose folders in the page)"); add_view_options(p)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=0, help="0 = a free port"); p.add_argument("--token")
    p.add_argument("--store")
    p.add_argument("--layouts", help="directory of layout files for the layout editor (/layout); default ./layouts")
    p.add_argument("--analysis", help="an analysis directory (its scalars.csv feeds the editor's time-series panels)")
    p.add_argument("--config", help="an analyze config: its [[map]] / [[profile]] products can be used by `product =` panels in the editor's preview")
    p.add_argument("--slurm", help='srun options for the page\'s long jobs (movie, prefetch), e.g. "-c 20 --mem=32G -t 01:00:00": a free node is used at once, else the job runs here')
    p.add_argument("--node-wait", type=int, default=10, help="seconds Slurm may take to start a job on a node before it runs here instead (default 10)")
    p.add_argument("--jobs-dir", help="where the jobs' folders go (default ~/.cache/splosh/jobs)")
    p.add_argument("--ffmpeg", help="path of ffmpeg for the movie jobs when it is not on PATH")
    p.add_argument("--root", help="the folder tree the page may browse and open (default: the home directory)")
    p.add_argument("--threads", type=int, help="cores for rendering (default: all of the pool; the pool size is NUMBA_NUM_THREADS, the page can change the count up to it)")
    p.add_argument("--no-auto-products", action="store_true", help="do not compute the plugins' snapshot products in the background when a run is opened")
    p.add_argument("--partial", choices=["auto", "on", "off"], default="auto", help="read only the gas a zoomed view needs (a chunk index per snapshot next to the store): auto = for snapshots of >= 2e6 gas particles ($SPLOSH_PARTIAL_MIN)")
    p.add_argument("--exit-when-orphaned", action="store_true", help="stop when the parent process goes away (started over ssh: the session closed)")
    p.add_argument("--idle-exit", type=float, metavar="MIN", help="stop after MIN minutes without a request")
    p.set_defaults(fn=cmd_serve)
    from . import prefetch
    prefetch.add_parser(sub)
    from . import dmd
    dmd.add_parser(sub)
    p = sub.add_parser("overlays", help="list the overlays and their options"); p.set_defaults(fn=cmd_overlays)
    p = sub.add_parser("inspect", help="particles contributing at a map position (JSON): X Y in the window coordinates of the view"); p.add_argument("snapshot")
    p.add_argument("x", type=float); p.add_argument("y", type=float); p.add_argument("--limit", type=int, default=20); add_view_options(p); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("analyze", help="per-run analysis pipeline: maps, profiles, scalars, dashboards, movies from a TOML config (incremental, cached)"); p.add_argument("run", help="run directory (or its output/)")
    p.add_argument("--config", help="analyze.toml (default: one map, one profile, the built-in scalars)"); p.add_argument("--out", help="output tree (default RUN/analysis)"); p.add_argument("--jobs", default="1", help="auto = NUMBA_NUM_THREADS / 5; worker processes over snapshots")
    p.add_argument("--only", help="comma-separated product names or kinds (their inputs are included)"); p.add_argument("--every", type=int, help="every Nth snapshot"); p.add_argument("--force", action="store_true", help="redo everything")
    p.add_argument("--dry-run", action="store_true", help="list what would be done"); p.add_argument("--store", help="mapstore directory (default OUT/.maps)"); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("compare", help="comparison figures of several runs (curves of all runs on one panel, map grids with a column per run), each run with its own cache"); p.add_argument("runs", nargs="+", help="run directories, optionally label=path")
    p.add_argument("--config", help="TOML with [[compare]] products (and the map / profile / scalar products they use)"); p.add_argument("--out", default="compare", help="output tree (default ./compare)")
    p.add_argument("--run-out", help="cache directory per run: DIR/<label> (default RUN/analysis)"); p.add_argument("--jobs", type=int, default=1); p.add_argument("--only", help="comparison names or types")
    p.add_argument("--every", type=int); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--align", choices=["time", "time_myr", "index", "offset"], help="override the time alignment of every product"); p.add_argument("--offset", action="append", metavar="LABEL=VALUE", help="shift of a run on the aligned axis (repeatable)")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("fields", help="list fields, presets, plugins"); p.set_defaults(fn=cmd_fields)
    p = sub.add_parser("store", help="mapstore statistics / clear"); p.add_argument("--store"); p.add_argument("--clear", action="store_true"); p.set_defaults(fn=cmd_store)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except ValueError as e:
        raise SystemExit(f"splosh {a.cmd}: {e}")


if __name__ == "__main__":
    main()
