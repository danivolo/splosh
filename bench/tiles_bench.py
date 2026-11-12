"""Tiled versus monolithic rendering: time and peak memory.
    python bench/tiles_bench.py SNAPSHOT [--npix 8192,16384] [--tile 256,512] [--workers 1,2,4,8] [--width 4] [--exact] [--stream] [--scratch DIR]
Every measurement runs in its own Python process (peak resident memory = ru_maxrss of that process, snapshot load included; the baseline after loading is printed
too). The monolithic render only exists up to View.npix = 8192. Tiled images are written to a memory-mapped .npy in --scratch (default $SPLOSH_TMP, else the
current directory; never the default temp directory). Print the machine load first: on a busy machine the times mean little."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time


def one(a):
    import splosh
    from splosh import View, render_map
    from splosh.maps import required_gas_fields
    from splosh import tiles
    v = View(center="bh", align="bh", width=a.width, npix=a.npix, exact=a.exact)
    snap = splosh.load_snapshot(a.snapshot, gas_fields=tuple(sorted(required_gas_fields(v))))
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    render_map(snap, v.with_(npix=64))                                    # warm the compiled kernels
    t0 = time.perf_counter()
    if a.mode == "mono":
        img, _ = render_map(snap, v)
    elif a.mode == "stream":                                              # tiles rendered and dropped: the working set without the output image
        for _ in tiles.iter_tiles(snap, v, tiles.window_grid(v, a.tile), workers=a.workers):
            pass
        img = None
    else:
        path = os.path.join(a.scratch, f"tiles_bench_{os.getpid()}.npy")
        img, meta = tiles.render_tiled(snap, v, tile=a.tile, out=path, workers=a.workers)
    dt = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    fin = float(img.max()) if img is not None else None
    if a.mode == "tiled":
        del img
        os.remove(path)
    print(json.dumps({"mode": a.mode, "npix": a.npix, "tile": a.tile if a.mode != "mono" else None, "workers": a.workers if a.mode != "mono" else None, "exact": a.exact,
                      "seconds": round(dt, 2), "peak_MB": round(peak), "baseline_MB": round(base), "max": fin}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot"); ap.add_argument("--npix", default="8192,16384"); ap.add_argument("--tile", default="256,512"); ap.add_argument("--workers", default="1,8")
    ap.add_argument("--width", type=float, default=4.0); ap.add_argument("--exact", action="store_true"); ap.add_argument("--scratch", default=os.environ.get("SPLOSH_TMP") or ".")
    ap.add_argument("--stream", action="store_true", help="also run the tiles-dropped variant (working set without the output image)")
    ap.add_argument("--one", choices=["mono", "tiled", "stream"], help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        a.mode, a.npix, a.tile, a.workers = a.one, int(a.npix), int(a.tile), int(a.workers)
        return one(a)
    print(f"host {os.uname().nodename}  cpus {os.cpu_count()}  load {os.getloadavg()[0]:.1f} {os.getloadavg()[1]:.1f}  width {a.width}  exact {a.exact}", flush=True)
    base = [sys.executable, os.path.abspath(__file__), a.snapshot, "--width", str(a.width), "--scratch", a.scratch] + (["--exact"] if a.exact else [])
    for n in a.npix.split(","):
        runs = ([("mono", 256, 1)] if int(n) <= 8192 else []) + [("tiled", int(t), int(w)) for t in a.tile.split(",") for w in a.workers.split(",")]
        runs += [("stream", int(t), int(w)) for t in a.tile.split(",") for w in a.workers.split(",")] if a.stream else []
        for mode, t, w in runs:
            r = subprocess.run(base + ["--one", mode, "--npix", n, "--tile", str(t), "--workers", str(w)], capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip()[-300:], flush=True)


if __name__ == "__main__":
    main()
