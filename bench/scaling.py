"""Thread scaling of render_map, split into its pieces: python bench/scaling.py SNAPSHOT [--threads 1,2,4,8,16] [--npix 1024] [--repeats 5]
For each width (zoom, medium, wide) and thread count: the whole render_map, the selection (frames.select_in_frame: the frame transform, the window cut
and the gathers in one parallel pass; until 2026-11-13 these were serial numpy, ~0.5 s per render at 9M particles), the splat on the particles that
remain, and the rest (field values, the final division, units). Minimum over repeats. Print the machine load first: on a busy machine
the numbers mean little. The thread count is changed in one process with numba.set_num_threads (at most NUMBA_NUM_THREADS / the CPUs seen)."""
import argparse
import os
import time

import splosh                    # first: it limits OpenBLAS to one thread before numpy loads it (see splosh/__init__.py)
import numba
import numpy as np
from splosh import View, make_frame, render_map
from splosh import splat as S


def best(f, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot"); ap.add_argument("--threads", default=""); ap.add_argument("--npix", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--widths", default="2,10,50")
    a = ap.parse_args()
    nmax = numba.config.NUMBA_NUM_THREADS
    threads = [int(t) for t in a.threads.split(",")] if a.threads else sorted({1, 2, 4, 8, 16, 32, nmax} & set(range(1, nmax + 1)))
    print(f"host {os.uname().nodename}  cpus seen {os.cpu_count()}  numba max threads {nmax}  load {os.getloadavg()[0]:.1f} {os.getloadavg()[1]:.1f}")
    snap = splosh.load_snapshot(a.snapshot)
    fr = make_frame(snap, "bh", "bh")
    print(f"{len(snap['SmoothingLength'])} gas particles, npix {a.npix}, min of {a.repeats}")
    print(f"{'thr':>4} {'width':>5}  {'render_map':>10}  {'select':>7} {'splat':>7} {'rest':>6}  {'particles':>9}  splat speed-up  render speed-up")
    base = {}
    for W in [float(w) for w in a.widths.split(",")]:
        v = View(center="bh", align="bh", width=W, npix=a.npix)
        for n in threads:
            numba.set_num_threads(n)
            render_map(snap, v, fr)                                              # compile / warm caches
            t_all = best(lambda: render_map(snap, v, fr), a.repeats)
            from splosh.frames import select_in_frame
            sel = lambda: select_in_frame(snap, fr, (0, 1, 2), W, 0.0, 0.0)
            t_sel = best(sel, a.repeats); x, y, z, hh, idx = sel()
            w = np.asarray(snap["Masses"], np.float64)[idx]; vals = np.zeros((0, len(idx)), np.float32)
            t_spl = best(lambda: S.splat_bands_pyr(x, y, hh, w, vals, (0.0, 0.0), W, a.npix), a.repeats)
            base.setdefault(W, (t_spl, t_all))
            print(f"{n:4d} {W:5.0f}  {t_all:8.1f}ms  {t_sel:7.1f} {t_spl:7.1f} {t_all - t_sel - t_spl:6.1f}  {len(idx):9d}  x{base[W][0] / t_spl:.2f}           x{base[W][1] / t_all:.2f}")
    print(f"load at the end {os.getloadavg()[0]:.1f}")


if __name__ == "__main__":
    main()
