"""Timing of the pieces of the viewer pipeline on one snapshot: python bench/time_pieces.py SNAPSHOT [--width 10] [--center bh] [--align bh] [--repeats 3]
Prints the minimum over repeats of: info, load, frame, projection / slice maps (512 and 1024 px), a cache hit, and a server round trip (HTTP, in process)."""
import argparse
import json
import os
import tempfile
import threading
import time
import urllib.request

import numpy as np

from splosh import View, load_snapshot, make_frame, render_map
from splosh.info import snapshot_info
from splosh.mapstore import MapStore
from splosh.server import make_server


def best(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot"); ap.add_argument("--width", type=float, default=10.0); ap.add_argument("--center", default="bh"); ap.add_argument("--align", default="bh")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    n = a.repeats
    rows = []
    v = View(center=a.center, align=a.align, width=a.width)
    t, info = best(lambda: snapshot_info(a.snapshot, plugin_sections=False), n); rows.append(("info (all statistics)", t))
    t, snap = best(lambda: load_snapshot(a.snapshot), n); rows.append((f"load ({info['counts'].get('gas', 0)} gas)", t))
    t, _ = best(lambda: make_frame(snap, a.center, a.align), n); rows.append(("frame (centre + alignment)", t))
    render_map(snap, v.with_(npix=64)); render_map(snap, v.with_(npix=64, mode="slice"))          # numba compile / cache load outside the timings
    for npix in (512, 1024):
        t, _ = best(lambda: render_map(snap, v.with_(npix=npix)), n); rows.append((f"projection rho {npix}^2", t))
        t, _ = best(lambda: render_map(snap, v.with_(npix=npix, mode="slice", field="u")), n); rows.append((f"slice u {npix}^2", t))
    t, _ = best(lambda: render_map(snap, v.with_(npix=512, field="vz")), n); rows.append(("projection mass-weighted vz 512^2", t))
    with tempfile.TemporaryDirectory(dir=os.environ.get("SPLOSH_TMP")) as d:
        st = MapStore(d)
        t0 = time.perf_counter(); st.get_or_render(a.snapshot, v.with_(npix=512)); rows.append(("mapstore miss (load + render + write)", time.perf_counter() - t0))
        t, _ = best(lambda: st.get_or_render(a.snapshot, v.with_(npix=512)), n); rows.append(("mapstore hit", t))
        srv, state = make_server([a.snapshot], token="t", store=st, defaults=v)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/map?token=t&snap=0&width={a.width}&npix=512"
        t0 = time.perf_counter(); urllib.request.urlopen(base + "&field=u&mode=slice").read(); rows.append(("server, first request (render + load)", time.perf_counter() - t0))
        t, _ = best(lambda: urllib.request.urlopen(base + "&field=u&mode=slice").read(), n); rows.append(("server, cached request (HTTP, 1 MB)", t))
        t0 = time.perf_counter(); urllib.request.urlopen(base + "&field=u&mode=slice&cx=1.0").read(); rows.append(("server, new view, snapshot in memory (slice u 512^2)", time.perf_counter() - t0))
        srv.shutdown()
    print(f"load average {os.getloadavg()[0]:.1f}, threads {os.environ.get('NUMBA_NUM_THREADS', 'all')}")
    for name, t in rows:
        print(f"{t * 1000:9.1f} ms  {name}")


if __name__ == "__main__":
    main()
