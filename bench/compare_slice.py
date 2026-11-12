"""Slice (cross-section) density of one snapshot: splosh vs pynbody (smooth = H/2 = the same kernel), same centre/frame/plane.
python bench/compare_slice.py SNAPSHOT [--width 10] [--npix 512] [--z0 0] [--center bh] [--align bh]"""
import argparse
import time
import warnings

import numpy as np

from splosh import View, load_snapshot, render_map
from splosh.frames import make_frame

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("--width", type=float, default=10.0); ap.add_argument("--npix", type=int, default=512); ap.add_argument("--z0", type=float, default=0.0)
ap.add_argument("--center", default="bh"); ap.add_argument("--align", default="bh"); ap.add_argument("--pyn-h", type=float, default=0.5)
a = ap.parse_args()
warnings.simplefilter("ignore")
snap = load_snapshot(a.snapshot)
v = View(center=a.center, align=a.align, mode="slice", field="rho", width=a.width, npix=a.npix, z0=a.z0, exact=True)
render_map(snap, v.with_(npix=32))
t0 = time.perf_counter(); ours, meta = render_map(snap, v); t_ours = time.perf_counter() - t0
t0 = time.perf_counter(); fast, _ = render_map(snap, v.with_(exact=False)); t_fast = time.perf_counter() - t0
import pynbody
fr = make_frame(snap, a.center, a.align)
p = fr.pos(snap["Coordinates"])
n = len(p)
s = pynbody.new(gas=n)
s.gas["pos"] = p; s.gas["pos"].units = "kpc"
s.gas["smooth"] = (snap["SmoothingLength"] * a.pyn_h).astype(np.float32); s.gas["smooth"].units = "kpc"
s.gas["mass"] = snap["Masses"].astype(np.float32); s.gas["mass"].units = "Msol"
s.gas["rho"] = snap["Density"].astype(np.float32); s.gas["rho"].units = "Msol kpc^-3"
t0 = time.perf_counter()
pyn = np.asarray(pynbody.sph.render_image(s.gas, "rho", width=f"{a.width} kpc", resolution=a.npix, z_camera=None, approximate_fast=False, threaded=True)).astype(np.float64)
t_pyn = time.perf_counter() - t0
assert a.z0 == 0.0, "pynbody renders the plane z = 0 of its own frame: only z0 = 0 is compared"
mk = ours > 1e-3 * ours.max()
r = np.abs(pyn[mk] - ours[mk]) / ours[mk]
rf = np.abs(fast[mk] - ours[mk]) / ours[mk]
print(f"slice density, pixels above 1e-3 of the max ({mk.sum()}): median |pynbody - exact| / exact = {np.median(r):.2e}, p99 = {np.percentile(r, 99):.2e}")
print(f"multi-resolution vs exact: median {np.median(rf):.2e}, p99 {np.percentile(rf, 99):.2e}")
print(f"times: exact {t_ours:.2f} s, multi-resolution {t_fast:.2f} s, pynbody (exact, threaded) {t_pyn:.2f} s")
