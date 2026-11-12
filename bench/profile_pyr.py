"""Stage timing of splat_bands_pyr (level selection, deposits per level, merging). Usage: python bench/profile_pyr.py SNAP [width npix big]"""
import os, sys, time
import numpy as np, numba
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import splosh
from splosh import splat
snap = splosh.load_snapshot(sys.argv[1]); W = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0; npix = int(sys.argv[3]) if len(sys.argv) > 3 else 1024; big = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
com = splosh.binary_com(snap); pos = snap["Coordinates"].astype(np.float64) - com
x, y = pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32); H = snap["SmoothingLength"].astype(np.float32); w = snap["Masses"].astype(np.float64) / snap["Density"]
vals = np.ascontiguousarray(np.stack([snap["Density"], snap["InternalEnergy"]]).astype(np.float32))
splat.splat_bands_pyr(x, y, H, w, vals, (0., 0.), W, npix, big=big)      # warm-up
nth = numba.get_num_threads(); dx = W / npix; hp = H / dx
print(f"load {os.getloadavg()[0]:.1f}, {nth} threads; width {W}, {npix}^2, big {big}")
T = {"select": 0.0, "deposit": 0.0, "upsample+add": 0.0}
t_all = time.perf_counter()
levels = []; rest = np.ones(len(x), bool); level, nl = 0, npix
while True:
    t0 = time.perf_counter(); last = nl // 2 < 64; sel = rest if last else (rest & (hp / 2**level <= big)); idx = np.nonzero(sel)[0]
    sub = lambda a: np.ascontiguousarray(a[idx]); xs, ys, hs, ws, vs = sub(x), sub(y), sub(H), sub(w), np.ascontiguousarray(vals[:, idx]); T["select"] += time.perf_counter() - t0
    t0 = time.perf_counter(); im = splat._bands_kernel(xs, ys, hs, ws, vs, 0., 0., W / nl, nl, nl, nth, 4, 6.0, splat._T2) if len(idx) else None; dt = time.perf_counter() - t0; T["deposit"] += dt
    print(f"  level {level}: grid {nl:5d}, {len(idx):7d} particles, deposit {dt * 1e3:7.1f} ms")
    levels.append((nl, im)); rest &= ~sel
    if last or not rest.any(): break
    level += 1; nl //= 2
t0 = time.perf_counter(); out = None
for n_l, im in reversed(levels):
    if out is not None:
        out = splat._upsample(out, n_l, im)
    else: out = im
T["upsample+add"] = time.perf_counter() - t0
print("  total %.1f ms: " % ((time.perf_counter() - t_all) * 1e3) + ", ".join(f"{k} {v * 1e3:.1f} ms" for k, v in T.items()))
