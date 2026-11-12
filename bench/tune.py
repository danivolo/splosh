"""Speed/accuracy trade-off of splat_bands_pyr (the `big` threshold) and thread scaling. Usage: python bench/tune.py SNAP [width npix]"""
import os, sys, time
import numpy as np, numba
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import splosh
from splosh import splat
snap = splosh.load_snapshot(sys.argv[1]); W = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0; npix = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
com = splosh.binary_com(snap)
pos = snap["Coordinates"].astype(np.float64) - com
x, y = pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32); H = snap["SmoothingLength"].astype(np.float32)
m = snap["Masses"].astype(np.float64); rho = snap["Density"].astype(np.float64); w = m / rho
vals = np.ascontiguousarray(np.stack([snap["Density"], snap["InternalEnergy"]]).astype(np.float32))
def best(fn, reps=4):
    fn(); t = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); t.append(time.perf_counter() - t0)
    return r, min(t)
print(f"load {os.getloadavg()[0]:.1f}, {numba.get_num_threads()} threads, width {W}, {npix}^2")
ex, t_ex = best(lambda: splat.splat_bands(x, y, H, w, vals, (0., 0.), W, npix), reps=1)
ref = ex[..., 1].astype(np.float64); mk = ref > 1e-3 * ref.max()
print(f"exact (bands, no pyramid): {t_ex:.2f} s")
print(f"{'big':>5s} {'t [s]':>8s} {'speed-up':>9s} {'median rel':>11s} {'p99 rel':>10s} {'max rel':>10s}")
for big in (24, 16, 12, 8, 6, 4):
    r, t = best(lambda: splat.splat_bands_pyr(x, y, H, w, vals, (0., 0.), W, npix, big=big))
    rel = np.abs(r[..., 1][mk] - ref[mk]) / ref[mk]
    print(f"{big:5d} {t:8.3f} {t_ex / t:9.1f} {np.median(rel):11.2e} {np.percentile(rel, 99):10.2e} {rel.max():10.2e}")
