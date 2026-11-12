"""Where do splosh and SPLASH differ? Render subsets of the particles (by H / pixel) with both and compare. Usage: python bench/subset_diff.py SNAP [width npix]"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import splosh
from splosh import splat
snap = splosh.load_snapshot(sys.argv[1]); W = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0; npix = int(sys.argv[3]) if len(sys.argv) > 3 else 512
dx = W / npix; com = splosh.binary_com(snap)
pos = snap["Coordinates"].astype(np.float64) - com; H = snap["SmoothingLength"].astype(np.float64); m = snap["Masses"].astype(np.float64); rho = snap["Density"].astype(np.float64)
out = os.environ.get("SPLOSH_BENCH_OUT", "bench_out"); drv = os.environ["SPLOSH_SPLASH_DRIVER"]; os.makedirs(out, exist_ok=True)
def mine(sel):
    return splat.splat_bands_pyr(pos[sel, 0].astype(np.float32).copy(), pos[sel, 1].astype(np.float32).copy(), H[sel].astype(np.float32).copy(), (m / rho)[sel].copy(),
                                 np.ascontiguousarray(rho[sel][None].astype(np.float32)), (0.0, 0.0), W, npix)[..., 1].astype(np.float64)
def splash(sel):
    hs = H[sel] / 2; n = int(sel.sum())
    with open(out + "/ss_in.bin", "wb") as f:
        f.write(np.array([n, npix], np.int32).tobytes()); f.write(np.array([-W / 2, -W / 2, dx], np.float64).tobytes())
        for arr in (pos[sel, 0], pos[sel, 1], pos[sel, 2], hs, (m / rho)[sel] / hs**3, rho[sel]): f.write(np.ascontiguousarray(arr, np.float64).tobytes())
    subprocess.run([drv, out + "/ss_in.bin", out + "/ss_out.bin", "1"], capture_output=True)
    return np.fromfile(out + "/ss_out.bin", np.float64).reshape(npix, npix)
print(f"width {W}, {npix}^2, pixel {dx:.4g}; columns: H/pixel range, N, mass(mine)/mass(SPLASH), median |rel|, p99 |rel|, max |rel|")
for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 32), (32, 128), (128, 1e9)):
    sel = (H / dx >= lo) & (H / dx < hi)
    if sel.sum() == 0: continue
    a, b = mine(sel), splash(sel)
    mk = b > 1e-3 * b.max()
    rel = np.abs(a[mk] - b[mk]) / b[mk]
    print(f"[{lo:>4g},{hi:>6g})  N={int(sel.sum()):7d}  mass ratio {a.sum() / b.sum():8.5f}   rel: median {np.median(rel):.2e}  p99 {np.percentile(rel, 99):.2e}  max {rel.max():.2e}")
