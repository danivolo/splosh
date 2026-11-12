#!/usr/bin/env python
"""Compare splosh's splatting variants with SPLASH's own projection routine and with pynbody on one GIZMO snapshot (column density).

    python bench/compare.py SNAP --width 10 --npix 1024 [--reps 5] [--threads 8] [--scale 1] [--out DIR]

All codes render the column density  Sigma = sum_i m_i W2D(r_i, H_i)  (+ the density-weighted u map) from the same particles:
  splosh : H = SmoothingLength = the kernel SUPPORT radius (GIZMO convention, checked against kernel.h)
  SPLASH    : interpolate3D_projection compiled from refs/splash (copy in scratch), h_sph = H/2 (as SPLASH's own GADGET reader does)
  pynbody   : smooth = pyn_h * SmoothingLength (default 0.5: kernel support 2*smooth = H; 1.0 = H itself, as some analysis scripts pass it)
--scale N makes a synthetic snapshot of N times the particles (resampled particles + a jitter of one H) for the timing at 1e7 particles.
Timings are wall times of the render call only (file reading, array preparation and kernel compilation excluded); the machine is shared:
the load average is printed and the minimum of --reps repeats is the number to look at.
"""
import argparse, os, subprocess, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import splosh
from splosh import splat

ap = argparse.ArgumentParser()
ap.add_argument("snap"); ap.add_argument("--width", type=float, default=10.0); ap.add_argument("--npix", type=int, default=1024)
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--scale", type=int, default=1)
ap.add_argument("--only", default="", help="comma list of: serial,tl,bands,pyr,splash,pynbody")
ap.add_argument("--pyn-h", type=float, default=0.5, help="pynbody smooth = pyn_h * SmoothingLength (0.5 = SPLASH/Monaghan convention; 1.0 = H itself, as some analysis scripts feeds it)")
ap.add_argument("--pyn-fast", type=int, default=1, help="pynbody approximate_fast (default on, as in its config)")
ap.add_argument("--out", default="bench_out", help="directory for the images and temporary files (default ./bench_out)")
ap.add_argument("--splash", default=os.environ.get("SPLOSH_SPLASH_DRIVER", ""), help="SPLASH projection driver built by bench/splash_driver/build.sh (or $SPLOSH_SPLASH_DRIVER); skipped if absent")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
want = set(a.only.split(",")) if a.only else {"serial", "tl", "bands", "pyr", "splash", "pynbody"}
if a.threads:
    import numba; numba.set_num_threads(a.threads)
import numba
nth = numba.get_num_threads()

snap = splosh.load_snapshot(a.snap)
com = splosh.binary_com(snap)
pos = snap["Coordinates"].astype(np.float64); H = snap["SmoothingLength"].astype(np.float64); m = snap["Masses"].astype(np.float64)
rho = snap["Density"].astype(np.float64); u = snap["InternalEnergy"].astype(np.float64)
if a.scale > 1:
    rng = np.random.default_rng(7)
    k = rng.integers(0, len(H), len(H) * a.scale)
    pos = pos[k] + rng.normal(size=(len(k), 3)) * H[k, None] * 0.5; H = H[k]; m = m[k] / a.scale; rho = rho[k]; u = u[k]
n = len(H)
x = (pos[:, 0] - com[0]).astype(np.float32); y = (pos[:, 1] - com[1]).astype(np.float32); z = (pos[:, 2] - com[2]).astype(np.float32)
Hf = H.astype(np.float32); vol = (m / rho)
vals = np.ascontiguousarray(np.stack([rho, u]).astype(np.float32))        # out[...,1] = column density (m W), out[...,2] = integral of (m/rho) u W
npix, W = a.npix, a.width
dx = W / npix
inside = (np.abs(x) < W / 2) & (np.abs(y) < W / 2)
print(f"{os.path.basename(a.snap)}: N = {n}  ({inside.sum()} centres inside the view), width {W}, {npix}^2 pixels, {nth} numba threads, load {os.getloadavg()[0]:.1f}")
print(f"H percentiles [0,50,99,100] = {np.percentile(H, [0, 50, 99, 100]).round(3)}   pixel = {dx:.4g}   H/pixel median {np.median(H) / dx:.2f}, max {H.max() / dx:.1f}")

def bench(fn, *args, reps=a.reps, **kw):
    t = []
    r = fn(*args, **kw)                 # warm-up (compilation / first-touch)
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(*args, **kw); t.append(time.perf_counter() - t0)
    return r, min(t), float(np.median(t))

res = {}
c = (0.0, 0.0)
wvol = vol.astype(np.float64)
for name, fn, kw in (("serial", splat.splat_serial, {}), ("tl", splat.splat_tl, {}), ("bands", splat.splat_bands, {}), ("pyr", splat.splat_bands_pyr, {})):
    if name not in want: continue
    img, tmin, tmed = bench(fn, x, y, Hf, wvol, vals, c, W, npix, reps=(1 if name == "serial" and n > 3e6 else a.reps), **kw)
    res[name] = (img[..., 1].astype(np.float64), tmin, tmed)

if "splash" in want and a.splash and os.path.exists(a.splash):
    hs = H / 2.0
    fin = os.path.join(a.out, "splash_in.bin"); fout = os.path.join(a.out, "splash_out.bin")
    with open(fin, "wb") as f:
        f.write(np.array([n, npix], np.int32).tobytes()); f.write(np.array([-W / 2, -W / 2, dx], np.float64).tobytes())
        for arr in (x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), hs, wvol / hs**3, rho):
            f.write(np.ascontiguousarray(arr, np.float64).tobytes())
    env = dict(os.environ); env["OMP_NUM_THREADS"] = str(nth)
    r = subprocess.run([a.splash, fin, fout, str(a.reps + 1)], capture_output=True, text=True, env=env)
    first = float(r.stdout.split("first call s:")[1].split()[0]); mean = float(r.stdout.split("mean s per call:")[1].split()[0])
    img = np.fromfile(fout, np.float64).reshape(npix, npix)
    res["splash"] = (img, first, mean)       # SPLASH: first call and the mean over the repeats (it reports no minimum)

if "pynbody" in want:
    import pynbody
    warnings.simplefilter("ignore")
    s = pynbody.new(gas=n)
    s.gas["pos"] = np.stack([x, y, z], 1); s.gas["pos"].units = "kpc"
    s.gas["smooth"] = (Hf * a.pyn_h).astype(np.float32); s.gas["smooth"].units = "kpc"
    s.gas["mass"] = m.astype(np.float32); s.gas["mass"].units = "Msol"
    s.gas["rho"] = rho.astype(np.float32); s.gas["rho"].units = "Msol kpc^-3"          # dummy units: the image is only a projection when out_units is a column density
    pynbody.config["number_of_threads"] = nth
    def pyn():
        return np.asarray(pynbody.sph.render_image(s.gas, "rho", width=f"{W} kpc", resolution=npix, out_units="Msol kpc^-2",
                                                   approximate_fast=a.pyn_fast, threaded=True)).astype(np.float64)
    img, tmin, tmed = bench(pyn, reps=min(a.reps, 3))
    res["pynbody"] = (img, tmin, tmed)

# ------------------------------------------------------------------------------------------------ report
true_mass_in = m[(np.abs(x) < W / 2) & (np.abs(y) < W / 2)].sum()
print(f"\n{'code':9s} {'t_min [s]':>10s} {'t_med [s]':>10s} {'mass in image / true':>22s}")
for k, (img, tmin, tmed) in res.items():
    print(f"{k:9s} {tmin:10.3f} {tmed:10.3f} {img.sum() * dx * dx / true_mass_in:22.5f}")
ref_name = "splash" if "splash" in res else ("serial" if "serial" in res else next(iter(res)))
ref = res[ref_name][0]
mask = ref > 1e-3 * ref.max()
print(f"\ndifference of the column-density image to {ref_name} (pixels above 1e-3 of its maximum: {mask.sum()}):")
print(f"{'code':9s} {'median |rel|':>13s} {'p99 |rel|':>11s} {'rms dlog10':>11s}  ")
for k, (img, _, _) in res.items():
    if k == ref_name: continue
    im = img
    rel = np.abs(im[mask] - ref[mask]) / ref[mask]
    dl = np.log10(np.maximum(im[mask], 1e-30)) - np.log10(ref[mask])
    print(f"{k:9s} {np.median(rel):13.2e} {np.percentile(rel, 99):11.2e} {np.sqrt(np.mean(dl**2)):11.2e}")
tag = f"{os.path.basename(a.snap)[:-5]}_w{W:g}_n{npix}_s{a.scale}"
np.savez_compressed(os.path.join(a.out, tag + ".npz"), **{k: v[0].astype(np.float32) for k, v in res.items()})
print("saved", os.path.join(a.out, tag + ".npz"))
