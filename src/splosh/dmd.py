"""Dynamic mode decomposition (DMD) of a run: the maps of all its snapshots at once -> spatial modes with their frequencies and growth rates.

    splosh dmd RUN [--field rho] [--width W | --width-a A] [--npix N] [--rank R] [--tmin T0] [--tmax T1] [--center C] [--align A] [--tilt T] [--name NAME]

Method (exact DMD: Schmid 2010, J. Fluid Mech. 656, 5; Tu et al. 2014, J. Comput. Dyn. 1, 391):
  1. one map per snapshot (any View: field, centre, alignment, tilt, window, resolution), by default log10 of the column density with a floor
     (1e-4 x the median of the positive pixels); the columns of the snapshot matrix X (npix^2, n_t). The time mean stays in (it is the f = 0 mode):
     with --subtract-mean it is removed first, which turns DMD into the temporal Fourier transform of the record (Chen, Tu & Rowley 2012, J. Nonlinear
     Sci. 22, 887) and biases the frequencies of a record that is not a whole number of periods;
  2. snapshots resampled linearly onto a uniform time grid when their spacing varies by more than 1% (DMD assumes a constant step dt);
  3. the rank-r truncated SVD of X1 = X[:, :-1] = U S V*; A~ = U* X2 V S^-1; its eigenvalues lambda and eigenvectors W; modes Phi = X2 V S^-1 W;
     amplitudes b from Phi b = x_0 (least squares); continuous-time omega = ln(lambda) / dt: frequency f = Im(omega) / 2 pi (cycles per time unit), growth
     rate Re(omega) (> 0 grows). Real data give conjugate pairs (+-f); the modes are ordered by |b| |Phi|.
Times are in units of the binary period P0 (the first snapshot's, two black holes) or in code time: frequencies in 1/P0 (f / f_orb) or 1/code time.
A pattern of m arms turning at the pattern speed Omega_p shows at f = m Omega_p / 2 pi; in the binary's corotating frame (--align bh) everything is
shifted by -m f_orb. Resolution in f ~ 1 / (time span); Nyquist 1 / (2 dt).

Output: <run>/analysis_viewer/dmd/<name>.npz (times, dt, period, unit, lam, freq, growth, amp, sv, rank, modes (npix, npix, rank) complex64, the mean
map, extent, the parameters) and <name>.png (the summary: eigenvalues and the unit circle, amplitude against frequency coloured by growth rate, the
singular values, the real parts of the four leading modes with f >= 0). The layout panel type `dmd` draws a saved result.

    snapshot_matrix(paths, view, log=True, floor_frac=1e-4, tmin=None, tmax=None) -> (times, X, extent)
    resample_uniform(times, X, tol=0.01) -> (times, X, dt)
    exact_dmd(X, dt, rank=20) -> {"lambda", "freq", "growth", "modes", "amp", "sv", "rank"}
    analyse(times, X, rank=20, period=None) -> result dict (ordered by amplitude, frequencies in 1/P0 when `period`)"""
import json
import os

import numpy as np


def snapshot_matrix(paths, view, log=True, floor_frac=1e-4, tmin=None, tmax=None, progress=None):
    """(times [code], X (npix^2, n), extent): the maps of the snapshots with tmin <= t <= tmax (code time), rows of each map flattened (origin lower-left)."""
    from .io import load_snapshot
    from .maps import render_map, required_gas_fields
    from .runs import snapshot_times
    times, cols, extent = [], [], None
    for k, p in enumerate(paths):
        t = snapshot_times(p)[0]
        if (tmin is not None and t < tmin) or (tmax is not None and t > tmax):
            continue
        s = load_snapshot(p, gas_fields=tuple(sorted(required_gas_fields(view))))
        img, meta = render_map(s, view)
        extent = meta["extent"]
        cols.append(np.asarray(img, np.float64).ravel())
        times.append(float(s["time"]))
        if progress:
            progress(k, len(paths), p)
    if len(cols) < 3:
        raise ValueError(f"DMD needs at least 3 snapshots in the time range, got {len(cols)}")
    X = np.array(cols).T
    if log:
        pos = X[np.isfinite(X) & (X > 0)]
        floor = floor_frac * float(np.median(pos)) if pos.size else 1e-30
        X = np.log10(np.maximum(np.nan_to_num(X, nan=floor), floor))
    else:
        X = np.nan_to_num(X)
    o = np.argsort(times, kind="stable")
    return np.asarray(times)[o], X[:, o], extent


def resample_uniform(times, X, tol=0.01):
    """linear resampling onto a uniform grid when the spacing varies by more than tol (relative). (times, X, dt)."""
    times = np.asarray(times, float)
    dts = np.diff(times)
    dt = float(np.median(dts))
    if not dt > 0:
        raise ValueError("DMD: the snapshots must have increasing times")
    if np.max(np.abs(dts - dt)) <= tol * dt:
        return times, X, dt
    tn = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    tn = tn[tn <= times[-1] + 1e-12 * abs(times[-1])]
    Xn = np.empty((X.shape[0], len(tn)))
    for i in range(X.shape[0]):
        Xn[i] = np.interp(tn, times, X[i])
    return tn, Xn, dt


def exact_dmd(X, dt, rank=20):
    """exact DMD of the snapshot matrix X (n_space, n_time) with step dt (see the module docstring)."""
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    r = int(min(rank, np.sum(s > s[0] * 1e-10))) if s.size and s[0] > 0 else 0
    if r < 1:
        raise ValueError("DMD: the maps do not change")
    U, sr, V = U[:, :r], s[:r], Vh[:r].conj().T
    At = U.conj().T @ X2 @ V / sr
    lam, W = np.linalg.eig(At)
    Phi = X2 @ V @ (W / sr[:, None])
    b = np.linalg.lstsq(Phi, X[:, 0].astype(complex), rcond=None)[0]
    omega = np.log(lam.astype(complex)) / dt
    return {"lambda": lam, "freq": omega.imag / (2 * np.pi), "growth": omega.real, "modes": Phi, "amp": b, "sv": s, "rank": r}


def analyse(times, X, rank=20, period=None, tol=0.01, subtract_mean=False):
    """exact DMD on a uniform time grid; times and rates in units of `period` when given. Modes ordered by |b| |Phi| (descending). The steady part
    stays in the data (a mode at f = 0, growth ~ 0) unless subtract_mean: subtracting the time mean makes DMD the temporal discrete Fourier transform
    of the record (Chen, Tu & Rowley 2012), whose frequencies are biased when the record is not a whole number of periods."""
    t, Xu, dt = resample_uniform(times, X, tol)
    mean = Xu.mean(axis=1)
    r = exact_dmd(Xu - mean[:, None] if subtract_mean else Xu, dt, rank)
    unit = float(period) if period else 1.0
    amp = np.abs(r["amp"]) * np.linalg.norm(r["modes"], axis=0)
    o = np.argsort(-amp, kind="stable")
    return {"times": t, "dt": dt, "period": period, "unit": "P0" if period else "code time", "lam": r["lambda"][o], "freq": r["freq"][o] * unit,
            "growth": r["growth"][o] * unit, "amp": amp[o], "b": r["amp"][o], "modes": r["modes"][:, o], "sv": r["sv"], "rank": r["rank"], "mean": mean,
            "resampled": len(t) != len(times) or not np.allclose(t, times), "subtract_mean": bool(subtract_mean)}


def result_dir(run_dir):
    return os.path.join(os.path.abspath(run_dir), "analysis_viewer", "dmd")


def save(res, path, npix, extent, params):
    tmp = path + ".tmp.npz"
    np.savez(tmp, times=res["times"], dt=res["dt"], period=np.nan if res["period"] is None else res["period"], unit=res["unit"], lam=res["lam"],
             freq=res["freq"], growth=res["growth"], amp=res["amp"], b=res["b"], sv=res["sv"], rank=res["rank"],
             modes=res["modes"].reshape(npix, npix, -1).astype(np.complex64), mean=res["mean"].reshape(npix, npix).astype(np.float32),
             extent=np.asarray(extent, float), params=json.dumps(params))
    os.replace(tmp, path)
    return path


def load(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["params"] = json.loads(str(out["params"]))
    out["unit"] = str(out["unit"])
    p = float(out["period"])
    out["period"] = None if not np.isfinite(p) else p
    return out


def leading(res, n=4, fmin=0.0):
    """indices of the n strongest modes with f >= fmin (one of each conjugate pair)."""
    return [i for i in range(len(res["freq"])) if res["freq"][i] >= fmin - 1e-12][:n]


def summary_figure(res, path, title=""):
    from matplotlib.figure import Figure
    u = "f_orb" if res["unit"] == "P0" else "1/t"
    fig = Figure(figsize=(14, 7.5), layout="constrained")
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(th), np.sin(th), color="0.6", lw=0.8)
    lam = np.asarray(res["lam"])
    sc = ax.scatter(lam.real, lam.imag, c=np.log10(np.maximum(res["amp"], 1e-300)), cmap="viridis", s=22, edgecolor="k", lw=0.3)
    ax.set_aspect("equal"); ax.set_xlabel("Re lambda"); ax.set_ylabel("Im lambda"); ax.set_title("eigenvalues (unit circle: neutral)", fontsize=9)
    fig.colorbar(sc, ax=ax, label="log10 amplitude", shrink=0.8)
    ax = fig.add_subplot(gs[0, 1])
    pos = res["freq"] >= 0
    g = np.asarray(res["growth"])
    gl = float(np.max(np.abs(g[pos]))) if pos.any() else 1.0
    sc = ax.scatter(res["freq"][pos], res["amp"][pos], c=g[pos], cmap="coolwarm", vmin=-gl or -1, vmax=gl or 1, s=30, edgecolor="k", lw=0.3)
    ax.set_yscale("log"); ax.set_xlabel(f"f [{u}]"); ax.set_ylabel("amplitude |b| |Phi|")
    span = (res["times"][-1] - res["times"][0]) / (res["period"] or 1.0)
    ax.set_title(f"{len(res['times'])} maps, dt = {res['dt'] / (res['period'] or 1.0):.3g}, span {span:.3g}, Nyquist {0.5 * (res['period'] or 1.0) / res['dt']:.3g}",
                 fontsize=9)
    fig.colorbar(sc, ax=ax, label=f"growth rate [{u}]", shrink=0.8)
    ax = fig.add_subplot(gs[1, 0:2])
    sv = np.asarray(res["sv"])
    ax.semilogy(sv / sv[0], "o-", ms=3)
    ax.axvline(int(res["rank"]) - 0.5, color="0.5", ls=":")
    ax.set_xlabel("index"); ax.set_ylabel("singular value / s_0"); ax.set_title(f"singular values (rank kept: {int(res['rank'])})", fontsize=9)
    modes = res["modes"]
    npix = modes.shape[0] if modes.ndim == 3 else int(round(np.sqrt(modes.shape[0])))
    e = res.get("extent")
    for k, i in enumerate(leading(res, 4)):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        mp = np.real(modes[:, :, i]) if modes.ndim == 3 else np.real(modes[:, i]).reshape(npix, npix)
        v = float(np.nanmax(np.abs(mp))) or 1.0
        ax.imshow(mp, origin="lower", extent=None if e is None else list(e), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"mode {i}: f = {res['freq'][i]:.3g} {u}, growth {res['growth'][i]:+.3g}", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=110)
    return path


def table(res, n=10):
    u = "f/f_orb" if res["unit"] == "P0" else "f [1/t]"
    lines = [f"  {u:>10}  {'amplitude':>10}  {'growth':>9}"]
    for i in leading(res, n):
        lines.append(f"  {res['freq'][i]:10.4g}  {res['amp'][i]:10.3g}  {res['growth'][i]:+9.3g}")
    return "\n".join(lines)


def main(a):
    """`splosh dmd RUN ...` (see the module docstring)."""
    import sys
    from .analyze import run_root
    from .io import load_snapshot
    from .maps import View
    from .runs import binary_period, run_snapshots
    paths = run_snapshots(a.run)
    if not paths:
        raise SystemExit(f"no snapshots in {a.run}")
    paths = paths[::max(1, int(a.every or 1))]
    run_dir = run_root(os.path.dirname(os.path.abspath(paths[0])))
    first = load_snapshot(paths[0], gas_fields=("Coordinates", "Masses"))
    nbh = len(first["bh_pos"]) if "bh_pos" in first else 0
    period = binary_period(paths[0]) if not a.code_time else None
    center = a.center or ("bh" if nbh else "mass-weighted")
    tilt = a.tilt or ("Lbh" if nbh >= 2 else "none")
    width = a.width
    if width is None:
        if nbh >= 2:
            from .modes import disc_frame
            width = float(a.width_a) * disc_frame(first, np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1))["a"]
        else:
            from .analyze import _auto_radius
            width = 2.2 * _auto_radius(first)
    view = View(field=a.field, center=center, align=a.align, tilt=tilt, width=float(width), npix=int(a.npix), mode=a.mode).validate()
    unit = period or 1.0
    tmin = None if a.tmin is None else a.tmin * unit
    tmax = None if a.tmax is None else a.tmax * unit
    log = (lambda k, n, p: print(f"{k + 1}/{n}  {os.path.basename(p)}", file=sys.stderr, flush=True)) if not a.quiet else None
    times, X, extent = snapshot_matrix(paths, view, log=not a.linear, tmin=tmin, tmax=tmax, progress=log)
    res = analyse(times, X, rank=int(a.rank), period=period, subtract_mean=a.subtract_mean)
    res["extent"] = extent
    d = a.out or result_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    params = {"field": view.field, "mode": view.mode, "center": view.center, "align": view.align, "tilt": view.tilt, "width": view.width, "npix": view.npix,
              "rank": int(a.rank), "subtract_mean": bool(a.subtract_mean), "tmin": a.tmin, "tmax": a.tmax, "log": not a.linear, "every": int(a.every or 1), "snapshots": len(times)}
    npz = save(res, os.path.join(d, a.name + ".npz"), view.npix, extent, params)
    frame = "corotating (align bh)" if view.align == "bh" else (f"align {view.align}" if view.align != "none" else "inertial")
    png = summary_figure(res, os.path.join(d, a.name + ".png"),
                         title=f"DMD of {'log10 ' if not a.linear else ''}{view.field}: {frame} frame, width {view.width:.3g}, {view.npix}^2 pixels, rank {res['rank']}")
    print(table(res))
    print(f"wrote {npz}\nwrote {png}")
    return res


def add_parser(sub):
    p = sub.add_parser("dmd", help="dynamic mode decomposition of a run's maps (all snapshots): modes, frequencies, growth rates -> <run>/analysis_viewer/dmd/")
    p.add_argument("run", help="the run directory (or its output/)")
    p.add_argument("--field", default="rho", help="the map's field (default rho: column density)")
    p.add_argument("--mode", default="proj", choices=["proj", "slice"])
    p.add_argument("--width", type=float, help="window width in code length (default: width-a x a with a binary, else 2.2 x the 95%% gas radius)")
    p.add_argument("--width-a", type=float, default=8.0, help="window width in units of the binary's a (default 8)")
    p.add_argument("--npix", type=int, default=128, help="pixels per side (default 128)")
    p.add_argument("--rank", type=int, default=20, help="SVD truncation rank (default 20)")
    p.add_argument("--tmin", type=float, help="first time (units of P0 with a binary, else code time)")
    p.add_argument("--tmax", type=float, help="last time (same units)")
    p.add_argument("--center", help="frames.py centre rule (default: bh with black holes, else mass-weighted)")
    p.add_argument("--align", default="none", help="none (inertial, default) | bh (corotating with the binary) | ...")
    p.add_argument("--tilt", help="frames.py tilt (default: Lbh with two black holes, face-on onto the orbit, else none)")
    p.add_argument("--linear", action="store_true", help="the map's values, not log10")
    p.add_argument("--subtract-mean", action="store_true", help="subtract the time-mean map first (DMD becomes the temporal DFT of the record)")
    p.add_argument("--code-time", action="store_true", help="frequencies in 1/code time even with a binary")
    p.add_argument("--every", type=int, default=1, help="every Nth snapshot")
    p.add_argument("--name", default="dmd", help="result name (file <name>.npz / .png; the dmd panel's `name`)")
    p.add_argument("--out", help="output directory (default <run>/analysis_viewer/dmd)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=main)
    return p
