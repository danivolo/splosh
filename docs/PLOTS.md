# Distributions, azimuthal modes, power spectra, DMD

Four plot capabilities on top of the snapshot products (products.py), the series sources and the layouts. Each has a snapshot product (computed once per
snapshot, read as the series source `prod_<name>`, its families `<prefix><number>` drawn as heatmaps against time or as `radial` panels) and / or a layout
panel type (drawn by both renderers: the bitmap dashboards and the matplotlib figures). Code: `src/splosh/hist.py`, `modes.py`, `pk.py`, `dmd.py`, the panel
types in `plotpanels.py`. Ready layouts in docs/layouts: `plots_distributions`, `plots_modes`, `plots_pk`, `plots_dmd`.

| what | panel type | product (series source) | command |
|---|---|---|---|
| histograms, CDFs | `hist` | `hist` (`prod_hist`) | |
| azimuthal modes A_m(R), phases, bending, gas quadrupole | (heatmap, radial, timeseries of the source) | `modes` (`prod_modes`) | |
| spatial power spectra P(k) | `pk` | `pk` (`prod_pk`, opt-in) | |
| dynamic mode decomposition | `dmd` | | `splosh dmd RUN` |

## Product parameters of a run
A product's parameters are its defaults, overridden by the run's own file `<run>/analysis_viewer/products/params.toml` (a table per product; `.json` also
read), e.g.

```toml
[hist]
quantities = ["density", "T", "R"]
ranges = {T = [1.0, 7.0]}        # axis units: log10 for a log quantity
r_max = 20.0

[pk]                             # an opt-in product: this table switches it on for the run
width = 16.0
npix = 512
```

The stored results are keyed by the parameters, so a changed table computes the product again. Products registered with `auto = False` (here `pk`, which
renders a map per snapshot) are computed when named (`splosh products RUN --names pk`) or when the run's params.toml has a table for them; the others
whenever `splosh products RUN` runs without `--names` (and in the viewer's background job).

## Histograms and CDFs (`hist.py`)
Quantities: those of the phase diagram (`density`, every per-particle field: `T`, `u`, `cs`, `pressure`, `vR`, `vphi`, ..., plugin fields, and the
geometry `x y z R r h v los`), in the view's unit system (`units`).
- **PDF**: the weight per unit of the axis (per dex for a log axis) divided by the total weight of the selection, so it integrates to the fraction inside
  the range. **CDF**: the fraction of the total weight at or below each bin's right edge, the weight below the range included (it starts at that fraction
  and ends at 1 minus the fraction above). The total counts the particles with a finite axis value (positive for a log axis).
- **Panel `hist`**: the gas of the panel's view window (the view keys of a map: `center`, `align`, `tilt`, `axes`, `width`, `cx`, `cy`, `mode`, `z0`,
  `units`, ...; the same selection as the `phase` panel). Keys: `quantities` (one name or a list), `hweight` (mass | count | volume), `bins` (128), `xr`
  ([lo, hi], log10 when `xlog`), `xlog` (true), `ylog`, `show` (pdf | cdf | both: the CDF on a second y axis), `snaps` (below).
- **`snaps`** (also for `pk`): unset = the current snapshot (a dashboard), or with `scope = "run"` four snapshots spread over the run; an integer N = N
  snapshots spread over the run; a list = these snapshot indices (negative from the end). One curve per snapshot (and per quantity), labelled by its time
  (t / P0 with a binary).
- **Product `hist`**: a region about the centre (spheres `r`, or cylinders `R` with `radius = "R"`): parameters `quantities` (["density", "T"]), `weight`,
  `per_dex` (10: bins of a log quantity on a fixed grid of 0.1 dex, so every snapshot has the same rows even when its range differs), `bins` (64, linear
  quantities), `ranges`, `log` ({name: bool}; default log when every value is positive), `center` (auto: the black holes' centre of mass, else the gas
  centre of mass), `tilt`, `r_min`, `r_max`, `radius`, `units`, `gamma`. Columns: `<q>_<x>` the PDF at the bin centre x (log10 for a log axis),
  `<q>cdf_<x>` the CDF at the right edge, `<q>_med`, `<q>_p16`, `<q>_p84` (weighted quantiles, in the quantity's units), `<q>_frac` (inside the range),
  `wtotal`. A quantity whose datasets a snapshot lacks (T without ElectronAbundance, ...) is left out. A heatmap panel with `prefix = "density_"` is the
  density distribution against time; `"densitycdf_"` its CDF.

## Azimuthal modes (`modes.py`, product `modes`)
- **Frame** (the rules of disc.py, reproduced in `modes.disc_frame`; disc.py is unchanged): two black holes: their centre of mass, z along the orbit
  normal, R in units of the osculating semi-major axis a; one black hole: about it, z along the gas angular momentum within `r_max`, R in code length;
  none: about the gas centre of mass (`center = "com"` forces it). The in-plane x axis is the simulation x projected on the plane (an inertial frame:
  phases comparable in time).
- **Families** per cylindrical shell (log-spaced, `r_min`, `r_max`, `nbins` = 0.5, 12, 32): `A<m>_<R>` = |sum m_i exp(i m phi_i)| / sum m_i (m = 1 ..
  `mmax` = 8; the surface density is Sigma ~ 1 + 2 sum A_m cos m(phi - phi_m), so a perturbation eps cos m(phi - phi0) has A_m = eps / 2),
  `phi<m>_<R>` = arg / m in degrees (the azimuth of one of the m maxima), `W_<R>` and `Wphi_<R>` the m = 1 vertical (bending) mode: W = 2 |sum m z
  exp(i phi)| / sum m R = tan of the tilt of a tilted ring, Wphi the azimuth of its highest point.
- **Scalars**: `A1` .. `A8`, `phi1` .. `phi8`, `W`, `Wphi` over the band `g_min` < R < `g_max` (1.5 - 5), `Mband`; with a black hole the gas tidal
  tensor at the centre (the l = 2 multipole of the gas potential): T_ij = G sum m (3 x_i x_j - r^2 delta_ij) / r^5 over the gas at r > `q_rmin` (1),
  in units of G M / a^3 (Omega_b^2 for a binary): `Qzz` (T_zz: -G M_ring / R^3 for a ring), `Q22` = |(T_xx - T_yy) / 2 + i T_xy| (in-plane m = 2) and
  `phiQ22` (its orientation, deg); with a binary `a`, `e`, `q`.
- Other parameters: `z_cut` (only |z| < z_cut R), `exclude_ids`.
- **Drawing**: a heatmap (`source = "prod_modes"`, `prefix = "A2_"`, `yscale = "log"`) is A_2(R, t); a radial panel (`prefixes = ["A1_", "A2_"]`,
  `windows`) the time-averaged A_m(R); the timeseries panel the global amplitudes and phases (`phi<m>` wrap at 360 / m degrees). disc.py's own `A1_<R>`
  (product `disc`) is the same m = 1 amplitude.

## Spatial power spectra (`pk.py`)
- **Definition**: the map q on N x N pixels of a window of width L, pixel d = L / N; the field f = `subtract` of q: `delta` (q / <q> - 1, the default),
  `mean` (q - <q>), `log` (log10 q - <log10 q>), `none`; pixels without gas (NaN, or q <= 0 for log) take the mean. f is multiplied by a `window`:
  `hann` (2D Hann taper, the default), `tukey` (25% cosine edges) or `none` (a periodic field only). F(k) = sum f w exp(-i k.x) d^2,
  P_2D(k) = |F|^2 / (L^2 <w^2>) (the window's loss of power restored). Then the variance of f is sum P_2D / L^2 = integral P_2D d^2k / (2 pi)^2
  (Parseval). P(k) is the mean of P_2D over annuli |k| in log-spaced bins from the fundamental k_f = 2 pi / L to the Nyquist pi / d (`nk` bins; bins
  without a Fourier mode, at the lowest k, are left out). k = 2 pi / wavelength in 1/length (the map's length unit).
- **Panel `pk`**: the map of the panel's view (every map view key, `post` transforms included); keys `window`, `subtract`, `nk`, `slopes` (reference
  lines, e.g. [-1.6667, -2, -3.6667]: drawn dashed through the curve's middle, labelled k^-5/3, k^-2, k^-11/3), `compensate` (draws k^c P(k): with
  c = 3.6667 a k^-11/3 spectrum is flat), `snaps`.
- **Product `pk`** (opt-in): columns `P_<n>` with n = k L / 2 pi (the number of wavelengths across the window, the same rows in every snapshot of the
  same `npix` and `nk`), `kf` (2 pi / L, 1/code length), `var`, `width`, `slope` (least-squares log-slope over `kn_lo` < n < `kn_hi`, 4 - 32).
  Parameters: `field` (rho), `mode`, `center` (auto), `align`, `tilt` (auto: Lbh, face-on onto the orbit, with two black holes), `axes`, `width` (code
  length; unset: `width_a` x a = 8 a with a binary, else 2.2 x the radius holding 95% of the gas: then fix it in params.toml, or the window changes
  from snapshot to snapshot), `npix` (256), `window`, `subtract`, `nk` (24). Heatmap: `prefix = "P_"`, `yscale = "log"`.

## Dynamic mode decomposition (`dmd.py`)
`splosh dmd RUN [--field rho] [--width W | --width-a 8] [--npix 128] [--rank 20] [--tmin T0] [--tmax T1] [--center C] [--align none|bh] [--tilt T]
[--linear] [--subtract-mean] [--every N] [--code-time] [--name dmd] [--out DIR]`

- **Snapshot matrix**: one map per snapshot (any View; default the column density, log10 with a floor of 1e-4 x the median positive pixel, centre on
  the black holes, face-on onto the orbit with two of them), flattened into the columns of X (npix^2, n_t). Snapshots unevenly spaced (> 1%) are
  resampled linearly onto a uniform grid.
- **Exact DMD** (Schmid 2010; Tu et al. 2014): X1 = X[:, :-1] = U S V* truncated to `rank` (and to the singular values above 1e-10 s_0),
  A~ = U* X2 V S^-1, eigenvalues lambda and eigenvectors W of A~, modes Phi = X2 V S^-1 W, amplitudes b from Phi b = x_0 (least squares),
  omega = ln(lambda) / dt: frequency f = Im(omega) / 2 pi, growth rate Re(omega) (> 0: growing). Modes are ordered by |b| |Phi|. Real data give conjugate
  pairs (+-f); the steady part is the mode at f = 0.
- **The mean**: kept by default (it is the f = 0 mode). `--subtract-mean` removes the time mean first, which makes DMD the temporal discrete Fourier
  transform of the record (Chen, Tu & Rowley 2012): its frequencies are then those of the record's Fourier grid, biased when the record is not a whole
  number of periods (the test shows both).
- **Units**: with two black holes, times in the binary period P0 of the first snapshot: f in f_orb = 1 / P0 and growth rates per P0 (`--tmin`,
  `--tmax` in P0); else code time (`--code-time` forces it). A pattern of m arms turning at Omega_p appears at f = m Omega_p / 2 pi; in the binary's
  corotating frame (`--align bh`) every frequency is shifted by -m f_orb. The resolution in f is about 1 / (time span), the Nyquist 1 / (2 dt): a few
  snapshots give a few modes only (rank <= n_t - 1).
- **Output**: `<run>/analysis_viewer/dmd/<name>.npz` (times, dt, period, unit, lam, freq, growth, amp, b, sv, rank, modes (npix, npix, rank) complex64,
  the mean map, extent, the parameters) and `<name>.png` (eigenvalues with the unit circle, amplitude against frequency coloured by growth rate, the
  singular values, the real parts of the four strongest modes with f >= 0), and a table of the leading modes on the terminal.
- **Panel `dmd`**: `name` (the result, default dmd), `show` = spectrum (amplitude against f >= 0, `fmax`) | eigs (the eigenvalues and the unit circle) |
  sv (singular values) | mode (`mode` = k: the k-th strongest mode with f >= 0, `part` = real | imag | abs | phase, `cmap`: a map with symmetric limits;
  its f and growth rate under it). A missing result is a blank panel that names the command.
- In Python: `dmd.snapshot_matrix(paths, view)`, `dmd.analyse(times, X, rank, period)`, `dmd.load(npz)`.

## Checks (tests/test_plots.py)
- hist: uniform values (PDF 1, CDF linear, median, quantiles), the weight below the range in the CDF, log-uniform values (1/4 per dex over 4 dex),
  weights; the panel on a box of gas (uniform x in the window, log-uniform density), several quantities and snapshots, both renderers; the product's fixed
  0.1-dex grid, the union of rows of two snapshots in a heatmap (each column integrates to 1); params.toml and the opt-in rule.
- modes: a ring with Sigma ~ 1 + 0.4 cos 2(phi - 30 deg) about a binary (A_2 = 0.2 at 30 deg, A_1 = A_3 = A_4 ~ 0; the tidal tensor T_zz = -M / R^3,
  |T_22| = 0.75 eps M / R^3 at 30 deg); a ring tilted by 10 deg (W = tan 10 deg, the highest point 90 deg from the node); a pattern turning by 0.2 rad per
  snapshot (phi2 against time), the heatmap and the radial panel.
- pk: a sine of 8 wavelengths across the window (the peak at k = 8 k_f), Parseval (the variance), Gaussian random fields with P ~ k^-3 (slope -3 for
  every window, and for the log of a lognormal field), the reference lines; the panel (k in 1/length from the fundamental to the Nyquist, both
  renderers) and the product (8 a window about a binary).
- DMD: a synthetic movie with an m = 2 pattern turning with temporal frequency 1.5 and a decaying (rate -0.8) m = 1 pattern turning at 0.4: both
  recovered to 1e-6, the steady part at f = 0; with --subtract-mean the DFT bias; the period scaling; resampling. The command on 16 snapshots of a
  particle ring with an m = 2 pattern turning at Omega = 1 (f = 2 Omega / 2 pi recovered to 2%, growth ~ 0), its files, and the four panel views in
  both renderers.
