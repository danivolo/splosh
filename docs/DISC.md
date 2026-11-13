# Disc analyses (product `disc`, layout panels `radial` and `spectrum`)

For runs with one or two black holes (PartType5). Computed once per snapshot like any product (`splosh products RUN`, or in the background when the viewer opens
the run) and read as the series source `prod_disc`. What every column means is in the docstring of `src/splosh/disc.py`; the essentials:

- **Frames.** Binary: circumbinary disc (CBD) about the centre of mass, z along the orbit normal, R in units of the semi-major axis a; each minidisc about its
  black hole in the plane of its own gas, R in units of the Roche radius R_L. Single black hole: R in code length.
- **Profiles** (families `<name>_<R>`): sigma, rhomid, hr, cs2, vphi, vR, vrat (v_phi / v_K with the binary quadrupole), mdot, vertical balance (vsup, vsupA,
  gauss), effective alpha (aR, aM), eccentricity (ecc, eccphi), m = 1 mode (A1, A1phi); with a magnetic field beta, Brms, vA2, Psi, vRPsi, Qz, Qphi, epsP.
  Minidiscs: the same with `_mini0_` / `_mini1_` in the name.
- **Angular-momentum budget** (CBD; outward fluxes through the cylinder R in units of a^2 Omega_b): Fadv, Frey, Fmax, dTdR (binary torque density on the gas),
  Tout (torque on the gas outside R) and jdot = -(sum of the fluxes) - Tout. Divided by mdot: the angular momentum carried per unit accreted mass; in steady
  state jdot / mdot = ell_0 at every R. A code's own physics (an explicit viscosity, ...) adds a term with `splosh.disc.register_flux(name, fn)` and the
  product parameter `fluxes`.
- **Effective viscosity:** nu (from all the stresses, F = 3 pi nu Sigma R^2 Omega), alphaeff, tvisc (R^2 / nu in binary orbits); scalar r_steady (the
  largest R whose viscous time is shorter than the time so far).
- **Resolution:** hH (<h> / H), nH (cells per H), toomre. Map fields (any map, any frame): `hres` = (c_s / Omega_K) / h, `ncellH` = cells per thermal scale
  height, `Qz`, `Qphi` = MRI quality of the vertical / azimuthal field. The product's Qz / Qphi are mass-weighted geometric means per shell; a projected
  map shows the mass-weighted arithmetic mean along the line of sight (larger where a few cells are well resolved).
- **Excluded gas** (`exclude_ids`, e.g. injected wind cells): mdotx, Fx (its advective angular-momentum flux), jdotx = jdot - Fx, mdott = mdot + mdotx.
- **Scalars:** r_cav, e_disc, varpi_disc, A1_lump, phi_lump, mdot_3a, Mcbd, Mmini0/1, r50/r90_mini0/1 (minidisc truncation), a, e, q.
- Parameters: `gamma`, `r_min`, `r_max`, `nbins`, `exclude_ids` (e.g. injected wind cells), `z_cut`, `fluxes`.

## Layout panels
- `type = "radial"`: a family of a series source (`source`, `prefix` or `prefixes`) averaged over time windows and drawn against its R. `windows = N` (the run in
  N equal parts, coloured by time), `"last:K"` (the last K orbits) or `[[a, b], ...]` (orbits with `xaxis = "orb"`); `divide = "<prefix>"` divides by
  another family (ratio of the window means: `jdot_` / `mdot_` = ell_0); `slope = true` draws d ln|y| / d ln R; `band` (16-84% over the window's
  snapshots, default on), `scale`, `xscale` (log), `yscale` (auto / lin / log / symlog), `hlines`, `labels` ({prefix: label}).
  `refs = true` on a slope panel adds the log-slopes of standard steady-disc closures for that quantity (sigma, hr, cs2, rhomid, Brms: Shakura & Sunyaev
  zones B and C, a magnetically supported disc with B ~ rho^(2/3), the pure flux-advection solution; disc.CLOSURE_SLOPES). `fit = "power"` or `"torqued"`
  (Sigma = A R^-n [1 - ell_0 / sqrt(R/a)], nu ~ R^n) over `fit_range` draws the fit with its parameters and errors.
- `type = "spectrum"`: Lomb-Scargle power of series columns (`source`, `columns`, `ids`) against frequency in 1/P0 (orbits^-1); `fmin`, `fmax`, `nf`,
  `detrend` (subtract a running mean of this many P0), `tmin` / `tmax` (in P0: e.g. skip the transient), `vlines` (e.g. [1, 2, 0.2]).

- `type = "scatter"`: y against x, two families of a source (a point per snapshot and R, coloured by R or time: `color`) or two columns, within `windows`;
  the binned median as a line; `xlog`, `ylog`. E.g. alpha_M against beta.
- `type = "parray"`: a 2D array a product stores per snapshot (`product`, `array`, `xedges`, `yedges`: names of its arrays), summed over `windows`;
  `norm = "column"` (each x a distribution) or none; `ytransform` none | log10 | slog. E.g. P(alpha | R) summed over time.

In the viewer, `radial` and `spectrum` are also pane kinds ("radial profile (time-averaged)", "spectrum"), saved to and opened from layouts.

Ready layouts in docs/layouts: `disc_profiles`, `disc_budget`, `disc_morphology`, `disc_mhd`, `resolution`.

Checks: tests/test_disc.py (a Keplerian disc with Sigma ~ 1/R and H/R = 0.1: slope, H/R, v_phi/v_K, vertical support, Gaussian profile, no inflow; an eccentric
ring e = 0.2 with its periapsis; a circumbinary ring; the panels on a toy product; the extra-flux hook).
