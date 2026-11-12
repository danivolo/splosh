# Using SPLOSH

## Commands
| Command | What it does |
|---|---|
| `splosh info SNAP [--json] [--units code\|cgs\|astro]` | summary (with the unit system and, if the snapshot has ElectronAbundance, the temperature): counts per particle type, datasets, header units and time, box, gas mass / H / density / u percentiles, densest particle, centre of mass, black holes (ids, masses, positions). Plugins add sections. |
| `splosh render SNAP --out FILE` | one map to `.png` (`--cmap`, `--scale log\|linear\|asinh[:F]\|histeq`, `--post`, `--vmin/--vmax`), `.npz` (`img`, `meta`) or `.npy`. Goes through the mapstore, so a repeated request is a cache hit. `--no-cache` bypasses it. |
| `splosh render SNAP --tiles 512 --out FILE` | the same map assembled from 512 x 512 pixel tiles (`--tiles` 64 ... 1024; `--workers N`): bounded memory and `--npix` above 8192 (see Tiles). `.npy` is written as a memory-mapped file, `.png` is coloured and compressed band by band. |
| `splosh tiles SNAP [--level L or A:B] [--tile 256] [--list] [--out DIR]` | list, or pre-render into a tile store (cache warm-up), the tiles covering the view window at a zoom range (see Tiles). |
| `splosh shot SNAP --out img.png` | a finished picture (see Headless pictures). |
| `splosh explore SNAP --out e.png` | the discovery sheet: one map shown six ways (log, asinh, histogram-equalised, contrast against the azimuthal mean, against a smoothed copy, gradient); see Finding faint structure. |
| `splosh sheet RUNDIR_or_SNAPS --out sheet.png` | contact sheet of many snapshots with shared colour limits. |
| `splosh movie RUNDIR_or_SNAPS --out m.mp4` | frames with the same limit policies, encoded with ffmpeg if available. |
| `splosh serve PATH...` | local server + browser viewer. Each directory (or `label=path`) is a RUN, listed in natural sort order (snap_2 before snap_10); loose snapshot files together form one run; a run directory without snapshots of its own but with an `output` subdirectory uses that. Several runs can be shown in the panes with time alignment (see Viewer and server). |
| `splosh overlays` / `splosh inspect SNAP X Y [--limit N]` | the overlays with their options; the gas particles contributing at a map position (JSON, see Overlays). |
| `splosh fields`, `splosh store [--clear]` | list fields with their cgs unit and the PartType0 datasets each needs, presets, plugins; mapstore statistics. |

View options (shared by `render`, `shot`, `sheet`, `movie` and `serve`, which uses them as defaults): `--center`, `--align`, `--axes xy|xz|yz`, `--mode proj|slice`, `--field`, `--width`, `--npix`,
`--cx --cy` (pan), `--z0` (slice plane), `--weight auto|mass|volume|column`, `--units code|cgs|astro`, `--gamma`, `--xh`, `--mu`, `--bunit`, `--post SPEC`, `--exact`, `--preset NAME`, `--view k=v,k=v`.

## Frames
Centre (`--center`): `com` (mass-weighted gas + black holes), `mass-weighted` (gas), `bh` (black-hole particles; the binary centre of mass for two), `peak` (densest gas particle),
`ids:A,B,...` (alias `pair:A,B`), `xyz:x,y,z`. Black-hole masses are the dynamical `Masses` dataset. The centre velocity is subtracted from velocity fields.
Alignment (`--align`): a rotation about z only, applied after centring: `bh` (the vector from the lowest-ID to the second-lowest-ID black hole becomes +x), `pair:A,B` (A to B becomes +x),
`vec:x,y,z`, `none`. The rotation is the active rotation by -theta with theta = atan2(dy, dx) of that vector, the same as `rotate_z(-theta in degrees)` in pynbody (tested).

Tilt (`--tilt`, the viewer's "tilt" menu, `tilt` in profiles): any 3D rotation after the alignment, so any line of sight can be the view's z axis. Terms joined by `+`
apply left to right, angles in degrees: `incl:I[,PA]` (rotate by PA about z, then by I about x: 0 face-on, 90 edge-on), `euler:A,B,C` (z-x-z), `los:x,y,z` (that
direction becomes the line of sight of an xy view), `L[:R]` (the gas angular momentum within R of the centre becomes +z: face-on onto the disc), `Lbh` (the black
holes' orbital angular momentum becomes +z). Velocities rotate with the positions. Example: `--tilt L:2+incl:90` shows the disc within R = 2 edge-on.

## Maps
- Projection: `rho` = column density (conserves the mass); other fields are the weighted mean along the line of sight, default weight = mass (density-weighted).
- Slice at `z0` along the line of sight: `rho` = SPH density `sum m W`; other fields = normalised SPH interpolation `sum (m/rho) q W / sum (m/rho) W` (default weight volume), pixels where
  the weight sum is under 5% are NaN. A slice samples the kernel at pixel centres: use pixels finer than the typical H.
- `weight=column`: the sum `sum m q K` without the normalisation: in a projection the column density of q (e.g. `tau_bin`, the torque on the binary per unit gas mass, becomes the
  torque per unit area: where the gas widens (> 0) or shrinks (< 0) the orbit; use a diverging colour map with symmetric limits), in a slice `rho q`.
- Fields: `rho`, `u`, `vx`, `vy`, `vz` (aligned frame), `vlos`, `speed`, `H`, the derived fields below; plugins register more.

## Units and derived fields
`splosh.units.Units` reads the code units from the header (`UnitLength_In_CGS`, `UnitMass_In_CGS`, `UnitVelocity_In_CGS`, also spelled `UnitLength_in_cm`, `UnitMass_in_g`, `UnitVelocity_in_cm_per_s`;
`HubbleParam`, `ComovingIntegrationOn`). A missing scale falls back to 1 kpc, 1e10 Msun, 1 km/s (GADGET customary) and `info` warns. For a comoving snapshot the flags are reported but no a or h factor
is applied: the factors are those of the stored code units. `splosh info` prints the unit system (time, density, column density, energy per mass, pressure scales in cgs).

`--units code|cgs|astro` (`View.units`, default `code`: nothing changes) converts the map values: cgs (g cm^-2 or g cm^-3, cm s^-1, ...), astro (Msun pc^-2 or Msun pc^-3, km s^-1, K, Myr, pc).
The window (`--width`, `--cx`, `--z0`) always stays in code length units. Every map's metadata has `quantity_unit` (`unit`, `factor` from code units, `cgs_unit`, `cgs_factor`, `code_unit`; `known` is false
for a plugin field registered without a unit, which is then never converted), `length_unit`, `time_unit`, `time_physical` and `extent_physical` in the chosen system. Temperature is always in K.

Derived fields (`splosh fields` lists them with their datasets; the loader reads only those): `T`, `cs`, `pressure`, `entropy`, `mach`, `jz`, `j`, `vR`, `vphi`, `beta`.
`T = (gamma-1) mu m_p / k_B u UnitVelocity^2`, `mu = 4 / (1 + 3 X + 4 X x_e)` (x_e = `ElectronAbundance`, X = `--xh` when given, otherwise 1 - Z - Y per particle from the snapshot's `Metallicity` columns 0 and 1 when present, else 0.76; other metals neglected), `--mu` replaces it;
`cs = sqrt(gamma (gamma-1) u)`; `pressure = (gamma-1) rho u`; `entropy = P / rho^gamma`; `mach = |v - v_frame| / cs` (frame velocity from the centre rule); `jz = x vy - y vx` and `j = |r x v|` about the
frame centre in the aligned frame; `vR`, `vphi` cylindrical about the z axis through the centre; `beta` (needs `MagneticField`) follows GIZMO's convention: internally mu_0 = 1 so P_B = B_int^2/2, and the snapshot stores B_out = B_int * sqrt(4 pi P_unit) / UnitMagneticField_in_gauss
(gizmo `io.c`, `allvars.h`). Hence B_gauss = B_out * UnitMagneticField_in_gauss and beta = P_cgs / (B_gauss^2 / 8 pi). The snapshot header does not carry `UnitMagneticField_in_gauss`: `--bunit` gives it, otherwise
it is read from a `*usedvalues*` or `gizmo.params` file in the snapshot's directory or its parent, otherwise GIZMO's default sqrt(4 pi) is used. Check `--bunit` for runs that set it (often 1).
`--gamma` (default 5/3) applies to all of them. No neighbour searches (no vorticity or divergence). A missing dataset gives an error naming it.

## Kernel convention flag
The convention everywhere is `hscale = 1`: H = `SmoothingLength` = the kernel's support radius (GIZMO). `--hscale S` (View.hscale) multiplies the stored smoothing length before splatting;
`hscale = 2` is a LEGACY option only: it reproduces a reader that treats H as the Monaghan h (support 2h), e.g. pynbody given `smooth = H`, so old figures made that way can be compared. It is not
the code's kernel. Every map's metadata carries a `kernel` note and says so when hscale is not 1. Black-hole centring (`bh`, `pair`, `ids`, `com`, `splosh.binary_com`) weights with the dynamical `Masses`, never `BH_Mass`.

## Headless pictures: `shot`, `sheet`, `movie`
No GUI and no matplotlib: the pictures are drawn with numpy and a small built-in bitmap font (`bitmapfont.py`) and written with the existing PNG writer. The maps come from the mapstore, so reruns are fast.

    splosh shot SNAP --out img.png [view options] [--cmap --scale --vmin --vmax --pct lo,hi --title TEXT --no-scalebar --no-colorbar --font-scale N --no-cache --store DIR]
    splosh sheet RUNDIR_or_SNAPS... --out sheet.png [--every N | --snaps a,b,c] [--ncols K] [--limits global|frame|fixed] [--pct lo,hi] [--vmin --vmax] [view/style options]
    splosh movie RUNDIR_or_SNAPS... --out m.mp4 [--fps 12] [--every N | --snaps a,b,c] [--limits ...] [--keep-frames --verbose] [view/style options]

- `shot`: the map in the chosen colour map and stretch, a colour bar (label from the map metadata and the unit of the field in code units; the unit names are taken from the header, e.g. `pc`,
  `km/s`, `Msun`, otherwise `L`, `M`, `V` with their cgs value in the footer), a scale bar (1-2-5 length, ~20% of the window), the snapshot time (code units, and physical time if the header
  has length and velocity units), an optional title and a footer with the view (mode, field, axes, centre rule, align, width; a `hscale` other than 1 is flagged as the legacy kernel convention).
  Limits: percentiles `--pct` (default 1,99.9) of the map, or `--vmin/--vmax`. `--out x.pdf` / `x.svg` or `--backend mpl` use matplotlib if it is installed (imported only then; a clear error
  otherwise): axes in physical units, same colours, stretch and limits.
- `sheet`: one tile per snapshot (time label per tile, scale bar on the first tile), `--ncols` default ~sqrt(N). `RUNDIR` may be a directory of snapshots, a run directory with an `output/` subdirectory,
  or files; `--every N` keeps every Nth file, `--snaps` selects by the number in the file name. Colour limits (`--limits`):
  `global` (default): one pair locked over the selection, min/max of all maps, or with `--pct lo,hi` the percentiles of all selected maps together (min/max exact; percentiles exact up to
  300k pixels per map, then from a regular subsample of each map); shared colour bar. `frame`: percentiles (`--pct`, default 1,99.9) of each map; each tile prints its own range, no shared bar.
  `fixed`: `--vmin --vmax`. A `--vmin` or `--vmax` with `global`/`frame` replaces that bound only. Log stretches use the positive pixels.
- `movie`: the same frames (colour bar, scale bar, time, footer; one size for all frames, padded to even dimensions) with the same policies; `global` needs two passes (the second is served by the
  mapstore, no frame is held in memory). With `ffmpeg` on PATH (or from the `ffmpeg=` argument of the API) the frames are written to a hidden directory next to `--out`, encoded
  (`ffmpeg -y -framerate F -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out`; other extensions get no codec flags) and removed. Without ffmpeg, or if it fails, they stay as
  `<out stem>_frames/frame_00000.png ...` and the message prints the command to encode them. Nothing is ever installed.
- Sidecar: every image gets `<out stem>.json` (if that name already belongs to another output, `<out>.json`): schema, SPLOSH version, git commit of the source tree (null for an installed copy),
  creation time, wall time, command, the snapshots (absolute path, name, time, content hash), the full `view` (all `View` fields, so centre rule, align, axes, mode, field, width, npix, `hscale`)
  plus a summary with the kernel note, the limits actually used (policy, `pct`, `vmin/vmax`, or `per_frame` for the frame policy), colour map, stretch, title, layout, image size, cache hits,
  and for movies fps, the ffmpeg command and the frames directory.
- Reproduce: `splosh shot --from-sidecar img.json` (also `sheet` / `movie`) redoes the picture with the recorded snapshot paths, view and the limits as used (so a percentile policy is not
  recomputed) and writes `<stem>_repro.<ext>` (or `--out`), never over the original; with the same snapshot files the PNG is byte-identical (tested). A changed snapshot gives a warning (hash).

Python API (`splosh.shot`, `splosh.contact_sheet`, `splosh.movie`, `splosh.figures.replay`):

    pic = splosh.shot("snap_010.hdf5", View(field="u", width=5), cmap="inferno", scale="linear", title="u")   # Picture
    pic.rgba          # (ny, nx, 4) uint8;  pic.png() -> PNG bytes;  pic.limits;  pic.meta;  pic.sidecar (dict)
    pic.save("u.png") # writes u.png and u.json
    splosh.contact_sheet(["run/output"], View(width=5), every=5, limits="global", pct=(1, 99.9), ncols=4).save("sheet.png")
    splosh.movie(["run/output"], "m.mp4", View(width=5), fps=12)   # dict: encoded, frames_dir, command, message, sidecar

## Overlays and click-to-inspect
`shot`, `sheet` and `movie` take `--overlay SPEC` (repeatable; Python: `overlays=[...]`). Overlays are drawn onto the map before the labels by a small numpy rasteriser (anti-aliased lines, arrow heads, discs, rings; no matplotlib), and the spec strings are
recorded in the sidecar, so `--from-sidecar` redraws them. The same data is served to the browser: `GET /overlay?snap=I&spec=vectors:v&spec=particles:bh,star&<view options>` and `GET /inspect?snap=I&x=..&y=..&limit=N&<view options>`.
A spec is `name[:arg,arg,k=v,...]`; `splosh overlays` lists the options and defaults.

| Spec | What |
|---|---|
| `vectors:v` / `vectors:B` | arrow glyphs on a regular `grid` x `grid` (default 24). The grid values are the view's own weighted projection (slice: weighted slice) of the two components along the horizontal and vertical axes of the view (`render_map` with field vx, vy, or Bx, By at npix = grid): v is the velocity in the aligned frame with the view centre's velocity subtracted, B the magnetic field rotated with the frame (needs `MagneticField`; a clear error otherwise). Arrows are centred on the cell centre. `arrows=prop` (default): length = `len` x cell x \|v\| / `ref`, where `ref` (option; default the 99th percentile of the drawn magnitudes) is the magnitude drawn as `len` cell spacings (default 0.9); `arrows=unit`: all arrows `len` cells long. `min`: cells below this magnitude are not drawn; `max`: lengths saturate at it. `color`, `lw` (px), `halo` (dark outline, px), `alpha`, `head`, `legend` (a reference arrow with its value, top right, in km/s when the header says so; B in G). |
| `streamlines:v` / `streamlines:B` | RK4 integration of dx/ds = v/\|v\| (arc length) on the same gridded field (bilinear, `grid` default 32), both directions from seeds on a regular `seeds` x `seeds` grid (default 14), visited from the window centre outwards. Density limit: `density` occupancy cells per axis hold one line each (a seed in an occupied cell is skipped, a line stops where it meets another one); length limit `maxlen` (fraction of the window, per direction); stops at the window edge, in cells without a value, at stagnation (\|v\| < `stag` x p99 of the field, or `min` as an absolute value) and when a loop closes. Lines shorter than `minlen` are dropped. Items are in flow order with an arrow head at the middle (`heads=false` to omit). |
| `particles:bh,star,dm` | points by type: bh = PartType5, dm = PartType1, dm2 = 2, type3 = 3, star = PartType4 (or pt1..pt5). Loaded read-only (`load_snapshot(types=(1, 4))`, existing loads unchanged); a type the file lacks gives a note, not an error. At most `max` (default 2000) per type inside the window, a deterministic random subsample (`seed`); `size` px, `color` (name or #rrggbb), `style` dot or ring, `depth` (only within +- depth of the slice plane); per type `star.color=gold,bh.size=8`. |
| a plugin name | any function registered with `plugins.register_overlay(name, fn)`, `fn(snap, frame, view) -> list of dict`; options are only passed to functions that declare them. |

`inspect(snap, view, x, y)` (Python, CLI `splosh inspect`, server `/inspect`): x, y are window coordinates. Slice: the gas particles whose kernel reaches the point (x, y, z0) (3D distance < H), nearest first; projection: the particles whose projected kernel covers (x, y),
ranked by contribution w f(s, H) (f: the column integral of the cubic spline); the contribution is normalised over all of them (w = mass, or mass / density for the volume weight of a slice of a quantity). Each entry has index, id (if loaded), pos (frame), H, rho, u, T (when ElectronAbundance or `--mu` allow it), vel (frame), mass, weight, kernel, contribution, rank and the smoothing circle (centre and radius in window coordinates; in a slice the cross-section radius sqrt(H^2 - dz^2); `H` is the full support radius).

### Overlay data (for other front ends)
Everything is plain JSON in window coordinates (the frame's horizontal / vertical axes, code length; window = `extent` = [x0, x1, y0, y1]; the map pixel (i, j), j from the bottom, centre is x0 + (i + 0.5) w / npix). `/overlay` returns `{"extent", "npix", "time", "items": [...], "notes": [...]}`; items:
- `{"kind": "arrow", x, y, dx, dy, magnitude, head, color, linewidth, alpha, halo, overlay}`: tail (x, y), tip (x + dx, y + dy), `head` = head length / arrow length, magnitude in code units.
- `{"kind": "polyline", points: [[x, y]...] (flow order), magnitude: [...], head: index or null, color, linewidth, alpha, halo, overlay}`.
- `{"kind": "points", type, x: [...], y: [...], depth: [...], id: [...], mass: [...], size_px, style: "dot"|"ring", color, alpha, n_total, n_drawn, overlay}`; `{"kind": "circle", x, y, r, fill, color, linewidth, alpha}`; `{"kind": "legend", text, length, value, color, head, linewidth, overlay}` (reference arrow of length `length`).
- items without `kind` (old markers: `x`, `y` and anything else) are drawn as rings. Colours are [r, g, b] 0..255. `overlays.draw_items(rgb, items, extent)` is the reference rasteriser (row 0 = top).

## Finding faint structure: transforms, stretches, contours, `explore` [verified: tests/test_explore.py, tools/browser_smoke.py section_explore]
A colour map has a few dozen distinguishable steps. A linear or log stretch spends them evenly over the VALUE range, so a few-percent ripple on a steep or plateau background (spiral arms, a gap, a
bar in a disc) gets one colour. The tools below spend them where the information is. Look first, quantify second: a picture suggests, only a measurement shows (see the last paragraph).

**`View.post` (`--post SPEC`, `post=` in the server, layouts and analyze maps)** transforms the finished map. It is part of the View, so it is in the cache key, the sidecar and `--from-sidecar`.

| Spec | Map | Default colours |
|---|---|---|
| `resid` = `resid:azim` | `f / f_bar(R) - 1`, `f_bar` the mean of the finite pixels in rings one pixel wide about the frame origin (the black hole, or what `--center` gives). The profile is interpolated at each pixel's own radius, so a steep profile does not bias the ring | RdBu_r, linear, limits (-v, v) with v the 99th percentile of \|f\| |
| `resid:mean` | `f / mean(f) - 1` over the window | the same |
| `resid:smooth:S` | `f / G_S(f) - 1`, a Gaussian of standard deviation S (code length): an unsharp mask. NaN-aware | the same |
| `grad` / `grad:S` | `\|grad log10 f\|` in dex per length unit; the map is first smoothed with a Gaussian (S in code length, default two pixels), so the noise of an SPH map does not dominate | magma, asinh |

For fields that are positive by nature (rho, u, H, speed, T, cs, pressure, entropy, mach, beta) a residual is a dimensionless contrast and the gradient is of the logarithm. Any other field (vx, vlos, jz, vR, vphi,
plugin fields) has no safe ratio: the residual is the difference `f - f_bar` in the field's own unit and the gradient is `\|grad f\|` per length. The azimuthal mean only makes sense for something that is roughly a disc about the frame origin (on the merger snapshot of the spin run, centred on the centre of mass, it is dominated by the binary itself: use `resid:smooth` there, which needs no geometry). The azimuthal mean is taken over the pixels IN THE WINDOW: in a zoom
that shows only part of each ring the reference is the average of the visible part, so the contrast is relative to that. Say so in a caption when it matters. A transform needs the whole window: it is
never tiled (`/api/tile` and `render --tiles` refuse it; the viewer then uses `/api/map`).

**Stretches** (`--scale`, the viewer's scale menu, `scale =` in analyze and layouts): `log`, `linear`, `asinh[:F]` and `histeq`. `asinh:F` is linear below, logarithmic above a scale `F x mean(\|vmin\|, \|vmax\|)`;
F = 0.1 is the old fixed behaviour (plain `asinh`), smaller values are more logarithmic and lift the faint end, a large F is linear. `histeq` (histogram equalisation) maps each pixel to its RANK, so every populated
range of values gets colours in proportion to its pixel count: no value range can be squeezed into one colour. It is for discovery, not for quantitative figures (the colour bar shows quantiles, 0, 10, 25, 50, 75, 90,
100 %, and the stretch belongs to one picture: `shot` and the viewer only, not sheets, movies, products or layouts). Pixels equal to exactly 0 (the empty background of a projection) are left out of the ranking.
The colour limits: `limits.SYM_PCT` (99) is the default upper percentile for symmetric limits; `--pct` replaces it.

**Contours** (`--overlay contours:FIELD,levels=log,n=8,step=0.5,smooth=2,top=0,post=none,color=white,lw=1`; the viewer's overlay panel) are lines of constant value of a map of the same window: absolute, they
do not depend on how colours are perceived. `levels=log` gives `n` levels `step` dex apart, the highest at `top` (default: the 99.5th percentile of the map); `levels=lin` n levels evenly between the 5th and
99.5th percentile; `levels=1e-3;1e-2` explicit values. `smooth` (pixels, default 2) smooths first, because raw SPH contours are ragged. The default FIELD is the view's own; the contoured map is the raw one
unless `post=` says otherwise, so contours of the density lie on a contrast map. Marching squares, linear interpolation along the cell edges, saddles resolved by the cell mean (`overlays.contour_paths`).

**`splosh explore SNAP --out e.png [view options] [--ncols 3]`** draws the discovery sheet: the same map as log with percentile limits, asinh F=0.02, histogram-equalised, the contrast against the azimuthal
mean and against a smoothed copy (W/25), and the gradient of the logarithm (smoothing W/150). The map is rendered once, the transforms are applied to it. Every tile has its own colours and prints its range. Make it
routine for a new dataset, then choose the representation that shows the physics for the figure and state the normalisation in the caption. Python: `splosh.figures.explore(snap, view, panels=[{title, post, scale, cmap}, ...])`;
the sidecar records the panels, so `--from-sidecar` redoes it.

**In the viewer:** the `transform` menu (with a width box for the smoothed and gradient maps) switches a pane to a transform, with its default colours; turning it off restores the previous colours. `asinh F` sets the
softening, `histeq` is in the scale menu, the pixel histogram under the limit boxes (bins in log10 for a log stretch) has two draggable lines for the colour limits (manual limits; the `auto` box restores the percentiles;
for a contrast map the lines stay symmetric about 0), and the colour bar labels five stretch positions (0, 25, 50, 75, 100 %) whatever the stretch. `contours` is in the overlay panel. The difference menu gained
`A / B - 1` (a relative difference; the `d` key still cycles off, A - B, log10(A/B)).

**Measure it (`analyze`):** `[[profile]]` now also gives `<q>_std`, the weighted scatter in each bin (`sqrt(<q^2> - <q>^2)`); the quantities `A<m>` and `phi<m>` (m = 1, 2, 3, ...) are the azimuthal Fourier amplitude
`\|sum w exp(i m phi)\| / sum w` and the phase of the m-fold pattern in each ring about the frame's z axis (A1: one-armed or off-centre, A2: bar or two arms); `geometry = "pdf"` makes it a probability density
function of its first quantity (`rmin`, `rmax` = the range, no `auto`; columns `rbins`, `pdf` per dex or per unit, `frac`, plus the other quantities averaged per bin, e.g. the temperature at each density); and the opt-in
scalar `modes` (`[[scalar]] names = ["modes"], params = {center = "bh", rmin = 0.5, rmax = 5, orders = [1, 2]}`) puts A1, A2, phi1, phi2 ... of the annulus into `scalars.csv`, to plot against time. See `docs/analyze.example.toml`.

## Tables and profiles (library)
`splosh.tables`: reader for per-task text logs (`read_rows`, `load_table`: several layouts of one file family, truncated lines, restart overlap dropped, `Schema` for named columns).
`splosh.profiles.radial_profile`: particle-binned radial profile (cylindrical or spherical, log or linear bins, weighted means), bins as pynbody's `Profile`.

## Mapstore
`$SPLOSH_STORE`, else `~/.cache/splosh/maps`, or `--store DIR`: one `.npz` per map (float32 image + JSON metadata: time, window, frame, weighting, kernel note, timings).
Key = hash of (snapshot signature, view with floats rounded to 6 significant digits). The snapshot signature hashes the header, the size and 8 sampled 64 kB blocks, so it is cheap for GB files
and survives copying, but a change outside the sampled blocks of a same-size file would be missed.

## Tiles (deep zoom, very large images) [verified: tests/test_tiles.py, bench/tiles_bench.py]
`splosh.tiles` renders a View as square tiles of `tile` pixels (256 default; 64, 128, 512, 1024 allowed), each independently and in parallel, each cached in the mapstore.
- Tile grid: tile (ix, iy) covers [ox + ix tw, ox + (ix+1) tw) x [oy + iy tw, oy + (iy+1) tw) in the frame coordinates of the view (after centring and alignment, like `--cx --cy`) and holds `tile` x `tile` pixels; indices may be negative.
  **Row 0 of a tile is its lowest y**, like maps (flip for display); column 0 is the lowest x.
- Global grid (the one the server and `splosh tiles` use): origin (0, 0); zoom level L (any integer) has tw = `base_width` / 2^L, pixel size base_width / (tile 2^L): widths are powers of two of the base width, tile edges of
  neighbouring levels coincide, and every window at that level reuses the same tiles (pan and small zooms hit the cache). `base_width` defaults to the view width rounded up to a power of two; a server fixes one value.
  `tile_grid(view, tile, base_width=None, level=None)` gives the tiles (`grid.tiles()`, `grid.extent(ix, iy)`) covering the window at the coarsest level whose pixel is at most `width / npix`.
- Window grid (`render_tiled`, `render --tiles`): origin at the window's lower-left corner, pixel = width / npix, so the tiles are the pixels of `render_map(view)`.
- A tile is rendered by `render_map` on the particles whose support (H x hscale) touches it, found with a binning by tile cell built once; particles straddling a tile edge or corner contribute to every tile they touch. Each tile is rendered with a 16 pixel
  margin that is cropped: the engine renormalises the weight of particles smaller than 6 pixels over the part of their footprint inside the image, which would otherwise overcount at every tile edge. Tiles run on `workers` Python threads, each with
  ncpu / workers numba threads (nothing nested).

      import splosh.tiles as T
      img, meta = T.render_tiled(snap_or_path, View(npix=16384, width=4, ...), tile=512, out="big.npy")   # (16384, 16384) float32 memory-mapped, row 0 = lowest y
      g = T.tile_grid(view, 256, base_width=16.0)            # global grid: g.level, g.tw, g.tiles()
      tile, meta = T.render_tile(snap, view, g.level, ix, iy, tile=256, base_width=16.0, store=store)
      for ix, iy, tile, meta, hit in T.iter_tiles(snap, view, g, store=store): ...                  # progressive, bounded memory

- Accuracy against the monolithic `render_map` (pixels farther than 16 from the window border: `render_map` itself renormalises and so overcounts particles of 1-6 pixels cut by ITS window border, the tiles do not): the exact engine (`--exact`) is identical up to float32 summation order
  (largest difference 1e-6 of the maximum on the real snapshot); the default multi-resolution engine merges the large particles on coarser grids of the tile instead of the whole image, so the two differ by the engine's own error,
  1e-3 level of the maximum (the tiled result is as close to the exact engine as the monolithic one: 2.9e-4 / 1.7e-3 of the maximum at widths 10 and 2). Slices and weighted fields likewise; a slice with `--weight mass` of a non-density field masks pixels under 1e-6 of the tile's own weight maximum
  (monolithic: of the image's), so only pixels that faint can differ.
- Speed: see docs/VALIDATION.md. Tiling is for memory (images above 8192 pixels, a bounded working set) and caching (reusing tiles); it is not faster than the monolithic render when that fits in memory.
- Server: `GET /api/tile?snap=I&level=L&ix=X&iy=Y[&tile=256][&base=B]` plus the usual view options (`field`, `mode`, `axes`, `center`, `align`, `z0`, `weight`, `units`, ...; `cx`, `cy`, `width`, `npix` are ignored). The answer is
  `tile x tile` float32 little endian (gzip if accepted), row 0 = lowest y, `X-Splosh-Meta` = the map metadata plus `n` (= tile), `stats` and `tile` = `{tile, level, ix, iy, extent [x0, x1, y0, y1], tile_width, base_width}`; the same headers as `/api/map`. `base` defaults to the server's default width rounded up to a power of two
  (`state.defaults.width`), so a client that wants other base widths must pass it on every request. Token and Host rules are those of every endpoint; 400 for a missing or invalid parameter (tile not in 64 ... 1024, |level| > 40, unknown field).
  A client at level L over a window needs the tiles `ix` from floor((x0 - 0) / tw) to ceil(x1 / tw) - 1, same for y, with tw = base / 2^L (`tile_grid` does this in Python); tile (ix, iy) is placed at pixel (ix * tile, iy * tile) of the level's global image.
- `splosh tiles SNAP --level 0:3 --width 40 --out DIR`: renders the tiles covering the window at levels 0 to 3 into the tile store DIR (a mapstore directory, so `splosh serve --store DIR` finds them when it uses the same view options and base width: `serve`'s default `--width` rounded up to a power of two, or `--base-width` here) and writes `DIR/tiles_index.json` (level, ix, iy, extent, key). `--list` only prints them and
  how many are cached; `--max-tiles` (default 4096) guards against an unintended request.

## Viewer and server
`splosh serve` binds to 127.0.0.1 only, requires a random token on every request (query `token=` or header `X-Splosh-Token`) and rejects other `Host` headers. The page: pan (drag), zoom
(wheel, re-requests a map for the new window after 200 ms; the old map is transformed meanwhile), double click to centre, snapshot slider and arrow keys, field / projection-slice / axes / centre and align
rules, colour map, log / linear / asinh, auto or fixed limits, hover read-out of the map value, black-hole markers, scale bar. Maps arrive as raw float32 (gzip if the browser accepts it), so changing
colour map, stretch or limits never calls the server.

Panes, blink, difference, keys (`?` or `h` in the page shows them):
- Panes: `panes` selector or `p` (1 to 4). Each pane has its own snapshot, field, mode, axes, colour map, stretch and limits; the centre / align rule, pixels and units are shared. The selected pane (click, or keys `1`..`4`)
  is the one the controls edit. `link pan/zoom` (key `l`): the panes share centre and width, so pan, zoom, double click and `r` act on all of them, and the hover position is shown in every pane (cross-hair in the others). Unlinked, each pane has its own window.
  Arrow keys (and `<` `>`) step all panes by one, keeping their offsets (alignment `none`) or aligned in time (below), when `step all panes` is on (Shift+arrow does the opposite: only the selected pane).
- Blink: `a` / `b` store the selected pane's state (snapshot, field, mode, axes, slice plane, colours, view) as A / B; `space` flips the pane between them. The `snap`, `field` and `view` boxes say which parts are applied (default snapshot and field, so
  pan and zoom stay). A pane that is blinking shows a `BLINK A` / `BLINK B` badge and a coloured frame; maps already seen are kept in the browser (about 200 MB), so the flip is immediate and sends no request.
- Difference: `d` cycles off, `A - B`, `log10 (A / B)` (also the selector). A and B are the stored states, at the pane's current window, so they can be two snapshots, or two fields of one snapshot (a ratio of different fields is allowed; read the unit with care).
  Computed in the browser from the two float32 maps with the diverging map `RdBu_r` and symmetric limits +-L (auto: the 99.5 percentile of |d|; `min`/`max` give L = the larger absolute value); pixels where a ratio is undefined (a value <= 0 or NaN) are grey.
- Hover: the value under the cursor in the unit system of the `units` selector (code, cgs, astro: the server converts the map, `quantity_unit.unit` of the metadata is printed after the value; dimensionless fields print none), for projections, slices and derived fields alike;
  slice pixels with no data (weight under 5%) read `no data`. In a difference pane it shows the difference and both values.
- Overlays (panel `overlays`, per pane; the selected pane's checkboxes build the `spec=` strings sent to `GET /overlay`, one request per spec so that a failing one does not remove the others): `vectors v` / `vectors B` (`grid`, `arrows` prop | unit),
  `streamlines v` / `streamlines B` (`density`), `particles` bh / star / dm, and a free field `more specs` (specs separated by `;`, e.g. `particles:dm,max=500;vectors:v,color=red`, the full syntax of "Overlays and click-to-inspect"). B overlays are cyan.
  They are drawn on a canvas above the map from the plain window-coordinate data (arrows, streamlines with their flow heads, points by type, circles, the reference-arrow legends top right): while panning and zooming the old overlay is transformed with the map and a new one is requested
  after the same debounce as the map. An error (no `MagneticField` for a B overlay, an unknown option) is a short red message under the panel and the other overlays stay; a missing particle type is a note.
- Click to inspect: a click on the map (no drag; it waits 280 ms so that a double click, which centres, does not inspect; shift+click inspects at once) calls `GET /inspect` at that window position and fills the `inspect` panel with the top 10 gas particles: rank, id, contribution %, H, rho, u, T, position and
  velocity (frame coordinates). Projection: the particles whose projected kernel covers the point, ranked by their share of the column; slice: the particles whose kernel reaches (x, y, z0), nearest first, with their kernel share (the circle is the cross-section at the plane). The smoothing circle of the selected particle (the first;
  click a card to select another) is drawn in the pane with a cross-hair at the clicked point. `Esc` clears; the result is dropped when snapshot, field, mode, axes, slice plane or the centre / align rule change.
- Deep zoom (`deep-zoom tiles` checkbox, on by default): a window much smaller than the base width is assembled from `GET /api/tile` tiles (see Tiles) instead of one `/api/map`. `GET /api/state` gives `tile = {base, size}` (the server's tile base width, the default
  `--width` rounded up to a power of two, and the tile size 256), and the page sends `base` with every tile request. **Switch rule** (`S.tilePlan(pane)` in the page): tiles are used when the pane shows no difference, its width is at most base / 4, the level
  L = ceil(log2(base npix / (tile width))) (the coarsest level whose pixel base / (tile 2^L) is at most width / npix, the pane's `pixels`) is at most 24 and the window needs at most 36 tiles; otherwise (full view, moderate zoom, a tiny window, difference mode, the checkbox off) the normal `/api/map` path is used.
  The covering tiles (power-of-two grid, edges shared by all windows of the level) are fetched in parallel, cached in the browser (a Map keyed by the tile URL without the token, about 150 MB) and composited into one square map on the level's pixel grid covering the window (pixels outside the fetched tiles are blank); that
  map goes through the normal colouring, so the colour limits, the hover read-out and the overlays are those of any map, with one pair of limits for all tiles (auto limits: percentiles of the composite; `lock` and fixed limits work as usual). Until the tiles arrive the previous map stays on the canvas, transformed to the
  new window (the coarse placeholder), and a spinner shows in the pane. Panning by one tile requests only the new tile column / row; `prefetch` also requests the ring of tiles around the covering block (nearest first, at most 16, one at a time). Black-hole markers are requested once per snapshot and frame rule. The composite is at the level's pixel size (between 1 and 2 times finer than the pane's `pixels`) and
  equals the plain `/api/map` of the same window up to the engine tolerance of the Tiles section (tools/browser_smoke.py checks 5e-3 of the maximum on a pixel-aligned window; 9e-4 measured).
- Several runs and time alignment (`splosh serve RUN1 RUN2 label=RUN3`; one directory gives the single run of earlier versions, and several directories are several runs, no longer one merged list). The page asks `GET /api/runs` (labels and for every run its snapshots with index, name, `time` in code units
  and `time_myr` = time x the header's length / velocity unit in Myr, null for comoving runs or headers without units). Each pane has its own run (selector `run`, selected pane) and snapshot; its caption starts with the run label, and each pane has a clock (top centre) with its code time and, when the header allows, the time in Myr.
  Alignment (`align panes by`): `none: keep index offsets` (the older behaviour: stepping adds one to every pane's snapshot; default with a single run), `time (code)` (default with several runs), `time (Myr)`, `snapshot index`, `time + run offsets`; the aligned value of a snapshot is its time (or Myr, or index) minus the run's `offset`
  (the `offset of each run` box, comma separated in run order, like `splosh compare --offset`). With `step all panes` (arrow keys, `<` `>`, the slider) the selected pane steps in its own run and every other pane moves to the snapshot of ITS run whose aligned value is nearest to the selected pane's new one, if it lies within the
  `tolerance` (the box; empty: half of that run's median snapshot spacing in the aligned unit, as in `compare`); a pane with no snapshot that close shows `no snapshot at this time` (with the nearest time) and requests nothing, and finds its snapshot again when stepping brings the time back. Snapshots with equal times go to the one nearest to the pane's current index.
  Shift+arrow (or the toggle off) steps the selected pane only. Pan and zoom link as before, across runs. `GET /api/align?mode=time&at=T[&tol=X][&offset=o1,o2][&prefer=i1,i2]` gives the same choice per run (library: `splosh.runs.align`, `nearest`), so other front ends and the tests share one rule.
- URL: the page is `http://127.0.0.1:PORT/?token=T`; nothing else is kept in the URL (no hash state: panes, overlays and alignment are per page load). Every data endpoint takes `run=LABEL` (or the run's number) and defaults to the first run, so URLs of earlier versions are unchanged: `/api/map`, `/api/png`, `/api/bh`, `/api/tile`, `/overlay`, `/inspect`; an unknown run is a 404.
- Keys: left / right previous / next snapshot (all panes or Shift: the selected one); `a` `b` store A / B, `space` blink, `d` difference; `p` number of panes, `1`..`4` select a pane; `l` link pan / zoom; `r` reset the view; drag pans, wheel zooms about the cursor, double click centres;
  click inspects (shift+click at once), `Esc` clears the inspection (and closes the help); `?` or `h` the help.
- Colouring is done in JavaScript on a 2D canvas (no WebGL).
- Pane kinds (selector at the pane's bottom right, top right on a plot): `map` (default), `radial profile` and `time series` [verified: tests/test_series.py, scratch browser check].
  `radial profile` is the azimuthal statistics of the pane's own map (its field, mode, units and window; the controls still edit it) in 48 log-spaced annuli about the centre of the
  frame (the centre rule's point, not the window centre): mean (line), 16-84 percentiles (band), dashed where the window covers under half of the annulus. For `rho` in projection that
  is the surface density Sigma(R). Hover on the profile draws the circle of that radius on every map pane; hover on a map marks its radius on the profiles.
  `time series` plots columns of one **source** of the run (`series.py`; `GET /api/series?run=` lists them, `&source=S` describes one, `&columns=a,b[&ids=][&smooth=][&tmin=&tmax=][&n=]`
  gives the curves): `snapshots` (the newest `<run>/analysis/scalars.csv` or `<run>/analysis_viewer/*/scalars.csv`), GIZMO's `energy.txt` (E_int, E_kin, E_tot, |E-E0|/|E0|, per type),
  `blackholes.txt`, and the performance logs (`perf.py`: `cpu` wall clock, hours per unit time, module and waiting shares over windows of 1/400 of the run; `timebin` dt_sys, bins, active
  fraction, bin occupancy; `timings` tree-walk balance and speed; `balance` useful and waiting seconds per module; read incrementally: a running simulation's new lines only;
  a figure of them: `docs/layouts/gizmo_performance.toml`), and sources that plugins register (`plugins.register_series(name, fn, files, label, version, defaults, priority)`). Each source is parsed once and cached as
  `<run>/analysis_viewer/cache/series_<source>.npz` (keyed by the size and mtime of its input files; `↻` reads them again). Curves are thinned on the server by min/max per bucket, so spikes
  and gaps survive; columns with style `events` are sent whole. Controls: source, columns (grouped; several share the y axis), one curve per id (black hole) with checkboxes, running mean
  of 0.01 / 0.1 / 1 binary period P0 (P0 of the first snapshot), x axis t/P0, Myr or code time, y auto (log when positive over more than 100x, symlog when both signs span decades, linthresh =
  20th percentile of |y|) or fixed. Angles with a wrap break their line at the wrap, are drawn in [0, 360) when that wraps less, and get no running mean. The legend gives each curve's value at
  the current snapshot (yellow line) or under the cursor; the wheel zooms in time (the window is fetched again at full detail), a double click shows the whole run, a click goes to the nearest
  snapshot. A run without snapshot scalars gets a `build` button: a background job (`POST /api/jobs {"kind": "series", "run": ...}`, analyze with only the built-in scalars, incremental) writes
  `<run>/analysis_viewer/series/scalars.csv`, and the pane fills in when it is done.

Remote data (data on a login node, browser on your laptop):

    # on the machine that holds the snapshots (run it inside a batch/interactive job if rendering is heavy; the login node is shared)
    splosh serve /path/to/output --port 8765 --center bh --align bh --width 10
    #   prints:  open  http://127.0.0.1:8765/?token=...
    # on your laptop, in another terminal:
    ssh -L 8765:localhost:8765 <user>@<login-node>
    # then open the printed URL in the laptop browser (the token is in the URL)

If the server runs on a compute node (it listens only on that node's loopback, so a plain `-L port:<node>:port` through the login node does not reach it): `ssh -J <user>@<login-node> -L 8765:localhost:8765 <user>@<compute-node>`.

## Snapshot products (`splosh products RUN [--names a,b] [--workers N]`, products.py)
Quantities a plugin computes once per snapshot (`plugins.register_product`; see ADAPTING.md), stored in `<run>/analysis_viewer/products/<name>/` under a key (the
snapshot's signature, the product's version and parameters) and read as the time-series source `prod_<name>` (a family `<prefix><number>` is a heatmap against
time). `splosh serve` computes what is missing in the background when a run is opened (a `products` job in the job list, on a free node when `--slurm` is set;
`--no-auto-products` turns it off; `POST /api/jobs {kind: "products", run}` starts it by hand). Parallel over snapshots: at most the physical cores, and as many
processes as leave ~4x a snapshot's size of memory to each.

## Analysis pipeline and run comparison
`splosh analyze RUN [--config analyze.toml] [--out DIR] [--jobs N] [--only NAMES] [--every N] [--force] [--dry-run]` computes, per snapshot, the products a TOML config declares and keeps them in a
tree (default `RUN/analysis`): maps (`NAME/SNAP.png` + the float32 map in `.maps`), radial profiles (`.npz`, `.csv`), scalar diagnostics (one `scalars.csv`), dashboards, run-wide time-series
figures (`figures/NAME.png`) and movies (`movies/NAME.mp4`), plus `manifest.json` (inputs hash, config hash, outputs, wall time, commit) and `digest.txt` (under 60 lines of text, made to be fetched).
A second run does nothing; a changed snapshot, product config or SPLOSH version redoes only what it affects; snapshots that cannot be opened yet (a run in progress) are skipped and reported, and
nothing is ever deleted. `--jobs N` uses N spawn workers (numba threads divided between them). Without a config: one projection of the density, one radial profile, the built-in scalars.
Everything about the config is in `docs/analyze.example.toml` (annotated), the module doc of `analyze.py` (incremental rule, colour limits `global | frame | fixed | viewport`) and, for dashboards, `layout.py`.

Layouts are data: a dashboard or time-series figure is a versioned declarative file (`docs/layouts/overview.toml`: panels of type map | profile | timeseries | text on a grid, derived curves from safe
expressions such as `bh_m1 / bh_m2`) read and written by `splosh.load_layout / save_layout / validate_layout`; `splosh.layout.registry()` lists what a panel can refer to (fields, scalars including
plugin ones registered with `plugins.register_scalar`, profile quantities, colour maps). Keys starting with `x_` are kept and ignored, for a GUI's own use.

**Time in binary orbits.** For a run whose first snapshot holds two bound black holes, P0 = their orbital period there (vis-viva on the two lowest-ID black holes,
dynamical `Masses`, G in the header's code units; `runs.binary_period`). The clocks show `t/P0`, the panes can be aligned by `orbits`, `/api/runs` gives `period` and
each snapshot's `orbits`, and `shot --orbits` / `movie --orbits` (and the movie job, when the run has P0) label the time as `t = ... (... Myr, 2.5 P0)`.
[verified: tests/test_runs.py, a circular orbit of known period]

**Choosing folders in the page.** `splosh serve` without paths starts empty and the page opens a folder picker; the **folders** link in the side panel opens it at any
time. It browses the tree under `--root` (default: the home directory; nothing outside it can be listed or opened, symlinks out of it included), shows the snapshot
files per folder, and with "show all subfolders with snapshots" lists every folder below that has any (hidden folders skipped, symlinks not followed, stops after a few
thousand folders). Tick folders: **open** replaces the runs, **add as runs** adds them (each folder is a run, a run directory's `output/` is found as on the command
line); the page then reloads. `GET /api/browse?dir=REL[&deep=1]`, `POST /api/open {paths, add}`. [verified: tests/test_server.py, headless browser run]

**Long jobs from the page** (side panel, "long jobs"): **movie of this pane** (the selected pane's view and colours over a snapshot range, every N, fps, colour limits the same
for all frames / per frame / as now) and **prefetch all** (every snapshot of every pane's run, with the pane's view, into the map store; afterwards stepping is instant).
**analyze these panes** writes an analyze config from the panes (a `[[map]]` per pane with its view and colours and global limits, a radial profile of `u` about the first
pane's centre, the built-in scalars, a dashboard: the maps, the profile, `gas_mass` and `rho_max` over time) and runs `splosh analyze RUN --jobs auto` into
`<run>/analysis_viewer/viewer/` (incremental: a second job adds only what is new); its layout also goes to the editor's layouts folder as `<run>-viewer.json`, and
the job's **open in the layout editor** link opens it with that pipeline's output as the preview's data. The movie job passes the pane's overlays (BH markers, vectors, ...).
With `serve --slurm "-c 20 --mem=32G -t 01:00:00"` each job first asks Slurm for a node with `srun --immediate=10` (`--node-wait`): it starts at once on a free node or
not at all, and then runs on the server's machine instead (nice 10, the page's cores, one at a time). The server only watches the processes; closing it cancels them.
The list shows where each job runs, its progress, the last log lines, cancel, and links to its files (`movie.mp4`, `movie.json`, `log.txt`; folders under `--jobs-dir`,
default `~/.cache/splosh/jobs`). The map store must be on a file system the nodes share (the default under `~/.cache` is, on a cluster with /home on NFS).
Movies need ffmpeg on PATH, `--ffmpeg PATH` or `SPLOSH_FFMPEG`, else the frames stay PNGs. `splosh prefetch --plan plan.json [--procs auto]` is the command the
prefetch job runs (auto: one snapshot per ~5 threads). [verified: tests/test_jobs.py with a fake srun that starts / refuses / waits; headless browser run]

**On a remote login node, from Windows** (`tools/remote/splosh-remote.ps1`, double-click wrapper `splosh-remote.cmd`): `splosh-remote.cmd [-Run "~/runs/x/output"] -Cores 4 -MaxCores 8 [-SshHost cluster] [-Splosh PATH] [-Extra "--layouts ..."]`.
It makes the token and a random port on the laptop, runs `ssh -tt -L P:localhost:P HOST "NUMBA_NUM_THREADS=MaxCores nice -n 10 splosh serve RUN --port P --token T --threads Cores --exit-when-orphaned --idle-exit 120"`,
opens the printed URL in the browser (and copies it), and stops the server when the window closes or on Ctrl+C (the terminal's hangup; `--exit-when-orphaned` stops it too when ssh dies without one,
`--idle-exit MIN` after MIN minutes without a request). The viewer's **cores** box changes the rendering threads between 1 and the pool size fixed at start (`NUMBA_NUM_THREADS`; `POST /api/threads`),
without a restart. `serve --threads N` sets the starting count. [verified on the laptop: tests/test_server.py, the orphan exit with a killed parent shell; the ssh part is not tested here]

**Layout editor.** `splosh serve RUN --layouts layouts [--analysis OUT] [--config analyze.toml]` also serves `/layout` (the URL is printed; same token, same `ssh -L`). Open a layout file, drag a
panel on the grid to move it, drag its corner to resize, add map / profile / time-series / text panels, set their options in the inspector (drop-downs from `registry()`: fields, colour maps, limit
policies; the `↺` button removes a key so it falls back to its default), edit the derived curves, undo with ctrl+z, and save. The editor writes the *minimal* layout (only the keys you set, in the file's
format: TOML for a new file) after checking that it validates; nothing invalid is ever written. The preview is the same drawing code as a dashboard, for one snapshot of the served run, refreshed after every
change: maps with inline settings are rendered from the snapshot (through the map store, so repeated previews are fast); `product =` panels and profiles need the analyze config (`--config`); time series need
`--analysis DIR` (the served run's analysis: its `scalars.csv`; the marker is the row with the same snapshot name, none for a `scope = "run"` layout); `global` colour limits cannot be known from one
snapshot, so the preview uses the frame's percentiles (a note under the picture says so). The config and `scalars.csv` are read again at every preview, profiles and `auto` radii are kept per snapshot.
A panel that cannot be drawn is blank with its reason, never a failed preview. Older layout schemas go through `layout.MIGRATIONS` when read (none exist yet).
[verified: tests/test_layout_editor.py, tools/browser_smoke.py section_layout]

`splosh compare RUN1 RUN2 ... [label=path] [--config compare.toml] [--out DIR] [--run-out DIR] [--align ...] [--offset LABEL=VALUE]` draws comparison figures from the runs' own caches (each run is analysed
with its own pipeline first): curves of all runs on one panel (`type = "timeseries"` with scalar columns or expressions, `"profile"`), and map grids with a column per run and the same View and colour
limits (`"maps"`). Time alignment per product: `time`, `time_myr`, `index` or `offset` (per-run shifts); runs with different snapshot spacing use the nearest snapshot within `tol`, else a blank tile with a note.

## Smoke test of the viewer
`tools/browser_smoke.py SNAPDIR --server-python PYTHON [--shots DIR] [--synthetic]` drives the page with Playwright (not a dependency, install it in a separate environment): canvas not blank, markers, zoom and stepping re-request maps, blink, difference
(a state minus itself is 0; with `--synthetic`, where snap_001 is snap_000 with doubled masses, A - B = -A and log10 (A/B) = -log10 2), linked panes, hover value and unit against an independent fetch of the map, the help, a narrow window, no console errors; screenshots with `--shots`.
With `--extra-run DIR` (another run: the test serves a second run with another snapshot spacing) it also checks the clocks, stepping all panes to the nearest snapshot in time, the tolerance and offsets, `no snapshot at this time`, the alignment modes, the page's rule against `/api/align`, and a pane showing the other run; the overlay, inspect and deep-zoom sections run in every case.
With `--layouts DIR --analysis DIR` (a `seed.toml` with a map `m` at column 0 and a text panel `t` at column 1 of 4; a `scalars.csv` with `gas_mass`) it also drives the layout editor: open, drag, refused
overlap, resize, inspector edits and resets, product versus inline settings, undo, an id clash shown and undone, the series from `scalars.csv`, save and read back. `--only layout` runs just that part.
`tests/test_viewer_browser.py` runs it with the python in `SPLOSH_BROWSER_PYTHON` (or the current one) and skips when Playwright or its chromium is missing. Setup that needs no root:

    uv venv /path/to/browser-env && uv pip install --python /path/to/browser-env/bin/python playwright
    PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers /path/to/browser-env/bin/playwright install chromium     # downloads chromium-headless-shell, no system packages needed on a normal desktop/WSL

## Not done yet
Colour-coded vectors and streamlines, mpl-backend overlays, contours and transforms in the matplotlib backend (`shot --backend mpl` takes `--post` and `asinh:F`, not overlays), a layout panel for a PDF's `pdf` column with a log y axis by default, GPU colouring (the colour map is applied in JavaScript on a 2D canvas), other kernels
(Wendland, quintic), periodic boxes, multi-file snapshots, editing profile products and the whole analyze config in the editor (only layouts so far), undo history across page reloads, viewer state in the URL hash.


## Phase diagrams and selections (viewer panel "phase diagram", `/api/phase`, `/api/selection`, `/api/map?sel=`)
"show the diagram" draws a 2D histogram of the gas in the selected pane's window (all along the line of sight for a projection, within H of the plane for a
slice): any two of `density`, the per-particle fields (u, T, velocities, vR, vphi, jz, ...) and the geometry `x y z R r h v los`, log or linear axes, weighted by
mass, count or volume, in the view's unit system. Drag on the diagram to draw a selection (lasso): the selected gas is highlighted on the map (magenta points,
at most 20000 drawn), the panel gives its count and mass, and "map only the selection" renders the pane from those particles alone (not stored in the map
store; deep-zoom tiles are off meanwhile). The selection is a region of the diagram, so it follows the pane to other snapshots and runs.

## GPU colouring (viewer: "GPU colouring", on by default)
The browser's GPU colours the maps: each float map becomes a WebGL2 texture and the stretch (linear, log, asinh), the limits and the colour table run in a
fragment shader, so dragging the limits, switching colour maps or stretches, differences and blinking never loop over the pixels in JavaScript. It runs on the
laptop (e.g. an Intel Iris Xe), not on the cluster. The CPU loop stays: it colours histeq maps, it is used when the browser has no WebGL2 or the switch is off, and it
is the reference (the browser tests compare both: identical except a rare pixel one colour-table step away). The status line says which one coloured the map and
how long it took. Saved figures are made on the server, so they do not depend on this.

## Partial loading (`serve --partial auto|on|off`, region.py)
For big snapshots the server reads only the gas a zoomed view needs. GIZMO writes the gas in spatial order (along its space-filling curve), so a small window
touches a few runs of consecutive rows. An index per snapshot (the bounding box and largest H of every 4096 consecutive particles, in `<store>/index/`) is built
the first time (it reads the positions and H once; `splosh prefetch` builds it too) and kept. A map or tile then loads the chunks that reach the window enlarged
by half a window on each side (tiles: two rings of tiles), so panning reuses the region; the maps are the same as from the whole snapshot (tests compare them).
`auto` does this for snapshots of at least 2e6 gas particles (`SPLOSH_PARTIAL_MIN`). Centre rules and tilts that need all the gas (`com`, `mass-weighted`,
`peak`, gas `ids`, `tilt L`) and windows that would read more than half of the rows load everything. The gain is in the I/O and the memory of the login node:
measured on a 0.9e6-particle snapshot, a window 1% of the box wide around the binary reads 46% of the rows (the gas is concentrated there), farther out much less.

## Volume rendering (`--mode volume`, viewer mode "volume (opacity)", `--kappa`)
The gas seen as a semi-transparent medium from +line of sight: particles sorted front to back, each absorbs the light from behind with optical depth
tau = kappa x its column density (the same projected kernel as the projections) and emits its value: E += T (1 - e^-tau) S, T *= e^-tau. For `rho` the map is the
opacity 1 - T; for any other field it is E / (1 - T), the value of the gas you see (cold dense gas in front hides hot gas behind). `kappa` is an opacity per
unit mass in code units (length^2 / mass); 0 (the default) means tau = 1 at the window's mean column density, so dense structures are opaque and diffuse gas
transparent; larger values show only the front layer. Combine it with `--tilt` (e.g. `incl:80`) for depth and inclination. Orthographic only (no perspective);
no deep-zoom tiles (the compositing needs the whole line of sight); about 2x the time of a projection with the exact engine (0.9e6 particles, 512^2: 3-5 s on
the laptop). Tests: the two-screen law, the optically thin limit (= the projection), opacity in [0, 1].