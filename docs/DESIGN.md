# SPLOSH design (general part)

Goal: a fast, light viewer for SPH / meshless-hydro snapshots (GIZMO, GADGET-like HDF5), usable on a laptop or on the machine that holds the data.

## Pieces
1. **Kernel and splatting engine** (`kernel.py`, `splat.py`): GIZMO cubic spline (support radius H = `SmoothingLength`), column projection and slices, several quantities in one pass,
   multi-resolution merging for large particles. Validated against SPLASH and the exact kernel (`bench/`, `tests/`).
2. **Frames and maps** (`frames.py`, `maps.py`): centre rules and an alignment rotation about z applied before splatting; `View` -> map (projection or slice, weighted quantities). **Loaders** (`io.py`): snapshot -> arrays in code units plus the header. Others (other formats, code-specific datasets) are plugins.
3. **Mapstore** (prototype, `mapstore.py`; format below is the plan, the current one is one `.npz` per map, see USAGE): one file per run with the rendered maps (quantised log values per snapshot, quantity and projection) and an `index.json`
   (times, field of view, centre rule, units, kernel, parameters hash). Written by `splosh render`, read by the viewer.
4. **Server** (prototype, `server.py`): a small Python process that serves the mapstore and renders on request (deep zoom from the snapshot, on-disk cache). Runs on the laptop
   or on the machine holding the data (reached through an ssh port forward); bind to localhost only, with a token.
5. **Viewer** (prototype, `viewer.html`: plain JS + 2D canvas, colouring in JS, no WebGL yet): frame stepping, quantity switching, colormap/stretch/limits, blink, difference (client-side from two float32 maps, a browser-side map cache), 1 to 4 panes with optional linked pan/zoom, hover values with units, keyboard help. Not done yet:
   physical-time clock across runs with different output spacing.
6. **Headless** (`figures.py`, `bitmapfont.py`, `limits.py`, `provenance.py`): `splosh shot / sheet / movie` and the Python API `shot`, `contact_sheet`, `movie`: the same maps (through the mapstore) turned into pictures with a built-in bitmap font and the PNG writer, shared colour-limit policies (global / frame / fixed), a JSON sidecar per image from which the picture is redone. matplotlib (PDF) and ffmpeg (video) are used only if present.
7. **Tiles** (`tiles.py`, `mapstore.tile_key`, `/api/tile`): the deep-zoom tier, see "Tiles: decisions" below and USAGE.

## Extension points (the plugin interface; main defines them, plugins implement them)
| Group | What a plugin provides |
|---|---|
| `splosh.loaders` | a reader for a file format or extra datasets: `load(path) -> dict of arrays` |
| `splosh.fields` | derived quantities: `name -> function(snapshot dict) -> array` |
| `splosh.centers` / `splosh.frames` | centring rules (BH, binary, density peak, ...) and orientation rules (align to a vector, a pair axis, an angular momentum) |
| `splosh.overlays` | markers, vectors, particle sets drawn on a map: `plugins.register_overlay(name, fn)`, `fn(snap, frame, view) -> list of dict` (schema in `overlays.py` / USAGE) |
| `splosh.panels` | extra linked panels (time series, profiles) with a click-to-snapshot hook |
| `splosh.presets` | named bundles of settings (units, fields, view) |
Plugins are discovered through Python entry points of the group `splosh.plugins` (or the environment variable `SPLOSH_PLUGINS`), so the main package never imports them by name; `plugins.py` is the registry (fields, info sections, overlays, presets).

## Units and derived fields (`units.py`, `fields.py`)
- A quantity's unit is a dimension tuple (exponents of code length, mass, velocity), so the cgs and astro factors follow from the header scales with no unit library (astropy and pint are deliberately not used).
  Temperature is the one exception (always K). A field registers its dimension with `register_field(..., unit=...)`; a callable covers units that depend on the view (entropy depends on gamma). Without a
  declared unit the map says `known: false` and is not converted, so unit conversion can never silently mislabel a plugin field.
- `View.units` multiplies the finished map by one scalar; the splat and the cache key are otherwise unchanged (the key includes the new View fields).
- Derived fields are registered through the same registry as plugin fields (`builtin=True`, so a plugin of the same name wins). `needs` may be a callable of the view (T needs ElectronAbundance only when `mu` is not given);
  `required_gas_fields` merges them, and the loader raises a KeyError naming a dataset the file lacks.
- Fields that need neighbours (vorticity, divergence, gradients) are out of scope here: they need a neighbour search and belong in a separate module.

## Conventions
- Lengths in the snapshot's code units unless a unit conversion is given; the header is the source of truth for time.
- `H` is the kernel support radius (GIZMO). Readers for codes with another convention convert on loading.
- Column density conserves the particle mass; the quantity images are `sum w q K` with weight `w` chosen by the caller (mass or volume).
- Overlays (`overlays.py`): computing and drawing are separate. Computing returns JSON data in window coordinates (so the browser draws the same thing); `draw_items` is the headless numpy rasteriser (coverage-based anti-aliasing, no matplotlib).
  Vector fields are not a new splat: they are two weighted maps of the view (`render_map`, fields vx/vy or the registered Bx/By/Bz) on a coarse grid, so they inherit the weighting, the frame and the slice / projection modes; streamlines integrate that grid.
  Alternatives rejected: sampling particles directly (noisy, not equal to the map's projection) and a second splat engine for vectors. Click-to-inspect uses the exact kernel tables of the splat (`kernel.tables`). [verified: tests/test_overlays.py]

## Tiles: decisions (`tiles.py`)
| Choice | Alternatives | Reason |
|---|---|---|
| each tile = `render_map` of the culled particles, with a 16 pixel margin cropped | call `splat_*` directly with a tile-specific engine; no margin | reuses units, NaN rules, weighting and every plugin field unchanged. The margin is needed because the engine renormalises particles under 6 pixels over the part of the footprint inside the image (without it every tile edge overcounts: 1.4x the mass in the first test). A margin of 16 = 2 x 6 + 2 rounded to keep tile + 2 margin divisible by 2^levels of the multi-resolution merge (14 gave 7e-3 errors with 512 tiles) |
| particles binned once by tile cell; 3 x 3 cells for H <= tile width, a vector test for the larger ones | per-tile test of all N particles (O(N) per tile); a tree | O(N) once then O(candidates) per tile; the 3 x 3 rule is exact because H <= tile width |
| frame (centre, rotation) computed once and passed to `render_map`; per-tile sub-snapshot (every per-particle array indexed) | recompute per tile | the centre rules (`com`, `peak`) are O(N) |
| two grids: global (origin 0, base_width / 2^L) for the cache and the server, window-anchored for `render_tiled` | only global, resampling to the window | resampling would add an interpolation error to the "equals the monolithic render" claim; a window grid is exact, the global grid is what makes neighbouring requests share tiles |
| key = snapshot signature + view without its window + tile size + grid geometry (tile width, origin) + indices + level | key on the window | the same tile serves any window |
| parallelism: Python threads over tiles, each with ncpu / workers numba threads | numba `prange` over tiles (needs a new splat entry point: splat.py is shared); processes | the existing kernels release the GIL and are band-parallel inside; numba's thread count is per Python thread, so nothing nests |
Reference: the engine and its tolerances are in VALIDATION.md; the single-particle and corner tests in `tests/test_tiles.py`.
Finding [verified]: `render_map` renormalises particles of 1-6 pixels over the part of the footprint inside ITS window, so its first and last 6-12 pixel rows and columns are slightly too bright where such particles are cut by the border. Tiles do not do this (they render the margin), so assembled tiles and `render_map` differ there; this is a defect of the monolithic path, not of the tiles, and it is left alone because `splat.py` is shared.

## Transforms, stretches, contours (`transforms.py`, `imaging.py`, `overlays.py`)
| Choice | Alternatives | Reason |
|---|---|---|
| the transform is a View field (`post`), applied at the end of `render_map` | a separate step in every consumer (shot, server, layouts, analyze) | one implementation; the mapstore key, the sidecar and the layouts follow for free. Cost: the raw map is rendered again for each post (`explore` avoids it: it renders once and calls `transforms.finish`) |
| azimuthal mean from the window's own pixels, profile interpolated at each pixel's radius | a separate wide render for the reference; plain ring means | no second render, works for every field and frame; the one-pixel-ring mean is biased by `1.5 dR / R` on an R^-1.5 profile (test: 6.8 % against < 2 % interpolated). Documented limit: partial rings in a zoom |
| a ratio only for fields positive by nature (`POSITIVE_FIELDS`), a difference otherwise | decide from the data (all pixels > 0) | the choice would flip between frames of a movie |
| a `Stretch` object carries what a stretch needs (the quantile table of `histeq`, the F of `asinh`); `==` compares with the plain name | thread a table through every call; global state | string scales keep working everywhere (`scale == "asinh"` in the tick code), `str()` is the spec for sidecars |
| histeq not in sheets, movies, products, layouts | one table over all frames | a per-frame rank would make frames incomparable and a global one needs a pass that the incremental pipeline avoids; it is a discovery tool |
| symmetric limits default to the 99th percentile of \|f\|, not 99.9 | the largest value | the noise tail of an SPH residual would set the range (shot of the synthetic spiral: +-1.2 at 99.9 against +-0.93 at 99, test image in the notes) |
| contours by marching squares on the (smoothed) map of the same window, items are plain polylines | matplotlib `contour`; contours in the browser | no dependency, the existing rasteriser and the browser's polyline drawing take them unchanged |
| PDF as `geometry = "pdf"` of the profile product | a new product kind | the writers, the manifest, the incremental rule and the profile panels are reused; `rbins` is then the value axis |
| the `modes` scalar is opt-in | part of `builtin` | `builtin` would add columns to every existing `scalars.csv` |
| the layout editor is a second page (`layout.html`) of the same server, POST for validate / save / preview | one big viewer page; a separate tool | one token, one port, one ssh tunnel; the preview shares the loaded snapshots and the map store with the viewer. POST needs `Content-Type: application/json`, which a cross-site form cannot send |
| the editor keeps the layout in its minimal form and saves it as it is (`save_layout(completed=False)`) | save the completed layout | the file stays readable by hand and diffs show only what the user changed; the server's `validate_layout` is the only authority for defaults and placement |
| layout file names are plain stems (`[A-Za-z0-9][A-Za-z0-9_.-]*`, no extension) inside the layouts directory | accept paths | no way to write outside the directory through the page |
| the preview resolves `product =` panels from an analyze config and says what it cannot show | run the pipeline for the preview | a preview must be seconds; the notes under the picture list approximations (global limits) and blanks |

Finding [verified: tests/test_explore.py::test_azimuthal_residual_recovers_the_perturbation]: a map of R^-1.5 (1 + 0.1 cos 2 phi) is recovered to within 0.02 of the true contrast; the gradient of exp(a x) is a / ln 10 per length to 0.2 %.
