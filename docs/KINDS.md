# Kinds of runs: mesh data in SPLOSH

SPLOSH's own snapshots are GIZMO/GADGET particle files. A **kind** is how a plugin teaches it another kind of run: a folder whose snapshots are
block-structured mesh data (adaptive or fixed mesh refinement, or a uniform grid as one level). Main holds the generic parts (`src/splosh/kinds.py`,
`src/splosh/mesh.py`): the run picker, the data model, the slice renderer, the overlays. A plugin holds the reader of its file format and its quantities.
A complete small example is the tests' reference format, `tests/meshref.py`.

## Registering a kind
```python
from splosh import plugins
from splosh.mesh import Level

plugins.register_kind(name, detect, snapshots, open, fields=None, units=None, default_view=None, label=None, time=None, doc=None)
```
| argument | |
|---|---|
| `detect(folder) -> bool` | the kind claims this folder (absolute path). It runs for every folder the picker lists: keep it cheap (a glob, no file reads). It is also asked about the folder holding a snapshot and that folder's parent. |
| `snapshots(folder) -> [paths]` | the folder's snapshot files of this kind (only these are offered: other files next to them, e.g. particle files, are not); sorted naturally by SPLOSH |
| `open(path) -> object` | the opened snapshot (the data model below). Opened objects are cached (8, by path, mtime and size) and closed (`close()`) when dropped. |
| `fields` | `{name: {"needs": (var, ...), "fn": fn(v) -> array or None, "label": str, "unit": str}}`. `fn` gets a dict of the needs sampled at the pixels (float64 arrays of the map's shape, NaN outside the mesh) and the pixel positions `"x"`, `"y"`, `"z"`; it is computed after interpolation. `fn` None: the value of the one variable in `needs`. Every variable of the file is a field by its own name as well. |
| `units` | `{"length": "M", "time": "M"}`: names of the length and time units (axis labels, scale bar, time label). Values are never converted (`--units` does not apply). |
| `default_view` | View settings a run of this kind opens with, e.g. `{"center": "origin", "mode": "slice", "field": "density", "width": 20}` (the viewer and the command line use them where no option is given; also a preset of the kind's name) |
| `label` | the name the picker shows ("12 mesh test snapshots in output/") |
| `time(path) -> float` | the snapshot's time without opening it fully (default: the attribute `Time` of `/Header`, first element) |

## The opened object (duck-typed)
| attribute / method | |
|---|---|
| `time` | float |
| `levels` | `[Level, ...]` in any order (SPLOSH sorts by `dx`) |
| `variables` | names of the stored variables |
| `read(var, level, patch, lo, hi) -> ndarray` | the values of `var` on patch `patch` (an index into `levels[level].patches`) at the index box `lo..hi` (inclusive, level index space, inside the patch; tuples of 3 ints), **axes ordered (x, y, z)**, shape `hi - lo + 1`. Slice the file here (h5py partial read): a slice asks for the few layers around its plane. For a file stored `[k][j][i]`: `ds[k0:k1, j0:j1, i0:i1].transpose(2, 1, 0)`. |
| `header` (optional) | dict, kept as the snapshot's header (JSON-able values) |
| `points()` (optional) | `{"pos": (N, 3), "mass": (N,), "vel": (N, 3), "id": (N,)}` (mass, vel, id optional) or None: point masses the frame rules see as black holes would be in a particle snapshot: `--center bh`, `--align bh`, `--tilt Lbh`, the viewer's markers |
| `markers()` (optional) | `[{"name", "pos": (N, 3), "labels": [str] (optional), "color": name or #rrggbb (optional)}]`: points drawn by the overlay `mesh_markers` |
| `close()` (optional) | release the file |

`Level(dx, origin=(0, 0, 0), patches=[(lo, hi), ...], centring="vertex", periodic=False, boxes=None)`:
- `origin`: the position of index (0, 0, 0); `centring` `"vertex"`: values at `origin + i dx`, `"cell"`: at `origin + (i + 1/2) dx`;
- `patches`: inclusive integer index ranges, the pieces `read` reads (they may be pieces of one refinement box, e.g. one per process);
- `boxes`: the refinement boxes drawn by `mesh_boxes` (default: the patches merged where together they form a larger box);
- `periodic`: the level wraps; its period is its index box (N points of spacing dx per axis).
Coarse data under a finer level may be stored or not: the finest level covering a point is always the one sampled.

## What SPLOSH does with it
- **Picker and runs.** A folder a kind claims is one run of that kind: its snapshot count and time range in the picker (`/api/browse`: `kind` per entry),
  `splosh serve FOLDER`, `sheet`, `movie` use `snapshots(folder)`. A run folder whose `output/` is claimed opens that. Particle folders are unchanged.
- **Slices** (`--mode slice`, any axes, centre, align and tilt: the plane is the frame's, `centre + a e_a + b e_b + z0 e_los`): every pixel takes the finest level
  covering it (vertex-centred: all corners of its cell are stored, also across patch seams; cell-centred: its cell is stored), with trilinear
  (`--interp linear`, the default; exact for linear functions) or nearest-point (`--interp nearest`) sampling. Only patches reaching the plane are read,
  and only the layers near it when the plane is thin in an index direction; patch arrays stay in an LRU (`SPLOSH_MESH_CACHE_MB`, default 512).
  A NaN stored in the file reads as "not covered" (the next coarser level is used). Projections and volume renderings of mesh data are not implemented.
- **Everything downstream** takes the slice like any map (`maps.render_map` output: float32 image and meta with `extent`, `label`, `length_unit`,
  `quantity_unit`, `kind`, and `mesh.pixels_per_level`): colour scales and limits, the map store (keyed by the file signature and the view), `shot`,
  `sheet`, `movie`, the viewer's panes, tiles, A/B differences, `post` transforms, layouts.
- **Overlays.** `mesh_boxes[:levels=0;2,labels=1,color=...,lw=..]`: the outline of every refinement box where the plane cuts it (a polygon for a tilted
  plane), coloured per level. `mesh_markers[:labels=0]`: the object's `markers()`. The viewer offers both when a mesh run is open. A plugin overlay
  (`register_overlay`) sees the snapshot dict with `snap["mesh"]` (the opened object) and `snap["kind"]`; `splosh.mesh.to_window(frame, view, pos)` maps
  world positions to window coordinates.
- `splosh info SNAP` lists the levels, patches, boxes, variables and fields; `splosh fields` the registered kinds.

## Snapshot products of a mesh run
`register_product`'s functions receive the paths of the run's snapshots, i.e. the kind's files. `splosh.kinds.open_mesh(path)` gives the opened object,
`splosh.mesh.sample(obj, path, points, [vars], interp="linear")` the values at arbitrary points (e.g. a 1D cut), with the same finest-level rule as the maps.

## Tests
`tests/meshref.py` (the format and its kind) and `tests/test_mesh.py`: synthetic files with three levels, two boxes on one level, a patch seam,
coarse data under fine data, analytic fields (a linear function: exact everywhere; 1 - 1/(2r): exact at grid points and against an independent trilinear
interpolation), arbitrary planes, box outlines, the map store, the server; `tests/test_viewer_browser.py::test_mesh_run_opens_and_shows_a_map` opens a
mesh run in a browser.
