# SPLOSH

# SPLOSH

Fast maps of GIZMO / GADGET-like SPH snapshots (numba splatting of the SPH kernel: projections, slices, volume rendering), a light browser viewer that
works through an ssh tunnel, publication figures and movies from declarative layouts, time series of the code's own logs, and an incremental analysis
pipeline. The name nods to [SPLASH](https://github.com/danieljprice/splash), Daniel Price's SPH visualisation tool, which SPLOSH's maps are validated
against ([docs/VALIDATION.md](docs/VALIDATION.md)); SPLOSH is a separate code.

**Status: research code under active development.** It is used daily on GIZMO simulations, and the tests pass, but interfaces, file formats of the
caches and layouts, and command-line options may still change without notice. The documentation is a reference (a proper user guide is planned).

Documentation: [docs/USAGE.md](docs/USAGE.md) (everything, by topic), [docs/KEYS.md](docs/KEYS.md) (viewer keys), [docs/DESIGN.md](docs/DESIGN.md),
[docs/VALIDATION.md](docs/VALIDATION.md), [docs/ADAPTING.md](docs/ADAPTING.md) (your own code's fields and logs as plugins, running on a cluster),
[docs/BRANCHES.md](docs/BRANCHES.md) (what belongs in the core and what in a plugin).

    pip install -e .                                   # numpy, numba, h5py; optional: matplotlib (figures, layouts), ffmpeg on PATH (movies)
    python -m pytest -q tests

    splosh info snapshot_005.hdf5 [--json]          # counts, datasets, units, H / density percentiles, centre of mass, black holes
    splosh render snapshot_005.hdf5 --center bh --align bh --width 10 --npix 1024 --out map.png      # cached in the mapstore
    splosh render snapshot_005.hdf5 --mode slice --field u --out u.npz
    splosh serve run_dir/ --center bh --align bh --width 10                                         # prints http://127.0.0.1:PORT/?token=...

```python
import splosh
snap = splosh.load_snapshot("snapshot_005.hdf5")
img, meta = splosh.render_map(snap, splosh.View(center="bh", align="bh", mode="slice", field="u", width=10, npix=512))
```
`H` is GIZMO's `SmoothingLength`, the kernel support radius. The column density conserves the particle mass. Maps are `(npix, npix)` float32 arrays, row 0 = lowest y.
The lower-level engine (`splat_bands_pyr`, `splat_slice`) takes particle arrays directly, see `src/splosh/splat.py`.

Optional: the `astroplot` figure styles (PGPLOT-like looks with their fonts) are a separate package, not published yet; without it the layouts' style
`default` (matplotlib's) is used and the other styles say so.

## Problems and questions
Please open an issue on the project's GitHub page.

## License and credits
SPLOSH is free software under the GNU General Public License, version 3 or (at your option) any later version: see [LICENSE](LICENSE).
The colour tables in `src/splosh/colormaps.py` are sampled from matplotlib's: viridis, inferno and magma (Stéfan van der Walt and Nathaniel Smith,
CC0), RdBu (Cynthia Brewer's ColorBrewer, Apache License 2.0), coolwarm (Kenneth Moreland), hot and gray (matplotlib).
