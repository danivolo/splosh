# Validation of the splatting engine (phase 0)

Test snapshot: a GIZMO snapshot with 467k gas particles and two black holes, 1024^2 pixels, view centred on the black-hole centre of mass. Timings on a shared 8-core laptop under
load 10-16 (minimum of 3-4 repeats; probably 1.5-2x pessimistic for all codes alike). `bench/compare.py`, `bench/tune.py`, `bench/profile_pyr.py` reproduce them.

## Conventions [verified]
- GIZMO writes `SmoothingLength = H`, the kernel support radius (cubic spline, normalisation 8/pi, cut at r = H). SPLASH halves it when reading GADGET files (h_sph = H/2, support 2h): the same kernel.
- pynbody's kernel has support 2 x `smooth`; `render_image` returns a slice unless `out_units` is a column density (then a projection, integrating to 1.01 for one particle).
- A single particle rendered here and by SPLASH equals the exact projected kernel to 1e-4.

## Speed (column density + one more quantity)
| view | H/pixel (median) | naive serial | row bands (exact) | **multi-resolution (default)** | SPLASH projection | pynbody (approximate_fast) |
|---|---|---|---|---|---|---|
| wide (50 length units) | 12 | n/a | 3.5 s | 1.7 s (1.0 s at `big`=8) | 2.6 s | 1.4 s |
| medium (10) | 59 | 41.5 s | 5.6 s | 0.52 s | 4.0 s | 0.33 s |
| zoom (2) | 295 | n/a | 2.8 s | 0.43 s | 2.2 s | 0.28 s |
One thread: 1.03 s (medium view). 4.7M particles, wide view: 15 s here, 11.7 s pynbody, 27 s SPLASH. Optimisations measured: nearest-bin table 1.4x, fused separable bicubic upsampling 1.75x on that stage.

## Accuracy against SPLASH (median / p99 relative difference of the column density)
| code | medium view | zoom view |
|---|---|---|
| exact row bands | 2.7e-5 / 1.3e-4 | 5.7e-5 / 2.6e-4 |
| multi-resolution, `big` = 12 | 7.0e-5 / 9.7e-4 | 3.2e-4 / 3.2e-3 |
| pynbody, smooth = H/2 | 1.0e-2 / 0.18 | 1.1e-2 / 0.17 |
| pynbody, smooth = H | 1.1e-2 / 0.30 | 6.6e-2 / 0.55 |
SPLASH's projection routine is not mass conserving for particles smaller than half a pixel (1.26x the mass in the wide view); this engine conserves mass to 1e-5.

## Slices, weighted maps (prototype, tests/test_maps.py, bench/compare_slice.py)
- Reference: `tests/refmaps.py` evaluates every pixel against every particle with the exact kernel (closed form for the slice, quadrature + cubic spline for the column integral). [verified by pytest]
  Slice: exact engine within 5e-3 of the maximum (the 4096-bin table), multi-resolution within 2e-2; projection: exact engine 4e-3, multi-resolution 3e-2 (particles above 6 pixels, where the small-particle renormalisation is off).
- Weighted maps (`u`, projection with mass weight, slice with volume or mass weight) equal the brute-force ratio `sum w q K / sum w K` to 6e-3 of the maximum; a constant field is reproduced to 2e-5; velocity fields follow the frame rotation (tested against pynbody's `rotate_z`).
- Real snapshot (649k gas, density slice through the black-hole centre of mass, 512^2, pixels above 1e-3 of the maximum), against pynbody with smooth = H/2 (its table is accurate to about 1%): median difference 7.5e-3 at widths 2, 10 and 40 (p99 1.5e-2, 1.7e-2 and 0.12 at width 40 where few pixels are above the cut) [verified: `python bench/compare_slice.py SNAPSHOT --width W`].
- The multi-resolution merge was not faster than the exact engine for slices (only particles within H of the plane contribute), so slices always use the exact engine; `splat_slice(pyramid=True)` stays available (median 4e-4 / p99 5e-3 against exact at width 2).

## Conventions fixed 2026-10-09 (tests/test_frames.py, tests/test_plugins.py)
- Kernel: `hscale = 1` (H = support radius) is the only convention used for new maps; 2 is a documented legacy switch (comparison with readers that give pynbody `smooth = H`).
- Black-hole centre of mass: dynamical `Masses`. Tested with a snapshot whose `BH_Mass` ratio differs from the `Masses` ratio, for the `bh`, `pair`, `ids`, `com` rules, `binary_com` and the markers (before this date `binary_com` used `BH_Mass`; fixed). [verified by pytest]
- Plugin discovery (entry-point group `splosh.plugins`, `SPLOSH_PLUGINS`) is tested with a toy plugin through the variable and through a fake `.dist-info` on `sys.path`. [verified by pytest]

## Tiles (deep zoom tier; tests/test_tiles.py, bench/tiles_bench.py)
Accuracy of the assembled tiles against `render_map` (pixels farther than 16 from the window border, see DESIGN "Tiles: decisions") [verified by pytest and by the runs below]:
- exact engine, projection, slice and weighted fields, synthetic snapshots with particles on tile corners and edges, larger than a tile (H = 3.5 tile = 2) and smaller than a pixel (cloud-in-cell): identical to float32 summation order (1e-7 of the maximum; test limit 1e-5).
  Single particles of H from 0.004 (a seventh of a pixel) to 4 (two tiles) on corners and edges: same. Real snapshot (465k gas), widths 10 and 2, 512^2 pixels, 256 and 512 pixel tiles: 1e-6 of the maximum.
- multi-resolution engine (default): tiled versus monolithic differ by the engine's own error. Against the exact monolithic map, real snapshot, 2048^2: monolithic 2.9e-4 (width 10) and 1.7e-3 (width 2) of the maximum, tiled with 256 or 512 pixel tiles 2.9e-4 and 1.7e-3, p99 of the relative error above 1e-3 of the maximum
  1.1e-3 / 1.8e-3 (monolithic 1.5e-3 / 3.8e-3): the tiles are as accurate as the monolithic render. Test limit against the monolithic multi-resolution map: 5e-3 of the maximum.
- weighted maps are compared where the weight image is above 1e-3 of its maximum (the ratio of two vanishing sums is rounding noise elsewhere).
- Defect found in the monolithic path [verified]: particles of 1-6 pixels are renormalised over the part of their footprint inside the image, so `render_map` overcounts them where its own window border cuts them (e.g. 0.245 versus 0.013 at an edge pixel). The tiles render a 16 pixel margin and are free of it (a first version without the margin had 1.4x the mass at tile edges).

Speed and memory, real snapshot (465k gas, projection of the density, width 4, centre `bh`, multi-resolution), 8 CPUs, shared laptop at load 5-7 (so pessimistic and noisy, one run each; peak = ru_maxrss of the process, snapshot load included, baseline after loading 124 MB; the
tiled images are written to a memory-mapped `.npy`, whose dirty pages count in the peak; `stream` drops the tiles, i.e. the working set):
| npix | engine | tile / workers | time | peak memory |
|---|---|---|---|---|
| 8192 | monolithic | - | 1.4 s | 996 MB |
| 8192 | tiled | 256 / 1 | 48 s | 490 MB |
| 8192 | tiled | 256 / 8 | 10.8 s | 523 MB |
| 8192 | tiled | 512 / 1 | 7.6 s | 489 MB |
| 8192 | tiled | 512 / 8 | 4.6 s | 549 MB |
| 16384 | monolithic | not possible (View.npix <= 8192; the image alone is 1 GB and the engine's two-channel working arrays several times that [not measured]) | | |
| 16384 | tiled | 256 / 1 | 62 s | 1242 MB |
| 16384 | tiled | 256 / 8 | 43 s | 1278 MB |
| 16384 | tiled | 512 / 1 | 34 s | 1243 MB |
| 16384 | tiled | 512 / 8 | 18 s | 1320 MB |
| 16384 | stream (tiles dropped) | 512 / 8 | 13 s | 290 MB |
Reading: tiling does not make a render that fits in memory faster (4.6 s against 1.4 s at 8192^2): every tile pays for its 16 pixel margin and for each particle larger than a tile, and particles many tiles wide are deposited again in every tile they touch (1781 particles per 256 pixel tile here,
most of the 465k particles are larger than a tile at this zoom). Parallel over tiles does pay (512 tiles: 7.6 s on one worker, 4.6 s on 8 on a loaded machine; 34 s to 18 s at 16384^2), and tiles of 512 are about 2-6x faster than 256 (fewer margins, less per-tile overhead; 256 with one worker was 48 s because each of the 1024
tiles launches parallel numba regions that compete with other processes: use `--tiles 512`). What tiling buys: (1) images above 8192 pixels, 16384^2 in 1.2 GB (of which 1 GB is the output file's pages; the working set is 170 MB above the 124 MB baseline), (2) cached tiles that panning and small zooms reuse
(the monolithic path re-renders the whole window), which is the point for the viewer. Repeat on an idle machine before quoting the times.
