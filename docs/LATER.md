# Features for later (proposed 2026-10-27; all five built 2026-10-28, see the status lines)

Source: a comparison of SPLOSH with other visualisers (swiftsimio, Sarracen, yt/widgyts, Firefly, ParaView/VisIt, CosmoVis, py-sphviewer, glnemo2).
Its description of SPLOSH was checked against the code; its estimates below are not measured. In the order we would do them.

## 1. Colour the maps on the laptop GPU (WebGL2)
**Built 2026-10-28** (ce0ae2b): the 'GPU colouring' switch; CPU loop kept for histeq and as the reference. The GPU unsharp mask / gradients are not done.
- **Today:** `viewer.html` draws each pane on a 2D canvas: a JavaScript loop over every pixel (stretch, limits, colormap), then `putImageData`. The server only sends
  `float32` maps. (`VIEWER_PLAN.md` had planned the GPU for this.)
- **Proposal:** upload each map once as an `R32F` texture and do the stretch (log, asinh, linear), the limits, the colormap, the differences (A−B, log A/B, A/B−1)
  and the blink in a fragment shader. Dragging the limits or switching colormaps then costs two uniforms, not a pass over 4 × 1024² pixels.
  - Deep-zoom tiles become textured quads (no composite array in JavaScript).
  - Later: an unsharp mask and gradients as shaders (a live width slider).
- It runs on the browser's GPU (the laptop's Iris Xe): the cluster needs no GPU.
- **Keep:**
  - the 2D-canvas path as the fallback (headless renders, the browser tests);
  - the server transforms as the reference for anything saved, so figures keep their byte-identical provenance; a GPU transform is a preview only.
- **Size:** a few hundred lines in `viewer.html`, plus browser tests comparing the WebGL and canvas output pixel by pixel.

## 2. Phase diagrams linked to the map (lasso selection)
**Built 2026-10-28** (143d2f7): phase.py, /api/phase, /api/selection, /api/map?sel=, the viewer's phase panel.
- A pane with a 2D histogram over the particles in the current window: ρ–T, R–v_R, j_z–E, and so on, weighted by mass.
  - Server: `histogram2d` or a Numba binner returning a 256² `float32` grid, drawn with the existing colormap code.
- A box or polygon drawn on it is sent to the server, which selects those particles. They come back:
  - as a `particles:` overlay on the map; or
  - as a map of only that subset (`render_map` on the selection): "where does this gas live".
- It builds on `/inspect`, the particle overlays and the layout editor (a new pane kind).

## 3. Arbitrary viewing angles
**Built 2026-10-28** (40ebd9b): `tilt` (incl, euler, los, L[:R], Lbh). The drag-to-rotate is not done (type the angles).
- **Today:** a rotation about z (`--align`) plus the xy/xz/yz planes (`frames.py`); `BRANCHES.md` lists arbitrary rotation matrices for main.
- **Proposal:** a rotation by Euler angles, or by a line of sight given as a vector, e.g. the disc's angular momentum: edge-on or inclined views of warped discs and streams.
  - Velocities rotate with the frame (as `--align` does now, tested against pynbody).
  - The time alignment and the centre rules are unchanged.
  - In the viewer: angle fields, and later a drag to rotate (one re-render per release).

## 4. Read only the particles a zoomed view needs
**Built 2026-10-28** (455a264): region.py, `serve --partial`; spatial order verified on a binary-disc snapshot (consecutive particles 1.8% as far apart as random pairs). Cold-load timing of a 1 GB snapshot on the cluster: still to measure.
- **Why:** the largest snapshots are about 10⁷ particles and about 1 GB. A cold load reads every array of the requested types: about 10 s on the cluster's NFS
  `/home` (about 100 MB/s shared by the whole cluster), and about 1 GB of memory on the login node per open snapshot. A zoom on the binary needs a few percent of that.
- **How:**
  - GIZMO writes the particles of each task in domain order, along a space-filling curve [memory: to check in the GIZMO source], so a small box covers a few contiguous slices.
  - A sidecar index, made once per snapshot in the map store: the bounding box (+ the largest smoothing length) of every chunk of ~16k consecutive particles,
    about 50 KB. A view reads only the chunks that intersect its window plus H_max.
  - The centre rules that need the whole box (`com`) cache their result, or the BH positions, in the index.
- **Built** (`region.py`, partial loading): the order is coherent (GIZMO's space-filling-curve domain order; on a 0.9M-particle snapshot consecutive particles
  are 1.8% as far apart as random pairs), so no re-sort was needed; the index is per 4096 particles. Still to measure: the cold load of a 1 GB snapshot on the cluster.

## 5. 3D volume rendering (emission + absorption)
**Built 2026-10-28**: `mode=volume`, `kappa`. Perspective is not done.
- The same Numba splatting with particles sorted along the line of sight and alpha blending, `C ← C·exp(−τ_i) + c_i·(1 − exp(−τ_i))`, with `τ_i = κ m_i F_2D`.
  Optional perspective (`x' = f x / (z_cam − z)`, the same for H).
- Dense cold gas in front shadows hot gas behind it: an immediate sense of depth and inclination. Mostly for movies and outreach, not for quantitative work.
  Reuses the kernel tables and the row-band parallelisation.

## Rendering speed: what limits one map (measured 2026-11-13)
- Cluster node (2 x 10 cores + hyperthreads), 8.9M particles, 2048^2 [verified: bench/scaling.py runs]: one socket's 10 cores give x5.2 over 1 thread,
  both sockets x1.27 more, hyperthreads x1.05. A whole-box map 0.9 s, medium 0.3 s, zoom 0.1 s at 20 threads.
- Fixed since: the frame transform, cut and gathers (0.4 s serial, now a parallel pass), the splat's set-up passes, each small particle's normalisation once.
- What remains, phase by phase (laptop, 4 cores, 8.9M particles, whole box) [verified: scratch/astroview/mc/deep.py]: the bookkeeping is 0.8 s of 12.2 s at 4 threads
  (x2.6 from 1 thread); the band loop is balanced (busiest thread 1.5% above the mean) but runs ~25% slower than its bands run one at a time: threads compete for
  memory and caches, and the CPU clocks lower with all cores busy. Not a serial piece of code.
- The work itself: the 1024^2 pyramid level (footprints up to 24 x 24 pixels) is half of it. The pyramid threshold `big` trades speed for accuracy
  [verified: scratch/astroview/mc/big.py, reference big = 24]: 12 (default) median 5e-5, 99% 2e-4 (whole box) / 1.5e-3 (zoom); 8 is 1.7x faster, fine on the whole
  box (99% 1e-3) but 99% errors of x2-3 in a zoomed view (upsampling ringing at steep edges): not usable as a default.
- Possible next steps, not done: band-local contiguous copies of the particle data (less memory traffic); a "draft" render during drags only if its artefacts are acceptable.

## 6. A proper user guide (asked 2026-11-12, not started)
A real SPLOSH documentation, in the spirit of SPLASH's manual, instead of the reference-style USAGE.md:
- getting started (install, first map, first server session through ssh), with pictures;
- one chapter per tool: viewer (panes, keys, overlays, profiles, time series), figures and layouts, movies, the analysis pipeline, comparisons;
- the physics conventions (kernel and H, units, frames and tilt rules, what each field and overlay computes), with the validation results;
- extending it: plugins (fields, overlays, scalars, series), with a worked example;
- running on clusters (ADAPTING.md section 3, expanded);
- built as a static site (e.g. MkDocs) from `docs/`, so the same files read well on GitHub.
Name in prose: SPLOSH; the program, package and repository: splosh.

## 7. Customizable shortcuts (planned 2026-11-12, not built)
Asked: every key and mouse binding of the viewer configurable by the user, with conflict detection, and the help generated from one table.

**Today** [verified: viewer.html]: the dispatch is one `switch (e.key)` in the page's `keydown` handler (plus a few separate listeners: `ctrl+m`, `Esc` in menus,
the mouse handlers of the panes and plots, the scan on `ctrl` / `shift` + wheel). A first step is done: `KEYMAP` in viewer.html is one table
`[action ids, keys as shown, description]` grouped by section, and the help overlay (`h`) is generated from it (`fillHelp`). The switch does not read it yet.

**1. One key-map table.** Replace `KEYMAP` by a table with one row per action:
| field | example | |
|---|---|---|
| `id` | `snap.next`, `zoom.in`, `view.centre.cursor`, `panel.toggle` | stable, used by the overrides |
| `group` | `Snapshots` | section of the help and of the editor |
| `help` | "next snapshot; a number first steps that many" | |
| `keys` | `["space"]`, `["ArrowRight"]`, `["shift+ArrowRight"]` | normalised chords: modifiers in a fixed order `ctrl+alt+shift+meta+KEY`, KEY = `KeyboardEvent.key` (shifted characters as typed: `A`, `?`, `{`, so `shift` is implied for them) |
| `mouse` | `["button3"]`, `["middle"]`, `["ctrl+wheel"]`, `["shift+wheel"]`, `["right"]`, `["dblclick"]` | the mouse gestures the page knows (a fixed list: they need code, not just a name) |
| `run` | `(p, e, n) => step(n)` | the action: the selected pane, the event, the count prefix (`9 space`) |
| `where` | `map`, `plot`, `any`, `figure` | where it applies (`n`, `t`, `C` need the cursor on a map) |
| `count` | true | takes the SPLASH count / `z` x10 prefix |
The `keydown` handler becomes a lookup: normalise the event to a chord, find the row (`where` filters), call `run`; the count / `z` prefix and the
"any key closes the help" rule stay in front of it. The menu bar's `data-key` items name an action id instead of synthesising a key event.

**2. Defaults and overrides.** Defaults are the table (SPLASH's keys as in KEYS.md). The user's overrides are a small map `{action id: [chords]}` (an empty
list unbinds), stored on the server's machine next to the recent folders (`~/.local/state/splosh/recent.json`, `server.RECENT_FILE`): a new
`~/.local/state/splosh/keys.json` (env `SPLOSH_KEYS`), read and written by `GET / PUT /api/prefs/keys` (token-checked like every endpoint, JSON validated:
known ids, well-formed chords, size limit). Not `localStorage`: the page's origin changes with the port on every start, so per-page storage would forget them
(the same reason the recent folders are kept by the server). The page loads the overrides once at start, merges them over the defaults, and keeps a copy in
`localStorage` only as a fallback when the server has none. Unknown ids in the file (an action removed in a later version) are kept and reported, not dropped.

**3. Editor** (View > Keys..., or a button in the help overlay): the table grouped like the help, each row with its current chords as chips; click "+" and press a
key (or a mouse gesture from the fixed list) to add one, x to remove, "default" to restore the row; "restore all" at the bottom. **Conflicts** are checked
live: the same chord on two actions whose `where` overlap is marked red on both rows with the other action's name, and saving is refused until it is resolved
(or one of them is unbound); chords the browser keeps for itself (`ctrl+w`, `ctrl+t`, `ctrl+n`, `F5`, `ctrl+l`) or that typing needs inside inputs are refused
with a note; a shifted character with an explicit `shift+` is normalised first, so `?` and `shift+/` count as the same chord.

**4. Help and menus from the table.** The help overlay lists each action with its *current* chords (overrides included; changed ones marked); the menu bar's
`<kbd>` hints, the context menu's hints and KEYS.md's tables (a small script prints them from the table, so the doc cannot drift) use the same source.

**5. Tests.** A unit test of the chord normalisation and the conflict check (in the browser smoke: set an override through `/api/prefs/keys`, reload, the new key
does the action and the old one does not, the help shows the new key, a conflict blocks the save); a server test of the prefs endpoint (validation, atomic write).

**Size:** a few hundred lines in viewer.html (most of it the editor), about 40 in server.py, the tests. The mouse gestures stay a fixed set with configurable
modifiers; free mapping of arbitrary mouse buttons is not planned.

## Not planned
Voronoi or Delaunay reconstruction of the MFM/MFV cells: treating cells as kernels with support H is the standard and the fast way (SPLASH does the same).
