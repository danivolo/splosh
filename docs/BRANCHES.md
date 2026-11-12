# Branches and plugins: keeping the general tool general

## Rule
- **`main`** is a general tool for anyone with GIZMO/GADGET-like SPH snapshots. It never imports, names or assumes anything of one person's runs, paths, options or
  projects, and it carries no personal data (names, e-mail addresses, machine names, paths of one computer).
- What is specific to someone's simulations (their code's extra datasets and log files, presets for their runs, figures of their thesis, their cluster's tools)
  lives in **a plugin package of their own**, found through the entry-point group `splosh.plugins` (or `SPLOSH_PLUGINS`). See [ADAPTING.md](ADAPTING.md), section 2.
- A private branch or repository may hold such a package next to `main`. It only *adds* files and merges `main` (never the other way round). A feature found
  there that turns out to be general is rewritten without the personal names and committed to `main`.
- Guard: `tools/check_split.sh` greps the general part for markers of one maintainer's projects and fails on a hit; `tests/test_split.py` runs it.

## What goes where
| | main | a plugin package |
|---|---|---|
| splatting, maps, slices, volume rendering, frames, units, derived fields, tiles, partial loading | yes | |
| viewer, layouts and figures, movies, the analysis pipeline, comparisons | yes | |
| time-series sources of GIZMO's own logs (energy, blackholes, cpu, timebin, timings, balance) | yes | |
| the registries: fields, overlays, scalars, series sources, snapshot products, presets, info sections | yes | |
| datasets and logs of a modified code (extra particle fields, per-object log files) | | yes |
| presets and layouts for specific runs, figures of a thesis or paper | | yes |
| tools of one cluster or one person's workflow | | yes |
