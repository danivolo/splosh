# Adapting SPLOSH to your own code and machines

For a new user (and their coding assistant) taking `main` to another GIZMO/GADGET-like code, e.g. a GRMHD version of GIZMO without black-hole modules.
Read this first, then [USAGE.md](USAGE.md) (the full reference), [DESIGN.md](DESIGN.md) and [BRANCHES.md](BRANCHES.md).

## 1. Install and check
```bash
git clone <your copy of the repository> splosh && cd splosh
python -m venv ~/envs/splosh && . ~/envs/splosh/bin/activate     # python >= 3.10
pip install -e .                       # numpy, numba, h5py; optional: matplotlib (figures, layouts), ffmpeg on PATH (movies)
python -m pytest -q tests              # must pass before you change anything (a few tests skip without optional packages)
splosh info /path/to/snapshot_000.hdf5        # what the loader sees in your snapshots
splosh fields                                  # fields, presets, plugins it knows
```

## 2. How it is organised (where your changes go)
- **Keep `main` generic; put your code-specific parts in a plugin package**, as the author does with a private package:
  - a module that calls `splosh.plugins.register_*` on import, found through the entry point group `splosh.plugins` in its own `pyproject.toml`, or through `SPLOSH_PLUGINS=yourpkg`;
  - this way you can keep pulling `main` without conflicts.
  - The registries (`src/splosh/plugins.py`):
    - `register_field(name, fn, needs=..., unit=...)`: derived fields from snapshot datasets (e.g. your GR quantities, Lorentz factor, b^2, sigma);
    - `register_overlay(name, fn, options=...)`: drawn items on maps (arrows, polylines, points, text);
    - `register_scalar(name, fn)`: one number per snapshot for `splosh analyze` and the time-series panes;
    - `register_series(name, fn, files, ...)`: time series read from your code's text logs (series.py; GIZMO's `energy.txt` and `blackholes.txt` are the built-in examples);
    - `register_preset(name, settings)`: named view settings for your set-ups;
    - `register_info_section(fn)`: extra lines in `splosh info`.
- **What the loader assumes** (`src/splosh/io.py`): GADGET/GIZMO HDF5 with `PartType0` `Coordinates`, `SmoothingLength` (GIZMO's kernel support radius H), `Masses`, `Density`, `InternalEnergy`, `Velocities`; other datasets are read when a field needs them; header `Time`, `BoxSize`, units if present.
  - A different kernel convention (H = support vs H = 2 x support) is the `hscale` view option. Check one map against SPLASH or your own tool before trusting numbers (VALIDATION.md shows how).
- **Black holes are optional**: the centre rule `bh`, the `bh` markers and the binary period only apply when the snapshot has `PartType5`. Without it, use `--center com` (or `peak`, `mass-weighted`, or a particle ID; see `splosh render --help`), `--align none`, and `--tilt L` to look face-on onto a disc.
  Check that nothing in your set-up defaults to `bh` (the server's `--center`, presets, layouts).
- **Magnetic fields**: the fields `B`, `beta`, `divB` and the B overlays use `MagneticField` (and `DivergenceOfMagneticField`) when present.
  A GRMHD code may store B in other units or conventions (e.g. Heaviside-Lorentz, a comoving 4-vector b^mu): add your own fields rather than changing the built-ins, and set their units through `register_field(..., unit=...)`.
- **Units**: code units come from the header when present (`units.py`); "astro" units assume pc, Msun, km/s style headers. For geometric units (G = c = 1, M = 1), use `--units code` and label your fields yourself.

## 3. Running on a cluster (the part that depends on your machines)
The design: the server runs where the data are; the browser runs on your laptop; nothing is exposed to the network.
- `splosh serve` binds to **127.0.0.1 only**, and every request needs a **random token** (in the URL it prints, or the header `X-Splosh-Token`). Requests whose `Host` is not localhost are rejected.
  - Other users on the same node can reach 127.0.0.1, so the **token is the only protection**: treat the printed URL like a password (don't paste it into chats or shared logs; a new token on every start, or `--token` if you need a fixed one).
- **Reaching it**: `ssh -L PORT:localhost:PORT you@login-node`, then open the printed URL locally.
  - If the server runs on a compute node (an interactive job), the node only listens on its own loopback. Use `ssh -J you@login-node -L PORT:localhost:PORT you@compute-node` if compute nodes accept ssh from the login node; otherwise ask your centre for their supported tunnelling (some provide a portal, Open OnDemand or a proxy). See USAGE.md, "Remote data".
- **Policies of serious machines** to check before running anything:
  - Login nodes usually forbid long or heavy processes: run `serve` inside an interactive allocation, not on the login node. (`--exit-when-orphaned` and the idle exit stop it when the ssh session ends.)
  - Port forwarding may be disabled or restricted (`AllowTcpForwarding`). MFA or jump hosts may be needed: put them in your `~/.ssh/config` (ProxyJump, ControlMaster for one login per session), not in SPLOSH.
  - **`--slurm "srun options"`** lets the viewer start long jobs (movies, prefetch) on a free node with `srun --immediate`, falling back to the local process. Leave it off unless you want that; check that your account and partition flags are in the options. Other schedulers (PBS, LSF) need a small change in `src/splosh/jobs.py` (`JobManager._cmd`), the only place that knows about Slurm.
  - **Disk**: the map cache defaults to `~/.cache/splosh/maps` (`SPLOSH_STORE` or `--store` to move it), and jobs to `~/.cache/splosh/jobs` (`--jobs-dir`). Home quotas on HPC are small: point both at scratch.
    The time-series cache and the analysis outputs are written next to the run (`<run>/analysis_viewer/`); if the run directory is read-only, the time series fall back to `~/.cache/splosh/series`.
  - Threads: rendering uses numba threads (the page's cores control; `NUMBA_NUM_THREADS`). Match it to your allocation, not to the node size.
- No network access is needed at run time: the page loads nothing from the internet.

## 4. What to look at first when adapting (a checklist for you or your assistant)
1. `splosh info` on one snapshot of each kind you have: are the datasets, units and time read right?
2. One `splosh render` with `--center com --align none`, against a picture you trust. Check the kernel convention (`hscale`).
3. Your fields as a plugin (section 2). Start with the ones you plot most. Add a test per field with a tiny synthetic snapshot (see `tests/test_server_fields.py` for how they are made).
4. Remove or override BH-centred defaults in your own presets or layouts.
5. Your logs as `register_series` sources if your code writes per-step text files (copy the `energy` source in `src/splosh/series.py`).
6. Cluster: the commands of section 3 on your machine, a test with a small run, then the real ones.
7. Keep changes to core files minimal and general (if one is needed, make it an extension point usable by any plugin); keep the tests passing.
