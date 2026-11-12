"""splosh (formerly astroview): fast SPH/GIZMO map rendering (numba), a browser viewer, figures and an analysis pipeline."""
import os as _os
# the rename from astroview (2026-11-11): the old environment variables still work, and the old cache / state folders are reused (linked, not moved:
# a server of the old version may still be running on them)
for _k, _v in list(_os.environ.items()):
    if _k.startswith("ASTROVIEW_"):
        _os.environ.setdefault("SPLOSH_" + _k[len("ASTROVIEW_"):], _v)
for _base in (_os.environ.get("XDG_CACHE_HOME") or _os.path.expanduser("~/.cache"), _os.path.expanduser("~/.local/state")):
    _old, _new = _os.path.join(_base, "astroview"), _os.path.join(_base, "splosh")
    try:
        if _os.path.isdir(_old) and not _os.path.lexists(_new):
            _os.symlink(_old, _new)
    except OSError:
        pass
# OpenBLAS (numpy's `@`, e.g. the frame rotation) starts one thread per logical CPU, and they spin after each product, competing with numba's
# splat threads: on a 20-core node render_map took ~100 ms more at >= 12 threads (bench/scaling.py, 2026-10-18). One BLAS thread is as fast
# for the 3x3 products used here. Only effective when splosh is imported before numpy; set OPENBLAS_NUM_THREADS=1 yourself otherwise.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
from .io import load_snapshot, binary_com
from .splat import splat_serial, splat_tl, splat_bands, splat_bands_pyr
from .frames import Frame, make_frame
from .maps import View, render_map
from .figures import shot, contact_sheet, movie, Picture
from . import analyze
from .analyze import Pipeline
from . import layout, compare
from .layout import load_layout, save_layout, validate_layout
from .compare import compare as compare_runs
