"""Plugin registry: the seam between the general tool and code that is specific to one set of runs.

A plugin is an importable module that registers things when imported:
    from splosh import plugins
    plugins.register_field("T", fn, needs=("ElectronAbundance",))   # fn(ctx) -> per-particle array, see maps.FieldContext
    plugins.register_info_section(fn)      # fn(path, info_dict) -> dict merged into `splosh info` (key 'extra' by section name)
    plugins.register_overlay(name, fn)     # fn(snap, frame, view) -> list of markers
    plugins.register_preset(name, dict)    # named bundle of view settings
    plugins.register_series(name, fn, files=fn2, label=...)   # a time-series source of a run (series.py): fn(ctx) -> series.Table, files(ctx) -> input files (cache key)
    plugins.register_scalar(name, fn, needs=())   # fn(snap, **params) -> float or {column: float}, one row per snapshot of `splosh analyze` (snap: see io.load_snapshot; needs: extra PartType0 datasets, '?X' = optional)
Discovery (load_plugins): entry points of the group `splosh.plugins` (each names a module to import) and the environment
variable SPLOSH_PLUGINS (comma-separated module names). The core never imports a plugin by name."""
import importlib
import os

FIELDS = {}          # name -> (function(ctx) -> array, label, kind, needs)   (needs: tuple, or callable(view) -> iterable of dataset names)
FIELD_META = {}      # name -> dict(unit=<dimension tuple | 'K' | callable(view) | None>, builtin=bool, doc=str)
INFO_SECTIONS = []   # functions (path, info) -> dict or None
OVERLAYS = {}        # name -> function(snap, frame, view) -> list of dict
PRESETS = {}         # name -> dict of View settings
SCALARS = {}         # name -> (function(snap, **params) -> float | dict, needs)
SERIES = {}          # name -> dict(fn, files, label, version, doc): time-series sources (series.py)
PRODUCTS = {}        # name -> dict(fn, needs, version, label, applies, params, defaults, meta): snapshot products (products.py)
_loaded = {"done": False, "modules": [], "errors": []}


def register_field(name, fn, label=None, kind="quantity", needs=(), unit=None, builtin=False, doc=None):
    """needs: extra PartType0 datasets the field reads (the loader fetches them); a tuple, or a callable(view) -> iterable when it depends on the view.
    unit: dimension of the field's values in code units, a tuple (a, b, c) = exponents of (code length, code mass, code velocity) (see units.py),
    'K' for a temperature, a callable(view) -> either of those, or None (unknown: maps of the field carry no unit and are never converted)."""
    FIELDS[name] = (fn, label or name, kind, needs if callable(needs) else tuple(needs))
    FIELD_META[name] = {"unit": unit, "builtin": bool(builtin), "doc": doc or ""}


def register_info_section(fn):
    INFO_SECTIONS.append(fn)


OVERLAY_OPTIONS = {}  # name -> [(option, label, default bool)]: on/off options the viewer offers as sub-checkboxes of the overlay


def register_overlay(name, fn, options=None):
    """options: [(key, label, default)] boolean options of the overlay (spec name:key=0|1); the viewer shows them under the overlay's checkbox."""
    OVERLAYS[name] = fn
    if options:
        OVERLAY_OPTIONS[name] = [(str(k), str(lab), bool(d)) for k, lab, d in options]


def register_preset(name, settings):
    PRESETS[name] = dict(settings)


def register_scalar(name, fn, needs=()):
    SCALARS[name] = (fn, tuple(needs))


def register_series(name, fn, files, label=None, version=1, doc=None, defaults=(), priority=0):
    """A time-series source (see series.py): fn(ctx) -> series.Table; files(ctx) -> the input files (empty: the source is not available for this run;
    their sizes and mtimes, with `version`, key the cache, so bump `version` when fn changes what it computes). defaults: the columns a new pane shows
    (the first that exists); priority: the viewer opens the available source with the highest priority first."""
    SERIES[name] = {"fn": fn, "files": files, "label": label or name, "version": int(version), "doc": doc or "", "defaults": list(defaults), "priority": int(priority)}


def register_product(name, fn, needs=(), version=1, label=None, applies=None, params=None, defaults=(), meta=None):
    """A snapshot product (see products.py): fn(path, params) -> {"scalars": {column: float}, "arrays": {name: array}}, one snapshot. needs: what it
    reads (documentation); applies(path) -> bool for a run; params: parameter defaults (part of the stored results' key, with `version`);
    defaults: the columns a time-series pane shows first; meta: {column: {label, unit, group}}."""
    PRODUCTS[name] = {"fn": fn, "needs": tuple(needs), "version": int(version), "label": label or name, "applies": applies, "params": dict(params or {}),
                      "defaults": list(defaults), "meta": dict(meta or {})}


def load_plugins(force=False):
    """Import the plugin modules named by entry points and SPLOSH_PLUGINS (once). Returns the list of imported module names; failures are kept in `errors()`."""
    if _loaded["done"] and not force:
        return list(_loaded["modules"])
    names = []
    try:
        from importlib.metadata import entry_points
        names += [ep.value.split(":")[0] for ep in entry_points(group="splosh.plugins")] + [ep.value.split(":")[0] for ep in entry_points(group="astroview.plugins")]     # + the old group name
    except Exception as e:                                     # pragma: no cover
        _loaded["errors"].append(("entry_points", repr(e)))
    names += [n.strip() for n in os.environ.get("SPLOSH_PLUGINS", "").split(",") if n.strip()]
    for n in dict.fromkeys(names):
        try:
            importlib.import_module(n)
            _loaded["modules"].append(n)
        except Exception as e:
            _loaded["errors"].append((n, repr(e)))
    _loaded["done"] = True
    return list(_loaded["modules"])


def errors():
    return list(_loaded["errors"])
