"""Layouts as data: a dashboard or a time-series figure is a small declarative object that a GUI can read, edit and write back (`load_layout` / `save_layout` / `validate_layout`),
and `render_layout` draws it. One layout per file (TOML or JSON), e.g. `layouts/overview.toml`:

    schema = "splosh-layout-1"
    name = "overview"
    title = "run overview"      # optional
    scope = "snapshot"          # snapshot: one picture per snapshot (a dashboard) | run: one picture for the whole run (a time-series figure)
    ncols = 2                   # grid columns; a panel without row / col is placed in reading order
    cell = [480, 300]           # optional cell size in pixels (default: the size of the map tiles, else 520 x 260)

    [[derived]]                 # optional new curves from columns: a safe expression (see `eval_expr`), names of earlier columns / derived entries only
    name = "q"
    expr = "bh_m1 / bh_m2"

    [[panel]]                   # common keys: id (unique), type, row, col, rowspan, colspan (grid cells, 0-based), label
    id = "dens"
    type = "map"                # map | profile | timeseries | text
    product = "density"         # map / profile: a [[map]] / [[profile]] product of the analyze config ...
                                # ... or, for a map, the View fields and style inline instead of `product` (field, center, width, npix, cmap, limits, ...)
    [[panel]]
    id = "ts"
    type = "timeseries"         # columns (names of scalars.csv columns, derived entries or expressions), xaxis time | time_myr | index, xoffset, series all | past,
    columns = ["gas_mass", "q"] # xlog, ylog, marker (a vertical line + dots at the snapshot's time)

Keys starting with `x_` are free for a GUI's own use: they are kept as they are and never interpreted. The schema is versioned (`schema`); `registry()` lists everything a panel can refer to
(fields, scalars including plugins, profile quantities, colour maps, ...) and `panel_types()` the keys of each panel type with their defaults."""
import ast
import json
import math
import os
import re

import numpy as np

SCHEMA = "splosh-layout-1"
PANEL_TYPES_NAMES = ("map", "profile", "timeseries", "heatmap", "text")
SCOPES = ("snapshot", "run")
XAXES = ("time", "time_myr", "index", "orb")      # orb: t / P0 (the binary period at the first snapshot)
_COMMON = {"id": None, "type": None, "row": None, "col": None, "rowspan": 1, "colspan": 1, "label": None}
_MAP_STYLE = {"cmap": "viridis", "scale": None, "limits": "global", "pct": None, "vmin": None, "vmax": None, "scalebar": True, "colorbar": True, "viewport": None,
              "overlays": []}
_TOP = {"schema": SCHEMA, "name": None, "title": None, "scope": "snapshot", "ncols": 2, "cell": None, "figure": None, "format": None}
# format: the name of a format file (formats/NAME.toml next to the layouts): [figure] options for the LOOK (style, fonts, sizes, colour bar ticks, ...),
# shared by many layouts; the layout's own [figure] (the ARRANGEMENT: panel size, colour bar placement, titles, gaps) overrides it key by key
# [figure]: the publication renderer's choices (layout_mpl.py), each from a fixed list; the bitmap renderer ignores them
FIGURE = {"size": ("wide", ("single", "wide", "page")),            # width 3.4 / 7.1 / 7.1 in (page: taller), panels keep their aspect
          "colorbar": ("panel", ("panel", "inside", "row", "column", "none")),  # one per map beside it, one per map inside it (top right),
                                                                    # one per row / column of maps (same limits needed), none
          "letters": (False, (True, False)),                        # (a), (b), ... in the panels' corners
          "time": ("panel", ("panel", "first", "none")),            # the snapshot time in every map, in the first one only, or nowhere
          "style": ("default", ("default", "astroplot", "journal", "pgplot", "journal_mono")),   # astroplot: its look with its big sizes; journal / pgplot /
                                                                    # journal_mono: only that astroplot style (fonts, ticks), matplotlib's sizes
          "dpi": (200, None),                                       # raster outputs (PNG, frames)
          "panel_in": (None, None),                                 # inches per map panel (e.g. 10: large dashboard figures); replaces size
          "colorbar_ticks": ("splosh", ("splosh", "matplotlib")),   # matplotlib: the bar is matplotlib's own norm (log ticks 10^n)
          "header": ("none", ("none", "run")),                      # run: "[run] Snap: N | t = T | Width = W unit" above the panels
          "nodata": ("dark", ("dark", "white")),                    # pixels without gas (NaN): the viewer's dark background, or white
          "title_pt": (None, None), "header_pt": (None, None),
          "colorbar_label": (True, (True, False)),                  # the quantity and unit along the bar
          "titles": (True, (True, False)),                          # the panels' titles (their labels) above them
          "gap": ("auto", ("auto", "none")),                        # none: the maps touch (tick labels on the outer edges only)
          "label_pt": (None, None), "tick_pt": (None, None)}         # font sizes (points) of the axis labels and of the tick labels (also the bars')       # font sizes (points) of the panel titles and the header; None: the style's


def figure_options(layout):
    """the [figure] options of a layout with defaults filled in, checked against the fixed lists."""
    raw = layout.get("figure") or {}
    _check_keys("figure", raw, set(FIGURE))
    out = {}
    for k, (default, allowed) in FIGURE.items():
        v = raw.get(k, default)
        if allowed is not None and v not in allowed:
            raise LayoutError(f"figure.{k} must be one of {list(allowed)}, got {v!r}")
        out[k] = v
    if not (isinstance(out["dpi"], int) and 50 <= out["dpi"] <= 1200):
        raise LayoutError("figure.dpi must be an integer between 50 and 1200")
    for k, lo, hi in (("panel_in", 1.0, 40.0), ("title_pt", 4.0, 120.0), ("header_pt", 4.0, 120.0), ("label_pt", 4.0, 120.0), ("tick_pt", 4.0, 120.0)):
        if out[k] is not None and not (isinstance(out[k], (int, float)) and not isinstance(out[k], bool) and lo <= out[k] <= hi):
            raise LayoutError(f"figure.{k} must be a number between {lo:g} and {hi:g} (or unset)")
    return out


class LayoutError(ValueError):
    pass


def _view_keys():
    from .maps import View
    return tuple(View.__dataclass_fields__)


def panel_types():
    """{type: {key: default}} for every panel type (the common keys are included in each); None = unset / required."""
    from .maps import View
    v = View()
    view = {k: getattr(v, k) for k in _view_keys()}
    return {"map": {**_COMMON, "type": "map", "product": None, **view, **_MAP_STYLE},
            "profile": {**_COMMON, "type": "profile", "product": None, "quantities": None, "x": "rbins", "xlog": True, "ylog": True, **view, "bins": 48, "band": True},
            "timeseries": {**_COMMON, "type": "timeseries", "columns": [], "xaxis": "time", "xoffset": 0.0, "series": "all", "xlog": False, "ylog": False, "marker": True,
                           "source": "snapshots", "ids": None, "smooth": 0.0, "raw": True, "yscale": None, "twin": [], "twin_scale": None, "hlines": []},
            "heatmap": {**_COMMON, "type": "heatmap", "source": None, "prefix": None, "ids": None, "xaxis": "time", "norm": "none", "zscale": "log", "cmap": "magma",
                        "nx": 600, "overlay": [], "marker": True, "zlabel": None, "ylabel": None},
            "text": {**_COMMON, "type": "text", "text": ""}}


# ----------------------------------------------------------------------------- safe expressions

_FUNCS = {"log10": np.log10, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def parse_expr(text):
    """Check an expression against the whitelist (numbers, column names, + - * / **, unary +/-, parentheses, the functions log10 log sqrt abs exp) and return (tree, names). Never eval."""
    if not isinstance(text, str) or not text.strip() or len(text) > 200:
        raise LayoutError(f"expression {text!r}: must be a non-empty string of at most 200 characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise LayoutError(f"expression {text!r}: syntax error ({e.msg})")
    names, n = [], 0
    for node in ast.walk(tree):
        n += 1
        if n > 80:
            raise LayoutError(f"expression {text!r}: too long")
        if isinstance(node, (ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            continue
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and not isinstance(node.right, ast.Constant):
                raise LayoutError(f"expression {text!r}: the exponent of ** must be a number")
            continue
        if isinstance(node, ast.UnaryOp):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Name):
            if node.id not in _FUNCS:
                names.append(node.id)
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS and len(node.args) == 1 and not node.keywords):
                raise LayoutError(f"expression {text!r}: only {sorted(_FUNCS)} with one argument may be called")
            continue
        raise LayoutError(f"expression {text!r}: {type(node).__name__} is not allowed (numbers, column names, + - * / **, {sorted(_FUNCS)})")
    return tree, list(dict.fromkeys(names))


def eval_expr(text, columns):
    """Value of the expression with `columns` = {name: array or float}; unknown names are an error that lists the available ones."""
    tree, names = parse_expr(text)
    missing = [n for n in names if n not in columns]
    if missing:
        raise LayoutError(f"expression {text!r}: unknown column(s) {missing}; available: {sorted(columns)}")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return np.asarray(columns[node.id], dtype=float)
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call):
            return _FUNCS[node.func.id](ev(node.args[0]))
        return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
    with np.errstate(all="ignore"):
        return ev(tree)


def with_derived(layout, table):
    """The table (dict name -> array) plus the layout's `derived` entries, in order (each may use the ones before it)."""
    out = dict(table)
    n = len(next(iter(table.values()))) if table else 0
    for d in layout.get("derived", []):
        v = eval_expr(d["expr"], out)
        out[d["name"]] = np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
    return out


def curve_values(spec, table):
    """Values of a curve spec: a column name, or an expression of columns."""
    if spec in table:
        return np.asarray(table[spec], dtype=float)
    return np.broadcast_to(np.asarray(eval_expr(spec, table), dtype=float), (len(next(iter(table.values()))),)).copy()


# ----------------------------------------------------------------------------- validate / load / save

def _check_keys(where, d, allowed):
    bad = [k for k in d if k not in allowed and not str(k).startswith("x_")]
    if bad:
        raise LayoutError(f"{where}: unknown key(s) {sorted(bad)}; allowed: {sorted(allowed)}")


MIGRATIONS = {}          # {old schema: function(layout dict) -> layout dict of the next schema}; empty while splosh-layout-1 is the only schema


def migrate(layout):
    """A layout of an older schema brought to the current one by the registered `MIGRATIONS` (a copy; the current schema is returned unchanged)."""
    out = dict(layout)
    for _ in range(len(MIGRATIONS) + 1):
        have = out.get("schema", SCHEMA)
        if have == SCHEMA:
            return out
        if have not in MIGRATIONS:
            break
        out = dict(MIGRATIONS[have](out))
    raise LayoutError(f"layout schema {layout.get('schema')!r} is not {SCHEMA!r} and cannot be migrated (known: {sorted(MIGRATIONS) + [SCHEMA]})")


def validate_layout(layout):
    """A validated, completed copy of a layout dict (defaults filled in, panels placed on the grid). Raises LayoutError naming the panel and key at fault."""
    if not isinstance(layout, dict):
        raise LayoutError("a layout is a table / object")
    layout = migrate(layout)
    _check_keys("layout", layout, set(_TOP) | {"derived", "panel"})
    out = {**_TOP, **{k: v for k, v in layout.items() if k not in ("derived", "panel")}}
    if out["figure"] is not None:
        out["figure"] = figure_options(out)
    if not out["name"] or not isinstance(out["name"], str):
        raise LayoutError("layout: needs a `name`")
    if out["scope"] not in SCOPES:
        raise LayoutError(f"layout {out['name']!r}: scope must be one of {SCOPES}")
    if not (isinstance(out["ncols"], int) and out["ncols"] >= 1):
        raise LayoutError(f"layout {out['name']!r}: ncols must be an integer >= 1")
    if out["cell"] is not None:
        out["cell"] = [int(x) for x in out["cell"]]
        if len(out["cell"]) != 2 or min(out["cell"]) < 40:
            raise LayoutError(f"layout {out['name']!r}: cell = [width, height] in pixels (>= 40)")
    derived = []
    for k, d in enumerate(layout.get("derived", [])):
        _check_keys(f"derived[{k}]", d, {"name", "expr"})
        if not d.get("name") or not d.get("expr"):
            raise LayoutError(f"derived[{k}]: needs `name` and `expr`")
        parse_expr(d["expr"])
        derived.append({k2: v for k2, v in d.items()})
    out["derived"] = derived
    types, panels, seen = panel_types(), [], set()
    from .analyze import _SIGNED
    for k, raw in enumerate(layout.get("panel", [])):
        t = raw.get("type")
        where = f"panel[{k}] {raw.get('id')!r}"
        if t not in types:
            raise LayoutError(f"{where}: type must be one of {list(PANEL_TYPES_NAMES)}, got {t!r}")
        if not raw.get("id") or not isinstance(raw["id"], str) or raw["id"] in seen:
            raise LayoutError(f"{where}: needs a unique string `id`")
        seen.add(raw["id"])
        _check_keys(where, raw, set(types[t]))
        p = {**types[t], **raw}
        for key in ("rowspan", "colspan"):
            if not (isinstance(p[key], int) and p[key] >= 1):
                raise LayoutError(f"{where}: {key} must be an integer >= 1")
        for key in ("row", "col"):
            if p[key] is not None and not (isinstance(p[key], int) and p[key] >= 0):
                raise LayoutError(f"{where}: {key} must be an integer >= 0")
        if (p["row"] is None) != (p["col"] is None):
            raise LayoutError(f"{where}: give both row and col, or neither")
        if t == "map":
            inline = [key for key in set(types["map"]) - set(_COMMON) - {"product"} if key in raw]
            if p["product"] and inline:
                raise LayoutError(f"{where}: `product` and inline settings {sorted(inline)} are exclusive")
            if p["product"]:
                p = {**{k: p[k] for k in _COMMON}, "product": p["product"], **{k: v for k, v in raw.items() if str(k).startswith("x_")}}
            elif p["limits"] not in ("global", "frame", "fixed", "viewport"):
                raise LayoutError(f"{where}: limits must be global, frame, fixed or viewport")
            if not p["product"]:
                p["overlays"] = [p["overlays"]] if isinstance(p["overlays"], str) else list(p["overlays"] or [])
                from .overlays import parse_spec
                from . import plugins
                plugins.load_plugins()
                for spec in p["overlays"]:
                    try:
                        name = parse_spec(spec)["name"]
                    except ValueError as e:
                        raise LayoutError(f"{where}: overlay {spec!r}: {e}") from None
                    if name not in plugins.OVERLAYS:
                        raise LayoutError(f"{where}: overlay {spec!r}: unknown; known: {sorted(plugins.OVERLAYS)}")
                if p["limits"] == "fixed" and (p["vmin"] is None or p["vmax"] is None):
                    raise LayoutError(f"{where}: limits = fixed needs vmin and vmax")
                from . import transforms
                from .analyze import _check_scale
                pst = transforms.style(p.get("post", "none"))
                if p["scale"] is None:
                    p["scale"] = pst["scale"] if pst else ("asinh" if p["field"] in _SIGNED else "log")
                try:
                    p["scale"] = str(_check_scale(p["scale"], where))
                except ValueError as e:
                    raise LayoutError(str(e)) from None
                if pst and "cmap" not in raw:
                    p["cmap"] = pst["cmap"]
                from .maps import View
                try:
                    View(**{key: p[key] for key in _view_keys()}).validate()
                except ValueError as e:
                    raise LayoutError(f"{where}: {e}")
        elif t == "profile":
            if not p["product"]:                         # no product: the azimuthal statistics of the map of the panel's own view (the viewer's radial-profile pane)
                from .maps import View
                try:
                    View(**{key: p[key] for key in _view_keys()}).validate()
                except ValueError as e:
                    raise LayoutError(f"{where}: {e} (a profile panel needs `product`, or the view of a map: field, width, ...)")
                if not (isinstance(p["bins"], int) and 4 <= p["bins"] <= 400):
                    raise LayoutError(f"{where}: bins must be an integer between 4 and 400")
            if p["quantities"] is not None:
                p["quantities"] = [p["quantities"]] if isinstance(p["quantities"], str) else list(p["quantities"])
        elif t == "heatmap":                             # a family of indexed columns of a series source against time (series.heat_tile)
            if not p["source"] or not p["prefix"]:
                raise LayoutError(f"{where}: a heatmap panel needs `source` and `prefix` (columns <prefix><number>, e.g. timebin occ_)")
            if p["xaxis"] not in XAXES or p["norm"] not in ("none", "column") or p["zscale"] not in ("log", "lin"):
                raise LayoutError(f"{where}: xaxis one of {XAXES}, norm none|column, zscale log|lin")
            p["overlay"] = [p["overlay"]] if isinstance(p["overlay"], str) else list(p["overlay"] or [])
            if not (isinstance(p["nx"], int) and 10 <= p["nx"] <= 4000):
                raise LayoutError(f"{where}: nx must be an integer between 10 and 4000")
        elif t == "timeseries":
            cols = [p["columns"]] if isinstance(p["columns"], str) else list(p["columns"])
            if not cols:
                raise LayoutError(f"{where}: needs `columns`")
            twin = [p["twin"]] if isinstance(p["twin"], str) else list(p["twin"] or [])
            src = str(p["source"] or "snapshots")
            for c in cols + twin:
                if src == "snapshots":
                    parse_expr(c)
                elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(c)):
                    raise LayoutError(f"{where}: column {c!r}: a source other than snapshots takes column names, not expressions")
            p["columns"], p["twin"], p["source"] = cols, twin, src
            if p["xaxis"] not in XAXES or p["series"] not in ("all", "past"):
                raise LayoutError(f"{where}: xaxis must be one of {XAXES}, series all or past")
            for k in ("yscale", "twin_scale"):
                if p[k] not in (None, "auto", "lin", "log", "symlog"):
                    raise LayoutError(f"{where}: {k} must be auto, lin, log or symlog")
            try:
                p["smooth"] = float(p["smooth"] or 0.0)
                p["hlines"] = [float(h) for h in ([p["hlines"]] if isinstance(p["hlines"], (int, float)) else (p["hlines"] or []))]
                p["ids"] = None if p["ids"] in (None, "", []) else [int(i) for i in ([p["ids"]] if isinstance(p["ids"], (int, str)) else p["ids"])]
            except (TypeError, ValueError):
                raise LayoutError(f"{where}: smooth must be a number, hlines a list of numbers, ids a list of integers")
            if p["smooth"] < 0:
                raise LayoutError(f"{where}: smooth must be >= 0 (units of P0)")
        panels.append(p)
    out["panel"] = _place(out["name"], panels, out["ncols"])
    return out


def _place(name, panels, ncols):
    """Give every panel row / col: explicit ones first (checked for overlaps), the others in reading order into free cells."""
    taken = {}

    def cells(p, r, c):
        return [(r + i, c + j) for i in range(p["rowspan"]) for j in range(p["colspan"])]
    for p in panels:
        if p["row"] is not None:
            if p["col"] + p["colspan"] > ncols:
                raise LayoutError(f"layout {name!r}: panel {p['id']!r} does not fit in {ncols} columns")
            for cell in cells(p, p["row"], p["col"]):
                if cell in taken:
                    raise LayoutError(f"layout {name!r}: panels {taken[cell]!r} and {p['id']!r} overlap at row {cell[0]}, col {cell[1]}")
                taken[cell] = p["id"]
    for p in panels:
        if p["row"] is None:
            if p["colspan"] > ncols:
                raise LayoutError(f"layout {name!r}: panel {p['id']!r} is wider than the {ncols} columns")
            r = c = 0
            while True:
                if c + p["colspan"] <= ncols and not any(x in taken for x in cells(p, r, c)):
                    break
                c += 1
                if c >= ncols:
                    r, c = r + 1, 0
            p["row"], p["col"] = r, c
            for cell in cells(p, r, c):
                taken[cell] = p["id"]
    return panels


def layout_to_toml(layout):
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return repr(float(v))
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):                          # an inline table ([figure] options)
            return "{" + ", ".join(f"{k} = {val(x)}" for k, x in v.items() if x is not None) + "}"
        raise LayoutError(f"cannot write {v!r} to TOML")
    lines = []

    def table(d, order):
        for k in list(order) + sorted(k for k in d if k not in order):
            if k in d and d[k] is not None and k not in ("panel", "derived"):
                lines.append(f"{k} = {val(d[k])}")
    table(layout, list(_TOP))
    for d in layout.get("derived", []):
        lines += ["", "[[derived]]"]
        table(d, ["name", "expr"])
    types = panel_types()
    for p in layout.get("panel", []):
        lines += ["", "[[panel]]"]
        order = ["id", "type", "row", "col", "rowspan", "colspan", "label"] + [k for k in types[p["type"]] if k not in _COMMON]
        table(p, order)
    return "\n".join(lines) + "\n"


def _read(path):
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    from .analyze import read_toml
    return read_toml(path)


def format_dirs(layout_dir):
    return [os.path.join(layout_dir, "formats"), layout_dir]


def read_format(name, dirs):
    """the [figure] options of format NAME (formats/NAME.toml or .json: a [figure] table, or the options at the top level)."""
    if not name or not isinstance(name, str) or "/" in name or name.startswith("."):
        raise LayoutError(f"format {name!r}: a plain name (a file formats/NAME.toml next to the layouts)")
    for d in dirs:
        for ext in (".toml", ".json"):
            f = os.path.join(d, name + ext)
            if os.path.isfile(f):
                raw = _read(f)
                fig = raw.get("figure", raw)
                return {k: v for k, v in fig.items() if k in FIGURE}
    raise LayoutError(f"format {name!r}: no formats/{name}.toml in {dirs}")


def apply_format(layout, dirs):
    """the layout with its format's [figure] options under its own (the layout's keys win)."""
    if not layout.get("format"):
        return layout
    base = read_format(layout["format"], dirs)
    return dict(layout, figure={**base, **(layout.get("figure") or {})})


def list_formats(layout_dir):
    d = os.path.join(layout_dir, "formats")
    return sorted(os.path.splitext(f)[0] for f in (os.listdir(d) if os.path.isdir(d) else []) if f.endswith((".toml", ".json")))


def load_layout(path):
    """Read and validate a layout file (.toml or .json); a `format` is read from formats/ next to it."""
    try:
        return validate_layout(apply_format(_read(path), format_dirs(os.path.dirname(os.path.abspath(path)))))
    except LayoutError as e:
        raise LayoutError(f"{path}: {e}")


def save_layout(layout, path, completed=True):
    """Validate and write a layout (.toml or .json, by extension); `load_layout(path)` gives back the same layout. completed=False writes the layout as given (a GUI's minimal
    form: only the keys the user set, no filled-in defaults) after checking that it validates."""
    lay = validate_layout(layout)
    if not completed:
        lay = {**migrate(layout), "schema": SCHEMA}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(lay, indent=1) + "\n" if path.endswith(".json") else layout_to_toml(lay)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def find_layout(spec, dirs=()):
    """A layout from a dict (embedded), a file path, or a name looked up as `<dir>/<name>.toml|.json` in `dirs` (then ./layouts)."""
    if isinstance(spec, dict):
        return validate_layout(spec)
    if os.path.isfile(str(spec)):
        return load_layout(str(spec))
    for d in list(dirs) + ["layouts"]:
        for ext in (".toml", ".json"):
            p = os.path.join(d, str(spec) + ext)
            if os.path.isfile(p):
                return load_layout(p)
    raise LayoutError(f"layout {spec!r}: no such file, and no {spec}.toml / {spec}.json in {list(dirs) + ['layouts']}")


def registry():
    """What a GUI can offer: panel types and their keys, derived fields, scalars (built-in columns and plugin ones), profile quantities, colour maps, limit policies, x axes, expression syntax."""
    from . import analyze, colormaps, plugins
    from .maps import field_names
    plugins.load_plugins()
    fields = [{"name": n, "label": plugins.FIELDS[n][1] if n in plugins.FIELDS else n, "builtin": bool(plugins.FIELD_META.get(n, {}).get("builtin", n not in plugins.FIELDS)),
               "doc": plugins.FIELD_META.get(n, {}).get("doc", "")} for n in field_names()]
    scalars = [{"name": n, "columns": list(c), "builtin": True, "doc": (analyze.BUILTIN_SCALARS[n].__doc__ or "")} for n, c in analyze.BUILTIN_SCALAR_COLUMNS.items()]
    scalars += [{"name": n, "columns": None, "builtin": False, "doc": (f.__doc__ or "")} for n, (f, _) in sorted(plugins.SCALARS.items())]
    return {"schema": SCHEMA, "panel_types": panel_types(), "fields": fields, "scalars": scalars, "pipeline_columns": ["snapshot", "time", "time_myr", "index"],
            "profile_quantities": ["density"] + [f["name"] for f in fields if f["name"] != "rho"] + ["A1", "A2", "A3", "phi1", "phi2", "phi3"], "profile_weightings": ["mass", "volume", "number"], "profile_geometries": ["spherical", "cylindrical", "pdf"],
            "figure": {k: {"default": d, "choices": list(a) if a else None} for k, (d, a) in FIGURE.items()}, "overlays": sorted(plugins.OVERLAYS),
            "colormaps": colormaps.names(), "scales": ["log", "linear", "asinh", "asinh:F"], "limit_policies": ["global", "frame", "fixed", "viewport"], "xaxes": list(XAXES),
            "expression": {"operators": ["+", "-", "*", "/", "**"], "functions": sorted(_FUNCS)}, "scopes": list(SCOPES)}


# ----------------------------------------------------------------------------- time-series data

def timeseries_data(panel, table, cur=None, marker=None):
    """What a timeseries panel draws: {'curves': [(label, xs, ys)], 'xnow', 'xlabel', 'xlog', 'ylog'}. `table`: column -> array over the run's snapshots (with the pipeline columns `time`, `time_myr`,
    `index` and the layout's derived entries); `cur`: index of the current snapshot (None for a figure of the whole run); `series = past` keeps the points up to `cur`."""
    n = len(table["index"])
    sel = np.arange(n) if (cur is None or panel["series"] == "all") else np.arange(cur + 1)
    x = np.asarray(table[panel["xaxis"]], dtype=float) - float(panel["xoffset"])
    curves = [(c, x[sel], curve_values(c, table)[sel]) for c in panel["columns"]]
    on = panel["marker"] if marker is None else marker
    return {"curves": curves, "xnow": float(x[cur]) if (cur is not None and on) else None, "xlabel": panel["xaxis"] + (f" - {panel['xoffset']:g}" if panel["xoffset"] else ""),
            "xlog": panel["xlog"], "ylog": panel["ylog"]}


# ----------------------------------------------------------------------------- drawing

_COLORS = [(120, 180, 255), (255, 170, 60), (110, 220, 140), (240, 110, 130), (200, 150, 255), (230, 220, 100)]


def _draw_line(canvas, x0, y0, x1, y1, color):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    ok = (xs >= 0) & (xs < canvas.shape[1]) & (ys >= 0) & (ys < canvas.shape[0])
    canvas[ys[ok], xs[ok]] = color


def _frame(canvas, x, y, w, h, fg):
    canvas[y:y + h, x:x + w] = (28, 28, 34)
    for e in (y, y + h - 1):
        canvas[e, x:x + w] = fg.DIM
    for e in (x, x + w - 1):
        canvas[y:y + h, e] = fg.DIM


def draw_curves(canvas, x, y, w, h, curves, s, label=None, xnow=None, xlog=False, ylog=False, xlabel=None):
    """Several curves [(name, xs, ys)] in one framed panel: legend in the curve colours, y range at the right, x range at the bottom, optional marker at xnow. Only drawn ranges are labelled."""
    from . import figures as fg
    cw, ch = fg.CELL_W * s, fg.CELL_H * s
    _frame(canvas, x, y, w, h, fg)
    lines = ([(label, fg.FG)] if label else []) + [(n, _COLORS[k % len(_COLORS)]) for k, (n, _, _) in enumerate(curves) if len(curves) > 1 or n != label]
    for k, (txt, col) in enumerate(lines):
        fg.draw_text(canvas, x + 3 * s, y + 2 * s + k * (ch + s), txt[:max(4, (w // 2) // cw)], s, col)
    top = 2 * s + len(lines) * (ch + s)
    X0, X1, Y0, Y1 = x + 2 * s, x + w - 1 - 2 * s, y + max(top, ch + 4 * s) if len(lines) > 2 else y + top, y + h - 1 - (ch + 3 * s)
    if Y1 - Y0 < 8:
        Y0 = y + ch + 4 * s
    data = []
    for k, (n, xs, ys) in enumerate(curves):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys) & ((xs > 0) if xlog else True) & ((ys > 0) if ylog else True)
        data.append((k, xs[ok], ys[ok]))
    allx = np.concatenate([d[1] for d in data]) if data else np.array([])
    ally = np.concatenate([d[2] for d in data]) if data else np.array([])
    if allx.size == 0:
        fg.draw_text(canvas, x + 3 * s, y + h // 2, "no data", s, fg.DIM)
        return
    tx = (lambda v: np.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: np.log10(v)) if ylog else (lambda v: v)
    x0, x1 = float(tx(allx).min()), float(tx(np.append(allx, xnow) if (xnow is not None and (not xlog or xnow > 0)) else allx).max())
    y0, y1 = float(ty(ally).min()), float(ty(ally).max())
    x0, x1 = (x0, x1) if x1 > x0 else (x0 - 0.5, x0 + 0.5)
    y0, y1 = (y0, y1) if y1 > y0 else (y0 - 0.5 * (abs(y0) or 1.0), y0 + 0.5 * (abs(y0) or 1.0))

    def px(v):
        return X0 + (tx(v) - x0) / (x1 - x0) * (X1 - X0)

    def py(v):
        return Y1 - (ty(v) - y0) / (y1 - y0) * (Y1 - Y0)
    for k, xs, ys in data:
        col = _COLORS[k % len(_COLORS)]
        pts = list(zip(px(xs), py(ys)))
        for (a0, b0), (a1, b1) in zip(pts[:-1], pts[1:]):
            _draw_line(canvas, a0, b0, a1, b1, col)
        if len(pts) == 1:
            canvas[int(pts[0][1]) - 2 * s:int(pts[0][1]) + 2 * s + 1, int(pts[0][0]) - 2 * s:int(pts[0][0]) + 2 * s + 1] = col
    if xnow is not None and (not xlog or xnow > 0):
        mx = int(round(float(px(xnow))))
        _draw_line(canvas, mx, Y0, mx, Y1, (210, 210, 210))
        for k, xs, ys in data:
            if xs.size:
                j = int(np.argmin(np.abs(xs - xnow)))
                cx, cy = int(round(float(px(xs[j])))), int(round(float(py(ys[j]))))
                canvas[max(cy - s, 0):cy + s + 1, max(cx - s, 0):cx + s + 1] = _COLORS[k % len(_COLORS)]
    ylo, yhi = fg.fmt_tick(float(ally.min())), fg.fmt_tick(float(ally.max()))
    fg.draw_text(canvas, x + w - 3 * s - len(yhi) * cw, y + 2 * s, yhi, s, fg.DIM)
    fg.draw_text(canvas, x + w - 3 * s - len(ylo) * cw, y + h - 2 * s - ch, ylo, s, fg.DIM)
    xr = f"{fg.fmt_tick(float(allx.min()))} .. {fg.fmt_tick(float(max(allx.max(), xnow if xnow is not None else allx.max())))}" + (f"  {xlabel}" if xlabel else "")
    fg.draw_text(canvas, x + 3 * s, y + h - 2 * s - ch, xr[:max(4, (w - 3 * s) // cw // 2 + 6)], s, fg.DIM)


def blank_tile(size, note, s):
    from . import figures as fg
    h, w = size
    t = np.empty((h, w, 3), np.uint8)
    t[:] = (30, 30, 36)
    cw, ch = fg.CELL_W * s, fg.CELL_H * s
    for k, ln in enumerate(fg._wrap([note], max(8, (w - 8 * s) // cw))):
        fg.draw_text(t, 4 * s, 4 * s + k * (ch + s), ln, s, fg.DIM)
    return t


def render_layout(layout, tiles, header="", title=None):
    """Draw a validated layout (uint8 RGB). `tiles` maps a panel id to its data:
        map         {'img', 'meta', 'cmap', 'scale', 'lim': (vmin, vmax), 'scalebar'}   or   {'blank': 'note'}
        profile     {'curves': [(name, xs, ys)], 'xlog', 'ylog'}
        timeseries  the dict of `timeseries_data`
        text        {'text': str}
    A panel without data is drawn as a blank tile with a note."""
    from . import figures as fg, imaging
    panels = layout["panel"]
    maps = [p for p in panels if p["type"] == "map" and "img" in tiles.get(p["id"], {})]
    s = fg._scale_for(tiles[maps[0]["id"]]["img"].shape[1], None) if maps else fg._scale_for((layout["cell"] or [520, 260])[0], None)
    cw, ch = fg.CELL_W * s, fg.CELL_H * s
    framed = {}
    for p in maps:
        t = tiles[p["id"]]
        rgb, _ = imaging.colorize(t["img"], t["cmap"], t["scale"], t["lim"][0], t["lim"][1], nan_color=fg.NAN_COLOR)
        if t.get("overlays"):
            from . import overlays as ovl
            ovl.draw_items(rgb, t["overlays"], t["meta"]["extent"], s)
        fg.overlay_tile(rgb, t["meta"], s, None, t.get("scalebar", True))
        framed[p["id"]] = fg.frame_around(rgb, t["meta"], s, cmap=t["cmap"], scale=t["scale"], vmin=t["lim"][0], vmax=t["lim"][1], title=p["label"] or p["id"], colorbar=True, footer=(), label_chars=8)
    if framed:
        cell_h, cell_w = max(v.shape[0] for v in framed.values()), max(v.shape[1] for v in framed.values())
    else:
        cell_w, cell_h = layout["cell"] or [520, 260]
    if layout["cell"] and not framed:
        cell_w, cell_h = layout["cell"]
    m = 5 * s
    nc = layout["ncols"]
    nr = max((p["row"] + p["rowspan"] for p in panels), default=1)
    head_h = ch + 2 * m
    W = nc * cell_w + (nc + 1) * m
    short = max(9 * ch, cell_h * 2 // 5)                       # rows of curves only (next to map rows) are shorter than a map tile
    rows_h = []
    for r in range(nr):
        plain = framed and not any(p["type"] in ("map", "profile") and p["row"] <= r < p["row"] + p["rowspan"] for p in panels) and any(p["row"] <= r < p["row"] + p["rowspan"] for p in panels)
        rows_h.append(short if plain else cell_h)
    ys = [head_h + m + sum(rows_h[:r]) + r * m for r in range(nr + 1)]   # top of each row (the last entry: the end)
    H = ys[nr] + m
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = fg.BG
    fg.draw_text(canvas, m, m, ((title or layout["title"]) + "   " if (title or layout["title"]) else "") + header, s, fg.FG)
    for p in panels:
        x0, y0 = m + p["col"] * (cell_w + m), ys[p["row"]]
        w, h = p["colspan"] * cell_w + (p["colspan"] - 1) * m, ys[p["row"] + p["rowspan"] - 1] + rows_h[p["row"] + p["rowspan"] - 1] - y0
        t = tiles.get(p["id"])
        if t is None:
            t = {"blank": "no data for this panel"}
        lab = p["label"] or p["id"]
        if p["type"] == "map" and p["id"] in framed:
            canvas[y0:y0 + h, x0:x0 + w] = fg._fit(framed[p["id"]], (h, w))
        elif "blank" in t:
            canvas[y0:y0 + h, x0:x0 + w] = blank_tile((h, w), f"{lab}: {t['blank']}", s)
        elif p["type"] == "text":
            canvas[y0:y0 + h, x0:x0 + w] = blank_tile((h, w), t["text"], s)
        elif "heat" in t:                                 # the heatmap as an image (time to the right, y up), with the current-time marker
            from . import imaging
            hz = np.asarray(t["heat"]["z"], float)[::-1]
            fin = hz[np.isfinite(hz) & ((hz > 0) if t["heat"]["zscale"] == "log" else True)]
            lo_, hi_ = (float(np.percentile(fin, 1)), float(np.max(fin))) if fin.size else (0.0, 1.0)
            rgb, _ = imaging.colorize(hz.astype(np.float32), t["heat"]["cmap"], "log" if t["heat"]["zscale"] == "log" else "linear", lo_, hi_)
            canvas[y0:y0 + h, x0:x0 + w] = fg._fit(rgb, (h, w))
            fg.draw_text(canvas, x0 + 3 * s, y0 + 2 * s, lab, s, fg.FG)
            if t.get("xnow") is not None:
                xe = np.asarray(t["heat"]["x"], float)
                xn = int(x0 + (t["xnow"] - xe[0]) / max(xe[-1] - xe[0], 1e-300) * (w - 1))
                if x0 <= xn < x0 + w:
                    canvas[y0:y0 + h, xn] = (255, 210, 60)
        else:
            draw_curves(canvas, x0, y0, w, h, t["curves"], s, label=lab, xnow=t.get("xnow"), xlog=t.get("xlog", False), ylog=t.get("ylog", False), xlabel=t.get("xlabel"))
    return canvas
