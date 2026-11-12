"""Sidecar files: every picture written by shot / sheet / movie gets `<out stem>.json` describing how it was made, enough to redo it (`splosh <cmd> --from-sidecar`)."""
import json
import os
import subprocess
import time

SCHEMA = "splosh-sidecar-1"


def version():
    try:
        from importlib.metadata import version as v
        return v("splosh")
    except Exception:
        return "unknown"


def git_commit():
    """`git describe --always --dirty` of the splosh source tree, or None (installed copy, no git)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        top = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5).stdout.strip()
        if not top or os.path.realpath(os.path.join(top, "src", "splosh")) != os.path.realpath(here):
            return None
        r = subprocess.run(["git", "-C", top, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def sidecar_path(out):
    """`<stem>.json`; if that file already describes a different output with the same stem (img.png then img.pdf), `<out>.json` is used so that nothing is overwritten."""
    p = os.path.splitext(out)[0] + ".json"
    if os.path.exists(p):
        try:
            with open(p) as f:
                prev = json.load(f).get("outputs", [])
            if prev and os.path.abspath(out) not in prev:
                return out + ".json"
        except Exception:
            pass
    return p


def environment():
    return {"splosh_version": version(), "git_commit": git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def write_sidecar(out, data):
    """Write `data` (plus schema and environment) next to `out`; returns the sidecar path."""
    p = sidecar_path(out)
    doc = {"schema": SCHEMA, **environment(), **data, "outputs": [os.path.abspath(out)]}
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    return p


def read_sidecar(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not an splosh sidecar (schema {doc.get('schema')!r})")
    return doc
