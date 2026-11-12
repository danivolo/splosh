"""`splosh info SNAP`: a quick summary of a GIZMO / GADGET-like HDF5 snapshot (counts per type, datasets, units and time, box, gas mass / H / density
statistics, centre of mass, black holes). Machine readable through snapshot_info() / `--json`. Plugins add sections (plugins.register_info_section)."""
import json
import os

import h5py
import numpy as np

from . import plugins
from .units import Units, SYSTEMS, DENSITY, LENGTH, ENERGY_PER_MASS, MASS

PTYPE = ["gas", "type1", "type2", "type3", "type4", "bh"]       # GADGET/GIZMO convention: PartType0..5 (the roles of 1-4 depend on the run)
PCT = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def _stats(a, pct=PCT):
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    d = {"n_nonfinite": int((~ok).sum())}
    if ok.any():
        v = a[ok]
        d.update({"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "percentiles": {str(p): float(x) for p, x in zip(pct, np.percentile(v, pct))}})
    return d


def _wmean(pos, m):
    return (pos * m[:, None]).sum(0) / m.sum()


def snapshot_info(path, plugin_sections=True, units="code"):
    """dict describing the snapshot (JSON-serialisable). units: 'code' | 'cgs' | 'astro': gas totals and percentiles are also given in that system (info['gas']['in_units'])."""
    if units not in SYSTEMS:
        raise ValueError(f"units must be one of {list(SYSTEMS)}")
    path = str(path)
    info = {"file": os.path.abspath(path), "size_bytes": os.path.getsize(path)}
    warn = []
    with h5py.File(path, "r") as f:
        hd = f["Header"].attrs
        g = lambda k, d=None: (hd[k].tolist() if hasattr(hd[k], "tolist") else hd[k]) if k in hd else d
        info["time"] = g("Time")
        info["redshift"] = g("Redshift")
        info["comoving"] = bool(g("ComovingIntegrationOn", 0))
        box = g("BoxSize", 0.0)
        info["box_size"] = box
        info["periodic"] = bool(box and box > 0)
        L, M, V = g("UnitLength_In_CGS"), g("UnitMass_In_CGS"), g("UnitVelocity_In_CGS")
        info["units"] = {"length_cm": L, "mass_g": M, "velocity_cm_s": V, "time_s": (L / V if L and V else None),
                         "G_code": g("Gravitational_Constant_In_Code_Inits")}
        us = Units.from_header({k: hd[k] for k in hd})
        info["unit_system"] = us.describe()
        info["unit_system"]["display"] = units
        info["kernel_function_id"] = g("Kernel_Function_ID")
        info["n_files"] = g("NumFilesPerSnapshot", 1)
        if info["n_files"] and info["n_files"] > 1:
            warn.append(f"snapshot is split into {info['n_files']} files; only this file was read")
        npt = g("NumPart_ThisFile", [])
        counts, fields = {}, {}
        for t in range(6):
            key = f"PartType{t}"
            if key in f and "Coordinates" in f[key]:
                counts[PTYPE[t]] = int(f[key]["Coordinates"].shape[0])
                fields[PTYPE[t]] = [{"name": k, "shape": list(f[key][k].shape[1:]), "dtype": str(f[key][k].dtype)} for k in f[key]]
                if t < len(npt) and int(npt[t]) != counts[PTYPE[t]]:
                    warn.append(f"header NumPart_ThisFile[{t}] = {npt[t]} but {key} has {counts[PTYPE[t]]} rows")
        info["counts"] = counts
        info["fields"] = fields
        info["n_total"] = int(sum(counts.values()))
        masses_com = []
        if "gas" in counts:
            gas = f["PartType0"]
            pos = gas["Coordinates"][...].astype(np.float64)
            m = gas["Masses"][...].astype(np.float64)
            gi = {"mass": {**_stats(m), "total": float(m.sum()), "median": float(np.median(m))}}
            if "SmoothingLength" in gas:
                h = gas["SmoothingLength"][...]
                gi["H"] = _stats(h)
                gi["H"]["note"] = "SmoothingLength = kernel support radius (GIZMO); SPLASH/pynbody use h = H/2 (support 2h)"
                if (h <= 0).any():
                    warn.append(f"{int((h <= 0).sum())} gas particles have SmoothingLength <= 0")
            if "Density" in gas:
                rho = gas["Density"][...]
                gi["density"] = _stats(rho)
                ip = int(np.argmax(rho))
                gi["density_peak"] = {"value": float(rho[ip]), "position": pos[ip].tolist()}
            if "InternalEnergy" in gas:
                gi["u"] = _stats(gas["InternalEnergy"][...])
                if "ElectronAbundance" in gas:        # temperature of an ideal gas, gamma = 5/3, X = 0.76: the same as the `T` field with default options (fields.py)
                    from .fields import mean_molecular_weight, DEFAULT_XH, DEFAULT_GAMMA
                    mu = mean_molecular_weight(gas["ElectronAbundance"][...])
                    T = mu * us.temperature_scale(1.0, DEFAULT_GAMMA) * gas["InternalEnergy"][...].astype(np.float64)
                    gi["T"] = _stats(T)
                    gi["T"]["note"] = f"K; gamma = {DEFAULT_GAMMA:.4g}, mu = 4/(1+3X+4X x_e), X = {DEFAULT_XH}"
                    gi["mu"] = _stats(mu)
            if units != "code":
                conv = {}
                for key, dim, st in (("density", DENSITY, gi.get("density")), ("H", LENGTH, gi.get("H")), ("u", ENERGY_PER_MASS, gi.get("u")), ("mass", MASS, gi["mass"])):
                    if st is None:
                        continue
                    ustr, fac = us.quantity_unit(dim, units)
                    conv[key] = {"unit": ustr, "factor": fac, **{k: st[k] * fac for k in ("min", "max", "mean") if k in st},
                                 "percentiles": {k: v * fac for k, v in st.get("percentiles", {}).items()}}
                conv["mass"]["total"] = gi["mass"]["total"] * conv["mass"]["factor"]
                gi["in_units"] = conv
            gi["bbox"] = {"min": pos.min(0).tolist(), "max": pos.max(0).tolist()}
            gi["center_of_mass"] = _wmean(pos, m).tolist()
            if "Velocities" in gas:
                v = gas["Velocities"][...].astype(np.float64)
                gi["center_of_mass_velocity"] = _wmean(v, m).tolist()
            info["gas"] = gi
            masses_com.append((pos, m))
        if "bh" in counts:
            b = f["PartType5"]
            bp = b["Coordinates"][...].astype(np.float64)
            bm = (b["Masses"][...] if "Masses" in b else np.ones(len(bp))).astype(np.float64)
            ids = b["ParticleIDs"][...].astype(np.int64) if "ParticleIDs" in b else np.arange(len(bp))
            o = np.argsort(ids, kind="stable")[:16]
            bi = {"n": len(bp), "ids": ids[o].tolist(), "mass": bm[o].tolist(), "position": bp[o].tolist(), "mass_dataset": "Masses (dynamical)",
                  "center_of_mass": _wmean(bp, bm).tolist()}
            if "Velocities" in b:
                bi["velocity"] = b["Velocities"][...].astype(np.float64)[o].tolist()
            info["bh"] = bi
            masses_com.append((bp, bm))
        if masses_com:
            P = np.concatenate([p for p, _ in masses_com]); M_ = np.concatenate([m for _, m in masses_com])
            info["center_of_mass_all"] = _wmean(P, M_).tolist()
            info["mass_total"] = float(M_.sum())
    info["warnings"] = warn
    if plugin_sections:
        plugins.load_plugins()
        for fn in plugins.INFO_SECTIONS:
            try:
                res = fn(path, info)
            except Exception as e:
                info["warnings"].append(f"info plugin {getattr(fn, '__name__', fn)} failed: {e!r}")
                continue
            if res:
                info.setdefault("extra", {})[res[0]] = res[1]
    return info


def _f(x, p=5):
    return f"{x:.{p}g}"


def _vec(v):
    return "(" + ", ".join(_f(c, 6) for c in v) + ")"


def format_info(info):
    """human-readable text of snapshot_info()."""
    L = []
    u = info["units"]
    L.append(f"{info['file']}  ({info['size_bytes'] / 1e6:.1f} MB)")
    L.append(f"time {_f(info['time'], 8)} (code)   redshift {info['redshift']}   comoving {info['comoving']}   box {info['box_size']} ({'periodic' if info['periodic'] else 'not periodic'})")
    us = info.get("unit_system")
    if us:
        L.append(f"unit system: length {us['length_cm']:.6g} cm, mass {us['mass_g']:.6g} g, velocity {us['velocity_cm_s']:.6g} cm/s, time {us['time_s']:.6g} s; density {us['density_cgs']:.6g} g/cm^3, "
                 f"column density {us['column_density_cgs']:.6g} g/cm^2, u {us['energy_per_mass_cgs']:.6g} erg/g, pressure {us['pressure_cgs']:.6g} dyn/cm^2"
                 + (f"; h = {us['hubble_param']}" if us.get("hubble_param") is not None else "") + ("; COMOVING (stored code units, no a/h applied)" if us["comoving"] else ""))
        if us["defaulted"]:
            L.append(f"  WARNING: header has no {', '.join(us['defaulted'])}: customary GADGET defaults assumed")
    elif u["length_cm"]:
        L.append(f"units: length {u['length_cm']:.6g} cm, mass {u['mass_g']:.6g} g, velocity {u['velocity_cm_s']:.6g} cm/s, time {u['time_s']:.6g} s, G(code) {u['G_code']}")
    L.append("particles: " + ", ".join(f"{k} {v}" for k, v in info["counts"].items()) + f"   (total {info['n_total']})")
    for k, flds in info["fields"].items():
        L.append(f"  {k} datasets: " + " ".join(d["name"] + ("[" + "x".join(map(str, d["shape"])) + "]" if d["shape"] else "") for d in flds))
    g = info.get("gas")
    if g:
        def row(name, s):
            p = s["percentiles"]
            return f"  {name:8s} min {_f(p['0'])}  1% {_f(p['1'])}  50% {_f(p['50'])}  99% {_f(p['99'])}  max {_f(p['100'])}"
        L.append(f"gas mass: total {_f(g['mass']['total'], 8)}, per particle {_f(g['mass']['min'])} .. {_f(g['mass']['max'])} (median {_f(g['mass']['median'])})")
        if "H" in g:
            L.append(row("H", g["H"]) + "   [kernel support radius]")
        if "density" in g:
            L.append(row("density", g["density"]))
            L.append(f"  densest particle: {_f(g['density_peak']['value'])} at {_vec(g['density_peak']['position'])}")
        if "u" in g:
            L.append(row("u", g["u"]))
        if "T" in g:
            L.append(row("T [K]", g["T"]) + "   [ideal gas, gamma 5/3, mu from ElectronAbundance]")
        for key, c in g.get("in_units", {}).items():
            if key == "mass":
                L.append(f"  in {info['unit_system']['display']}: gas mass total {_f(c['total'], 8)} {c['unit']}")
            else:
                p = c["percentiles"]
                L.append(f"  in {info['unit_system']['display']}: {key:8s} 1% {_f(p['1'])}  50% {_f(p['50'])}  99% {_f(p['99'])}  [{c['unit']}]")
        L.append(f"  gas centre of mass {_vec(g['center_of_mass'])}   bbox {_vec(g['bbox']['min'])} .. {_vec(g['bbox']['max'])}")
    b = info.get("bh")
    if b:
        L.append(f"black holes: {b['n']} (ids sorted; masses from {b['mass_dataset']}); centre of mass {_vec(b['center_of_mass'])}")
        for k in range(len(b["ids"])):
            L.append(f"  id {b['ids'][k]}  mass {_f(b['mass'][k], 7)}  pos {_vec(b['position'][k])}")
    if "center_of_mass_all" in info:
        L.append(f"centre of mass (gas + black holes) {_vec(info['center_of_mass_all'])}   total mass {_f(info['mass_total'], 8)}")
    for name, sec in info.get("extra", {}).items():
        L.append(f"[{name}]")
        txt = sec.get("text") if isinstance(sec, dict) else None
        L.extend(("  " + t) for t in (txt or json.dumps(sec, indent=1).splitlines()))
    for w in info["warnings"]:
        L.append("WARNING: " + w)
    return "\n".join(L)
