"""Frames: where the map is centred and how the system is oriented, applied to the particles before splatting.

Centre rules (`center` spec string):
    com               mass-weighted centre of all loaded particles (gas + black holes), with their mean velocity
    mass-weighted     mass-weighted centre of the gas only
    bh                mass-weighted centre of the black-hole particles (PartType5); for two black holes the binary centre of mass
    peak              position of the densest gas particle
    ids:A,B,...       mass-weighted centre of the particles (black holes, or gas if ParticleIDs were loaded) with these IDs; `pair:A,B` is an alias
    xyz:x,y,z         a fixed point (x,y,z or just 'x,y,z'); velocity 0
    origin            the coordinate origin (= xyz:0,0,0)
    box               the centre of the box (BoxSize / 2 along every axis; needs a header with BoxSize > 0); velocity 0
Black-hole masses are the dynamical `Masses` dataset (what gravity sees), not the sub-grid `BH_Mass`.

Alignment rules (`align` spec string): a rotation about the z axis (the disc normal stays z), applied after centring:
    none              no rotation
    bh                the vector from the lowest-ID to the second-lowest-ID black hole becomes +x
    pair:A,B          the vector from particle A to particle B becomes +x
    vec:x,y,z         the xy-projection of this vector becomes +x
Rotation angle theta = atan2(dy, dx) of the vector; positions and velocities are rotated by -theta (active rotation, matrix [[c,s],[-s,c]] with c = cos theta,
s = sin theta), the same as rotating a pynbody snapshot with rotate_z(-theta in degrees).
One alignment rule is a full 3D rotation instead (the orbital frame of one particle):
    orbit[:ID]        the particle ID (default: the only black hole, or the most massive one by dynamical mass) relative to the centre: first its orbital angular
                      momentum L = r x v about the centre (r, v relative to the centre's position and velocity) becomes +z (frames.to_z), then the particle is
                      put on the +x axis by a rotation about z. With centre `origin` (or `box`, `xyz:...`) this is the orbital frame of a body orbiting a
                      fixed central mass. When L = 0 (radial motion, or at rest) only the rotation about z is made. Positions and velocities are rotated
                      the same way: the velocities are those of the instantaneous rotated frame (no Omega x r is subtracted).

Tilt rules (`tilt` spec string): an arbitrary 3D rotation applied after centring and alignment, so any line of sight can be the view's z axis. Terms joined by '+'
are applied left to right (each acts on the result of the previous ones); angles in degrees; every rotation is active (p' = R p, velocities too):
    none              no rotation
    incl:I[,PA]       rotate by PA about z, then by I about x: I = 0 face-on, I = 90 edge-on (the old +z then points to -y)
    euler:A,B,C       z-x-z Euler angles: R = Rz(C) Rx(B) Rz(A)
    los:x,y,z         the smallest rotation that makes this direction (in the aligned frame) the +z axis, i.e. the line of sight of an xy view
    L:R               the gas angular momentum about the centre within radius R becomes +z (face-on onto the disc); 'L' alone: all loaded gas
    Lbh               the orbital angular momentum of the black holes about the centre becomes +z (face-on onto the binary's orbit)
Example: 'L:2+incl:60' views the disc within R = 2 at an inclination of 60 degrees."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angle: float = 0.0                      # radians; the alignment vector had this angle to +x before the rotation
    center_spec: str = "xyz:0,0,0"
    align_spec: str = "none"
    tilt: np.ndarray = field(default_factory=lambda: np.eye(3))     # the tilt rotation, applied after the alignment
    tilt_spec: str = "none"
    align_rot: object = None                # a full 3D alignment rotation (align 'orbit'), used instead of the rotation about z by `angle`

    @property
    def rot(self):
        if self.align_rot is not None:
            return self.tilt @ self.align_rot
        c, s = np.cos(self.angle), np.sin(self.angle)
        return self.tilt @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def pos(self, p):
        """positions (N,3) or (3,) in the snapshot frame -> this frame (float64)."""
        return (np.asarray(p, dtype=np.float64) - self.center) @ self.rot.T

    def vel(self, v):
        return (np.asarray(v, dtype=np.float64) - self.velocity) @ self.rot.T

    def describe(self):
        return {"center": [float(c) for c in self.center], "velocity": [float(c) for c in self.velocity], "angle_deg": float(np.degrees(self.angle)),
                "center_spec": self.center_spec, "align_spec": self.align_spec,
                "tilt_spec": self.tilt_spec, "tilt": [[float(x) for x in r] for r in self.tilt],
                **({"align_rot": [[float(x) for x in r] for r in self.align_rot]} if self.align_rot is not None else {})}


def _bh_mass(snap):
    return snap["bh_mdyn"] if "bh_mdyn" in snap else snap["bh_mass"]


def _wmean(pos, m):
    return (pos * m[:, None]).sum(0) / m.sum()


def _parse_floats(txt, n):
    v = [float(t) for t in txt.replace(";", ",").split(",") if t.strip()]
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {txt!r}")
    return np.array(v)


def _find_ids(snap, ids):
    """positions, velocities, masses of the particles with the given IDs (black holes first, then gas if its IDs were loaded)."""
    pos, vel, mas, left = [], [], [], set(int(i) for i in ids)
    if "bh_id" in snap:
        for k, i in enumerate(snap["bh_id"]):
            if int(i) in left:
                pos.append(snap["bh_pos"][k]); vel.append(snap["bh_vel"][k]); mas.append(_bh_mass(snap)[k]); left.discard(int(i))
    if left and "ParticleIDs" in snap:
        sel = np.isin(snap["ParticleIDs"], list(left))
        for k in np.nonzero(sel)[0]:
            pos.append(snap["Coordinates"][k]); vel.append(snap["Velocities"][k]); mas.append(snap["Masses"][k]); left.discard(int(snap["ParticleIDs"][k]))
    if left:
        raise KeyError(f"particle ids not found: {sorted(left)} (gas IDs are only searched if the snapshot was loaded with ParticleIDs)")
    return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64), np.array(mas, dtype=np.float64)


def resolve_center(snap, spec="bh"):
    """(centre, velocity) for a centre rule, see the module docstring."""
    spec = (spec or "bh").strip()
    key, _, arg = spec.partition(":")
    key = key.lower()
    if key == "bh":
        if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
            raise ValueError("centre 'bh': the snapshot has no black-hole particles")
        m = _bh_mass(snap).astype(np.float64)
        return _wmean(snap["bh_pos"], m), _wmean(snap["bh_vel"], m)
    if key == "mass-weighted":
        m = snap["Masses"].astype(np.float64)
        return _wmean(snap["Coordinates"].astype(np.float64), m), _wmean(snap["Velocities"].astype(np.float64), m)
    if key == "com":
        pos, vel, m = [snap["Coordinates"].astype(np.float64)], [snap["Velocities"].astype(np.float64)], [snap["Masses"].astype(np.float64)]
        if "bh_pos" in snap and len(snap["bh_pos"]):
            pos.append(snap["bh_pos"]); vel.append(snap["bh_vel"]); m.append(_bh_mass(snap).astype(np.float64))
        pos, vel, m = np.concatenate(pos), np.concatenate(vel), np.concatenate(m)
        return _wmean(pos, m), _wmean(vel, m)
    if key == "peak":
        i = int(np.argmax(snap["Density"]))
        return snap["Coordinates"][i].astype(np.float64), snap["Velocities"][i].astype(np.float64)
    if key in ("ids", "pair"):
        pos, vel, m = _find_ids(snap, [int(t) for t in arg.split(",") if t.strip()])
        return _wmean(pos, m), _wmean(vel, m)
    if key == "xyz" or (arg == "" and "," in key):
        return _parse_floats(arg if key == "xyz" else spec, 3), np.zeros(3)
    if key == "origin":
        return np.zeros(3), np.zeros(3)
    if key == "box":
        box = (snap.get("header") or {}).get("BoxSize", 0.0)
        box = np.broadcast_to(np.asarray(box, dtype=np.float64), (3,)).copy()
        if not (box > 0).all():
            raise ValueError("centre 'box': the header has no BoxSize > 0 (use origin or xyz:x,y,z)")
        return 0.5 * box, np.zeros(3)
    raise ValueError(f"unknown centre rule {spec!r} (com|bh|mass-weighted|peak|ids:A,B|xyz:x,y,z|origin|box)")


def resolve_angle(snap, spec="none"):
    """rotation angle theta (radians) of the alignment vector, or 0.0 for 'none'."""
    spec = (spec or "none").strip()
    key, _, arg = spec.partition(":")
    key = key.lower()
    if key in ("none", ""):
        return 0.0
    if key == "bh":
        if "bh_id" not in snap or len(snap["bh_id"]) < 2:
            raise ValueError("align 'bh' needs at least two black-hole particles")
        o = np.argsort(snap["bh_id"].astype(np.int64), kind="stable")
        d = snap["bh_pos"][o[1]] - snap["bh_pos"][o[0]]
    elif key == "pair":
        ids = [int(t) for t in arg.split(",") if t.strip()]
        if len(ids) != 2:
            raise ValueError("align 'pair:A,B' needs two ids")
        pos, _, _ = _find_ids(snap, ids[:1]); pos2, _, _ = _find_ids(snap, ids[1:])
        d = pos2[0] - pos[0]
    elif key == "vec":
        d = _parse_floats(arg, 3)
    elif key == "orbit":
        raise ValueError("align 'orbit' is a 3D rotation: use resolve_orbit (make_frame does)")
    else:
        raise ValueError(f"unknown align rule {spec!r} (none|bh|pair:A,B|vec:x,y,z|orbit[:ID])")
    if d[0] == 0 and d[1] == 0:
        raise ValueError("alignment vector has no xy component")
    return float(np.arctan2(d[1], d[0]))


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_z(n):
    """the smallest (active) rotation that maps the direction n onto +z."""
    n = np.asarray(n, dtype=np.float64)
    norm = np.linalg.norm(n)
    if not norm > 0 or not np.isfinite(norm):
        raise ValueError("tilt: the direction is zero or not finite")
    n = n / norm
    z = np.array([0.0, 0.0, 1.0])
    c = float(n @ z)
    if c > 1 - 1e-15:
        return np.eye(3)
    if c < -1 + 1e-15:
        return _rx(np.pi)                           # anti-parallel: half a turn about x
    k = np.cross(n, z)
    s = np.linalg.norm(k)
    k = k / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)       # Rodrigues


def tilt_needs(spec):
    """extra PartType0 datasets a tilt rule reads ('L' needs the gas velocities)."""
    return {"Velocities"} if any(t.strip().split(":")[0] == "L" for t in (spec or "none").split("+")) else set()


def resolve_tilt(snap, spec, frame):
    """the 3x3 tilt matrix of a tilt rule (see the module docstring), given the frame with its centre and alignment (and no tilt yet)."""
    spec = (spec or "none").strip()
    T = np.eye(3)
    if spec in ("", "none"):
        return T
    for term in spec.split("+"):
        key, _, arg = term.strip().partition(":")
        if key == "incl":
            v = [float(t) for t in arg.replace(";", ",").split(",") if t.strip()]
            if not 1 <= len(v) <= 2:
                raise ValueError("tilt 'incl:I[,PA]' needs one or two angles in degrees")
            R = _rx(np.radians(v[0])) @ _rz(np.radians(v[1] if len(v) > 1 else 0.0))
        elif key == "euler":
            a, b, c = _parse_floats(arg, 3)
            R = _rz(np.radians(c)) @ _rx(np.radians(b)) @ _rz(np.radians(a))
        elif key == "los":
            R = to_z(_parse_floats(arg, 3))
        elif key in ("L", "Lbh"):
            cur = Frame(center=frame.center, velocity=frame.velocity, angle=frame.angle, tilt=T, align_rot=frame.align_rot)
            if key == "L":
                if "Velocities" not in snap:
                    raise ValueError("tilt 'L' needs the gas Velocities (load them)")
                p, v, m = cur.pos(snap["Coordinates"]), cur.vel(snap["Velocities"]), np.asarray(snap["Masses"], dtype=np.float64)
                if arg.strip():
                    sel = np.sqrt((p ** 2).sum(1)) < float(arg)
                    p, v, m = p[sel], v[sel], m[sel]
                    if not len(m):
                        raise ValueError(f"tilt 'L:{arg}': no gas within R = {arg} of the centre")
            else:
                if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
                    raise ValueError("tilt 'Lbh': the snapshot has no black-hole particles")
                p, v, m = cur.pos(snap["bh_pos"]), cur.vel(snap["bh_vel"]), _bh_mass(snap).astype(np.float64)
            Lvec = (m[:, None] * np.cross(p, v)).sum(0)
            R = to_z(Lvec)
        else:
            raise ValueError(f"unknown tilt rule {term!r} (none|incl:I[,PA]|euler:A,B,C|los:x,y,z|L[:R]|Lbh, joined by '+')")
        T = R @ T
    return T


def orbit_particle(snap, arg=""):
    """(position, velocity) of the particle of align 'orbit[:ID]': that ID, else the only black hole, else the most massive one (dynamical mass)."""
    arg = (arg or "").strip()
    if arg:
        pos, vel, _ = _find_ids(snap, [int(arg)])
        return pos[0], vel[0]
    if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
        raise ValueError("align 'orbit': the snapshot has no black-hole particles (give a particle: orbit:ID)")
    k = int(np.argmax(_bh_mass(snap)))
    return np.asarray(snap["bh_pos"][k], np.float64), np.asarray(snap["bh_vel"][k], np.float64)


def resolve_orbit(snap, arg, center, velocity):
    """(3x3 rotation, angle about z in radians) of align 'orbit[:ID]' about this centre (see the module docstring)."""
    p, v = orbit_particle(snap, arg)
    r, u = p - np.asarray(center, np.float64), v - np.asarray(velocity, np.float64)
    if not np.linalg.norm(r) > 0:
        raise ValueError("align 'orbit': the particle sits on the centre (choose another centre, e.g. origin, box or xyz:...)")
    L = np.cross(r, u)
    R1 = to_z(L) if np.linalg.norm(L) > 1e-12 * np.linalg.norm(r) * max(np.linalg.norm(u), 1e-300) else np.eye(3)
    q = R1 @ r
    if q[0] == 0 and q[1] == 0:
        raise ValueError("align 'orbit': the particle lies on the z axis and does not move about it")
    phi = float(np.arctan2(q[1], q[0]))
    return _rz(-phi) @ R1, phi


def make_frame(snap, center="bh", align="none", tilt="none"):
    c, v = resolve_center(snap, center)
    key, _, arg = (align or "none").strip().partition(":")
    if key.lower() == "orbit":
        R, phi = resolve_orbit(snap, arg, c, v)
        f = Frame(center=c, velocity=v, angle=phi, center_spec=center, align_spec=align, align_rot=R)
    else:
        f = Frame(center=c, velocity=v, angle=resolve_angle(snap, align), center_spec=center, align_spec=align)
    if (tilt or "none").strip() not in ("", "none"):
        f.tilt = resolve_tilt(snap, tilt, f)
        f.tilt_spec = tilt
    return f


# ----------------------------------------------------------------------------- the frame transform and the window cut of a render, in one parallel pass
import numba as _nb


@_nb.njit(cache=True, parallel=True)
def _select_kernel(coords, center, rrow, h, hscale, cx, cy, half, z0, slice_, nchunk):
    """pass 1: per chunk, the particles whose kernel reaches the window (and, slice_, the plane); pass 2: their frame coordinates, H and index."""
    n = coords.shape[0]
    size = (n + nchunk - 1) // nchunk
    counts = np.zeros(nchunk + 1, np.int64)
    for c in _nb.prange(nchunk):
        k = 0
        for p in range(c * size, min(n, (c + 1) * size)):
            d0 = coords[p, 0] - center[0]; d1 = coords[p, 1] - center[1]; d2 = coords[p, 2] - center[2]
            a = rrow[0, 0] * d0 + rrow[0, 1] * d1 + rrow[0, 2] * d2
            b = rrow[1, 0] * d0 + rrow[1, 1] * d1 + rrow[1, 2] * d2
            hp = h[p] * hscale
            if abs(a - cx) < half + hp and abs(b - cy) < half + hp:
                if slice_:
                    l = rrow[2, 0] * d0 + rrow[2, 1] * d1 + rrow[2, 2] * d2
                    if abs(l - z0) >= hp:
                        continue
                k += 1
        counts[c + 1] = k
    start = np.cumsum(counts)
    m = start[-1]
    xa = np.empty(m, np.float64); ya = np.empty(m, np.float64); za = np.empty(m, np.float64)
    ha = np.empty(m, h.dtype); idx = np.empty(m, np.int64)
    for c in _nb.prange(nchunk):
        k = start[c]
        for p in range(c * size, min(n, (c + 1) * size)):
            d0 = coords[p, 0] - center[0]; d1 = coords[p, 1] - center[1]; d2 = coords[p, 2] - center[2]
            a = rrow[0, 0] * d0 + rrow[0, 1] * d1 + rrow[0, 2] * d2
            b = rrow[1, 0] * d0 + rrow[1, 1] * d1 + rrow[1, 2] * d2
            hp = h[p] * hscale
            if abs(a - cx) < half + hp and abs(b - cy) < half + hp:
                l = rrow[2, 0] * d0 + rrow[2, 1] * d1 + rrow[2, 2] * d2
                if slice_ and abs(l - z0) >= hp:
                    continue
                xa[k] = a; ya[k] = b; za[k] = l; ha[k] = hp; idx[k] = p
                k += 1
    return xa, ya, za, ha, idx


@_nb.njit(cache=True, parallel=True)
def gather(arr, idx):
    """arr[idx] for a 1-d array, in parallel."""
    out = np.empty(len(idx), arr.dtype)
    for k in _nb.prange(len(idx)):
        out[k] = arr[idx[k]]
    return out


def select_in_frame(snap, frame, axes_idx, width, cx, cy, hscale=1.0, z0=0.0, slice_=False):
    """(a, b, l, h, idx) of the gas whose kernel reaches the window: frame coordinates along the view's axes (ia, ib, il), H x hscale, and the indices
    into the snapshot arrays. Equal to Frame.pos followed by the cut of maps.render_map (same rotation, same tests), without the full (N, 3) array."""
    coords = np.asarray(snap["Coordinates"])
    h = np.asarray(snap["SmoothingLength"])
    rot = frame.rot
    rrow = np.ascontiguousarray(np.stack([rot[axes_idx[0]], rot[axes_idx[1]], rot[axes_idx[2]]]), dtype=np.float64)
    nchunk = max(1, min(4 * _nb.get_num_threads(), (len(h) + 9999) // 10000))
    return _select_kernel(coords, np.asarray(frame.center, np.float64), rrow, h, np.float32(hscale) if h.dtype == np.float32 else float(hscale),
                          float(cx), float(cy), 0.5 * float(width), float(z0), bool(slice_), nchunk)
