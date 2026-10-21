"""Sculpt an athletic footballer as a signed-distance field and mesh it with marching cubes.

The body is built from anatomical volumes (skull, jaw, brow, nose, ears, chest, pecs, lats,
abdomen, pelvis, glutes, deltoids, biceps, forearms, hands, quads, hamstrings, knees, calves,
boots) blended with smooth unions, the way a sculptor blocks out a figure. Left and right
limbs are joined to the torso but never to each other.

Bind pose: A-pose (arms 35 degrees out), facing +z, feet on y = 0, height about 1.80 m.
"""
import numpy as np
from skimage.measure import marching_cubes

ARM = np.radians(35)
ADIR = np.array([np.sin(ARM), -np.cos(ARM), 0.0])
SH = np.array([0.20, 1.43, 0.0])
EL = SH + ADIR * 0.29
WR = EL + ADIR * 0.25


def mirror(v):
    return np.array([-v[0], v[1], v[2]])


def length(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = length(q)
    k1 = length((p - c) / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sphere(p, c, r):
    return length(p - c) - r


def round_cone(p, a, b, r1, r2):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a; l2 = ba @ ba; rr = r1 - r2; a2 = l2 - rr * rr; il2 = 1.0 / l2
    pa = p - a; y = pa @ ba; z = y - l2
    xv = pa * l2 - np.outer(y, ba); x2 = np.sum(xv * xv, axis=-1)
    y2 = y * y * l2; z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    c1 = np.sign(z) * a2 * z2 > k
    c2 = np.sign(y) * a2 * y2 < k
    d = np.where(c2, np.sqrt(x2 + y2) * il2 - r1, d)
    d = np.where(c1, np.sqrt(x2 + z2) * il2 - r2, d)
    return d


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def head(p):
    d = ellipsoid(p, np.array([0, 1.695, -0.005]), np.array([0.091, 0.115, 0.104]))
    d = smin(d, ellipsoid(p, np.array([0, 1.715, -0.03]), np.array([0.094, 0.1, 0.1])), 0.03)            # back of skull
    d = smin(d, ellipsoid(p, np.array([0, 1.628, 0.03]), np.array([0.066, 0.05, 0.068])), 0.035)         # jaw
    d = smin(d, sphere(p, np.array([0, 1.598, 0.066]), 0.028), 0.03)                                     # chin
    d = smin(d, ellipsoid(p, np.array([0, 1.724, 0.08]), np.array([0.066, 0.016, 0.028])), 0.035)        # brow ridge
    for s in (1, -1):
        d = smin(d, ellipsoid(p, np.array([s * 0.047, 1.672, 0.066]), np.array([0.026, 0.016, 0.022])), 0.04)  # cheekbones
        d = smin(d, ellipsoid(p, np.array([s * 0.0145, 1.664, 0.102]), np.array([0.012, 0.009, 0.01])), 0.012)  # nostril wings
        d = smin(d, ellipsoid(p, np.array([s * 0.093, 1.685, -0.008]), np.array([0.013, 0.031, 0.022])), 0.012)  # ears
    d = smin(d, round_cone(p, [0, 1.706, 0.099], [0, 1.668, 0.112], 0.010, 0.015), 0.022)               # nose
    d = smin(d, ellipsoid(p, np.array([0, 1.633, 0.094]), np.array([0.025, 0.01, 0.012])), 0.014)      # lips
    # eye sockets: carve slightly
    for s in (1, -1):
        d = -smin(-d, ellipsoid(p, np.array([s * 0.037, 1.703, 0.117]), np.array([0.021, 0.01, 0.012])), 0.012)
    return d


def torso(p):
    d = ellipsoid(p, np.array([0, 1.33, 0.0]), np.array([0.172, 0.16, 0.112]))                        # ribcage
    d = smin(d, ellipsoid(p, np.array([0, 1.31, -0.03]), np.array([0.188, 0.17, 0.098])), 0.04)       # lats / upper back
    for s in (1, -1):
        d = smin(d, ellipsoid(p, np.array([s * 0.074, 1.355, 0.055]), np.array([0.083, 0.058, 0.058])), 0.03)  # pecs
        d = smin(d, round_cone(p, [s * 0.02, 1.535, -0.02], [s * 0.17, 1.455, -0.01], 0.048, 0.04), 0.04)   # trapezius
        d = smin(d, sphere(p, np.array([s * 0.198, 1.418, 0.0]), 0.064), 0.035)                            # deltoid
        d = smin(d, ellipsoid(p, np.array([s * 0.066, 0.9, -0.058]), np.array([0.079, 0.088, 0.07])), 0.03)    # glutes
    d = smin(d, ellipsoid(p, np.array([0, 1.13, 0.004]), np.array([0.142, 0.16, 0.104])), 0.06)        # abdomen / obliques
    d = smin(d, ellipsoid(p, np.array([0, 0.955, -0.005]), np.array([0.158, 0.11, 0.108])), 0.05)      # pelvis
    d = smin(d, round_cone(p, [0, 1.46, -0.006], [0, 1.62, -0.012], 0.058, 0.051), 0.04)               # neck
    return d


def arm(p, s):
    S = SH if s > 0 else mirror(SH); E = EL if s > 0 else mirror(EL); W = WR if s > 0 else mirror(WR)
    ad = ADIR if s > 0 else mirror(ADIR)
    d = round_cone(p, S, E, 0.054, 0.041)
    d = smin(d, round_cone(p, S + ad * 0.07 + [0, 0, 0.012], S + ad * 0.22 + [0, 0, 0.01], 0.05, 0.043), 0.03)   # biceps
    d = smin(d, round_cone(p, S + ad * 0.06 + [0, 0, -0.012], S + ad * 0.24 + [0, 0, -0.012], 0.048, 0.04), 0.03)  # triceps
    d = smin(d, round_cone(p, E, W, 0.041, 0.029), 0.02)
    d = smin(d, round_cone(p, E + ad * 0.05, E + ad * 0.14, 0.043, 0.038), 0.04)                                 # forearm belly
    d = smin(d, round_cone(p, W + ad * 0.02, W + ad * 0.1, 0.029, 0.027), 0.02)                                  # palm
    d = smin(d, round_cone(p, W + ad * 0.1, W + ad * 0.17, 0.024, 0.017), 0.015)                                 # fingers
    d = smin(d, round_cone(p, W + ad * 0.04 + [0, 0, 0.03], W + ad * 0.09 + [0, 0, 0.05], 0.014, 0.011), 0.012)  # thumb
    return d


def leg(p, s):
    x = 0.095 * s
    d = round_cone(p, [x * 1.05, 0.95, 0], [x, 0.5, 0.006], 0.087, 0.054)
    d = smin(d, ellipsoid(p, np.array([x * 1.05, 0.72, 0.028]), np.array([0.074, 0.165, 0.064])), 0.04)   # quads
    d = smin(d, ellipsoid(p, np.array([x, 0.72, -0.026]), np.array([0.066, 0.15, 0.058])), 0.04)           # hamstrings
    d = smin(d, ellipsoid(p, np.array([x, 0.505, 0.014]), np.array([0.042, 0.046, 0.04])), 0.06)            # knee
    d = smin(d, round_cone(p, [x, 0.5, 0.004], [x, 0.085, -0.004], 0.048, 0.032), 0.05)                     # shin
    d = smin(d, ellipsoid(p, np.array([x, 0.355, -0.026]), np.array([0.054, 0.105, 0.054])), 0.04)        # calf
    d = smin(d, sphere(p, np.array([x, 0.08, -0.008]), 0.037), 0.04)                                        # ankle
    boot = round_cone(p, [x, 0.052, -0.045], [x, 0.036, 0.175], 0.047, 0.034)                               # boot
    d = smin(d, boot, 0.03)
    d = np.maximum(d, -p[:, 1] + 0.001)                                                                      # flat sole
    return d


def body_sdf(p):
    core = smin(torso(p), head(p), 0.03)
    d = smin(core, arm(p, 1), 0.035)
    d = smin(d, arm(p, -1), 0.035)
    d = smin(d, leg(p, 1), 0.03)
    d = np.minimum(d, smin(core, leg(p, -1), 0.03))
    return d


BOUNDS = (np.array([-0.72, -0.01, -0.22]), np.array([0.72, 1.84, 0.3]))


def mesh(h):
    lo, hi = BOUNDS
    xs = np.arange(lo[0], hi[0] + h, h); ys = np.arange(lo[1], hi[1] + h, h); zs = np.arange(lo[2], hi[2] + h, h)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    for i, x in enumerate(xs):
        Y, Z = np.meshgrid(ys, zs, indexing='ij')
        P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=1)
        vol[i] = body_sdf(P).reshape(Y.shape)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    verts += lo
    # marching cubes winding faces inward for an SDF that is negative inside; flip to face outward
    faces = faces[:, ::-1].copy()
    # smooth normals from the field gradient
    e = h * 0.5
    n = np.stack([body_sdf(verts + [e, 0, 0]) - body_sdf(verts - [e, 0, 0]),
                  body_sdf(verts + [0, e, 0]) - body_sdf(verts - [0, e, 0]),
                  body_sdf(verts + [0, 0, e]) - body_sdf(verts - [0, 0, e])], axis=1)
    n /= np.maximum(length(n)[:, None], 1e-9)
    return verts.astype(np.float32), faces.astype(np.int32), n.astype(np.float32)


if __name__ == '__main__':
    import time
    for name, h in (('lod0', 0.0105), ('lod1', 0.02)):
        t = time.time()
        v, f, n = mesh(h)
        np.savez(f'./{name}.npz', v=v, f=f, n=n)
        print(name, len(v), 'verts', len(f), 'tris', f'{time.time() - t:.1f}s', 'y', v[:, 1].min(), v[:, 1].max())
