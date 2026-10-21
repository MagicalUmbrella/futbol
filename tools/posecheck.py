import numpy as np
from build import BONES, BI, SKIN, SHIRT, SLEEVE, SHORTS, SOCKS, BOOT, TRIM, SOLE, HAND, FOREARM, FACE, HAIR
from render import render
from PIL import Image


def euler_yxz(x, y, z):
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def pose(m, rot):
    W = []
    for i, (n, par, wp) in enumerate(BONES):
        pp = np.array(BONES[par][2]) if par >= 0 else np.zeros(3)
        L = np.eye(4); L[:3, 3] = np.array(wp) - pp
        r = rot.get(n, (0, 0, 0)); L[:3, :3] = euler_yxz(*r)
        W.append(L if par < 0 else W[par] @ L)
    skin = []
    for i, (n, par, wp) in enumerate(BONES):
        ib = np.eye(4); ib[:3, 3] = -np.array(wp); skin.append(W[i] @ ib)
    skin = np.array(skin)
    v = np.c_[m['v'], np.ones(len(m['v']))]
    out = np.zeros((len(v), 3)); nout = np.zeros((len(v), 3))
    for k in range(4):
        Mk = skin[m['j'][:, k]]
        out += m['w'][:, k:k + 1] * np.einsum('nij,nj->ni', Mk, v)[:, :3]
        nout += m['w'][:, k:k + 1] * np.einsum('nij,nj->ni', Mk[:, :3, :3], m['n'])
    nout /= np.linalg.norm(nout, axis=1, keepdims=True) + 1e-9
    return out, nout


pal = {SKIN: (.78, .6, .47), SHIRT: (.78, .15, .16), SLEEVE: (.78, .15, .16), SHORTS: (.95, .95, .95), SOCKS: (.78, .15, .16), BOOT: (.08, .08, .08),
       TRIM: (1, 1, 1), SOLE: (.9, .9, .9), HAND: (.78, .6, .47), FOREARM: (.78, .6, .47), FACE: (.78, .6, .47), HAIR: (.15, .11, .08)}
m = dict(np.load('./lod1_rig.npz'))
col = np.array([pal[int(k)] for k in m['part']])
A = np.radians(35)
poses = {
    'rest': {'uaL': (0, 0, -A + .1), 'uaR': (0, 0, A - .1), 'faL': (-.2, 0, 0), 'faR': (-.2, 0, 0)},
    'run': {'uaL': (.6, 0, -A + .12), 'uaR': (-.6, 0, A - .12), 'faL': (-1.2, 0, 0), 'faR': (-1.1, 0, 0), 'thL': (-.7, 0, 0), 'shinL': (.3, 0, 0),
            'thR': (.5, 0, 0), 'shinR': (1.2, 0, 0), 'spine': (.15, 0, 0), 'chest': (0, -.2, 0), 'head': (0, .3, 0)},
    'kick': {'uaL': (-.3, 0, -A + 1.0), 'uaR': (0, 0, A - .3), 'thR': (-1.1, 0, 0), 'shinR': (.1, 0, 0), 'thL': (-.1, 0, 0), 'shinL': (.3, 0, 0), 'chest': (0, -.2, 0)},
}
ims = []
for nm, rot in poses.items():
    v, n = pose(m, rot)
    for yaw in (25, 100):
        render(v, m['f'].astype(int), n, col, yaw, size=(300, 600), out='/tmp/claude-0/p.png'); ims.append(Image.open('/tmp/claude-0/p.png').copy())
sheet = Image.new('RGB', (300 * len(ims), 600))
for i, im in enumerate(ims): sheet.paste(im, (300 * i, 0))
sheet.save('./pose_sheet.png'); print('ok')
