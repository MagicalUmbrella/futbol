"""Rig, UV and export the sculpted footballer.

Outputs
  matchday_player.glb      textured, skinned glTF binary (opens in Blender / three.js / any glTF viewer)
  player_data.js           compact quantised copy embedded in the game (two LODs)
"""
import base64, json, struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import marching_cubes
import sculpt as S

# ---------------- skeleton (bind pose = A-pose, matches the game's bone order) ----------------
SH, EL, WR, AD = S.SH, S.EL, S.WR, S.ADIR
M = S.mirror
BONES = [  # name, parent, world position
    ('root', -1, [0, 0, 0]), ('hips', 0, [0, .96, 0]), ('spine', 1, [0, 1.06, 0]), ('chest', 2, [0, 1.26, 0]),
    ('neck', 3, [0, 1.48, 0]), ('head', 4, [0, 1.58, 0]),
    ('uaL', 3, list(SH)), ('faL', 6, list(EL)), ('uaR', 3, list(M(SH))), ('faR', 8, list(M(EL))),
    ('thL', 1, [.095, .93, 0]), ('shinL', 10, [.095, .5, 0]), ('footL', 11, [.095, .08, 0]),
    ('thR', 1, [-.095, .93, 0]), ('shinR', 13, [-.095, .5, 0]), ('footR', 14, [-.095, .08, 0]),
]
BI = {b[0]: i for i, b in enumerate(BONES)}
HAND_END = WR + AD * .17
SEGS = {  # bone -> (a, b, radius) used for automatic skin weights
    'hips': ([0, .85, 0], [0, 1.0, 0], .14), 'spine': ([0, 1.0, 0], [0, 1.2, 0], .14), 'chest': ([0, 1.2, 0], [0, 1.46, 0], .16),
    'neck': ([0, 1.46, 0], [0, 1.585, 0], .055), 'head': ([0, 1.585, 0], [0, 1.82, 0], .1),
    'uaL': (SH, EL, .05), 'faL': (EL, HAND_END, .036), 'uaR': (M(SH), M(EL), .05), 'faR': (M(EL), M(HAND_END), .036),
    'thL': ([.095, .95, 0], [.095, .5, 0], .075), 'shinL': ([.095, .5, 0], [.095, .085, 0], .045), 'footL': ([.095, .085, -.04], [.095, .03, .18], .04),
    'thR': ([-.095, .95, 0], [-.095, .5, 0], .075), 'shinR': ([-.095, .5, 0], [-.095, .085, 0], .045), 'footR': ([-.095, .085, -.04], [-.095, .03, .18], .04),
}
# part ids (the game colours each part per player: skin tone, kit colours, boots...)
SKIN, SHIRT, SLEEVE, SHORTS, SOCKS, BOOT, TRIM, SOLE, HAND, FOREARM, FACE, HAIR = range(12)


def seg_dist(p, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + np.outer(t, ab)), axis=1), t


def arm_info(p):
    """distance from the arm axis and distance travelled along the arm, for the side each vertex is on"""
    side = np.where(p[:, 0] >= 0, 1, -1)
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    a = SH; full = HAND_END - SH; L = np.linalg.norm(full)
    t = np.clip(((q - a) @ full) / (L * L), -1, 2) * L
    d = np.linalg.norm(q - (a + np.outer(np.clip(t / L, 0, 1), full)), axis=1)
    return side, d, t


def classify(p):
    side, ad, at = arm_info(p)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    part = np.full(len(p), SKIN)
    isarm = (ad < .08) & (at > .11) & (np.abs(x) > .17)
    part[(y > .99) & (y <= 1.53)] = SHIRT
    part[(y > .73) & (y <= .99)] = SHORTS
    part[(y > .73) & (y <= .99) & (np.abs(x) > .148) & (np.abs(z) < .03)] = TRIM
    part[(y > .465) & (y <= .73)] = SKIN
    part[(y > .43) & (y <= .465)] = TRIM
    part[(y > .085) & (y <= .43)] = SOCKS
    part[(y <= .085) | ((y <= .11) & (z > .05))] = BOOT
    part[(y <= .012)] = SOLE
    neck_r = np.hypot(x, z + .006)
    part[(y > 1.53) & (y <= 1.585)] = SKIN
    part[(y > 1.522) & (y <= 1.538) & (neck_r > .05) & (neck_r < .076)] = TRIM               # collar ring
    part[(y > 1.585)] = FACE
    # arms override
    arm = np.where(isarm)[0]
    t = at[arm]
    ap = np.full(len(arm), SKIN)
    ap[t < .14] = SLEEVE
    ap[(t >= .135) & (t < .16)] = TRIM                                                       # cuff
    ap[(t >= .29) & (t < .52)] = FOREARM
    ap[t >= .52] = HAND
    part[arm] = ap
    return part, isarm, side


def weights(p, part, isarm, side):
    names = [b[0] for b in BONES]
    N = len(p); score = np.full((N, len(BONES)), 1e9)
    for n, (a, b, r) in SEGS.items():
        d, _ = seg_dist(p, a, b)
        score[:, BI[n]] = d / r
    x = p[:, 0]
    allowed = np.ones_like(score, bool)
    allowed[:, BI['root']] = False
    for n in names[1:]:
        j = BI[n]
        if n.endswith('L'): allowed[x < -.02, j] = False
        if n.endswith('R'): allowed[x > .02, j] = False
    legs = ['thL', 'shinL', 'footL', 'thR', 'shinR', 'footR']
    arms = ['uaL', 'faL', 'uaR', 'faR']
    a_idx = np.where(isarm)[0]
    for n in names:
        if n not in ('chest',) + tuple(arms): allowed[a_idx, BI[n]] = False
    for n in arms: allowed[~isarm & (p[:, 1] < 1.3), BI[n]] = False
    lower = (p[:, 1] < .86) & ~isarm
    for n in names:
        if n not in ('hips',) + tuple(legs): allowed[lower, BI[n]] = False
    for n in legs: allowed[p[:, 1] > 1.02, BI[n]] = False
    head = part == FACE
    hair = part == HAIR
    for n in names:
        if n != 'head': allowed[head | hair, BI[n]] = False
    score[~allowed] = 1e9
    mn = score.min(1, keepdims=True)
    w = np.exp(-(score - mn) * 7.0); w[~allowed] = 0
    idx = np.argsort(-w, axis=1)[:, :4]
    ww = np.take_along_axis(w, idx, 1)
    ww[ww < .02] = 0
    ww /= ww.sum(1, keepdims=True)
    return idx.astype(np.uint8), ww.astype(np.float32)


def hair_mesh(h=.006):
    lo = np.array([-.13, 1.58, -.15]); hi = np.array([.13, 1.84, .14])
    xs = np.arange(lo[0], hi[0] + h, h); ys = np.arange(lo[1], hi[1] + h, h); zs = np.arange(lo[2], hi[2] + h, h)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij'); P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    def f(P):
        # the hair follows the actual skull: a 1.6 cm thick layer on top of the head surface,
        # so no part of the scalp can poke through it
        hd = S.head(P)
        shell = hd - .016
        inner = -(hd + .002)
        # hairline: high at the forehead, down to the nape at the back, lifted clear of the ears at the sides
        fz = np.clip((P[:, 2] - -.06) / .14, 0, 1)
        hl = 1.635 + fz * .1 + np.clip(np.abs(P[:, 0]) - .05, 0, None) * 1.2
        cut = hl - P[:, 1]
        return np.maximum(np.maximum(shell, cut), inner)
    vol = f(P).reshape(X.shape)
    v, fc, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h)); v += lo; fc = fc[:, ::-1].copy()
    e = h * .5
    n = np.stack([f(v + [e, 0, 0]) - f(v - [e, 0, 0]), f(v + [0, e, 0]) - f(v - [0, e, 0]), f(v + [0, 0, e]) - f(v - [0, 0, e])], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return v.astype(np.float32), fc.astype(np.int32), n.astype(np.float32)


def uv_and_groups(v, f, part):
    maj = part[f]  # (F,3)
    def vote(k): return (maj == k).sum(1) >= 2
    group = np.ones(len(f), np.int8)
    group[vote(SHIRT)] = 0
    group[vote(FACE)] = 2
    uv = np.zeros((len(v), 2), np.float32)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # shirt: cylinder around the torso, front = u 0, back = u 0.5 (same layout as the game's kit textures)
    us = (np.arctan2(x, z) / (2 * np.pi)) % 1.0
    vs = np.clip((y - .99) / .54, 0, 1)
    # face: sphere around the head, front = u 0.25 (three.js sphere layout)
    q = (v - [0, 1.69, 0]) / [.1, .123, .112]
    th = np.arccos(np.clip(q[:, 1] / (np.linalg.norm(q, axis=1) + 1e-9), -1, 1))
    uf = (np.arctan2(q[:, 2], -q[:, 0]) / (2 * np.pi)) % 1.0
    vf = 1 - th / np.pi
    # split vertices along the texture seams so no triangle wraps across the whole texture
    v = list(v); extra = {}
    newf = f.copy()
    vert_uv = {}
    out_uv = []
    out_src = []
    def get(vi, uvv, g):
        key = (vi, int(g), round(float(uvv[0]), 5) >= 1.0)
        if key not in vert_uv:
            vert_uv[key] = len(out_src); out_src.append(vi); out_uv.append(uvv)
        return vert_uv[key]
    for fi in range(len(f)):
        g = group[fi]; tri = f[fi]
        if g == 0: uu = us[tri].copy(); vv = vs[tri]
        elif g == 2: uu = uf[tri].copy(); vv = vf[tri]
        else: uu = np.zeros(3); vv = np.zeros(3)
        if g != 1 and uu.max() - uu.min() > .5: uu[uu < .5] += 1.0
        for k in range(3): newf[fi, k] = get(tri[k], (uu[k], vv[k]), g)
    src = np.array(out_src); uvs = np.array(out_uv, np.float32)
    # vertices used by both a textured and an untextured face keep their first uv; untextured faces ignore uv
    return src, uvs, newf, group


def build(lod_npz, with_hair=True):
    d = np.load(lod_npz); v, f, n = d['v'], d['f'], d['n']
    part, isarm, side = classify(v)
    if with_hair:
        hv, hf, hn = hair_mesh(.0045 if 'lod0' in lod_npz else .007)
        hp = np.full(len(hv), HAIR)
        f = np.vstack([f, hf + len(v)]); v = np.vstack([v, hv]); n = np.vstack([n, hn])
        part = np.concatenate([part, hp]); isarm = np.concatenate([isarm, np.zeros(len(hv), bool)])
    src, uvs, f2, group = uv_and_groups(v, f, part)
    V = v[src]; Nn = n[src]; P = part[src]; A = isarm[src]
    J, W = weights(V, P, A, None)
    order = np.argsort(group, kind='stable')
    f2 = f2[order]; group = group[order]
    counts = [int((group == g).sum()) for g in range(3)]
    return dict(v=V.astype(np.float32), n=Nn.astype(np.float32), uv=uvs, part=P.astype(np.uint8), j=J, w=W, f=f2.astype(np.uint32), counts=counts)


# ---------------- textures for the standalone GLB ----------------
def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


def shirt_png(base=(200, 38, 42), trim=(255, 255, 255), ink=(255, 255, 255), num='9', name='MATCHDAY'):
    W = 512; im = Image.new('RGB', (W, W), base); g = ImageDraw.Draw(im)
    g.rectangle([116, 0, 140, W], fill=trim); g.rectangle([372, 0, 396, W], fill=trim)
    f1 = font(150); f2 = font(36); f3 = font(30)
    g.text((256, 244), num, font=f1, fill=ink, anchor='mm'); g.text((256, 124), name, font=f2, fill=ink, anchor='mm')
    for x in (0, W): g.text((x, 236), 'RIVERSIDE', font=f3, fill=ink, anchor='mm')
    g.text((64, 184), num, font=f3, fill=ink, anchor='mm')
    return im


def face_png(skin=(214, 168, 130), hair=(40, 30, 22), eye=(74, 46, 28)):
    W, H = 512, 256; im = Image.new('RGB', (W, H), tuple(int(c * .95) for c in skin)); g = ImageDraw.Draw(im)
    s = 2
    for sx in (-1, 1):
        ex, ey = 128 + sx * 27, 118
        g.ellipse([ex - 11, ey - 5, ex + 11, ey + 5], fill=(238, 232, 224))
        g.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=eye); g.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=(12, 8, 6))
        g.arc([ex - 12, ey - 7, ex + 12, ey + 7], 190, 350, fill=(25, 14, 8), width=2)
        g.line([(ex - sx * 14, 104), (ex + sx * 2, 98), (ex + sx * 14, 103)], fill=hair, width=5, joint='curve')
    lip = tuple(int(c * .78) for c in skin)
    g.ellipse([114, 166, 142, 176], fill=lip)
    g.rectangle([0, 0, W, 60], fill=hair); g.rectangle([236, 0, W, 110], fill=hair)
    return im


def write_glb(path, m, shirt_img, face_img):
    import io
    palette = {SKIN: (214, 168, 130), SHIRT: (200, 38, 42), SLEEVE: (200, 38, 42), SHORTS: (243, 243, 243), SOCKS: (200, 38, 42), BOOT: (20, 20, 20),
               TRIM: (255, 255, 255), SOLE: (235, 235, 235), HAND: (214, 168, 130), FOREARM: (214, 168, 130), FACE: (214, 168, 130), HAIR: (40, 30, 22)}
    col = np.array([palette[int(k)] for k in m['part']], np.float32) / 255.0
    col = col ** 2.2  # glTF vertex colours are linear
    bin_ = bytearray(); views = []; accs = []
    def add(data, target=None):
        while len(bin_) % 4: bin_.append(0)
        off = len(bin_); b = data.tobytes(); bin_.extend(b)
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(b)}
        if target: v['target'] = target
        views.append(v); return len(views) - 1
    def acc(data, ctype, typ, target=None, minmax=False, normalized=False):
        bv = add(data, target)
        a = {'bufferView': bv, 'componentType': ctype, 'count': int(data.shape[0]), 'type': typ}
        if normalized: a['normalized'] = True
        if minmax: a['min'] = data.min(0).tolist(); a['max'] = data.max(0).tolist()
        accs.append(a); return len(accs) - 1
    aP = acc(m['v'], 5126, 'VEC3', 34962, True); aN = acc(m['n'], 5126, 'VEC3', 34962)
    aT = acc(m['uv'], 5126, 'VEC2', 34962); aC = acc(col, 5126, 'VEC3', 34962)
    aJ = acc(m['j'].astype(np.uint8), 5121, 'VEC4', 34962); aW = acc(m['w'], 5126, 'VEC4', 34962)
    c0, c1, c2 = m['counts']; F = m['f']
    prims = []
    for gi, (s, e) in enumerate(((0, c0), (c0, c0 + c1), (c0 + c1, c0 + c1 + c2))):
        if e > s:
            ai = acc(F[s:e].reshape(-1).astype(np.uint32), 5125, 'SCALAR', 34963)
            prims.append({'attributes': {'POSITION': aP, 'NORMAL': aN, 'TEXCOORD_0': aT, 'COLOR_0': aC, 'JOINTS_0': aJ, 'WEIGHTS_0': aW}, 'indices': ai, 'material': gi})
    imgs = []
    for im in (shirt_img, face_img):
        bio = io.BytesIO(); im.save(bio, 'PNG')
        bv = add(np.frombuffer(bio.getvalue(), np.uint8)); imgs.append({'bufferView': bv, 'mimeType': 'image/png'})
    ibm = []
    for name, par, wp in BONES:
        mtx = np.eye(4, dtype=np.float32); mtx[:3, 3] = -np.array(wp, np.float32)
        ibm.append(mtx.T.reshape(-1))  # column-major
    aI = acc(np.array(ibm, np.float32), 5126, 'MAT4')
    nodes = []
    for i, (name, par, wp) in enumerate(BONES):
        pp = np.array(BONES[par][2]) if par >= 0 else np.zeros(3)
        nodes.append({'name': name, 'translation': (np.array(wp) - pp).tolist()})
    for i, (name, par, wp) in enumerate(BONES):
        kids = [k for k, b in enumerate(BONES) if b[1] == i]
        if kids: nodes[i]['children'] = kids
    nodes.append({'name': 'Footballer', 'mesh': 0, 'skin': 0})
    gltf = {
        'asset': {'version': '2.0', 'generator': 'Matchday sculpt pipeline'},
        'scene': 0, 'scenes': [{'nodes': [0, len(nodes) - 1]}], 'nodes': nodes,
        'meshes': [{'name': 'Footballer', 'primitives': prims}],
        'skins': [{'joints': list(range(len(BONES))), 'inverseBindMatrices': aI, 'skeleton': 0}],
        'materials': [
            {'name': 'Kit shirt', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': .75}},
            {'name': 'Body', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': .6}},
            {'name': 'Face', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 1}, 'metallicFactor': 0, 'roughnessFactor': .55}}],
        'textures': [{'source': 0, 'sampler': 0}, {'source': 1, 'sampler': 0}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 33071}],
        'images': imgs, 'accessors': accs, 'bufferViews': views, 'buffers': [{'byteLength': len(bin_)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4: js += b' '
    while len(bin_) % 4: bin_.append(0)
    out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(bin_), 0x004E4942) + bytes(bin_)
    open(path, 'wb').write(out)
    return col


def pack_game(m):
    """quantised arrays for the browser: int16 positions (0.1 mm), int8 normals, uint16 uv, uint8 joints/weights/part"""
    q = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    pos = np.round(m['v'] * 10000).astype(np.int16)
    nor = np.round(np.clip(m['n'], -1, 1) * 127).astype(np.int8)
    uv = np.round(np.clip(m['uv'], 0, 2) / 2 * 65535).astype(np.uint16)
    w = np.round(m['w'] * 255).astype(np.int32)
    w[:, 0] += 255 - w.sum(1)  # keep the four weights summing to exactly 255
    idx = m['f'].reshape(-1)
    itype = 'u16' if len(m['v']) < 65536 else 'u32'
    idx = idx.astype(np.uint16 if itype == 'u16' else np.uint32)
    return {'n': int(len(m['v'])), 'pos': q(pos), 'nor': q(nor), 'uv': q(uv), 'j': q(m['j']), 'w': q(w.astype(np.uint8)), 'part': q(m['part']), 'idx': q(idx), 'itype': itype, 'counts': m['counts']}


if __name__ == '__main__':
    import sys
    m0 = build('./lod0.npz'); m1 = build('./lod1.npz')
    for nm, m in (('lod0', m0), ('lod1', m1)):
        print(nm, len(m['v']), 'verts', len(m['f']), 'tris', 'groups', m['counts'])
        np.savez(f'./{nm}_rig.npz', **m)
    col = write_glb('./matchday_player.glb', m0, shirt_png(), face_png())
    data = {'lod0': pack_game(m0), 'lod1': pack_game(m1)}
    open('./player_data.js', 'w').write('const PLAYER_MODEL = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    import os
    print('glb', os.path.getsize('./matchday_player.glb') // 1024, 'KB', 'js', os.path.getsize('./player_data.js') // 1024, 'KB')
