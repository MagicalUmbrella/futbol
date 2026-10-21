"""Small z-buffered software renderer for checking the model (no GPU needed)."""
import sys
import numpy as np
from PIL import Image


def render(v, f, n, col, yaw_deg, size=(420, 820), center=(0, .9), span=1.95, out='x.png', pitch_deg=0):
    W, H = size
    a = np.radians(yaw_deg); b = np.radians(pitch_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R = Rx @ R
    vr = (v - [center[0], center[1], 0]) @ R.T
    nr = n @ R.T
    s = H / span
    px = vr[:, 0] * s + W / 2; py = -vr[:, 1] * s + H / 2; pz = vr[:, 2]
    L = np.array([-.4, .6, .7]); L /= np.linalg.norm(L)
    shade = np.clip(nr @ L, 0, 1) * .75 + .25 + np.clip(nr[:, 2], 0, 1) * .1
    vc = np.clip(col * shade[:, None], 0, 1)
    img = np.zeros((H, W, 3)) + [.13, .16, .2]
    zb = np.full((H, W), -1e9)
    tri = f
    x0, y0, z0 = px[tri[:, 0]], py[tri[:, 0]], pz[tri[:, 0]]
    x1, y1, z1 = px[tri[:, 1]], py[tri[:, 1]], pz[tri[:, 1]]
    x2, y2, z2 = px[tri[:, 2]], py[tri[:, 2]], pz[tri[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    keep = np.abs(area) > 1e-9
    for t in np.nonzero(keep)[0]:
        xs = (x0[t], x1[t], x2[t]); ys = (y0[t], y1[t], y2[t])
        xmin = max(int(min(xs)), 0); xmax = min(int(max(xs)) + 1, W - 1)
        ymin = max(int(min(ys)), 0); ymax = min(int(max(ys)) + 1, H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + .5, np.arange(ymin, ymax + 1) + .5)
        ar = area[t]
        w0 = ((x1[t] - gx) * (y2[t] - gy) - (x2[t] - gx) * (y1[t] - gy)) / ar
        w1 = ((x2[t] - gx) * (y0[t] - gy) - (x0[t] - gx) * (y2[t] - gy)) / ar
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        z = w0 * z0[t] + w1 * z1[t] + w2 * z2[t]
        sub = zb[ymin:ymax + 1, xmin:xmax + 1]
        m &= z > sub
        if not m.any():
            continue
        c = (w0[..., None] * vc[tri[t, 0]] + w1[..., None] * vc[tri[t, 1]] + w2[..., None] * vc[tri[t, 2]])
        sub[m] = z[m]
        img[ymin:ymax + 1, xmin:xmax + 1][m] = c[m]
    Image.fromarray((img * 255).astype(np.uint8)).save(out)


if __name__ == '__main__':
    d = np.load(sys.argv[1])
    v, f, n = d['v'], d['f'], d['n']
    col = d['c'] if 'c' in d.files else np.tile([.8, .72, .65], (len(v), 1))
    tag = sys.argv[2]
    views = [(0, 'front'), (90, 'side'), (35, 'three')]
    ims = []
    for yaw, nm in views:
        render(v, f, n, col, yaw, out=f'/tmp/claude-0/{tag}_{nm}.png'); ims.append(Image.open(f'/tmp/claude-0/{tag}_{nm}.png'))
    render(v, f, n, col, 20, size=(420, 420), center=(0, 1.67), span=.36, out=f'/tmp/claude-0/{tag}_face.png'); ims.append(Image.open(f'/tmp/claude-0/{tag}_face.png').resize((420, 420)))
    sheet = Image.new('RGB', (420 * 4, 820), (20, 24, 32))
    for i, im in enumerate(ims):
        sheet.paste(im, (420 * i, 0))
    sheet.save(f'/home/claude/model/{tag}_sheet.png')
    print('ok')
