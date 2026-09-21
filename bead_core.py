# -*- coding: utf-8 -*-
"""CeraBeads · 拼豆图纸生成器 —— 核心引擎

只依赖 numpy + Pillow（不需要 OpenCV / onnxruntime / 联网）。
- 色卡：config\\mard-221.json / config\\mard-291.json（内置，离线可用）
- 色彩匹配：CIEDE2000（自实现，用 Sharma 标准测试数据校验，误差 < 1e-4）
- 取色：按格取"主导色"（4bit/通道分箱 + 组内取均值，抗锯齿像素自动淡化）
- 背景移除：与边框颜色相似的连通区域洪水填充（自己实现，无外部依赖）
- 渲染：网格 + 色号 + 每 10 格红线 + 坐标刻度 + 底部图例/用量
"""
import csv
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 手机常见格式 HEIC/HEIF：装了 pillow-heif 就注册，之后 PIL 直接能读（没装也不影响别的格式）
try:
    from pillow_heif import register_heif_opener as _register_heif

    _register_heif()
    HEIF_OK = True
except Exception:                      # 没装 pillow-heif：只有这一个格式读不了
    HEIF_OK = False

# 能读的图片格式（都靠 PIL；HEIC/HEIF 需 pillow-heif，AVIF 由 Pillow 自带支持）
IMG_EXT = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.bmp', '.gif', '.webp', '.tif', '.tiff',
           '.ppm', '.pgm', '.pbm', '.avif', '.avifs', '.heic', '.heif', '.hif', '.ico', '.pcx',
           '.tga', '.dds', '.jp2', '.j2k', '.eps', '.im', '.sgi', '.xbm', '.ras')

APP_TITLE = 'CeraBeads · 拼豆图纸生成器'
APP_VERSION = 'v1.6.2'
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, 'config')
# 色卡清单（第一个是默认值，保持不变：MARD 221·MARD家）
PALETTE_FILES = [('MARD 221（MARD家·默认）', 'mard-221-alfonse.json'),
                 ('MARD 221（源码版·交叉校验）', 'mard-221-github.json'),
                 ('MARD 291', 'mard-291.json'),
                 ('COCO 291', 'coco-291.json'),
                 ('漫漫 278', 'manman-278.json'),
                 ('咪小窝 290', 'mixiaowo-290.json'),
                 ('盼盼 289', 'panpan-289.json'),
                 ('优肯 174（公开整理）', 'youken-174.json'),
                 ('优肯 C197（官方）', 'artkal-c197.json'),
                 ('优肯 M221（官方）', 'artkal-m221.json'),
                 ('优肯 418（官方合并）', 'artkal-418.json')]
PALETTE_DEFAULT = PALETTE_FILES[0][0]


def palette_labels():
    return [label for label, _ in PALETTE_FILES]
RED_EVERY = 10            # 每 10 格画红线
BG_TOL = 18               # 背景容差（0-255 最大通道差）
BG_CELL = 0.55            # 一格内背景像素占比 ≥ 此值 → 空格


def app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return HERE


def res_dir():
    return getattr(sys, '_MEIPASS', None) or app_dir()


# ---------------------------------------------------------------- 颜色
def hex2rgb(h):
    h = h.strip().lstrip('#')
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float64)


def srgb2lab(rgb):
    """单色 sRGB(0-255) → CIELAB(D65)"""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def srgb2lab_arr(arr):
    """(...,3) sRGB(0-255 float/uint8) → (...,3) LAB，向量化"""
    c = np.asarray(arr, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def de00_arr(l1, a1, b1, l2, a2, b2):
    """CIEDE2000，广播式向量化（kL=kC=kH=1）"""
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2.0
    Cb7 = Cb ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = l2 - l1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (l1 + l2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hs = h1p + h2p
    hbp = np.where((C1p * C2p) == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2.0,
                            np.where(hs < 360, (hs + 360) / 2.0, (hs - 360) / 2.0)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dTh = 30 * np.exp(-(((hbp - 275) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    Rc = 2 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dTh)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def de00(lab1, lab2):
    return float(de00_arr(np.float64(lab1[0]), np.float64(lab1[1]), np.float64(lab1[2]),
                          np.float64(lab2[0]), np.float64(lab2[1]), np.float64(lab2[2])))


_DE00_TEST = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
              ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
              ((50, 2.8361, -74.0200), (50, 0, -82.7485), 3.4412),
              ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
              ((50, 0, 0), (50, -1, 2), 2.3669),
              ((50, 2.49, -0.001), (50, -2.49, 0.0011), 7.2195),
              ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
              ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
              ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082)]


def de00_validate():
    return max(abs(de00(a, b) - e) for a, b, e in _DE00_TEST)


# ---------------------------------------------------------------- 色卡
class PaletteError(RuntimeError):
    """色卡文件缺失或损坏（提示用户把 exe 旁边的 config 文件夹放回来）"""


def _palette_dirs():
    """按优先级：程序旁边的 config（用户可改）→ 程序目录/config → 打包内 config → 源码目录/config（同一路径只留一个）"""
    ds = [os.path.join(app_dir(), 'config'), CFG,
          os.path.join(res_dir(), 'config'), os.path.join(HERE, 'config')]
    seen, out = set(), []
    for d in ds:
        k = os.path.normcase(os.path.abspath(d))
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def load_palette(name=None):
    """name 可为色卡标签（见 palette_labels）/ 'MARD 221'（=默认）/ json 路径；空 = 默认。
    某个文件找不到/坏了就自动换下一个候选位置；全都没有时给中文提示。"""
    if not name or name == 'MARD 221':
        name = PALETTE_DEFAULT
    p = name
    for label, fn in PALETTE_FILES:
        if name == label:
            p = fn
    cands = [p] if os.path.isabs(p) else [os.path.join(d, p) for d in _palette_dirs()]
    tried = []
    for c in cands:
        if not os.path.exists(c):
            tried.append(c + '（不存在）')
            continue
        try:
            with open(c, encoding='utf-8') as f:
                d = json.load(f)
            cols = d['colors']
            if not cols:
                raise ValueError('里面没有颜色')
            return {'name': d.get('name') or os.path.basename(c), 'count': len(cols),
                    'colors': cols, 'path': c}
        except Exception as e:
            tried.append('%s（打不开：%s）' % (c, e))
    raise PaletteError('找不到可用的色卡文件「%s」。\n\n找过这些地方：\n%s\n\n'
                       '解决：把程序旁边的 config 文件夹放回来（或重新解压一份）。'
                       % (p, '\n'.join(tried)))


class Matcher:
    """CIEDE2000 最近色匹配（分块计算，色卡可限定子集）"""

    def __init__(self, palette, allowed_codes=None):
        cols = palette['colors']
        if allowed_codes:
            keep = [c for c in cols if c['code'] in set(allowed_codes)]
            cols = keep or cols
        self.colors = cols
        self.codes = [c['code'] for c in cols]
        self.rgb = np.array([[int(c['rgb'][0]), int(c['rgb'][1]), int(c['rgb'][2])]
                             for c in cols], dtype=np.uint8)
        self.lab = np.array([srgb2lab(c['rgb']) for c in cols], dtype=np.float64)
        self.hexes = [c['hex'] for c in cols]
        self.groups = [c.get('group', '') for c in cols]

    def match(self, lab_q, chunk=2500):
        q = np.asarray(lab_q, dtype=np.float64).reshape(-1, 3)
        n = len(q)
        idx = np.empty(n, dtype=np.int32)
        dist = np.empty(n, dtype=np.float64)
        pl, pa, pb = self.lab[:, 0], self.lab[:, 1], self.lab[:, 2]
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d = de00_arr(q[s:e, 0][:, None], q[s:e, 1][:, None], q[s:e, 2][:, None],
                         pl[None, :], pa[None, :], pb[None, :])
            idx[s:e] = d.argmin(axis=1)
            dist[s:e] = d[np.arange(e - s), idx[s:e]]
        return idx, dist


# ---------------------------------------------------------------- 取色 / 抠背景
def srgb_to_linear(a):
    """sRGB(0..1) → 线性光"""
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(a):
    """线性光 → sRGB(0..1)"""
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / 2.4) - 0.055)


def pixelate_mean(arr, gw, gh):
    """按格取平均色：格内像素先在**线性光**空间平均（gamma 正确，不会偏暗），
    再转回 sRGB。照片、渐变、细节比「主导色」更接近原图。"""
    px = arr[:, :, :3].astype(np.float64) / 255.0
    lin = srgb_to_linear(px).reshape(-1, 3)
    h, w = px.shape[0], px.shape[1]
    ys = (np.arange(h, dtype=np.int64) * gh) // h
    xs = (np.arange(w, dtype=np.int64) * gw) // w
    cell = (ys[:, None] * gw + xs[None, :]).reshape(-1)
    n = gw * gh
    sums = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        sums[:, c] = np.bincount(cell, weights=lin[:, c], minlength=n)
    cnt = np.bincount(cell, minlength=n).astype(np.int64)
    mean_lin = sums / np.maximum(cnt, 1)[:, None]
    reps = np.clip(linear_to_srgb(mean_lin), 0.0, 1.0) * 255.0
    return reps.reshape(gh, gw, 3), cnt.reshape(gh, gw)


# 自动取色参数：图像整体"色块分离度"中位数 ≥ AUTO_GLOBAL_SEP → 整张图按色块/线稿处理（全用主导色）；
# 否则按照片处理（用平均色），但单格分离度 > AUTO_CELL_SEP 的强边缘格改用主导色（防脏边）。
MIN_PX_PER_CELL = 12.0      # 每格原图像素低于这个数就先放大（实测能显著提升还原度）
UPSCALE_MAX = 4             # 最多放大 4 倍
AUTO_GLOBAL_SEP = 20.0
AUTO_CELL_SEP = 40.0
AUTO_MIN_PX = 16          # 每格像素少于这个数 → 强制用平均色（主导色统计不可靠）
AUTO_DOM_ALL = 0.5        # 强边界格占比 ≥ 此值 → 整图按色块/线稿处理（全用主导色）
AUTO_DOM_NONE = 0.05      # 强边界格占比 < 此值 → 整图统一用平均色


def _kmeans2_sep(arr, gw, gh, chunk_px=1_500_000, iters=4):
    """每格 K-means(K=2) 两簇中心的 CIEDE2000 分离度（大＝这格跨在边界上）。
    整幅原图参与、不降采样；按块累加，峰值内存只与一块有关。"""
    h, w = arr.shape[0], arr.shape[1]
    n = gw * gh
    ys = (np.arange(h, dtype=np.int64) * gh) // h
    xs = (np.arange(w, dtype=np.int64) * gw) // w
    cell_all = (ys[:, None] * gw + xs[None, :]).reshape(-1)
    lin = srgb_to_linear(arr[:, :, :3].reshape(-1, 3).astype(np.float32)
                         / np.float32(255.0)).astype(np.float32)
    W3 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    step = max(1, int(chunk_px))

    lsum = np.zeros(n, dtype=np.float64)
    lcnt = np.zeros(n, dtype=np.float64)
    for s0 in range(0, lin.shape[0], step):
        e = min(lin.shape[0], s0 + step)
        c = cell_all[s0:e]
        lm = lin[s0:e] @ W3
        lsum += np.bincount(c, weights=lm.astype(np.float64), minlength=n)
        lcnt += np.bincount(c, minlength=n)
    lmean = (lsum / np.maximum(lcnt, 1)).astype(np.float32)

    s1 = np.zeros((n, 3))
    s2 = np.zeros((n, 3))
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    for s0 in range(0, lin.shape[0], step):
        e = min(lin.shape[0], s0 + step)
        c = cell_all[s0:e]
        blk = lin[s0:e]
        lm = blk @ W3
        low = (lm < lmean[c]).astype(np.float32)
        up = 1.0 - low
        for ch in range(3):
            s1[:, ch] += np.bincount(c, weights=blk[:, ch] * low, minlength=n)
            s2[:, ch] += np.bincount(c, weights=blk[:, ch] * up, minlength=n)
        k1 += np.bincount(c, weights=low, minlength=n)
        k2 += np.bincount(c, weights=up, minlength=n)
    c1 = s1 / np.maximum(k1, 1)[:, None]
    c2 = s2 / np.maximum(k2, 1)[:, None]
    same = (k1 == 0) | (k2 == 0)
    c1[same] = lmean[same, None]
    c2[same] = lmean[same, None]

    for _ in range(int(iters)):
        t1 = np.zeros((n, 3))
        t2 = np.zeros((n, 3))
        a1 = np.zeros(n)
        a2 = np.zeros(n)
        for s0 in range(0, lin.shape[0], step):
            e = min(lin.shape[0], s0 + step)
            c = cell_all[s0:e]
            blk = lin[s0:e]
            d1 = ((blk - c1[c]) ** 2).sum(1)
            d2 = ((blk - c2[c]) ** 2).sum(1)
            m1 = (d1 <= d2).astype(np.float32)
            m2 = 1.0 - m1
            for ch in range(3):
                t1[:, ch] += np.bincount(c, weights=blk[:, ch] * m1, minlength=n)
                t2[:, ch] += np.bincount(c, weights=blk[:, ch] * m2, minlength=n)
            a1 += np.bincount(c, weights=m1, minlength=n)
            a2 += np.bincount(c, weights=m2, minlength=n)
        c1 = t1 / np.maximum(a1, 1)[:, None]
        c2 = t2 / np.maximum(a2, 1)[:, None]

    l1, a1_, b1 = srgb2lab_arr(np.clip(linear_to_srgb(c1.astype(np.float64)), 0, 1) * 255).T
    l2, a2_, b2 = srgb2lab_arr(np.clip(linear_to_srgb(c2.astype(np.float64)), 0, 1) * 255).T
    return de00_arr(l1, a1_, b1, l2, a2_, b2).reshape(gh, gw)


def pixelate_photo(arr, gw, gh, tau=12.0):
    """照片专项优化：以「该格主导色」为基准做加权平均——
    跟主导色差别大的像素（边界另一侧、JPEG 噪点、缩放毛刺）权重自动降下来，
    所以既保留了平均色的层次过渡，又不会把边缘两侧混成一条灰边。
    这也是默认「自动」在处理照片时用的取色方式。"""
    dom, cnt = _pixelate_dominant(arr, gw, gh)
    h, w = arr.shape[0], arr.shape[1]
    ys = (np.arange(h, dtype=np.int64) * gh) // h
    xs = (np.arange(w, dtype=np.int64) * gw) // w
    cell = (ys[:, None] * gw + xs[None, :]).reshape(-1)
    idx = arr.reshape(-1, arr.shape[2])[:, :3].astype(np.float64)
    lin = srgb_to_linear(idx / 255.0)
    dlab = srgb2lab_arr(idx)                       # 每像素 Lab
    dcell = srgb2lab_arr(dom.reshape(-1, 3))       # 每格主导色 Lab
    dd = dlab - dcell[cell]
    dist = np.sqrt((dd * dd).sum(1))               # Lab 欧氏差（只要权重，够用且快）
    wgt = 1.0 / (1.0 + (dist / tau) ** 2)
    n = gw * gh
    sw = np.bincount(cell, weights=wgt, minlength=n)
    sums = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        sums[:, c] = np.bincount(cell, weights=lin[:, c] * wgt, minlength=n)
    m = sums / np.maximum(sw, 1)[:, None]
    reps = np.clip(linear_to_srgb(m), 0.0, 1.0) * 255.0
    return reps.reshape(gh, gw, 3), cnt


def pixelate_auto(arr, gw, gh, global_sep=AUTO_GLOBAL_SEP, cell_sep=AUTO_CELL_SEP,
                  min_px=AUTO_MIN_PX, dom_all=AUTO_DOM_ALL, dom_none=AUTO_DOM_NONE):
    """自动：三个维度一起决定，避免局部风格不一致——
      ① 每格像素数 < min_px → 整图用平均色（格子太小，主导色统计不可靠）
      ② 强边界格（sep > cell_sep）占比 ≥ dom_all → 整图用主导色（线稿/色块型）
      ③ 占比 < dom_none → 整图用平均色（照片型，避免零星“硬化”点）
      ④ 介于两者之间 → 逐格混合（仅强边界格用主导色）"""
    reps_mean, cnt = pixelate_mean(arr, gw, gh)
    reps_photo, _cntp = pixelate_photo(arr, gw, gh)     # 照片分支用「主导色引导的加权平均」
    px_per_cell = float(arr.shape[0]) * float(arr.shape[1]) / float(gw * gh)
    if px_per_cell < min_px:
        return reps_photo if px_per_cell >= 4 else reps_mean, cnt
    sep = _kmeans2_sep(arr, gw, gh)                  # 整幅原图，不缩图
    reps_dom, _ = _pixelate_dominant(arr, gw, gh)
    frac = float((sep > cell_sep).mean())
    if frac >= dom_all:
        return reps_dom, cnt
    if frac < dom_none or float(np.median(sep)) >= global_sep:
        return reps_photo, cnt
    m = (sep > cell_sep)[:, :, None]
    return np.where(m, reps_dom, reps_photo), cnt


def pixelate(arr, gw, gh, sample='auto', mask=None):
    """按格取色。sample='auto'（默认）＝自动判断（照片走照片专项优化）；
    'photo'＝照片专项优化（主导色引导的加权平均，保层次又不出灰边）；
    'mean'＝格内线性光平均色；'dominant'＝4bit 分箱主导色（色块/线稿最干净）。
    返回 (reps(gh,gw,3) float 0..255, cover(gh,gw) 有效像素数)"""
    if sample == 'dominant':
        return _pixelate_dominant(arr, gw, gh, mask)
    if sample == 'mean':
        return pixelate_mean(arr, gw, gh)
    if sample == 'photo':
        return pixelate_photo(arr, gw, gh)
    return pixelate_auto(arr, gw, gh)



# ============ 优化模式（通用 / 图片优化 / 动漫优化）============

PRESETS = [('general', '通用'), ('photo', '图片优化'), ('anime', '动漫优化')]
EDGE_THRESH = 55.0          # Sobel 梯度幅值阈值：超过就算“边缘像素”
PHOTO_BILATERAL_RADIUS = 2
PHOTO_BILATERAL_SIGMA_R = 26.0


def preset_labels():
    return ['%s%s' % (lab, '（默认）' if k == 'general' else '') for k, lab in PRESETS]


def bilateral_lite(arr, radius=PHOTO_BILATERAL_RADIUS, sigma_s=2.0,
                   sigma_r=PHOTO_BILATERAL_SIGMA_R, band_px=2_000_000):
    """小半径双边滤波：磨掉传感器噪点/JPEG 块效应，但保住边缘（图片优化用）。
    整幅原图参与、不缩图；按行分块累加，峰值内存只与一块有关。返回 uint8。"""
    a = np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)
    h, w = a.shape[:2]
    pad = int(radius)
    p = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    sp = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma_s * sigma_s)).astype(np.float32)
    rho = np.float32(1.0 / (2.0 * sigma_r * sigma_r))
    out = np.empty_like(a)
    band = max(1, int(band_px) // max(1, w))
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        cur = a[y0:y1]
        num = np.zeros_like(cur)
        den = np.zeros((y1 - y0, w, 1), dtype=np.float32)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                sh = p[y0 + dy + pad:y1 + dy + pad, dx + pad:dx + pad + w]
                d2 = ((sh - cur) ** 2).sum(2, keepdims=True)
                wgt = sp[dy + pad, dx + pad] * np.exp(-d2 * rho)
                num += sh * wgt
                den += wgt
        out[y0:y1] = num / np.maximum(den, np.float32(1e-9))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _conv3(m, k):
    p = np.pad(m, 1, mode='edge')
    out = np.zeros_like(m)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i:i + m.shape[0], j:j + m.shape[1]]
    return out


def sobel_edge_mask(arr, thresh=EDGE_THRESH):
    """梯度幅值 > 阈值的像素 = 边缘像素（抗锯齿过渡带）。
    动漫/线稿优化用它把这些过渡像素从“主导色投票”里剔掉，轮廓就不会被拉向中间调。"""
    g = arr[:, :, :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    kx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    mag = np.sqrt(_conv3(g, kx) ** 2 + _conv3(g, kx.T) ** 2)
    return mag > thresh


def prepare_by_preset(arr, preset):
    """按优化模式做预处理。返回 (处理后的图, 边缘掩膜或 None)。"""
    mask = None
    if preset == 'photo':
        arr = bilateral_lite(arr)                # 整幅原图参与，绝不缩图
    elif preset == 'anime':
        mask = sobel_edge_mask(arr)
    return arr, mask


PRESET_SAMPLE = {'general': 'mean', 'photo': 'photo', 'anime': 'dominant'}


def resolve_sample(preset, sample):
    """sample 为 None/''/auto 时跟随优化模式；显式指定则用指定的。"""
    if not sample or sample == 'auto':
        return PRESET_SAMPLE.get(preset, 'mean')
    return sample


def _pixelate_dominant(arr, gw, gh, mask=None):
    """按格取主导色：4bit/通道分箱，取格内占比最大的箱子，色值用该箱的均值。"""
    h, w = arr.shape[0], arr.shape[1]
    ys = (np.arange(h, dtype=np.int64) * gh) // h
    xs = (np.arange(w, dtype=np.int64) * gw) // w
    cell = (ys[:, None] * gw + xs[None, :]).reshape(-1)
    px = arr.reshape(-1, 3).astype(np.int32)
    if px.shape[1] == 4:
        px = px[:, :3]
    if mask is not None:
        # 只用“非边缘像素”投票；整格非边缘像素不足 25% 的格子回退成整格像素
        mk = np.asarray(mask).reshape(-1).astype(bool)
        if mk.shape[0] != cell.shape[0]:
            mk = np.ones(cell.shape[0], dtype=bool)
        keep = ~mk
        area = np.bincount(cell, minlength=gw * gh)
        vcnt = np.bincount(cell[keep], minlength=gw * gh)
        weak = np.where(vcnt < 0.25 * np.maximum(area, 1))[0]
        if weak.size:
            add = np.isin(cell, weak)
            cell = np.concatenate([cell[keep], cell[add]])
            px = np.concatenate([px[keep], px[add]])
        else:
            cell = cell[keep]
            px = px[keep]
    q = px >> 4
    key = cell * 4096 + ((q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2])
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uk), 3), dtype=np.float64)
    np.add.at(sums, inv, px)
    means = sums / cnt[:, None]
    ucell = (uk // 4096).astype(np.int64)
    order = np.lexsort((-cnt, ucell))
    uc_sorted = ucell[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = uc_sorted[1:] != uc_sorted[:-1]
    sel = order[first]
    reps = np.zeros((gw * gh, 3), dtype=np.float64)
    cover = np.zeros(gw * gh, dtype=np.int64)
    reps[ucell[sel]] = means[sel]
    cover[ucell[sel]] = cnt[sel]
    return reps.reshape(gh, gw, 3), cover.reshape(gh, gw)


def bg_remove(arr, tol=BG_TOL, bg_color=None, max_iter=3000):
    """与边框颜色接近、且与边框连通的区域 → 背景。
    返回 (mask(h,w) True=背景, bg_color)。自己实现（形态学重建），不依赖 OpenCV。"""
    a = arr[..., :3].astype(np.int16)
    if bg_color is None:
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg_color = np.median(border, axis=0)
    bg = np.asarray(bg_color, dtype=np.int16)
    cand = np.abs(a - bg).max(axis=2) <= tol
    seed = np.zeros(cand.shape, dtype=bool)
    seed[0, :] = seed[-1, :] = True
    seed[:, 0] = seed[:, -1] = True
    cur = seed & cand
    for _ in range(max_iter):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt &= cand
        if nxt.sum() == cur.sum():
            break
        cur = nxt
    return cur, bg_color


def cells_empty(mask, gw, gh, thresh=BG_CELL):
    """每格里背景像素占比 ≥ thresh → 该格空"""
    if mask is None:
        return np.zeros((gh, gw), dtype=bool)
    h, w = mask.shape
    ys = (np.arange(h, dtype=np.int64) * gh) // h
    xs = (np.arange(w, dtype=np.int64) * gw) // w
    cell = (ys[:, None] * gw + xs[None, :]).reshape(-1)
    bgc = np.bincount(cell, weights=mask.reshape(-1).astype(np.float64),
                      minlength=gw * gh)
    tot = np.bincount(cell, minlength=gw * gh)
    frac = bgc / np.maximum(tot, 1)
    return (frac >= thresh).reshape(gh, gw)


# ---------------------------------------------------------------- 生成图案
def limit_colors(idx, dist, matcher, max_colors):
    """色号数超限时：保留用量最多的 N 个，其余重映射到最近的保留色。
    返回 (新 idx, 被合并掉的色号数)"""
    if not max_colors or max_colors <= 0:
        return idx, 0
    flat = idx.reshape(-1)
    used = flat[flat >= 0]
    if len(used) == 0:
        return idx, 0
    usage = np.bincount(used, minlength=len(matcher.codes))
    codes_used = np.where(usage > 0)[0]
    if len(codes_used) <= max_colors:
        return idx, 0
    order = codes_used[np.argsort(-usage[codes_used])]
    keep = order[:max_colors]
    drop = order[max_colors:]
    lab_keep = matcher.lab[keep]
    for c in drop:
        m = np.abs(matcher.lab[c][None, :] - lab_keep)  # 粗筛
        d = de00_arr(lab_keep[:, 0][None, :], lab_keep[:, 1][None, :], lab_keep[:, 2][None, :],
                     np.full((1, len(keep)), matcher.lab[c][0]),
                     np.full((1, len(keep)), matcher.lab[c][1]),
                     np.full((1, len(keep)), matcher.lab[c][2]))
        tgt = keep[int(d.argmin())]
        flat[flat == c] = tgt
    return flat.reshape(idx.shape), len(drop)


def despeckle(idx, min_cells=3, max_rounds=4, max_cells=400000):
    """清理碎点：把面积 < min_cells 的连通色块并到相邻最多的另一种颜色。
    idx 里 -1 表示空格（不参与）。返回 (新 idx, 被清理的色块数)。
    网格太大（>max_cells）时直接跳过，避免卡顿。"""
    idx = np.array(idx, dtype=np.int32, copy=True)
    gh, gw = idx.shape
    if min_cells <= 1 or gh * gw > max_cells:
        return idx, 0
    from collections import Counter
    cleaned = 0
    seen = np.zeros((gh, gw), dtype=bool)
    NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r0 in range(gh):
        for c0 in range(gw):
            v0 = idx[r0, c0]
            if v0 < 0 or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in NB:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gh and 0 <= nx < gw and not seen[ny, nx] and idx[ny, nx] == v0:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) >= min_cells:
                continue
            nb = Counter()
            for y, x in comp:
                for dy, dx in NB:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gh and 0 <= nx < gw:
                        v = idx[ny, nx]
                        if v >= 0 and v != v0:
                            nb[int(v)] += 1
            if nb:
                nv = nb.most_common(1)[0][0]
                for y, x in comp:
                    idx[y, x] = nv
                cleaned += 1
    return idx, cleaned


def _pair_de(lab_a, lab_b):
    """两组 Lab 之间的逐对 CIEDE2000 距离矩阵（n×m）。"""
    n, m = len(lab_a), len(lab_b)
    A = np.repeat(np.asarray(lab_a, dtype=np.float64), m, axis=0)
    Bx = np.tile(np.asarray(lab_b, dtype=np.float64), (n, 1))
    return de00_arr(A[:, 0], A[:, 1], A[:, 2], Bx[:, 0], Bx[:, 1], Bx[:, 2]).reshape(n, m)


def harmonize(idx, matcher, cell_lab, lam=0.35, tie=3.0, rounds=2, max_cells=120000):
    """防伪杂色（严格版）：只有当「邻居在用的某个色号」与本格当前色号几乎一样（ΔE <= tie）时，
    才可能把本格统一成邻居那个色号。所以只会合并近重复色号，绝不会引入新色号、不会抹掉细节。
    cell_lab：每格（非空格）的原图像素 Lab 颜色。返回 (新 idx, 改动格数)。"""
    idx = np.array(idx, dtype=np.int32, copy=True)
    gh, gw = idx.shape
    NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
    flat = idx.reshape(-1)
    ok = flat >= 0
    n = int(ok.sum())
    if n == 0 or n > max_cells or len(cell_lab) != n:
        return idx, 0
    lab = matcher.lab
    cl = np.asarray(cell_lab, dtype=np.float64)
    moved = 0

    def shifted(dy, dx):
        sh = np.full((gh, gw), -1, dtype=np.int32)
        ys = slice(max(0, dy), gh + min(0, dy))
        xs = slice(max(0, dx), gw + min(0, dx))
        ys2 = slice(max(0, -dy), gh + min(0, -dy))
        xs2 = slice(max(0, -dx), gw + min(0, -dx))
        sh[ys, xs] = idx[ys2, xs2]
        return sh.reshape(-1)[ok]

    for _ in range(int(rounds)):
        own = flat[ok]
        own_lab = lab[own]
        d_own = de00_arr(cl[:, 0], cl[:, 1], cl[:, 2],
                         own_lab[:, 0], own_lab[:, 1], own_lab[:, 2])
        cand = np.repeat(own[:, None], 5, axis=1)
        dsel = np.repeat(np.inf, 5 * n).reshape(n, 5)
        dsel[:, 0] = d_own
        nbs = [shifted(dy, dx) for dy, dx in NB]
        dnbs = []
        for k, sh in enumerate(nbs):
            v = sh >= 0
            nbl = lab[np.maximum(sh, 0)]
            dk = de00_arr(cl[:, 0], cl[:, 1], cl[:, 2], nbl[:, 0], nbl[:, 1], nbl[:, 2])
            dn = de00_arr(own_lab[:, 0], own_lab[:, 1], own_lab[:, 2],
                          nbl[:, 0], nbl[:, 1], nbl[:, 2])
            use = v & (dn <= tie) & (dk <= d_own + 1.0)      # 邻居色必须“几乎一样”，否则不当候选
            cand[:, 1 + k] = np.where(use, sh, own)
            dsel[:, 1 + k] = np.where(use, dk, np.inf)
            dnbs.append((v, nbl))
        nbterm = np.zeros((n, 5))
        for (v, nbl) in dnbs:
            for j in range(5):
                cj = lab[cand[:, j]]
                nbterm[:, j] += np.where(v, de00_arr(cj[:, 0], cj[:, 1], cj[:, 2],
                                                     nbl[:, 0], nbl[:, 1], nbl[:, 2]), 0.0)
        cost = dsel + lam * nbterm
        cost[:, 0] -= 1e-6                     # 平手时优先保持原样，避免来回抖
        best = np.take_along_axis(cand, np.argmin(cost, axis=1)[:, None], 1)[:, 0]
        new = flat.copy()
        new[ok] = best
        moved += int((new[ok] != flat[ok]).sum())
        if np.array_equal(new, flat):
            break
        flat = new
        idx = flat.reshape(gh, gw)
    return idx, moved


def build_pattern(img, gw=48, gh=48, palette=None, max_colors=None,
                  allowed_codes=None, remove_bg=False, bg_tol=BG_TOL, bg_color=None,
                  sample='auto', despeckle_min=0, harmonize_on=True, match_tol=None,
                  preset='general', upscale=True):
    """主流程：图片 → 图案数据。img 可为路径或 PIL.Image。
    返回 pattern dict：gw/gh/idx(gh,gw, -1=空)/counts(色号→颗数)/colors(用到的色卡项)/…
    """
    if isinstance(img, str):
        img = Image.open(img)
    im = img.convert('RGB')

    # 小图先放大：每格覆盖的原图像素太少时，Lanczos 放大到够用（实测比不放大更接近原图）
    up_k = 1
    if upscale:
        _ppc = im.width * im.height / float(max(1, int(gw) * int(gh)))
        if _ppc < MIN_PX_PER_CELL:
            up_k = min(UPSCALE_MAX, max(2, int((MIN_PX_PER_CELL / max(_ppc, 0.01)) ** 0.5) + 1))
            im = im.resize((im.width * up_k, im.height * up_k), Image.LANCZOS)
    arr = np.asarray(im)
    mask = None
    bgc = None
    if remove_bg:
        mask, bgc = bg_remove(arr, tol=bg_tol, bg_color=bg_color)
    arr, _emask = prepare_by_preset(arr, preset)
    _sample = resolve_sample(preset, sample)
    reps, cover = pixelate(arr, gw, gh, _sample, _emask)
    empty = cells_empty(mask, gw, gh)
    matcher = Matcher(load_palette(palette), allowed_codes)
    labs = srgb2lab_arr(reps.reshape(-1, 3))
    idx, dist = matcher.match(labs)
    idx = idx.reshape(gh, gw).astype(np.int32)
    dist = dist.reshape(gh, gw)
    idx[empty] = -1
    if match_tol is not None:                      # 离所有色都太远的格 → 留空
        idx[(dist > match_tol) & (idx >= 0)] = -1
    moved = 0
    if harmonize_on:                                # 量化阶段的空间一致性（防伪杂色）——先做，再限色
        idx, moved = harmonize(idx, matcher,
                                  srgb2lab_arr(reps.reshape(-1, 3)))
    idx, dropped = limit_colors(idx, dist, matcher, max_colors)
    speck = 0
    if despeckle_min and despeckle_min > 1:          # 第二道防线：清理孤立小色块
        idx, speck = despeckle(idx, min_cells=despeckle_min)
    flat = idx.reshape(-1)
    used = flat[flat >= 0]
    usage = np.bincount(used, minlength=len(matcher.codes)).astype(int)
    codes = [(i, int(u)) for i, u in enumerate(usage) if u > 0]
    codes.sort(key=lambda t: -t[1])
    return {'gw': gw, 'gh': gh, 'idx': idx, 'matcher': matcher,
            'counts': codes, 'total': int(usage.sum()), 'dropped': dropped,
            'speck': speck, 'despeckle_min': despeckle_min, 'harmonized': moved,
            'sample': sample, 'bg_color': bgc, 'empty': empty, 'dist': dist,
            'src_wh': (im.width, im.height),
            'upscaled': up_k, 'px_per_cell': im.width * im.height / float(gw * gh),
            'palette_name': load_palette(palette)['name']}


# ---------------------------------------------------------------- 渲染
_TIMES_FONT = ['msyh.ttc', 'simhei.ttf', 'simsun.ttc', 'DejaVuSans.ttf', 'arial.ttf']


def _font(size, bold=False):
    names = (['msyhbd.ttc'] if bold else []) + _TIMES_FONT
    for n in names:
        for d in (r'C:\Windows\Fonts', '/usr/share/fonts'):
            p = os.path.join(d, n)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def _lum(c):
    v = [x / 255.0 for x in c]
    v = [(x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4) for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def _txt_color(rgb):
    """按 WCAG 对比度选白字/深字（哪个对比高用哪个）"""
    l1 = _lum(rgb)
    def ratio(fg):
        l2 = _lum(fg)
        hi, lo = max(l1, l2), min(l1, l2)
        return (hi + 0.05) / (lo + 0.05)
    return (255, 255, 255) if ratio((255, 255, 255)) >= ratio((35, 35, 35)) else (35, 35, 35)


def render(pattern, cell=30, red_every=RED_EVERY, title=None, show_codes=True,
           show_coords=True, show_legend=True, pad=26, legend_cols=None):
    gw, gh, idx = pattern['gw'], pattern['gh'], pattern['idx']
    matcher = pattern['matcher']
    margin = pad
    u = cell / 24.0

    def M_(v):
        return max(1, int(round(v * u)))

    lab_h = M_(34) if show_coords else 0
    lab_w = M_(34) if show_coords else 0
    title_h = M_(56) if title else 0
    grid_w, grid_h = gw * cell, gh * cell
    W = margin * 2 + lab_w + grid_w
    items = [(matcher.codes[i], matcher.hexes[i], n) for i, n in pattern['counts']]
    lcols = legend_cols or max(1, min(8, (W - 2 * margin) // M_(190)))
    lrows = int(math.ceil(len(items) / lcols)) if (items and show_legend) else 0
    legend_h = (lrows * M_(30) + M_(74)) if lrows else 0
    H = margin * 2 + title_h + lab_h + grid_h + legend_h
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    f_title = _font(M_(24), bold=True)
    f_sub = _font(M_(13))
    f_coord = _font(M_(12))
    f_code = _font(max(8, int(cell * 0.46)))
    f_leg = _font(M_(13))
    y0 = margin + title_h + lab_h
    x0 = margin + lab_w
    # 标题
    if title:
        d.text((margin, margin + M_(6)), title, font=f_title, fill=(20, 20, 20))
        d.text((margin, margin + M_(34)),
               '尺寸 %d×%d ｜ 用色 %d 种 ｜ 共 %d 颗 ｜ 色卡 %s'
               % (gw, gh, len(items), pattern['total'], pattern['palette_name']),
               font=f_sub, fill=(110, 110, 110))
    # 色块 + 色号
    for r in range(gh):
        for c in range(gw):
            i = int(idx[r, c])
            if i < 0:
                continue
            rgb = tuple(int(v) for v in matcher.rgb[i])
            x, y = x0 + c * cell, y0 + r * cell
            d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=rgb)
            if show_codes and cell >= 14:
                t = matcher.codes[i]
                bb = d.textbbox((0, 0), t, font=f_code)
                d.text((x + (cell - (bb[2] - bb[0])) / 2 - bb[0],
                        y + (cell - (bb[3] - bb[1])) / 2 - bb[1]), t,
                       font=f_code, fill=_txt_color(rgb))
    # 网格线：普通细灰线 + 每 red_every 格红线
    for c in range(gw + 1):
        x = x0 + c * cell
        red = (c % red_every == 0)
        d.line([x, y0, x, y0 + grid_h], fill=(214, 40, 40) if red else (205, 205, 205),
               width=max(2, int(round(cell * 0.085))) if red else 1)
    for r in range(gh + 1):
        y = y0 + r * cell
        red = (r % red_every == 0)
        d.line([x0, y, x0 + grid_w, y], fill=(214, 40, 40) if red else (205, 205, 205),
               width=max(2, int(round(cell * 0.085))) if red else 1)
    d.rectangle([x0, y0, x0 + grid_w, y0 + grid_h], outline=(60, 60, 60), width=2)
    # 坐标刻度
    if show_coords:
        for c in range(0, gw, red_every):
            t = str(c + 1)
            bb = d.textbbox((0, 0), t, font=f_coord)
            d.text((x0 + c * cell + 2, y0 - lab_h + M_(8)), t, font=f_coord, fill=(90, 90, 90))
        for r in range(0, gh, red_every):
            t = str(r + 1)
            d.text((margin + M_(4), y0 + r * cell + 2), t, font=f_coord, fill=(90, 90, 90))
    # 底部图例
    if legend_h:
        ly = y0 + grid_h + M_(16)
        d.line([margin, ly, W - margin, ly], fill=(220, 220, 220), width=1)
        d.text((margin, ly + M_(8)), '用量清单（按用量排序）', font=f_leg, fill=(70, 70, 70))
        col_w = max(M_(150), (W - 2 * margin) // lcols)
        sw = M_(20)
        for k, (code, hx, n) in enumerate(items):
            cx = margin + (k % lcols) * col_w
            cy = ly + M_(34) + (k // lcols) * M_(30)
            rgb = tuple(int(v) for v in matcher.rgb[matcher.codes.index(code)])
            d.rectangle([cx, cy, cx + sw, cy + sw], fill=rgb, outline=(120, 120, 120))
            d.text((cx + sw + M_(7), cy + M_(3)), '%s ×%d' % (code, n), font=f_leg,
                   fill=(45, 45, 45))
    return im


def save_csv(pattern, path):
    m = pattern['matcher']
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['色号', 'HEX', 'RGB', '颗数'])
        for i, n in pattern['counts']:
            rgb = m.rgb[i]
            w.writerow([m.codes[i], m.hexes[i], '%d,%d,%d' % tuple(rgb), n])
        w.writerow(['合计', '', '', pattern['total']])
    return path


def layout(pattern, cell=30, red_every=RED_EVERY, title=None, show_coords=True,
           show_legend=True, pad=26, legend_cols=None):
    """与 render() 同一套版面算法，返回几何信息（供界面把点击坐标换算成行列）"""
    gw, gh = pattern['gw'], pattern['gh']
    u = cell / 24.0

    def M_(v):
        return max(1, int(round(v * u)))

    margin = M_(pad)
    lab_h = M_(34) if show_coords else 0
    lab_w = M_(34) if show_coords else 0
    title_h = M_(56) if title else 0
    grid_w, grid_h = gw * cell, gh * cell
    W = margin * 2 + lab_w + grid_w
    items = pattern['counts']
    lcols = legend_cols or max(1, min(8, (W - 2 * margin) // M_(190)))
    lrows = int(math.ceil(len(items) / lcols)) if (items and show_legend) else 0
    legend_h = (lrows * M_(30) + M_(74)) if lrows else 0
    H = margin * 2 + title_h + lab_h + grid_h + legend_h
    return {'W': W, 'H': H, 'x0': margin + lab_w, 'y0': margin + title_h + lab_h,
            'cell': cell, 'grid_w': grid_w, 'grid_h': grid_h, 'legend_h': legend_h,
            'u': u}


def sub_pattern(pat, r0, r1, c0, c1):
    """裁一块子图案（PDF 分板用）"""
    idx = pat['idx'][r0:r1, c0:c1]
    m = pat['matcher']
    flat = idx.reshape(-1)
    used = flat[flat >= 0]
    usage = np.bincount(used, minlength=len(m.codes)) if len(used) else np.zeros(len(m.codes), int)
    counts = sorted([(i, int(u)) for i, u in enumerate(usage) if u > 0], key=lambda t: -t[1])
    d = dict(pat)
    d.update({'gw': idx.shape[1], 'gh': idx.shape[0], 'idx': idx,
              'counts': counts, 'total': int(usage.sum())})
    return d


def save_pdf(pattern, path, cell=30, tile=None, title=None, **kw):
    """导出 PDF。tile=(板宽格, 板高格) 时按板分页（每页一块，方便打印拼装）"""
    pages = []
    if not tile:
        pages.append(render(pattern, cell=cell, title=title, **kw))
    else:
        tw, th = tile
        gw, gh = pattern['gw'], pattern['gh']
        rows = int(math.ceil(gh / float(th)))
        cols = int(math.ceil(gw / float(tw)))
        total = rows * cols
        n = 0
        for ir in range(rows):
            for ic in range(cols):
                r0, c0 = ir * th, ic * tw
                sub = sub_pattern(pattern, r0, min(gh, r0 + th), c0, min(gw, c0 + tw))
                n += 1
                t = '%s  第 %d/%d 块（行 %d-%d，列 %d-%d）' % (
                    title or 'CeraBeads 图纸', n, total, r0 + 1, min(gh, r0 + th),
                    c0 + 1, min(gw, c0 + tw))
                pages.append(render(sub, cell=cell, title=t, **kw))
    pages[0].save(path, 'PDF', resolution=150.0, save_all=True, append_images=pages[1:])
    return path


def render_band(pattern, cell, red_every, r0, r1, show_codes=True):
    """只画“第 r0~r1 格行”那几条格行（供超大图流式写盘，逐行拼出来一模一样）。"""
    gw = pattern['gw']
    idx, matcher = pattern['idx'], pattern['matcher']
    W = gw * cell
    H = (r1 - r0) * cell
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    f_code = _font(max(8, int(cell * 0.46)))
    for r in range(r0, r1):
        y = (r - r0) * cell
        for c in range(gw):
            i = int(idx[r, c])
            if i < 0:
                continue
            rgb = tuple(int(v) for v in matcher.rgb[i])
            d.rectangle([c * cell, y, c * cell + cell - 1, y + cell - 1], fill=rgb)
            if show_codes and cell >= 14:
                t = matcher.codes[i]
                bb = d.textbbox((0, 0), t, font=f_code)
                d.text((c * cell + (cell - (bb[2] - bb[0])) / 2 - bb[0],
                        y + (cell - (bb[3] - bb[1])) / 2 - bb[1]), t,
                       font=f_code, fill=_txt_color(rgb))
    lw = max(2, int(round(cell * 0.085)))
    for c in range(gw + 1):
        red = (c % red_every == 0)
        d.line([c * cell, 0, c * cell, H], fill=(214, 40, 40) if red else (205, 205, 205),
               width=lw if red else 1)
    for r in range(r0, r1 + 1):
        y = (r - r0) * cell
        red = (r % red_every == 0)
        d.line([0, y, W, y], fill=(214, 40, 40) if red else (205, 205, 205),
               width=lw if red else 1)
    return im


def save_png_stream(pattern, path, cell=30, red_every=RED_EVERY, show_codes=True,
                    band_cells=64):
    """流式写 PNG：逐条格行渲染、逐行压缩落盘。
    所以**再大的图纸也能写完**（峰值内存只与一条格带有关），不会“图太大做不了”。"""
    import struct
    import zlib

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    gw, gh = pattern['gw'], pattern['gh']
    W, H = gw * cell, gh * cell
    comp = zlib.compressobj(6)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)))
        buf = bytearray()
        for r0 in range(0, gh, max(1, int(band_cells))):
            r1 = min(gh, r0 + max(1, int(band_cells)))
            im = render_band(pattern, cell, red_every, r0, r1, show_codes)
            raw = im.tobytes()
            stride = W * 3
            for y in range(im.height):
                buf += b'\x00' + raw[y * stride:(y + 1) * stride]
                if len(buf) > (1 << 20):
                    f.write(chunk(b'IDAT', comp.compress(bytes(buf))))
                    buf.clear()
        if buf:
            f.write(chunk(b'IDAT', comp.compress(bytes(buf))))
        f.write(chunk(b'IDAT', comp.flush()))
        f.write(chunk(b'IEND', b''))
    return path


# ============ 本机性能探测：决定预览能画到多大 ---------===

def machine_profile():
    """本机可用内存 / 总内存 / 逻辑核数（纯标准库，Windows 用 GlobalMemoryStatusEx）。"""
    avail_mb = total_mb = 0.0
    try:
        import ctypes

        class _MEM(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        m = _MEM()
        m.dwLength = ctypes.sizeof(_MEM)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            total_mb = m.ullTotalPhys / 1048576.0
            avail_mb = m.ullAvailPhys / 1048576.0
    except Exception:
        pass
    if not avail_mb:                                  # 非 Windows 兜底
        try:
            with open('/proc/meminfo') as fh:
                for ln in fh:
                    if ln.startswith('MemAvailable:'):
                        avail_mb = float(ln.split()[1]) / 1024.0
                    elif ln.startswith('MemTotal:'):
                        total_mb = float(ln.split()[1]) / 1024.0
        except Exception:
            pass
    cores = os.cpu_count() or 1
    return {'avail_mb': avail_mb, 'total_mb': total_mb, 'cores': cores}


def preview_cell_cap(gw, gh, cell_wanted, prof=None, max_share=0.35, hard_mp=400.0):
    """按本机情况算「预览最多能按每格多少像素画」。
    一张 RGB 图纸在预览里会同时存在 3 份（PIL 图 + QImage + QPixmap），约 9 字节/像素；
    因此 像素上限 ≈ 可用内存 × max_share / 9。返回 (cell, 信息dict)。"""
    prof = prof or machine_profile()
    cells = max(1, int(gw) * int(gh))
    cell = int(cell_wanted)
    avail_mb = prof['avail_mb']
    if avail_mb <= 0:
        budget_mp = hard_mp
    else:
        budget_mp = min(hard_mp, avail_mb * max_share * 1048576.0 / 9.0 / 1e6)
    budget_mp = max(4.0, budget_mp)
    need_mp = cells * cell * cell / 1e6
    capped = False
    if need_mp > budget_mp:
        cell = int((budget_mp * 1e6 / cells) ** 0.5)
        cell = max(4, min(int(cell_wanted), cell))
        capped = True
    info = dict(cell=cell, wanted=int(cell_wanted), capped=capped, cells=cells,
                budget_mp=budget_mp, need_mp=need_mp, avail_mb=avail_mb,
                total_mb=prof['total_mb'], cores=prof['cores'])
    return cell, info


# ============ 诊断类小工具：线宽预警 / 孤岛 / 色号预算 ============

def line_width_px(arr, thresh=EDGE_THRESH):
    """估算「最细线的过渡带宽度」（像素）。做法：Sobel 边缘带沿行/列做游程统计取中位。
    意义：每格覆盖的原图像素 < 过渡带宽度时，细线会被格子抹掉。"""
    m = sobel_edge_mask(arr, thresh)
    if not m.any():                      # 浅色/低对比线稿：改用梯度分位数兜底
        g = np.asarray(arr.convert('L') if hasattr(arr, 'convert') else arr).astype(np.float64)
        if g.ndim == 3:
            g = g.mean(axis=2)
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        mag = np.zeros_like(g)
        mag[:, :-1] += gx
        mag[:-1, :] += gy
        q = np.percentile(mag, 92)
        if q <= 0:
            return 0.0, 0.0
        m = mag > q
    runs = []
    for axis in (0, 1):
        mm = m if axis == 0 else m.T
        for row in mm:
            idx = np.flatnonzero(row)
            if idx.size == 0:
                continue
            brk = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([0], brk + 1))
            ends = np.concatenate((brk, [idx.size - 1]))
            for a, b in zip(starts, ends):
                runs.append(idx[b] - idx[a] + 1)
    if not runs:
        return 0.0, 0.0
    runs = np.asarray(runs, dtype=np.float64)
    return float(np.median(runs)), float(np.percentile(runs, 25))


def islands(idx, max_cells=2):
    """找出「孤岛」：同色且只有 ≤ max_cells 格的连通小块（4 邻接）。返回 bool 掩膜。
    只标记、不合并——要不要清由你决定（清理碎点开关）。"""
    h, w = idx.shape
    out = np.zeros((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or idx[r0, c0] < 0:
                continue
            code = idx[r0, c0]
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and idx[rr, cc] == code:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if len(comp) <= max_cells:
                for r, c in comp:
                    out[r, c] = True
    return out


def suggest_max_colors(counts, cover=0.98, low=4, high=120):
    """按用量曲线推荐色号上限：把用量从多到少累加，到 cover（默认 98%）就行。
    返回 (建议种数, 建议种数时的覆盖率)。"""
    tot = float(sum(n for _, n in counts)) or 1.0
    acc = 0.0
    k = 0
    for k, (_i, n) in enumerate(counts, 1):
        acc += n
        if acc / tot >= cover:
            break
    return max(low, min(high, k)), acc / tot


def save_jpg(pattern, path, cell=30, quality=92, **kw):
    im = render(pattern, cell=cell, **kw).convert('RGB')
    im.save(path, 'JPEG', quality=quality, subsampling=0)
    return path


def selftest(outdir=None):
    out = []
    err = de00_validate()
    out.append('CIEDE2000 校验（9 组 Sharma 标准值）最大误差 = %.6f %s'
               % (err, 'OK' if err < 1e-3 else 'FAIL'))
    for label, _ in PALETTE_FILES:
        p = load_palette(label)
        _want = None
        _mm = re.search(r'(\d{2,4})', label)
        if _mm:
            _want = int(_mm.group(1))
        _okp = (p['count'] > 0) and (_want is None or p['count'] == _want)
        out.append('%s：%d 色 %s' % (label, p['count'], 'OK' if _okp else 'FAIL'))
    # 造一张测试图：白底 + 几个色块，跑全流程
    a = np.full((240, 320, 3), 255, dtype=np.uint8)
    a[40:140, 30:150] = (220, 60, 60)
    a[70:180, 170:290] = (40, 90, 200)
    a[150:220, 60:260] = (250, 200, 60)
    im = Image.fromarray(a)
    pat = build_pattern(im, gw=40, gh=30, palette='MARD 221', max_colors=10, remove_bg=True)
    out.append('测试图 → %d×%d，用色 %d 种，总 %d 颗，背景格 %d'
               % (pat['gw'], pat['gh'], len(pat['counts']), pat['total'],
                  int(pat['empty'].sum())))
    out.append('用量前 5：%s' % [(pat['matcher'].codes[i], n) for i, n in pat['counts'][:5]])
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        p1 = os.path.join(outdir, 'selftest_pattern.png')
        p2 = os.path.join(outdir, 'selftest_用量.csv')
        render(pat, cell=18, title='CeraBeads selftest').save(p1)
        save_csv(pat, p2)
        out.append('渲染 PNG %s (%d 字节)；CSV %s (%d 字节)'
                   % (p1, os.path.getsize(p1), p2, os.path.getsize(p2)))
    out.append('结果 = %s' % ('OK' if err < 1e-3 else 'FAIL'))
    return '\n'.join(out)


if __name__ == '__main__':
    txt = selftest(sys.argv[2] if len(sys.argv) > 2 else None)
    print(txt)
    try:
        txt = '%s\npython=%s numpy=%s PIL=%s\n%s' % (
            APP_TITLE + ' ' + APP_VERSION, sys.version.split()[0], np.__version__,
            Image.__version__, txt)
    except Exception:
        pass
    with open(os.path.join(HERE, '_selftest.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
