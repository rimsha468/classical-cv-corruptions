import cv2
import numpy as np
from skimage.feature import hog as _hog
from skimage.feature import local_binary_pattern

_HIST_BINS = 16
_LBP_POINTS = 8
_LBP_BINS = _LBP_POINTS + 2  # "uniform" patterns plus one bin for everything else


def _gray(images):
    return np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images])


def tiny(images, size=16):
    small = [cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in images]
    return np.stack(small).reshape(len(images), -1).astype(np.float32) / 255.0


def color_hist(images):
    n, h, w, c = images.shape
    offsets = np.arange(n)[:, None] * _HIST_BINS
    out = np.empty((n, c * _HIST_BINS), dtype=np.float32)
    for ch in range(c):
        bins = (images[..., ch] >> 4).astype(np.int64).reshape(n, -1) + offsets
        counts = np.bincount(bins.ravel(), minlength=n * _HIST_BINS)
        out[:, ch * _HIST_BINS:(ch + 1) * _HIST_BINS] = counts.reshape(n, _HIST_BINS) / (h * w)
    return out


def hog(images):
    feats = [
        _hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
        for g in _gray(images)
    ]
    return np.stack(feats).astype(np.float32)


def lbp(images, grid=4):
    gray = _gray(images)
    n, h, w = gray.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = np.zeros((n, grid * grid * _LBP_BINS), dtype=np.float32)
    for i, g in enumerate(gray):
        codes = local_binary_pattern(g, _LBP_POINTS, 1, method="uniform").astype(np.int64)
        k = 0
        for a in range(grid):
            for b in range(grid):
                cell = codes[ys[a]:ys[a + 1], xs[b]:xs[b + 1]]
                hist = np.bincount(cell.ravel(), minlength=_LBP_BINS)
                out[i, k * _LBP_BINS:(k + 1) * _LBP_BINS] = hist / cell.size
                k += 1
    return out


BASE_FEATURES = {"tiny": tiny, "color_hist": color_hist, "hog": hog, "lbp": lbp}


def parse_spec(spec):
    parts = spec.split("+")
    for part in parts:
        if part not in BASE_FEATURES:
            raise ValueError(f"unknown feature {part!r} in {spec!r}, choose from {sorted(BASE_FEATURES)}")
    return parts


def validate_spec(spec):
    parse_spec(spec)


def extract(images, specs):
    """Return {spec: (N, D) array}. A spec like "hog+color_hist" concatenates base features."""
    needed = {part for spec in specs for part in parse_spec(spec)}
    base = {name: BASE_FEATURES[name](images) for name in needed}
    return {spec: np.concatenate([base[p] for p in parse_spec(spec)], axis=1) for spec in specs}
