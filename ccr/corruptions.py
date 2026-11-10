import zlib

import cv2
import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)

# severity levels loosely follow the spirit of Hendrycks & Dietterich (2019)
# but are tuned for 32x32 images and are not the CIFAR-10-C values
_NOISE_SIGMA = (0.02, 0.04, 0.06, 0.08, 0.10)
_BLUR_SIGMA = (0.5, 0.75, 1.0, 1.5, 2.0)
_BRIGHTNESS_SHIFT = (0.1, 0.2, 0.3, 0.4, 0.5)
_CONTRAST_FACTOR = (0.75, 0.5, 0.4, 0.3, 0.15)
_JPEG_QUALITY = (60, 40, 25, 15, 8)
_OCCLUDED_SIDE = (0.1, 0.2, 0.3, 0.4, 0.5)


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gaussian_noise(img, severity, rng):
    sigma = _NOISE_SIGMA[severity - 1]
    return _to_uint8(img + rng.normal(0.0, sigma * 255, img.shape))


def gaussian_blur(img, severity, rng):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=_BLUR_SIGMA[severity - 1])


def brightness(img, severity, rng):
    return _to_uint8(img + 255 * _BRIGHTNESS_SHIFT[severity - 1])


def contrast(img, severity, rng):
    mean = img.mean(axis=(0, 1), keepdims=True)
    return _to_uint8((img - mean) * _CONTRAST_FACTOR[severity - 1] + mean)


def jpeg(img, severity, rng):
    quality = _JPEG_QUALITY[severity - 1]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def occlusion_side(size, severity):
    return max(1, round(_OCCLUDED_SIDE[severity - 1] * size))


def occlusion(img, severity, rng):
    h, w = img.shape[:2]
    side = occlusion_side(min(h, w), severity)
    top = rng.integers(0, h - side + 1)
    left = rng.integers(0, w - side + 1)
    out = img.copy()
    out[top:top + side, left:left + side] = 0
    return out


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "contrast": contrast,
    "jpeg": jpeg,
    "occlusion": occlusion,
}


def corrupt_batch(images, name, severity, entropy):
    """Corrupt an (N, H, W, 3) uint8 batch.

    Every image gets its own generator derived from (entropy, name, severity,
    position in the batch), so results do not depend on how a dataset is chunked.
    """
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}, choose from {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    if images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError("expected a uint8 array of shape (N, H, W, C)")
    fn = CORRUPTIONS[name]
    seq = np.random.SeedSequence([*entropy, zlib.crc32(name.encode()), severity])
    out = np.empty_like(images)
    for i, child in enumerate(seq.spawn(len(images))):
        out[i] = fn(images[i], severity, np.random.default_rng(child))
    return out


def augment_batch(images, names, severities, p, entropy):
    """Corrupt each image with probability p using a random (corruption, severity) pair."""
    rng = np.random.default_rng(list(entropy))
    n = len(images)
    hit = rng.random(n) < p
    which = rng.integers(len(names), size=n)
    level = rng.choice(severities, size=n)
    out = images.copy()
    for k, name in enumerate(names):
        for s in severities:
            idx = np.flatnonzero(hit & (which == k) & (level == s))
            if len(idx):
                out[idx] = corrupt_batch(images[idx], name, s, entropy)
    return out
