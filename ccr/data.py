import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import load_digits as _sk_digits
from sklearn.model_selection import train_test_split

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_batch(tar, name):
    batch = pickle.load(tar.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
    x = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(batch[b"labels"], dtype=np.int64)


def load_cifar10(root, download=True, md5=CIFAR10_MD5):
    root = Path(root)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"{archive} not found and downloading is disabled")
        root.mkdir(parents=True, exist_ok=True)
        part = archive.with_name(archive.name + ".part")
        urllib.request.urlretrieve(CIFAR10_URL, part)
        part.replace(archive)
    # the batches are pickles, so check the archive before reading them
    if _md5(archive) != md5:
        raise ValueError(f"checksum mismatch for {archive}; delete the file and try again")
    with tarfile.open(archive) as tar:
        train = [_read_batch(tar, f"data_batch_{i}") for i in range(1, 6)]
        x_test, y_test = _read_batch(tar, "test_batch")
    x_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    return x_train, y_train, x_test, y_test


def load_digits():
    """sklearn's 8x8 digits, upsampled to 32x32 RGB. Meant for smoke tests, not for results."""
    d = _sk_digits()
    gray = np.rint(d.images / 16.0 * 255).astype(np.uint8)
    x = np.stack([
        cv2.cvtColor(cv2.resize(g, (32, 32), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        for g in gray
    ])
    return train_test_split(x, d.target, test_size=0.4, stratify=d.target, random_state=0)


def load_dataset(name, root, download=True):
    if name == "cifar10":
        return load_cifar10(root, download)
    if name == "digits":
        x_train, x_test, y_train, y_test = load_digits()
        return x_train, y_train, x_test, y_test
    raise ValueError(f"unknown dataset {name!r}")


def stratified_subset(y, n, seed):
    """Indices of a class-proportional random subset of size n, in random order."""
    if n >= len(y):
        return np.random.default_rng(seed).permutation(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return idx
