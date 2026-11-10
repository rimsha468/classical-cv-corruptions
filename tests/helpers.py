import io
import pickle
import tarfile

import numpy as np


def random_images(n=8, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(40, 216, size=(n, size, size, 3)).astype(np.uint8)


def write_fake_cifar(path, per_batch=12, seed=0):
    """Write a tiny archive laid out like cifar-10-python.tar.gz. Returns the test batch."""
    rng = np.random.default_rng(seed)
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    test_batch = None
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            batch = {
                b"data": rng.integers(0, 256, (per_batch, 3072), dtype=np.uint8),
                b"labels": rng.integers(0, 10, per_batch).tolist(),
            }
            if name == "test_batch":
                test_batch = batch
            payload = pickle.dumps(batch)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return test_batch
