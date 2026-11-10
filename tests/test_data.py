import tempfile
import unittest
from pathlib import Path

import numpy as np

from ccr import data
from tests.helpers import write_fake_cifar


class CifarLoaderTests(unittest.TestCase):
    def test_reads_archive_and_fixes_axis_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "cifar-10-python.tar.gz"
            test_batch = write_fake_cifar(archive, per_batch=12)
            x_tr, y_tr, x_te, y_te = data.load_cifar10(tmp, download=False, md5=data._md5(archive))
            self.assertEqual(x_tr.shape, (60, 32, 32, 3))
            self.assertEqual(x_te.shape, (12, 32, 32, 3))
            self.assertEqual(x_tr.dtype, np.uint8)
            self.assertEqual(y_tr.shape, (60,))
            np.testing.assert_array_equal(y_te, test_batch[b"labels"])
            # first 1024 values of a row are the red plane, then green, then blue
            row = test_batch[b"data"][3]
            np.testing.assert_array_equal(x_te[3, :, :, 0].ravel(), row[:1024])
            np.testing.assert_array_equal(x_te[3, :, :, 1].ravel(), row[1024:2048])
            np.testing.assert_array_equal(x_te[3, :, :, 2].ravel(), row[2048:])

    def test_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fake_cifar(Path(tmp) / "cifar-10-python.tar.gz")
            with self.assertRaises(ValueError):
                data.load_cifar10(tmp, download=False)

    def test_missing_archive_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                data.load_cifar10(tmp, download=False)


class DigitsTests(unittest.TestCase):
    def test_shapes_and_split(self):
        x_tr, y_tr, x_te, y_te = data.load_dataset("digits", "unused")
        self.assertEqual(x_tr.shape[1:], (32, 32, 3))
        self.assertEqual(x_tr.dtype, np.uint8)
        self.assertEqual(len(x_tr) + len(x_te), 1797)
        self.assertEqual(set(y_tr), set(range(10)))
        np.testing.assert_array_equal(x_tr[..., 0], x_tr[..., 1])

    def test_split_is_fixed(self):
        a = data.load_digits()
        b = data.load_digits()
        np.testing.assert_array_equal(a[0], b[0])

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            data.load_dataset("mnist", ".")


class SubsetTests(unittest.TestCase):
    def test_class_proportions_and_determinism(self):
        y = np.repeat(np.arange(10), 100)
        idx = data.stratified_subset(y, 200, seed=3)
        self.assertEqual(len(idx), 200)
        self.assertEqual(len(set(idx.tolist())), 200)
        np.testing.assert_array_equal(np.bincount(y[idx]), 20)
        np.testing.assert_array_equal(idx, data.stratified_subset(y, 200, seed=3))
        self.assertFalse(np.array_equal(idx, data.stratified_subset(y, 200, seed=4)))

    def test_oversized_request_returns_everything(self):
        y = np.repeat(np.arange(3), 5)
        idx = data.stratified_subset(y, 100, seed=0)
        self.assertEqual(sorted(idx.tolist()), list(range(15)))


if __name__ == "__main__":
    unittest.main()
