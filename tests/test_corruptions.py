import unittest

import numpy as np

from ccr.corruptions import (CORRUPTIONS, SEVERITIES, augment_batch, corrupt_batch,
                             occlusion_side)
from tests.helpers import random_images


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.images = random_images()

    def test_shape_and_dtype_preserved(self):
        for name in CORRUPTIONS:
            for s in SEVERITIES:
                out = corrupt_batch(self.images, name, s, (0,))
                self.assertEqual(out.shape, self.images.shape, (name, s))
                self.assertEqual(out.dtype, np.uint8, (name, s))

    def test_input_is_not_modified(self):
        before = self.images.copy()
        for name in CORRUPTIONS:
            corrupt_batch(self.images, name, 5, (0,))
        np.testing.assert_array_equal(self.images, before)

    def test_damage_grows_with_severity(self):
        for name in CORRUPTIONS:
            diffs = [
                np.abs(corrupt_batch(self.images, name, s, (0,)).astype(float) - self.images).mean()
                for s in SEVERITIES
            ]
            self.assertTrue(all(a < b for a, b in zip(diffs, diffs[1:])), (name, diffs))

    def test_same_entropy_same_output(self):
        for name in CORRUPTIONS:
            a = corrupt_batch(self.images, name, 3, (4, 2))
            b = corrupt_batch(self.images, name, 3, (4, 2))
            np.testing.assert_array_equal(a, b)

    def test_different_entropy_changes_random_corruptions(self):
        for name in ("gaussian_noise", "occlusion"):
            a = corrupt_batch(self.images, name, 3, (0,))
            b = corrupt_batch(self.images, name, 3, (1,))
            self.assertFalse(np.array_equal(a, b), name)

    def test_chunking_does_not_matter_for_first_images(self):
        full = corrupt_batch(self.images, "gaussian_noise", 2, (0,))
        head = corrupt_batch(self.images[:3], "gaussian_noise", 2, (0,))
        np.testing.assert_array_equal(full[:3], head)

    def test_occlusion_area(self):
        img = np.full((4, 32, 32, 3), 200, dtype=np.uint8)
        for s in SEVERITIES:
            side = occlusion_side(32, s)
            out = corrupt_batch(img, "occlusion", s, (0,))
            zeros = (out == 0).reshape(4, -1).sum(axis=1)
            np.testing.assert_array_equal(zeros, side * side * 3)

    def test_contrast_leaves_flat_image_alone(self):
        img = np.full((2, 32, 32, 3), 123, dtype=np.uint8)
        np.testing.assert_array_equal(corrupt_batch(img, "contrast", 5, (0,)), img)

    def test_brightness_shift(self):
        img = np.full((1, 8, 8, 3), 100, dtype=np.uint8)
        out = corrupt_batch(img, "brightness", 1, (0,))
        self.assertTrue(np.all(np.abs(out.astype(int) - 125.5) <= 1))
        out = corrupt_batch(np.full((1, 8, 8, 3), 250, dtype=np.uint8), "brightness", 5, (0,))
        self.assertTrue(np.all(out == 255))

    def test_jpeg_keeps_channel_order(self):
        img = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        img[..., 0], img[..., 1], img[..., 2] = 200, 60, 30
        for s in SEVERITIES:
            out = corrupt_batch(img, "jpeg", s, (0,)).astype(int)
            self.assertTrue((out[..., 0] > out[..., 1]).all(), s)
            self.assertTrue((out[..., 1] > out[..., 2]).all(), s)

    def test_noise_level_matches_sigma(self):
        img = np.full((50, 32, 32, 3), 128, dtype=np.uint8)
        out = corrupt_batch(img, "gaussian_noise", 5, (0,)).astype(float)
        self.assertAlmostEqual(out.std(), 0.10 * 255, delta=1.0)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            corrupt_batch(self.images, "fog", 1, (0,))
        for bad in (0, 6):
            with self.assertRaises(ValueError):
                corrupt_batch(self.images, "jpeg", bad, (0,))
        with self.assertRaises(ValueError):
            corrupt_batch(self.images.astype(np.float32), "jpeg", 1, (0,))
        with self.assertRaises(ValueError):
            corrupt_batch(self.images[0], "jpeg", 1, (0,))


class AugmentTests(unittest.TestCase):
    def setUp(self):
        self.images = random_images(n=400, size=16)

    def test_p_controls_fraction_changed(self):
        out = augment_batch(self.images, ["gaussian_noise"], [3], 0.5, (0, 1))
        changed = (out != self.images).reshape(len(out), -1).any(axis=1).mean()
        self.assertTrue(0.4 < changed < 0.6, changed)

    def test_p_one_changes_everything(self):
        out = augment_batch(self.images, ["gaussian_noise"], [2], 1.0, (0, 1))
        self.assertTrue((out != self.images).reshape(len(out), -1).any(axis=1).all())

    def test_only_listed_corruptions_are_used(self):
        out = augment_batch(self.images, ["occlusion"], [1, 2, 3], 1.0, (0, 1))
        self.assertTrue(np.all(out <= self.images))

    def test_input_unchanged_and_deterministic(self):
        before = self.images.copy()
        a = augment_batch(self.images, ["jpeg", "occlusion"], [1, 2], 0.5, (3, 1))
        b = augment_batch(self.images, ["jpeg", "occlusion"], [1, 2], 0.5, (3, 1))
        np.testing.assert_array_equal(self.images, before)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
