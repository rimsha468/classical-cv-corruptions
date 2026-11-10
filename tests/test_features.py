import unittest

import numpy as np

from ccr.features import BASE_FEATURES, extract, parse_spec
from tests.helpers import random_images


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.images = random_images(n=5)

    def test_dimensions_for_32px_images(self):
        dims = {"tiny": 16 * 16 * 3, "color_hist": 48, "hog": 324, "lbp": 160}
        for name, fn in BASE_FEATURES.items():
            out = fn(self.images)
            self.assertEqual(out.shape, (5, dims[name]), name)
            self.assertEqual(out.dtype, np.float32, name)
            self.assertTrue(np.isfinite(out).all(), name)

    def test_color_hist_rows_are_normalised_per_channel(self):
        out = BASE_FEATURES["color_hist"](self.images)
        np.testing.assert_allclose(out.reshape(5, 3, 16).sum(axis=2), 1.0, atol=1e-6)

    def test_color_hist_puts_flat_image_in_one_bin(self):
        img = np.full((1, 32, 32, 3), 100, dtype=np.uint8)
        out = BASE_FEATURES["color_hist"](img).reshape(3, 16)
        np.testing.assert_array_equal(out.argmax(axis=1), 100 >> 4)
        np.testing.assert_allclose(out.max(axis=1), 1.0)

    def test_color_hist_handles_every_intensity(self):
        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
        img = np.repeat(ramp[None, :, :, None], 3, axis=3)
        out = BASE_FEATURES["color_hist"](img).reshape(3, 16)
        np.testing.assert_allclose(out, 1 / 16)

    def test_lbp_cells_sum_to_one(self):
        out = BASE_FEATURES["lbp"](self.images).reshape(5, 16, 10)
        np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-6)

    def test_lbp_cells_sum_to_one_when_grid_does_not_divide_image(self):
        out = BASE_FEATURES["lbp"](random_images(n=3, size=30)).reshape(3, 16, 10)
        np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-6)

    def test_hog_of_flat_image_is_zero(self):
        img = np.full((1, 32, 32, 3), 77, dtype=np.uint8)
        np.testing.assert_array_equal(BASE_FEATURES["hog"](img), 0)

    def test_batch_equals_single_images(self):
        for name, fn in BASE_FEATURES.items():
            whole = fn(self.images)
            single = np.concatenate([fn(self.images[i:i + 1]) for i in range(5)])
            np.testing.assert_allclose(whole, single, atol=1e-6, err_msg=name)

    def test_combined_spec_is_concatenation(self):
        out = extract(self.images, ["hog", "color_hist", "hog+color_hist"])
        np.testing.assert_array_equal(out["hog+color_hist"],
                                      np.concatenate([out["hog"], out["color_hist"]], axis=1))

    def test_unknown_feature(self):
        with self.assertRaises(ValueError):
            parse_spec("hog+sift")


if __name__ == "__main__":
    unittest.main()
