import conseal as cl
import jpeglib
import logging

from PIL import Image
from conseal.jsideinfo import Method
from parameterized import parameterized
import unittest
import numpy as np

from . import defs
STEGO_DIR = defs.ASSETS_DIR / 'jsideinfo'


class TestJSIDEINFO(unittest.TestCase):
    """Test suite for J-SIDEINFO embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.NAIVE_DCT),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.NAIVE_DCT),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.NAIVE_DCT),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.NAIVE_DCT),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.NAIVE_DCT),
    ])
    def test_unquantized_coefficients_extraction(self, pgm_image, jpg_image, method):
        with Image.open(STEGO_DIR / "pgm" / pgm_image) as img:
            spatial = np.array(img.convert("L"), "uint8")
            jpeg = jpeglib.read_dct(STEGO_DIR / "jpg" / jpg_image)
            coeffs = cl.jsideinfo._costmap.compute_unquantized_coefficients(
                x0=spatial,
                qt=jpeg.qt[0],
                method=method
            )
            # Difference can be off by a maximum of 0.5 for the
            # quantization error + 0.0625 (2^-4) because of differences in precision
            np.testing.assert_allclose(coeffs, jpeg.Y, atol=0.5625)


    @parameterized.expand([
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg'),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg'),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg'),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg'),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg'),
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg'),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg'),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg'),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg'),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg'),
    ])
    def test_naive_vs_islow(self, pgm_image, jpg_image):
        with Image.open(STEGO_DIR / "pgm" / pgm_image) as img:
            spatial = np.array(img.convert("L"), "uint8")
            jpeg = jpeglib.read_dct(STEGO_DIR / "jpg" / jpg_image)
            coeffs_islow = cl.jsideinfo._costmap.compute_unquantized_coefficients(
                x0=spatial,
                qt=jpeg.qt[0],
                method=Method.LIBJPEG_ISLOW
            )
            coeffs_naive = cl.jsideinfo._costmap.compute_unquantized_coefficients(
                x0=spatial,
                qt=jpeg.qt[0],
                method=Method.NAIVE_DCT
            )
            # Difference can be off by a maximum of 0.0625 (2^-4), due to of differences in precision
            np.testing.assert_allclose(coeffs_islow, coeffs_naive, atol=0.0625)


    @parameterized.expand([
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.LIBJPEG_ISLOW),
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.NAIVE_DCT),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.NAIVE_DCT),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.NAIVE_DCT),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.NAIVE_DCT),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.NAIVE_DCT),
    ])
    def test_costmap(self, pgm_image, jpg_image, method):
        with Image.open(STEGO_DIR / "pgm" / pgm_image) as img:
            spatial = np.array(img.convert("L"), "uint8")
            jpeg = jpeglib.read_dct(STEGO_DIR / "jpg" / jpg_image)
            cost = cl.jsideinfo.compute_cost_adjusted(
                x0=spatial,
                y0=jpeg.Y,
                qt=jpeg.qt[0],
                method=method
            )
            self.assertGreater(np.min(cost), 0)
            self.assertNotEqual(np.min(cost), 0)

            rho_p1, rho_m1 = cost
            qe = cl.jsideinfo._costmap.compute_unquantized_coefficients(
                x0=spatial,
                qt=jpeg.qt[0],
                method=method
            ) - jpeg.Y

            ignore = (jpeg.Y > -1023) & (jpeg.Y < 1023)
            rho_p1, rho_m1, qe = rho_p1[ignore], rho_m1[ignore], qe[ignore]

            # This is only true for symmetric cost, we are testing with juniward
            self.assertTrue(np.all(rho_p1[qe > 0] <= rho_m1[qe > 0]))
            self.assertTrue(np.all(rho_p1[qe < 0] >= rho_m1[qe < 0]))


    @parameterized.expand([
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.LIBJPEG_ISLOW, 0.1),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.LIBJPEG_ISLOW, 0.2),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.LIBJPEG_ISLOW, 0.3),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.LIBJPEG_ISLOW, 0.4),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.LIBJPEG_ISLOW, 0.5),
        ('BOSSbase_1.pgm', 'BOSSbase_1.jpg', Method.NAIVE_DCT, 0.1),
        ('BOSSbase_2.pgm', 'BOSSbase_2.jpg', Method.NAIVE_DCT, 0.2),
        ('BOSSbase_3.pgm', 'BOSSbase_3.jpg', Method.NAIVE_DCT, 0.3),
        ('BOSSbase_4.pgm', 'BOSSbase_4.jpg', Method.NAIVE_DCT, 0.4),
        ('BOSSbase_5.pgm', 'BOSSbase_5.jpg', Method.NAIVE_DCT, 0.5),
    ])
    def test_simulation(self, pgm_image, jpg_image, method, alpha):
        with Image.open(STEGO_DIR / "pgm" / pgm_image) as img:
            spatial = np.array(img.convert("L"), "uint8")
            jpeg = jpeglib.read_dct(STEGO_DIR / "jpg" / jpg_image)
            new_coeffs = cl.jsideinfo.simulate_single_channel(
                x0=spatial,
                y0=jpeg.Y,
                qt=jpeg.qt[0],
                alpha=0,
                method=method
            )
            np.testing.assert_array_equal(jpeg.Y, new_coeffs)
            new_coeffs = cl.jsideinfo.simulate_single_channel(
                x0=spatial,
                y0=jpeg.Y,
                qt=jpeg.qt[0],
                alpha=alpha,
                method=method
            )
            np.testing.assert_allclose(jpeg.Y, new_coeffs, atol=1)


__all__ = ['TestJSIDEINFO']
