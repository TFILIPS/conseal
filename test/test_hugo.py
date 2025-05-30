import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import tempfile
import unittest

from . import defs


STEGO_DIR = defs.ASSETS_DIR / 'hugo'
COST_DIR = STEGO_DIR / 'costmap-matlab'


class TestHUGO(unittest.TestCase):
    """Test suite for HUGO embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_hugo_costmap(self, f: str):
        self._logger.info(f'TestHUGO.test_compare_hugo_matlab({f})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        rho_p1, rho_m1 = cl.hugo.compute_cost_adjusted(x)
        # compare to matlab reference
        mat = scipy.io.loadmat(COST_DIR / f'{f}.mat')
        np.testing.assert_allclose(rho_p1, mat['rhoP1'])
        np.testing.assert_allclose(rho_m1, mat['rhoM1'])

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_hugo_stego(self, f: str):
        self._logger.info(f'TestHUGO.test_hugo_stego({f})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        x1 = cl.hugo.simulate_single_channel(
            x0=x0,
            alpha=.4,
            order='F',
            generator='MT19937',
            seed=139187,
        )
        # compare to matlab reference
        x1_ref = np.array(Image.open(STEGO_DIR / f'{f}.png'))
        np.testing.assert_allclose(x1, x1_ref)
