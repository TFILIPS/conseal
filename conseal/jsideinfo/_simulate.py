"""
Author: Benedikt Lorch, Martin Benes, Thomas Filips
Affiliation: University of Innsbruck
"""

from typing import Callable

import numpy as np
from ._costmap import compute_cost_adjusted, Method
from .. import simulate
from .. import tools


def simulate_single_channel(
    x0: np.ndarray,
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    cost_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
    *,
    method: Method = Method.LIBJPEG_ISLOW,
    dry_cost: float = 0.1,
    wet_cost: float = 10**13,
    generator: str = None,
    seed: int = None,
) -> np.ndarray:
    """My (Thomas) iterpretation of the proposed embedding sceme from the paper
    `Side-Informed Steganography with Additive Distortion <https://ieeexplore.ieee.org/abstract/document/7368589>`__
     by T. Denemark, et. al.

    The implementation is designed for JPEG images and uses the JUNIWARD cost function by default.
    To calculate quantization errors, we work with the unquantized DCT coefficients.
    These can be obtained either through a naive DCT implementation or by extracting them directly from libjpeg.

    :param x0: pixel values of pre-cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param y0: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate in bits per nzAC coefficient
    :type alpha: float
    :param cost_fn: Cost function used for calculating the base cost. It provides access to the
        unquntized coefficiets, which therefore can be integrated in the cost calculation.
    :type cost_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    :param method: choose the method that is used to extract the unquantized dct coefficients
    :param dry_cost: Limits the downscaling of the cost. Should not be set to 0
    :type dry_cost: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param generator: type of PRNG used by embedding simulator
    :type generator: `numpy.random.Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: quantized stego DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> with Image.open("pre-cover.pgm") as img:
    ...    spatial = np.expand_dims(np.array(img.convert("L"), "uint8"), axis=-1)
    ...    jpeg = jpeglib.from_spatial(spatial)
    ...    jpeg.write_spatial("cover.jpg", qt=quality)
    ...    jpeg = jpeglib.read_dct("cover.jpg")
    ...    jpeg.Y = cl.jsideinfo.simulate_single_channel(
    ...        x0=spatial[..., 0],
    ...        y0=jpeg.Y,
    ...        qt=jpeg.qt[0],
    ...        alpha=0.3,
    ...        method=Method.LIBJPEG_ISLOW
    ...    )
    ...    jpeg.write_dct("stego.jpg")
    """
    # Count the number of embeddable DCT coefficients
    nzAC = tools.dct.nzAC(y0)

    if nzAC == 0:
        raise ValueError('Expected non-zero AC coefficients')

    # Compute cost for embedding into the quantized DCT coefficients
    # of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    rho_p1, rho_m1 = compute_cost_adjusted(
        x0=x0,
        y0=y0,
        qt=qt,
        cost_fn=cost_fn,
        method=method,
        dry_cost=dry_cost,
        wet_cost=wet_cost,
    )

    # Rearrange from 4D to 2D
    rho_p1_2d = tools.dct.jpeglib_to_jpegio(rho_p1)
    rho_m1_2d = tools.dct.jpeglib_to_jpegio(rho_m1)

    # STC simulation
    delta_2d = simulate.ternary(
        rhos=(rho_p1_2d, rho_m1_2d),
        alpha=alpha,
        n=nzAC,
        generator=generator,
        seed=seed,
    )

    # Convert from 2D to 4D
    delta = tools.dct.jpegio_to_jpeglib(delta_2d)

    # stego = cover + stego noise
    return y0 + delta
