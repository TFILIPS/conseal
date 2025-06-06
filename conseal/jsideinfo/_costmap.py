"""
Author: Benedikt Lorch, Martin Benes, Thomas Filips
Affiliation: University of Innsbruck
"""  # noqa: E501
import enum
from typing import Callable

import jpeglib
from jpeglib import DCTMethod
import numpy as np
import typing

from .. import tools
from .. import juniward


class Method(enum.Enum):
    LIBJPEG_ISLOW = enum.auto()
    LIBJPEG_IFAST = enum.auto()
    LIBJPEG_FLOAT = enum.auto()

    NAIVE_DCT = enum.auto()


class MidpointHandling(enum.Enum):
    OMIT_EMBEDDING = enum.auto()
    CLIP_COST_SCALING = enum.auto()


def naive_dct(x0):
    """Naive scipy.fftpack block DCT implementation that mimics the JPEG DCT calculation

    :param x0: pixel values of pre-cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: unquantized_coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    block_size = 8

    x0 = x0.astype(float) - 128
    height, width = x0.shape
    assert height % block_size == 0 and width % block_size == 0, \
        f'No support for padding (image dimensions must be divisible by the {block_size})'

    height, width = height // block_size, width // block_size
    dct_coeffs = np.zeros((height, width, block_size, block_size))
    for x in range(height):
        for y in range(width):
            block = x0[x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size]
            dct_coeffs[x, y] = tools.dct.dct2(block)

    return dct_coeffs


def compute_unquantized_coefficients(
        x0: np.ndarray,
        qt: np.ndarray,
        method: Method = Method.LIBJPEG_ISLOW
) -> np.array:
    """Compute the unquantized coefficients of the pre-cover. Either by using a naive DCT implementation
    or by extracting them directly from libjpeg.

    :param x0: pixel values of pre-cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param method: choose the method that is used to extract the unquantized dct coefficients
    :return: unquantized_coefficients divided by the quantization table,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    if method == Method.LIBJPEG_ISLOW:
        jpeg = jpeglib.from_spatial(x0[:, :, None])
        jpeg.samp_factor = "4:4:4"
        unquantized_coefficients = jpeg.unquantized_coefficients(dct_method=DCTMethod.JDCT_ISLOW)[0]
    elif method == Method.NAIVE_DCT:
        unquantized_coefficients = naive_dct(x0)
    else:
        raise ValueError("DCT method not supported in this version.")

    return unquantized_coefficients / qt


def compute_cost_adjusted(
    x0: np.ndarray,
    y0: np.ndarray,
    qt: np.ndarray,
    cost_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = None,
    *,
    method: Method = Method.LIBJPEG_ISLOW,
    midpoint_handling = MidpointHandling.CLIP_COST_SCALING,
    min_qe_scale_factor: float = 10 ** -3,
    wet_cost: float = 10**13,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute the adjusted cost for J-SIDEINFO tenary embedding.

    :param x0: pixel values of pre-cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param y0: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cost_fn: Cost function used for calculating the base cost. It provides access to the
        unquntized coefficiets, which therefore can be integrated in the cost calculation.
    :type cost_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    :param method: choose the method that is used to extract the unquantized dct coefficients
    :param midpoint_handling: choose the midpoint (qe == 0.5) handling strategy
    :type midpoint_handling: MidpointHandling
    :param min_qe_scale_factor: Limits the downscaling of the cost.
    :type min_qe_scale_factor: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: embedding costs of +1 and -1 changes,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> with Image.open(input_path) as img:
    ...    spatial = np.expand_dims(np.array(img.convert("L"), "uint8"), axis=-1)
    ...    jpeg = jpeglib.from_spatial(spatial)
    ...    jpeg.write_spatial(output_file_name, qt=quality)
    ...    jpeg = jpeglib.read_dct(output_file_name)
    ...    rho_p1, rho_m1 = cl.jsideinfo.compute_cost_adjusted(
    ...        x0=spatial[..., 0],
    ...        y0=jpeg.Y,
    ...        qt=jpeg.qt[0],
    ...        method=Method.LIBJPEG_ISLOW
    ...    )
    """
    # Count the number of embeddable DCT coefficients
    assert tools.dct.nzAC(y0) > 0, 'Expected non-zero AC coefficients'

    unquantized_coefficients = compute_unquantized_coefficients(x0, qt, method=method)
    qe = unquantized_coefficients - y0

    if cost_fn is None:
        def default_cost(uc):
            return juniward.compute_cost_adjusted(x0, uc, qt)
        cost_fn = default_cost

    rho_p1, rho_m1 = cost_fn(unquantized_coefficients)

    scaling_p1 = 1 - 2 * np.abs(qe[qe > 0])
    scaling_m1 = 1 - 2 * np.abs(qe[qe < 0])

    if midpoint_handling == MidpointHandling.OMIT_EMBEDDING:
        rho_p1[qe > 0] = np.where(scaling_p1 > min_qe_scale_factor, rho_p1[qe > 0] * scaling_p1, wet_cost)
        rho_m1[qe < 0] = np.where(scaling_m1 > min_qe_scale_factor, rho_m1[qe < 0] * scaling_m1, wet_cost)
    else:
        # Costs, which approach 0, cause over-embedding and increase detectability for small embedding rates
        # Also prevents cost from getting negative because qe can be slightly higher than 0.5, due to
        # different precisions
        rho_p1[qe > 0] *= np.maximum(scaling_p1, min_qe_scale_factor)
        rho_m1[qe < 0] *= np.maximum(scaling_m1, min_qe_scale_factor)

    # Do not embed +1 if the DCT coefficient has max value
    rho_p1[y0 >= 1023] = wet_cost
    # Do not embed -1 if the DCT coefficient has min value
    rho_m1[y0 <= -1023] = wet_cost

    return rho_p1, rho_m1
