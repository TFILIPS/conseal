"""
Implementation of the J-UNIWARD steganography method as described in

V. Holub, J. Fridrich, T. Denemark
"Universal distortion function for steganography in an arbitrary domain"
EURASIP Journal on Information Security, 2014
http://www.ws.binghamton.edu/fridrich/research/uniward-eurasip-final.pdf

Author: Benedikt Lorch, Martin Benes, Thomas Filips
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find that license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2013 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
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


def naive_dct(x0):
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


def compute_unquantized_coefficients(x0, qt, method: Method = Method.LIBJPEG_ISLOW) -> np.array:
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
    dry_cost: float = 50.0,
    wet_cost: float = 10**13,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Count the number of embeddable DCT coefficients
    assert tools.dct.nzAC(y0) > 0, 'Expected non-zero AC coefficients'

    unquantized_coefficients = compute_unquantized_coefficients(x0, qt, method=method)
    qe = unquantized_coefficients - y0

    if cost_fn is None:
        def default_cost(uc):
            return juniward.compute_cost_adjusted(x0, uc, qt)
        cost_fn = default_cost

    rho_p1, rho_m1 = cost_fn(unquantized_coefficients)

    rho_p1[qe > 0] *= (1 - 2 * np.abs(qe[qe > 0]))
    # Costs, which approach 0, cause over-embedding and increase detectability for small embedding rates
    rho_p1[rho_p1 < dry_cost] = dry_cost
    # Do not embed +1 if the DCT coefficient has max value
    rho_p1[y0 >= 1023] = wet_cost

    rho_m1[qe < 0] *= (1 - 2 * np.abs(qe[qe < 0]))
    # Costs, which approach 0, cause over-embedding and increase detectability for small embedding rates
    rho_m1[rho_m1 < dry_cost] = dry_cost
    # Do not embed -1 if the DCT coefficient has min value
    rho_m1[y0 <= -1023] = wet_cost

    return rho_p1, rho_m1
