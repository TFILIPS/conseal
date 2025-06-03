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
