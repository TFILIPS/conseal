"""

Author: Benedikt Lorch, Martin Benes, Thomas Filips
Affiliation: University of Innsbruck
"""

import numpy as np

from .. import simulate
from .. import tools


def simulate(
    x0: np.ndarray,
    alpha: float,
    *,
    wet_cost: float = 10**13,
    dtype: np.dtype = np.float64,
    generator: str = None,
    seed: int = None,
) -> np.ndarray:
    # Count number of embeddable DCT coefficients
    nzAC = tools.dct.nzAC(y0)

    if nzAC == 0:
        raise ValueError('Expected non-zero AC coefficients')

    # Compute cost for embedding into the quantized DCT coefficients
    # of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    rho_p1, rho_m1 = compute_cost_adjusted(
        x0=x0,
        y0=y0,
        qt=qt,
        dtype=dtype,
        implementation=implementation,
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
