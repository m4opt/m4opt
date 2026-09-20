import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve


def athena_interp(points, values, xi):
    #
    # FIXME for Athena: fill in this function with your own Catmull-Rom
    # interpolation code.
    #
    # The default iterative solver stops on an absolute tolerance.
    return RegularGridInterpolator(
        points,
        np.asarray(values, dtype=float),
        method="cubic",
        bounds_error=False,
        solver=spsolve,
    )(xi)
