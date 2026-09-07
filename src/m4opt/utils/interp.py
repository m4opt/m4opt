import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve


def athena_interp(points, values, xi):
    #
    # FIXME for Athena: fill in this function with your own Catmull-Rom
    # interpolation code.
    #
    # The spline coefficients are solved directly. The default iterative solver
    # stops on an absolute tolerance, which misses the grid values by about
    # 1e-5 and underflows to zero for values smaller than that.
    return RegularGridInterpolator(
        points,
        np.asarray(values, dtype=float),
        method="cubic",
        bounds_error=False,
        solver=spsolve,
    )(xi)
