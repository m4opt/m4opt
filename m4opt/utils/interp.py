import scipy.interpolate


def athena_interp(points, values, xi):
    #
    # FIXME for Athena: fill in this function with your own Catmull-Rom
    # interpolation code.
    #
    return scipy.interpolate.interpn(
        points,
        values,
        xi,
        method="linear",
        bounds_error=False,
    )
