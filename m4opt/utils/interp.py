import numpy as np

def athena_interp_1d(points, y, intp):
    """
    Perform Catmull-Rom interpolation on a regularly sampled series.

    Parameters
    ----------
    points : the integer values at which a function is sampled 
    y : numpy.ndarray
        The function f(x) sampled at regularly spaced values (points), such that
        y[0] = f(points[0]), y[1] = f(points[1]), etc.
    intp : numpy.ndarray
        The abscissae at which to evaluate the interpolant.

    Returns
    -------
    yinterp : numpy.ndarray
        The interpolated function, f(t)
    """
    
    #make everything into numpy arrays
    points = np.asarray(points)
    y = np.asarray(y)
    intp = np.asarray(intp)

    scalar_input = intp.ndim == 0
    intp = np.atleast_1d(intp)

    #sort input arrays
    idx = np.argsort(points)
    points = points[idx]
    y = np.take(y, idx, axis=0)

    N = len(points)

    #spacing of input points
    dx = np.diff(points)[0]

    #ensure interpolation can be performed
    if N < 3:
        raise ValueError("athena_interp requires at least 3 points")
    if not np.allclose(np.diff(points), np.diff(points)[0]):
        raise ValueError("Interpolation must be over regularly sampled grid")
     
    #bounds
    lo, hi = points[0], points[-1]

    #interval indices
    i = np.searchsorted(points, intp) - 1
    i = np.clip(i, 0, N - 2)
    valid = (intp >= lo) & (intp <= hi)

    #base indices
    i0 = i - 1
    i1 = i
    i2 = i + 1
    i3 = i + 2
    
    #three cases: left edge, middle (normal), right edge
    left = (i == 0)
    right = (i >= N - 2)
    middle = ~(left | right)

    yinterp = np.full(intp.shape + y.shape[1:], np.nan, dtype=float) 

    #Left Case Interpolation
    if np.any(left):
        m1 = (y[i2[left]] - y[i1[left]]) / dx
        m2 = (y[i3[left]] - y[i1[left]]) / (2 * dx)
        y0 = y[i1[left]]
        y1 = y[i2[left]]
        dxl = (intp[left] - points[i1[left]]) / (points[i2[left]] - points[i1[left]])

        yinterp[left] = (
            (2*dxl**3 - 3*dxl**2 + 1)[...,None] * y0
            + (dxl**3 - 2*dxl**2 + dxl)[...,None] * (m1 * dx)
            + (-2*dxl**3 + 3*dxl**2)[...,None] * y1
            + (dxl**3 - dxl**2)[...,None] * (m2 * dx)
        )

    #Middle Case --> Traditional Catmull-Rom Formula
    if np.any(middle):
        m1 = (y[i2[middle]] - y[i0[middle]]) / (2 * dx)
        m2 = (y[i3[middle]] - y[i1[middle]]) / (2 * dx)
        y0 = y[i1[middle]]
        y1 = y[i2[middle]]
        dxm = (intp[middle] - points[i1[middle]]) / (points[i2[middle]] - points[i1[middle]])

        yinterp[middle] = (
            (2*dxm**3 - 3*dxm**2 + 1)[...,None] * y0
            + (dxm**3 - 2*dxm**2 + dxm)[...,None] * (m1 * dx)
            + (-2*dxm**3 + 3*dxm**2)[...,None] * y1
            + (dxm**3 - dxm**2)[...,None] * (m2 * dx)
        )

    #Right Case
    if np.any(right):
        m1 = (y[i2[right]] - y[i0[right]]) / (2 * dx)
        m2 = (y[i2[right]] - y[i1[right]]) / dx
        y0 = y[i1[right]]
        y1 = y[i2[right]]
        dxr = (intp[right] - points[i1[right]]) / (points[i2[right]] - points[i1[right]])

        yinterp[right] = (
            (2*dxr**3 - 3*dxr**2 + 1)[...,None] * y0
            + (dxr**3 - 2*dxr**2 + dxr)[...,None] * (m1 * dx)
            + (-2*dxr**3 + 3*dxr**2)[...,None] * y1
            + (dxr**3 - dxr**2)[...,None] * (m2 * dx)
        )

    #accounting for exact matches between intp and points
    exact = np.isclose(intp, points[i1])
    yinterp[exact] = y[i1[exact]] 
    #i1 is not allowed to be N - 1 so the below accounts for an exact match of the last value in points
    yinterp[np.isclose(intp, points[-1])] = np.take(y, -1, axis=0)    
    yinterp[~valid] = np.nan
    if scalar_input:
        return np.squeeze(yinterp)
    return yinterp


def athena_interp(points, values, xi):
    """
    N-Dimensional Catmull-Rom interpolation.

    Parameters
    ----------
    points : list of n arrays [x_points, y_points, ...]
    values : N-D array (Nx, Ny, ...)
    xi : (..., n) array of query points

    Returns
    -------
    interpolated values at xi
    """
    
    xi = np.asarray(xi)
    out = values

    #interpolates across each dimension of the input xi
    for dim in reversed(range(len(points))):
        out = np.moveaxis(out, dim, 0)
        out = athena_interp_1d(points[dim], out, xi[..., dim])
        out = np.moveaxis(out, 0, dim)

    return out
