import numpy as np

#coefficients for interpolation
def cr_coeff(t, ndim):
    hshape = t.shape + (1,) * ndim
    h0 = (2*t**3 - 3*t**2 + 1).reshape(hshape)
    h1 = (t**3 - 2*t**2 + t).reshape(hshape)
    h2 = (-2*t**3 + 3*t**2).reshape(hshape)
    h3 = (t**3 - t**2).reshape(hshape)
    return h0, h1, h2, h3

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
    #if y.ndim == 1:
    #    y = y[:,None]
    intp = np.asarray(intp)

    #account for a scalar intp input
    scalar_input = (intp.ndim == 0)
    intp = np.atleast_1d(intp)

    N = len(points) #number of sample points on the grid

    #ensure sampled points are sorted
    idx = np.argsort(points)
    points = points[idx]
    y = y[idx]

    #spacing of input points
    dpoints = np.diff(points)
    dx = dpoints[0]

    #ensure interpolation can be performed
    if N < 3:
        raise ValueError("athena_interp requires at least 3 points")
    if not np.allclose(dpoints, dx):
        raise ValueError("Interpolation must be over regularly sampled grid")

    #bounds
    lo, hi = points[0], points[-1]
    valid = (intp >= lo) & (intp <= hi)

    #interval indices
    i = np.searchsorted(points, intp) - 1

    #base indices
    i0 = i - 1
    i1 = i
    i2 = i + 1
    i3 = i + 2
    
    #four cases: exact match, left edge, middle (normal), right edge
    exact_mat = np.isclose(intp[:, None], points[None, :])
    exact = exact_mat.any(axis=1)
    left = (i == 0) & (~exact)
    right = (i == N - 2) & (~exact)
    middle = (i > 0) & (i < N - 2) & (~exact)

    outshape = (len(intp),) + y.shape[1:]
    yinterp = np.full(outshape, np.nan, dtype=float) #empty interpolated points array

    #Left Case Interpolation
    if np.any(left):
        m1 = (y[i2[left]] - y[i1[left]]) / 2
        m2 = (y[i3[left]] - y[i1[left]]) / 2
        y0 = y[i1[left]]
        y1 = y[i2[left]]
        dxl = (intp[left] - points[i1[left]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxl, y0.ndim - 1)
        yinterp[left] = h0 * y0 + h1 * m1 + h2 * y1 + h3 * m2

    #Middle Case Interpolation (Traditional Catmull-Rom Formula)
    if np.any(middle):
        m1 = (y[i2[middle]] - y[i0[middle]]) / 2
        m2 = (y[i3[middle]] - y[i1[middle]]) / 2
        y0 = y[i1[middle]]
        y1 = y[i2[middle]]
        dxm = (intp[middle] - points[i1[middle]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxm, y0.ndim - 1)
        yinterp[middle] = h0 * y0 + h1 * m1 + h2 * y1 + h3 * m2

    #Right Case
    if np.any(right):
        m1 = (y[i2[right]] - y[i0[right]]) / 2
        m2 = (y[i2[right]] - y[i1[right]]) / 2
        y0 = y[i1[right]]
        y1 = y[i2[right]]
        dxr = (intp[right] - points[i1[right]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxr, y0.ndim - 1)
        yinterp[right] = h0 * y0 + h1 * m1 + h2 * y1 + h3 * m2

    #accounting for exact matches between intp and points
    exact_idx = exact_mat.argmax(axis=1)
    if np.any(exact):
        yinterp[exact] = y[exact_idx[exact]]
    yinterp[~valid] = np.nan
    if scalar_input:
        return yinterp[0]
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
    ndim = len(points)
    if xi.shape[-1] != ndim:
        raise ValueError("Sample points are incorrectly entered.")
    
    orig_shape = xi.shape[:-1]
    xi_flat = xi.reshape(-1, ndim)

    for dim in reversed(range(ndim)):
        out = np.moveaxis(out, dim, 0)
        out = athena_interp_1d(points[dim], out, xi_flat[:, dim])
        out = np.moveaxis(out, 0, dim)
    out = np.asarray(out)
    out = out.reshape(orig_shape + out.shape[1:])
    return out
