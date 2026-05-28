import numpy as np

#coefficients for interpolation
def cr_coeff(t):
    t2 = t * t
    t3 = t2 * t
    h0 = 2*t3 - 3*t2 + 1
    h1 = t3 - 2*t2 + t 
    h2 = -2*t3 + 3*t2
    h3 = t3 - t2
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
    intp = np.asarray(intp)

    #account for a scalar intp input
    scalar_input = (intp.ndim == 0)
    intp = np.atleast_1d(intp)

    N = len(points) #number of sample points on the grid

    if y.ndim == 1:
        y = y[:,None]

    #ensure sampled points are sorted
    if not np.all(points[:-1] <= points[1:]):
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
    exact_idx = np.clip(i + 1, 0, N - 1)
    exact = valid & (points[exact_idx] == intp)
    left = valid & (i == 0) & (~exact)
    right = valid & (i == N - 2) & (~exact)
    middle = valid & (i > 0) & (i < N - 2) & (~exact)

    yinterp = np.full((len(intp), y.shape[-1]), np.nan, dtype=float) #empty interpolated points array

    #Left Case Interpolation
    if np.any(left):
        m1 = (y[i2[left]] - y[i1[left]]) / 2
        m2 = (y[i3[left]] - y[i1[left]]) / 2
        y0 = y[i1[left]]
        y1 = y[i2[left]]
        dxl = (intp[left] - points[i1[left]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxl)
        yinterp[left] = h0[:, None] * y0 + h1[:, None] * m1 + h2[:, None] * y1 + h3[:, None] * m2

    #Middle Case Interpolation (Traditional Catmull-Rom Formula)
    if np.any(middle):
        m1 = (y[i2[middle]] - y[i0[middle]]) / 2
        m2 = (y[i3[middle]] - y[i1[middle]]) / 2
        y0 = y[i1[middle]]
        y1 = y[i2[middle]]
        dxm = (intp[middle] - points[i1[middle]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxm)
        yinterp[middle] = h0[:, None] * y0 + h1[:, None] * m1 + h2[:, None] * y1 + h3[:, None] * m2

    #Right Case
    if np.any(right):
        m1 = (y[i2[right]] - y[i0[right]]) / 2
        m2 = (y[i2[right]] - y[i1[right]]) / 2
        y0 = y[i1[right]]
        y1 = y[i2[right]]
        dxr = (intp[right] - points[i1[right]]) / dx
        h0, h1, h2, h3 = cr_coeff(dxr)
        yinterp[right] = h0[:, None] * y0 + h1[:, None] * m1 + h2[:, None] * y1 + h3[:, None] * m2

    #accounting for exact matches between intp and points
    if np.any(exact):
        yinterp[exact] = y[exact_idx[exact]]
    yinterp[~valid] = np.nan
    return yinterp[0] if scalar_input else yinterp

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
    ndim = len(points)

    if xi.shape[-1] != ndim:
        raise ValueError("Sample points are incorrectly entered.")
    
    orig_shape = xi.shape[:-1]
    xi_flat = xi.reshape(-1, ndim)
    M = len(xi_flat)
    
    out = np.asarray(values, dtype=float)

    for dim in range(ndim - 1, -1, -1):
        out = np.moveaxis(out, dim, 0)
        grid_size = out.shape[0]
        rest_shape = out.shape[1:]
        out = out.reshape(grid_size, -1)
        t = xi_flat[:, dim]
        out = athena_interp_1d(points[dim], out, t)
        out = out.reshape((M,) + rest_shape)
        out = np.moveaxis(out, 0, dim)
    idx = (np.arange(M),) * ndim
    out = out[idx]
    return out.reshape(orig_shape)
