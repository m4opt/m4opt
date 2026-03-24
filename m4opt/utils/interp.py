import numpy as np

def athena_interp_1d(points, y, intp):
    """
    Perform Catmull-Rom interpolation on a regularly sampled series.

    Parameters
    ----------
    points : the integer values at which a function is sampled 
    y : numpy.ndarray
        The function f(x) sampled at integer values (points), such that
        y[0] = f(0), y(1) = f(1), etc.
    intp : numpy.ndarray
        The abscissae at which to evaluate the interpolant.

    Returns
    -------
    yinterp : numpy.ndarray
        The interpolated function, f(t)
    """
    
    points = np.asarray(points)
    y = np.asarray(y)
    intp = np.asarray(intp)
    scalar_input = intp.ndim == 0
    intp = np.atleast_1d(intp)
    idx = np.argsort(points)
    points = points[idx]
    y = np.take(y, idx, axis=0)

    N = len(points)
    if N < 3:
        raise ValueError("athena_interp requires at least 3 points")
    if not np.allclose(np.diff(points), np.diff(points)[0]):
        raise ValueError("Interpolation must be over regularly sampled grid")
    if np.any(np.diff(points) == 0):
        raise ValueError("points must be strictly increasing")
    
    # bounds
    lo = points[0]
    hi = points[-1]

    out = (intp < lo) | (intp > hi)

    # find interval indices
    t_low = np.searchsorted(points, intp, side="right") - 1
    t_low = np.clip(t_low, 0, N - 2)

    y_shape = y.shape[1:]
    yinterp = np.full(intp.shape + y_shape, np.nan, dtype=float)
    exact = np.isclose(intp, points[t_low])    
    if np.any(exact):
        vals = np.take(y, t_low[exact], axis=0)
        yinterp[exact] = vals
        
    #Bounds for Left Case, Middle Case, Right Case
    not_exact = ~exact
    left = (t_low >= 0) & (t_low < 1) & not_exact
    middle = (t_low >= 1) & (t_low <= N - 3) & not_exact
    right = (t_low > N - 3) & (t_low < N - 1) & not_exact

    #Left Case
    if np.any(left):
        t_il = t_low[left]
        t0 = points[t_il]
        t1 = points[t_il + 1]
        t_dl = ((intp[left] - t0) / (t1 - t0))
        m1 = (y[t_il + 1] - y[t_il]) 
        m2 = (y[t_il + 2] - y[t_il]) / 2
        y0 = y[t_il]
        y1 = y[t_il + 1]

        yinterp[left] = (
            (2*t_dl**3 - 3*t_dl**2 + 1) * y0
            + (t_dl**3 - 2*t_dl**2 + t_dl) * m1
            + (-2*t_dl**3 + 3*t_dl**2) * y1
            + (t_dl**3 - t_dl**2) * m2
        )

    #Middle Case --> Traditional Catmull-Rom Formula
    if np.any(middle):
        t_im = t_low[middle]
        t0 = points[t_im]
        t1 = points[t_im + 1]
        t_dm = ((intp[middle] - t0) / (t1 - t0))
        m1 = (y[t_im + 1] - y[t_im - 1]) / 2
        m2 = (y[t_im + 2] - y[t_im]) / 2

        y0 = y[t_im]
        y1 = y[t_im + 1]

        yinterp[middle] = (
            (2*t_dm**3 - 3*t_dm**2 + 1) * y0
            + (t_dm**3 - 2*t_dm**2 + t_dm) * m1
            + (-2*t_dm**3 + 3*t_dm**2) * y1
            + (t_dm**3 - t_dm**2) * m2
        )

    #Right Case
    if np.any(right):
        t_ir = t_low[right]
        t0 = points[t_ir]
        t1 = points[t_ir + 1]
        t_dr = ((intp[right] - t0) / (t1 - t0))
        m1 = (y[t_ir + 1] - y[t_ir - 1]) / 2
        m2 = (y[t_ir + 1] - y[t_ir])
        y0 = y[t_ir]
        y1 = y[t_ir + 1]

        yinterp[right] = (
            (2*t_dr**3 - 3*t_dr**2 + 1) * y0
            + (t_dr**3 - 2*t_dr**2 + t_dr) * m1
            + (-2*t_dr**3 + 3*t_dr**2) * y1
            + (t_dr**3 - t_dr**2) * m2
        )

    yinterp[np.isclose(intp, points[-1])] = np.take(y, -1, axis=0)    
    yinterp[out] = np.nan
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
    dim = len(points)

    if dim == 1:
        return athena_interp_1d(points[0], values, xi[...,0])
    
    results = []

    for i in range(values.shape[0]):
        sub_values = values[i]
        interp = athena_interp(points[1:], sub_values, xi[..., 1:])
        results.append(interp)

    results = np.stack(results, axis=0)  
    out = athena_interp_1d(points[0], results, xi[..., 0])
    return out.reshape(xi.shape[:-1])
