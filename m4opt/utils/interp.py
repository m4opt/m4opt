import numpy as np 

def athena_interp_1d(tensors, t):
    """
    1-Dimensional Catmull-Rom interpolation.

    Parameters
    ----------
    tensors : list of values associated with 4 points neighboring query points
    t : normalized positions of query points between two closest points

    Returns
    -------
    interpolated values at t
    """
    t2 = t * t
    t3 = t2 * t
    h0, h1, h2, h3 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t, -2*t3 + 3*t2, t3 - t2
    y0, y1, y2, y3 = tensors[:, 0], tensors[:, 1], tensors[:, 2], tensors[:, 3]
    m1 = (y2 - y0) / 2 #tangent 1
    m2 = (y3 - y1) / 2 #tangent 2
    intp = h0 * y1 + h1 * m1 + h2 * y2 + h3 * m2 #Catmull-Rom Formula
    return intp

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
    #array everything
    points = [np.asarray(p) for p in points]
    values = np.asarray(values)
    xi = np.asarray(xi)
    
    xi_orig_shape = xi.shape 
    dim = len(points) #dimension of points

    if xi.ndim == 1 and dim == 1:
        xi_flat = xi[:, np.newaxis]
    elif xi.ndim == 1 and dim > 1:
        xi_flat = xi[np.newaxis, :]
    else: 
        xi_flat = xi.reshape(-1, dim)

    num_queries = xi_flat.shape[0] #number of query points
    mins = [] #minima by axis of points
    maxs = [] #maxima by axis of points
    t_list = [] #list of normalized positions between 2 closest points
    idx_arr = [] #array of tensor indices
    offsets = np.arange(-2, 2) 
    
    for i in range(dim):
        p_i = points[i]
        if len(p_i) < 3:
            raise ValueError("At least 3 points in each dimension are required.")
        dpoints = np.diff(p_i)
        dx = dpoints[0]
        if not np.allclose(dpoints, dx):
            raise ValueError("Interpolation must be over regularly sampled grid.")
        mins.append(p_i[0])
        maxs.append(p_i[-1])
        idx_i = np.searchsorted(p_i, xi_flat[:, i])  #index of query point on axis
        idx_i_clip = np.clip(idx_i, 1, len(p_i) - 1)
        t_i = (xi_flat[:, i] - p_i[idx_i_clip - 1]) / dx #normalize position of query point
        t_list.append(t_i)
        grid_i = idx_i_clip[:, np.newaxis] + offsets #indices of all neighboring points
        grid_i = np.clip(grid_i, 0, values.shape[i] - 1)
        target_shape = [num_queries] + [1] * dim #target shape of 1s
        target_shape[i + 1] = 4 #replace the given axis's position with 4
        idx_arr.append(grid_i.reshape(target_shape)) #reshape and store
    
    out_of_bounds = np.any(xi_flat < mins, axis=1) | np.any(xi_flat > maxs, axis=1)
    tens = np.array(values[tuple(idx_arr)]) #tensor array

    for i in reversed(range(dim)):
        t = t_list[i] 
        remaining_spatial_axes = i 
        t_expanded = t[(slice(None),) + (None,) * remaining_spatial_axes]
        t_broadcasted = np.broadcast_to(t_expanded, tens.shape[:-1])
        tensors_2d = tens.reshape(-1, 4)
        t_1d = t_broadcasted.ravel()
        interp_flat = athena_interp_1d(tensors_2d, t_1d)
        tens = interp_flat.reshape(tens.shape[:-1])
    if not np.issubdtype(tens.dtype, np.floating):
        tens = tens.astype(np.float64)
    tens[out_of_bounds] = np.nan
    if len(xi_orig_shape) == 1:
        return tens.reshape(1)
    
    out_shape = xi_orig_shape[:-1] + values.shape[dim:]
    return tens.reshape(out_shape if out_shape else ())
